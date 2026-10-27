"""The HTTP fetcher: the ONLY module that opens network connections for the collector (G-1).

`httpx2` and `socket` may be imported here and nowhere else (an import-linter contract pins it).
What it guarantees, per request:

- Only `http`/`https` URLs without credentials; nothing is sent but a GET with a fixed
  `User-Agent`. No cookies, no `Authorization`, no `Referer`, and proxy settings from the
  environment are ignored (a proxy would see, and could redirect, the connection).
- Every host is resolved here and ALL its answers must be public addresses (`collector.netpolicy`).
  The connection then goes to the validated IP address, with the original name in the `Host`
  header and as the TLS server name, so the name is never resolved a second time and a changing
  DNS answer (rebinding) cannot lead the request elsewhere.
- Redirects are followed by hand, at most `max_redirects`, and each hop is validated and pinned
  again: a redirect to a private address, another scheme or a disguised numeric host fails.
- The body is capped at `max_bytes` AFTER decompression (only `gzip` is requested; a
  decompression bomb stops at the cap) and the whole fetch has a wall-clock deadline, so a
  server that trickles bytes cannot hold the worker.
- An operator may allow named private hosts (`private_hosts`); every use is logged with the host
  name only. It is off by default.
- Every failure is a `FetchError` with a fixed message: no URL, header or body text escapes.
"""

import ipaddress
import logging
import math
import socket
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import httpx2

from social_tracker.collector.domain import (
    FetchError,
    FetchErrorKind,
    FetchRequest,
    FetchResponse,
    parse_retry_after,
)
from social_tracker.collector.netpolicy import (
    FetchTarget,
    is_blocked_address,
    parse_fetch_url,
    redirect_target,
    usable_addresses,
)
from social_tracker.shared.clock import Clock

USER_AGENT = "social-tracker/0.1 (local observation; read-only)"
_ACCEPT = (
    "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5"
)
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_ADDRESSES_TRIED = 3
_ETAG_MAX = 512
_LAST_MODIFIED_MAX = 100
_LOG = logging.getLogger("social_tracker.collector")

type Resolver = Callable[[str, int], list[str]]


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


@dataclass(frozen=True)
class FetchLimits:
    """Bounds of one fetch. `timeout_s` applies to each network operation, `deadline_s` to the
    whole fetch (redirects and body included), so a fetch ends within about their sum."""

    timeout_s: float = 10.0
    deadline_s: float = 15.0
    max_bytes: int = 2 * 1024 * 1024
    max_redirects: int = 5

    def __post_init__(self) -> None:
        if not _positive_number(self.timeout_s) or not _positive_number(self.deadline_s):
            raise ValueError("timeout_s and deadline_s must be positive, finite numbers")
        if (
            isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(self.max_redirects, bool)
            or not isinstance(self.max_redirects, int)
            or self.max_redirects < 0
        ):
            raise ValueError("max_redirects must be a non-negative integer")


def system_resolver(host: str, port: int) -> list[str]:
    """The operating system's answer for `host`: distinct addresses in the order given."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        raise FetchError(FetchErrorKind.DNS) from None
    found: list[str] = []
    for info in infos:
        address = str(info[4][0])
        if address not in found:
            found.append(address)
    if not found:
        raise FetchError(FetchErrorKind.DNS)
    return found


def _translate(exc: Exception) -> FetchError:
    if isinstance(exc, httpx2.TimeoutException):
        return FetchError(FetchErrorKind.TIMEOUT)
    if isinstance(exc, httpx2.InvalidURL):
        return FetchError(FetchErrorKind.BAD_URL)
    return FetchError(FetchErrorKind.PROTOCOL)


def _validator(value: str | None, max_length: int) -> str | None:
    """A conditional-request validator worth keeping: short printable ASCII, else dropped."""
    text = (value or "").strip()
    if not text or len(text) > max_length or not all(32 <= ord(ch) < 127 for ch in text):
        return None
    return text


@dataclass(frozen=True)
class _Redirect:
    url: str


class HttpFetcher:
    def __init__(
        self,
        clock: Clock,
        *,
        limits: FetchLimits | None = None,
        resolver: Resolver = system_resolver,
        transport: httpx2.BaseTransport | None = None,
        private_hosts: Iterable[str] = (),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._limits = limits or FetchLimits()
        self._resolver = resolver
        self._transport = transport
        self._private_hosts = frozenset(host.strip().lower() for host in private_hosts)
        self._monotonic = monotonic

    def fetch(self, request: FetchRequest) -> FetchResponse:
        started = self._monotonic()
        url = request.url
        with httpx2.Client(
            transport=self._transport,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx2.Timeout(self._limits.timeout_s),
        ) as client:
            for _ in range(self._limits.max_redirects + 1):
                self._check_deadline(started)
                target = parse_fetch_url(url)
                outcome = self._hop(client, target, self._addresses(target), request, started)
                if isinstance(outcome, FetchResponse):
                    return outcome
                url = outcome.url
        raise FetchError(FetchErrorKind.TOO_MANY_REDIRECTS)

    def _check_deadline(self, started: float) -> None:
        if self._monotonic() - started > self._limits.deadline_s:
            raise FetchError(FetchErrorKind.TIMEOUT)

    def _addresses(self, target: FetchTarget) -> list[str]:
        allow_private = target.host in self._private_hosts
        try:
            ipaddress.ip_address(target.host)
            answer: Iterable[str] = [target.host]
        except ValueError:
            answer = self._resolver(target.host, target.port)
        addresses = usable_addresses(answer, allow_private=allow_private)
        if allow_private and any(is_blocked_address(ipaddress.ip_address(a)) for a in addresses):
            _LOG.warning("private address allowed by configuration for host %s", target.host)
        return addresses

    def _hop(
        self,
        client: httpx2.Client,
        target: FetchTarget,
        addresses: list[str],
        request: FetchRequest,
        started: float,
    ) -> FetchResponse | _Redirect:
        for address in addresses[:_MAX_ADDRESSES_TRIED]:
            try:
                return self._request_once(client, target, address, request, started)
            except httpx2.ConnectError:
                continue  # try the next validated address
            except (httpx2.HTTPError, httpx2.StreamError, httpx2.InvalidURL) as exc:
                raise _translate(exc) from None
        raise FetchError(FetchErrorKind.CONNECT)

    def _request_once(
        self,
        client: httpx2.Client,
        target: FetchTarget,
        address: str,
        request: FetchRequest,
        started: float,
    ) -> FetchResponse | _Redirect:
        shown = f"[{address}]" if ":" in address else address
        pinned_url = f"{target.scheme}://{shown}:{target.port}{target.target}"
        headers = {
            "Host": target.authority,
            "User-Agent": USER_AGENT,
            "Accept": _ACCEPT,
            "Accept-Encoding": "gzip",
        }
        if request.etag:
            headers["If-None-Match"] = request.etag
        if request.last_modified:
            headers["If-Modified-Since"] = request.last_modified
        extensions = {"sni_hostname": target.host} if target.scheme == "https" else {}
        with client.stream("GET", pinned_url, headers=headers, extensions=extensions) as response:
            client.cookies.clear()  # no cookie ever travels to a later hop
            return self._outcome(response, target, started)

    def _outcome(
        self, response: httpx2.Response, target: FetchTarget, started: float
    ) -> FetchResponse | _Redirect:
        status = response.status_code
        etag = _validator(response.headers.get("etag"), _ETAG_MAX)
        modified = _validator(response.headers.get("last-modified"), _LAST_MODIFIED_MAX)
        if status == 304:
            return FetchResponse(b"", etag, modified, not_modified=True)
        if status in _REDIRECT_STATUSES:
            location = response.headers.get("location", "")
            if not location.strip():
                raise FetchError.http(status)
            return _Redirect(redirect_target(target.url, location))
        if not 200 <= status < 300:
            wait = parse_retry_after(response.headers.get("retry-after"), self._clock.now())
            raise FetchError.http(status, retry_after_s=wait)
        return FetchResponse(self._read_body(response, started), etag, modified)

    def _read_body(self, response: httpx2.Response, started: float) -> bytes:
        limit = self._limits.max_bytes
        encoding = response.headers.get("content-encoding", "").strip().lower() or "identity"
        if encoding not in ("identity", "gzip"):
            raise FetchError(FetchErrorKind.DECODE)
        declared = response.headers.get("content-length", "").strip()
        if declared.isascii() and declared.isdigit() and int(declared) > limit:
            raise FetchError(FetchErrorKind.TOO_LARGE)
        decoder = zlib.decompressobj(zlib.MAX_WBITS | 16) if encoding == "gzip" else None
        parts: list[bytes] = []
        received = size = 0
        for chunk in response.iter_raw():
            self._check_deadline(started)
            received += len(chunk)
            if received > limit:
                raise FetchError(FetchErrorKind.TOO_LARGE)
            if decoder is None:
                parts.append(chunk)
                continue
            pending = chunk
            while pending:
                try:
                    produced = decoder.decompress(pending, limit - size + 1)
                except zlib.error:
                    raise FetchError(FetchErrorKind.DECODE) from None
                size += len(produced)
                if size > limit:
                    raise FetchError(FetchErrorKind.TOO_LARGE)
                parts.append(produced)
                pending = decoder.unconsumed_tail
        if decoder is not None and not decoder.eof:
            raise FetchError(FetchErrorKind.DECODE)
        return b"".join(parts)
