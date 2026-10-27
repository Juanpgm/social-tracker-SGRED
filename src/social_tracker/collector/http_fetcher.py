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
- Only the ports in `allowed_ports` (default 80, 443, 8080, 8443) are contacted, redirects
  included; a feed has no business on a database or mail port.
- Redirects are followed by hand, at most `max_redirects`, and each hop is validated and pinned
  again: a redirect to a private address, another scheme or a disguised numeric host fails. A
  redirect from `https` to `http` is refused (no downgrade), and the conditional-request
  validators (`If-None-Match`, `If-Modified-Since`) are sent only to the authority the fetch
  started at, never to a host or port a redirect led to.
- The body is capped at `max_bytes` AFTER decompression (only `gzip` is requested; a
  decompression bomb stops at the cap).
- The whole fetch has a wall-clock deadline (`deadline_s`) that is enforced INSIDE the I/O, not
  just between steps: every connect, TLS handshake, read and write gets at most the budget that
  is left (a per-read timeout alone restarts on every byte, so a peer sending one byte at a time
  would hold the worker for hours), and name resolution runs in a helper thread that is given up
  on at the deadline. A redirect chain, all the addresses of a host and the body share the one
  budget. A fetch therefore ends within `deadline_s` plus a small scheduling slack; the helper
  thread of a lookup the system never answers is a daemon and at most `MAX_PENDING_LOOKUPS` of
  them are tolerated at once. The response head is capped at 100 KiB by the HTTP layer.
- An operator may allow named private hosts (`private_hosts`); every use is logged with the host
  name only. It is off by default.
- Every failure is a `FetchError` with a fixed message: no URL, header or body text escapes.
"""

import ipaddress
import logging
import math
import socket
import ssl
import threading
import time
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import httpcore2
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
DEFAULT_ALLOWED_PORTS = frozenset({80, 443, 8080, 8443})
MAX_PENDING_LOOKUPS = 8
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
    """Bounds of one fetch. `timeout_s` caps each network operation; `deadline_s` is the wall-clock
    budget of the WHOLE fetch (name lookup, redirects, headers and body included), enforced inside
    every operation, so a fetch ends within `deadline_s` however slowly the peer sends."""

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


def _bare(request: FetchRequest) -> FetchRequest:
    """The same request without conditional-request validators."""
    return FetchRequest(request.url)


@dataclass(frozen=True)
class _Redirect:
    url: str


class _DeadlineStream(httpcore2.NetworkStream):
    """A connection whose every operation gets at most the budget that is left."""

    def __init__(self, inner: httpcore2.NetworkStream, remaining: Callable[[], float]) -> None:
        self._inner = inner
        self._remaining = remaining

    def _budget(self, timeout: float | None, expired: type[Exception]) -> float:
        left = self._remaining()
        if left <= 0:
            raise expired("fetch deadline reached")
        return left if timeout is None else min(timeout, left)

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._inner.read(max_bytes, self._budget(timeout, httpcore2.ReadTimeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._inner.write(buffer, self._budget(timeout, httpcore2.WriteTimeout))

    def close(self) -> None:
        self._inner.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore2.NetworkStream:
        budget = self._budget(timeout, httpcore2.ConnectTimeout)
        wrapped = self._inner.start_tls(ssl_context, server_hostname, budget)
        return _DeadlineStream(wrapped, self._remaining)

    def get_extra_info(self, info: str) -> Any:
        return self._inner.get_extra_info(info)


class _DeadlineBackend(httpcore2.NetworkBackend):
    def __init__(self, inner: httpcore2.NetworkBackend, remaining: Callable[[], float]) -> None:
        self._inner = inner
        self._remaining = remaining

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore2.NetworkStream:
        left = self._remaining()
        if left <= 0:
            raise httpcore2.ConnectTimeout("fetch deadline reached")
        budget = left if timeout is None else min(timeout, left)
        stream = self._inner.connect_tcp(host, port, budget, local_address, socket_options)
        return _DeadlineStream(stream, self._remaining)


class _DeadlineTransport(httpx2.HTTPTransport):
    """The default HTTP transport with its connections wrapped in the deadline of one fetch."""

    def __init__(self, ssl_context: ssl.SSLContext, remaining: Callable[[], float]) -> None:
        super().__init__(verify=ssl_context, trust_env=False)
        self._pool.close()  # replaced below by a pool over the deadline-aware backend
        self._pool = httpcore2.ConnectionPool(
            ssl_context=ssl_context,
            network_backend=_DeadlineBackend(httpcore2.SyncBackend(), remaining),
            http2=False,
            retries=0,
        )


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
        allowed_ports: Iterable[int] | None = DEFAULT_ALLOWED_PORTS,
    ) -> None:
        self._clock = clock
        self._limits = limits or FetchLimits()
        self._resolver = resolver
        self._transport = transport
        self._private_hosts = frozenset(host.strip().lower() for host in private_hosts)
        self._monotonic = monotonic
        self._allowed_ports = None if allowed_ports is None else frozenset(allowed_ports)
        self._ssl_context = (
            httpx2.create_ssl_context(trust_env=False) if transport is None else None
        )
        self._lookup_slots = threading.BoundedSemaphore(MAX_PENDING_LOOKUPS)

    def fetch(self, request: FetchRequest) -> FetchResponse:
        started = self._monotonic()
        url = request.url
        with httpx2.Client(
            transport=self._transport_for(started),
            follow_redirects=False,
            trust_env=False,
            timeout=httpx2.Timeout(self._limits.timeout_s),
        ) as client:
            origin: FetchTarget | None = None
            for _ in range(self._limits.max_redirects + 1):
                self._check_deadline(started)
                target = parse_fetch_url(url)
                self._check_port(target)
                if origin is None:
                    origin = target
                elif origin.scheme == "https" and target.scheme == "http":
                    _LOG.warning("redirect from https to http refused (downgrade)")
                    raise FetchError(FetchErrorKind.INSECURE_REDIRECT)
                hop_request = request if target.authority == origin.authority else _bare(request)
                outcome = self._hop(
                    client, target, self._addresses(target, started), hop_request, started
                )
                if isinstance(outcome, FetchResponse):
                    return outcome
                url = outcome.url
        raise FetchError(FetchErrorKind.TOO_MANY_REDIRECTS)

    def _remaining(self, started: float) -> float:
        return self._limits.deadline_s - (self._monotonic() - started)

    def _transport_for(self, started: float) -> httpx2.BaseTransport:
        if self._transport is not None:
            return self._transport
        assert self._ssl_context is not None
        return _DeadlineTransport(self._ssl_context, lambda: self._remaining(started))

    def _check_port(self, target: FetchTarget) -> None:
        if self._allowed_ports is not None and target.port not in self._allowed_ports:
            raise FetchError(FetchErrorKind.BLOCKED_ADDRESS)

    def _check_deadline(self, started: float) -> None:
        if self._remaining(started) <= 0:
            raise FetchError(FetchErrorKind.TIMEOUT)

    def _lookup(self, host: str, port: int, started: float) -> list[str]:
        """The resolver's answer, or a timeout once the budget is spent. The lookup runs in a
        daemon thread because the system resolver cannot be interrupted."""
        left = self._remaining(started)
        if left <= 0:
            raise FetchError(FetchErrorKind.TIMEOUT)
        if not self._lookup_slots.acquire(blocking=False):
            raise FetchError(FetchErrorKind.DNS)  # earlier lookups are still hanging
        outcome: list[tuple[bool, Any]] = []

        def work() -> None:
            try:
                outcome.append((True, self._resolver(host, port)))
            except BaseException as exc:  # noqa: BLE001 - handed to the caller below
                outcome.append((False, exc))
            finally:
                self._lookup_slots.release()

        thread = threading.Thread(target=work, name="social-tracker-dns", daemon=True)
        thread.start()
        thread.join(left)
        if not outcome:
            raise FetchError(FetchErrorKind.TIMEOUT)
        succeeded, value = outcome[0]
        if not succeeded:
            raise value
        return value  # type: ignore[no-any-return]

    def _addresses(self, target: FetchTarget, started: float) -> list[str]:
        allow_private = target.host in self._private_hosts
        try:
            ipaddress.ip_address(target.host)
            answer: Iterable[str] = [target.host]
        except ValueError:
            answer = self._lookup(target.host, target.port, started)
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
            self._check_deadline(started)
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
