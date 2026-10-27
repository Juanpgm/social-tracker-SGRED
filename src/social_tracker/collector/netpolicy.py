"""Which addresses the collector may contact. Pure, no I/O (name resolution lives in the adapter).

Only plain http(s) URLs without credentials are fetched, and a host is contacted only if EVERY
address it resolves to is a public one: a name that answers with a loopback, private, link-local
(cloud metadata included), carrier-grade NAT, multicast, reserved or documentation address is
refused whole, so a mixed answer cannot be used to smuggle one in. IPv6 addresses that wrap an
IPv4 one (`::ffff:a.b.c.d`, the IPv4-translated `::ffff:0:a.b.c.d`, the deprecated IPv4-compatible
`::a.b.c.d`, NAT64 `64:ff9b::/96`, 6to4 `2002::/16`) are judged by the IPv4 address inside; the
deprecated site-local range `fec0::/10` is refused. Numeric spellings of an IPv4 address
(`2130706433`, `0x7f.1`, `127.1`) are refused as URLs rather than interpreted. The operator may
allow named private hosts (see the fetcher).
"""

import ipaddress
import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from social_tracker.collector.domain import FetchError, FetchErrorKind

MAX_URL_CHARS = 2048
_SCHEMES = {"http": 80, "https": 443}
_NUMERIC_HOST_RE = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*")
_LABEL_RE = re.compile(r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")
_IPV4_TRANSLATED = ipaddress.ip_network("::ffff:0:0:0/96")
_SITE_LOCAL = ipaddress.ip_network("fec0::/10")
type IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _bad() -> FetchError:
    return FetchError(FetchErrorKind.BAD_URL)


@dataclass(frozen=True)
class FetchTarget:
    scheme: str
    host: str  # lower-case, IDNA-encoded, without brackets
    port: int
    target: str  # path and query, always starting with "/"
    authority: str  # what the Host header carries: host[:port], port omitted when default

    @property
    def url(self) -> str:
        return f"{self.scheme}://{self.authority}{self.target}"


def _hostname(raw: str) -> str:
    if "%" in raw:  # an IPv6 zone identifier or an escape: never a public host name
        raise _bad()
    try:
        ipaddress.ip_address(raw)
        return raw
    except ValueError:
        pass
    if _NUMERIC_HOST_RE.fullmatch(raw):
        raise _bad()
    try:
        ascii_host = raw.encode("idna").decode("ascii")
    except UnicodeError:
        raise _bad() from None
    labels = ascii_host.removesuffix(".").split(".")
    if not ascii_host or len(ascii_host) > 253 or not all(_LABEL_RE.fullmatch(x) for x in labels):
        raise _bad()
    return ascii_host


def parse_fetch_url(url: object) -> FetchTarget:
    """Validate `url` and split it, or raise `FetchError(BAD_URL)`. The fragment is dropped."""
    if (
        not isinstance(url, str)
        or not url
        or len(url) > MAX_URL_CHARS
        or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url)
    ):
        raise _bad()
    try:
        parts = urlsplit(url)
        port = parts.port
        raw_host = parts.hostname
    except ValueError:
        raise _bad() from None
    if parts.scheme.lower() not in _SCHEMES or not raw_host or "@" in parts.netloc:
        raise _bad()
    scheme = parts.scheme.lower()
    if port is not None and not 1 <= port <= 65535:
        raise _bad()
    host = _hostname(raw_host.lower())
    chosen_port = port if port is not None else _SCHEMES[scheme]
    shown_host = f"[{host}]" if ":" in host else host
    authority = shown_host if chosen_port == _SCHEMES[scheme] else f"{shown_host}:{chosen_port}"
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    return FetchTarget(scheme, host, chosen_port, target, authority)


def redirect_target(current_url: str, location: str) -> str:
    """The absolute address a `Location` header points to, resolved against `current_url`."""
    if (
        not isinstance(location, str)
        or not location.strip()
        or len(location) > MAX_URL_CHARS
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in location)
    ):
        raise _bad()
    absolute = urljoin(current_url, location.strip())
    return absolute.partition("#")[0]


def _embedded_ipv4(address: IPAddress) -> ipaddress.IPv4Address | None:
    if isinstance(address, ipaddress.IPv4Address):
        return None
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address in _NAT64 or address in _IPV4_COMPATIBLE or address in _IPV4_TRANSLATED:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return address.sixtofour


def is_blocked_address(address: IPAddress) -> bool:
    """True unless `address` is a globally routable unicast address."""
    inner = _embedded_ipv4(address)
    if inner is not None:
        return is_blocked_address(inner)
    return not address.is_global or address.is_multicast or address in _SITE_LOCAL


def usable_addresses(answer: Iterable[str], *, allow_private: bool) -> list[str]:
    """The distinct addresses of a name-resolution answer, in order, if all of them may be used.

    An empty answer or one with an unparsable entry is a DNS failure; one blocked entry makes the
    whole answer `BLOCKED_ADDRESS` unless `allow_private` (an operator allowlist entry) is set.
    """
    found: list[str] = []
    for text in answer:
        if not isinstance(text, str) or "%" in text:
            raise FetchError(FetchErrorKind.DNS)
        try:
            address = ipaddress.ip_address(text)
        except ValueError:
            raise FetchError(FetchErrorKind.DNS) from None
        if not allow_private and is_blocked_address(address):
            raise FetchError(FetchErrorKind.BLOCKED_ADDRESS)
        canonical = str(address)
        if canonical not in found:
            found.append(canonical)
    if not found:
        raise FetchError(FetchErrorKind.DNS)
    return found
