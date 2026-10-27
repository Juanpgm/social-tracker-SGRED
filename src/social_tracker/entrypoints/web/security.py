"""Protections shared by every request of the web UI. Pure functions, no framework types.

- Host check: only the loopback names and the bound host are served, which defeats DNS rebinding
  (a page on another origin that resolves its own name to 127.0.0.1).
- CSRF: a random per-session cookie plus a form token that is an HMAC of the cookie under a
  per-process secret. A token is only valid together with the cookie it was derived from, so one
  copied from another session, or issued before a restart, is refused.
- Loopback enforcement for `serve`.
"""

import hmac
import ipaddress
import re
import secrets
from collections.abc import Iterable
from hashlib import sha256

from social_tracker.shared.errors import ValidationError

LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
CSRF_COOKIE_NAME = "st_csrf"
CSRF_FIELD = "csrf_token"
_COOKIE_SHAPE = re.compile(r"[A-Za-z0-9_-]{43}")  # token_urlsafe(32)

CSP = (
    "default-src 'self'; base-uri 'none'; object-src 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)
# Every browser feature the page could ask for is switched off: it needs none of them.
PERMISSIONS_POLICY = ", ".join(
    f"{feature}=()"
    for feature in (
        "accelerometer",
        "autoplay",
        "camera",
        "clipboard-read",
        "display-capture",
        "encrypted-media",
        "fullscreen",
        "geolocation",
        "gyroscope",
        "hid",
        "magnetometer",
        "microphone",
        "midi",
        "payment",
        "picture-in-picture",
        "publickey-credentials-get",
        "screen-wake-lock",
        "serial",
        "usb",
        "xr-spatial-tracking",
    )
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "Permissions-Policy": PERMISSIONS_POLICY,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def is_loopback(host: str) -> bool:
    """True for `localhost` and any loopback IP literal."""
    name = host.strip().strip("[]").lower()
    if name in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def check_bind_host(host: str, *, allow_non_loopback: bool) -> None:
    """Refuse to listen beyond this machine unless the operator said so explicitly (G-5)."""
    if not allow_non_loopback and not is_loopback(host):
        raise ValidationError(
            "refusing to listen on a non-loopback address: the UI has no login. "
            "Pass --allow-non-loopback only on a network you fully trust"
        )


# A Host header is `name[:port]` or `[ipv6][:port]`, nothing else: no spaces, no `@`, no path.
# `fullmatch` and `[0-9]` on purpose: `$` accepts a trailing newline and `\d` accepts any script.
_HOST_HEADER = re.compile(
    r"(?:\[(?P<v6>[0-9a-f:]+)\]|(?P<name>[a-z0-9.-]+))(?::(?P<port>[0-9]{1,5}))?", re.ASCII
)
_HOST_NAME = re.compile(r"[a-z0-9.-]+", re.ASCII)
_MAX_PORT = 65535


def _canonical(name: str) -> str:
    """An IP literal in its one canonical text form (`0:0::1` is `::1`); any other name as is."""
    try:
        return str(ipaddress.ip_address(name))
    except ValueError:
        return name


def host_name(header: str) -> str | None:
    """The canonical host of a `Host` header, or None when the header is not well formed.

    Strict on purpose: the value is lower-cased but never stripped, so ` localhost` and
    `localhost:80@evil.com` are not `localhost`. One trailing dot (`localhost.`) is the same name.
    """
    if not header.isascii():  # some non-ASCII characters lower-case to ASCII letters
        return None
    found = _HOST_HEADER.fullmatch(header.lower())
    if found is None:
        return None
    if found["port"] and int(found["port"]) > _MAX_PORT:
        return None
    if found["v6"]:
        try:
            address = ipaddress.IPv6Address(found["v6"])
        except ValueError:
            return None
        return str(address)
    name = found["name"].removesuffix(".")
    return _canonical(name) if name else None


def _configured_host(raw: str) -> str | None:
    """A host the operator names (`--host`, `--allowed-host`), in canonical form.

    Brackets around an IPv6 address are optional here (`::1` is what `--host` takes). Returns
    None for a wildcard address: `0.0.0.0` and `::` say "listen everywhere", they are not a name
    anybody should be served under (a page may send `Host: 0.0.0.0` to reach a local server).
    """
    value = raw.lower()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    value = value.removesuffix(".")
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        if value and _HOST_NAME.fullmatch(value):
            return value
        raise ValidationError("a host must be a name or an IP address, without port") from None
    return None if address.is_unspecified else str(address)


def allowed_hosts(bound_host: str, extra: Iterable[str] = ()) -> frozenset[str]:
    """The names pages are served under: loopback, the bound host and the extra names given.

    A wildcard bind adds no name of its own; an extra name must be a real one.
    """
    names = {_configured_host(bound_host)}
    for host in extra:
        name = _configured_host(host)
        if name is None:
            raise ValidationError("a wildcard address is not a host name to allow")
        names.add(name)
    return LOOPBACK_NAMES | {name for name in names if name is not None}


def host_allowed(header: str | None, allowed: frozenset[str]) -> bool:
    name = host_name(header) if header else None
    return name is not None and name in allowed


class CsrfTokens:
    def __init__(self, secret: bytes | None = None) -> None:
        self._secret = secret if secret is not None else secrets.token_bytes(32)

    @staticmethod
    def new_cookie() -> str:
        return secrets.token_urlsafe(32)

    @staticmethod
    def well_formed(cookie: str | None) -> bool:
        return cookie is not None and _COOKIE_SHAPE.fullmatch(cookie) is not None

    def token_for(self, cookie: str) -> str:
        return hmac.new(self._secret, cookie.encode("ascii"), sha256).hexdigest()

    def valid(self, cookie: str | None, token: str | None) -> bool:
        if not self.well_formed(cookie) or not token:
            return False
        return hmac.compare_digest(self.token_for(cookie or "").encode(), token.encode())
