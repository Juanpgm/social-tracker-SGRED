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
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
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


def host_name(header: str) -> str:
    """The host part of a `Host` header: lower-cased, without port or IPv6 brackets."""
    value = header.strip().lower()
    if value.startswith("["):
        return value[1:].partition("]")[0]
    return value.partition(":")[0]


def allowed_hosts(bound_host: str) -> frozenset[str]:
    return LOOPBACK_NAMES | {host_name(bound_host)}


def host_allowed(header: str | None, allowed: frozenset[str]) -> bool:
    return bool(header) and host_name(header or "") in allowed


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
