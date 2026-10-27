"""Collector vocabulary: fetch requests, results and failures. Pure, no I/O.

A `FetchError` carries a machine-readable kind and a FIXED, short, human-readable message. The
message is what an operator sees as the "last error" of a source, so it is built only from this
vocabulary: never from an exception text, a URL, a header or a response body (those could carry a
secret or an attacker's text).
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

from social_tracker.control.domain import (
    MAX_RETRY_AFTER_S,
    CollectionState,
    FetchStatus,
    SourceState,
)
from social_tracker.shared.errors import DomainError
from social_tracker.sources.domain import NewsRssSource, Source

__all__ = ["MAX_RETRY_AFTER_S"]  # the clamp lives with the state machine that applies it
_MAX_RETRY_AFTER_TEXT = 100
_SECONDS_RE = re.compile(r"[0-9]+")
_DEGRADED_STATUSES = frozenset({401, 403, 429})


class FetchErrorKind(StrEnum):
    BAD_URL = "bad_url"
    DNS = "dns"
    BLOCKED_ADDRESS = "blocked_address"
    CONNECT = "connect"
    TIMEOUT = "timeout"
    TOO_MANY_REDIRECTS = "too_many_redirects"
    INSECURE_REDIRECT = "insecure_redirect"
    TOO_LARGE = "too_large"
    DECODE = "decode"
    PROTOCOL = "protocol"
    HTTP = "http"


_MESSAGES = {
    FetchErrorKind.BAD_URL: "invalid address",
    FetchErrorKind.DNS: "name lookup failed",
    FetchErrorKind.BLOCKED_ADDRESS: "address not allowed",
    FetchErrorKind.CONNECT: "connection failed",
    FetchErrorKind.TIMEOUT: "timeout",
    FetchErrorKind.TOO_MANY_REDIRECTS: "too many redirects",
    FetchErrorKind.INSECURE_REDIRECT: "redirect to plain http refused",
    FetchErrorKind.TOO_LARGE: "response too large",
    FetchErrorKind.DECODE: "undecodable response",
    FetchErrorKind.PROTOCOL: "protocol error",
}
_HTTP_HINTS = {401: " (access denied)", 403: " (access denied)", 429: " (rate limited)"}


class FetchError(DomainError):
    """A fetch that did not produce a response body. Never carries a URL or response text."""

    code = "fetch_failed"

    def __init__(
        self,
        kind: FetchErrorKind,
        *,
        status: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        if kind is FetchErrorKind.HTTP:
            message = f"http {status}{_HTTP_HINTS.get(status or 0, '')}"
        else:
            message = _MESSAGES[kind]
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.retry_after_s = retry_after_s

    @classmethod
    def http(cls, status: int, retry_after_s: float | None = None) -> "FetchError":
        return cls(FetchErrorKind.HTTP, status=status, retry_after_s=retry_after_s)

    @property
    def degraded(self) -> bool:
        """401, 403 and 429: the source answers but refuses us (CI-6)."""
        return self.kind is FetchErrorKind.HTTP and self.status in _DEGRADED_STATUSES


@dataclass(frozen=True)
class FetchRequest:
    url: str
    etag: str | None = None
    last_modified: str | None = None


@dataclass(frozen=True)
class FetchResponse:
    body: bytes
    etag: str | None = None
    last_modified: str | None = None
    not_modified: bool = False


def parse_retry_after(value: Any, now: datetime) -> float | None:
    """Seconds to wait from a `Retry-After` header: a whole number of seconds or an HTTP date.

    Negative, fractional, non-ASCII-digit and garbage values are ignored (None); a date in the
    past is 0; anything above `MAX_RETRY_AFTER_S` is clamped to it.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > _MAX_RETRY_AFTER_TEXT:
        return None
    if _SECONDS_RE.fullmatch(text):
        return float(min(int(text), MAX_RETRY_AFTER_S))
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    try:
        wait = (moment - now).total_seconds()
    except OverflowError:
        return None
    return float(min(max(wait, 0.0), MAX_RETRY_AFTER_S))


def is_due(
    source: Source, state: SourceState | None, fetch: FetchStatus | None, now: datetime
) -> bool:
    """Whether the worker should collect `source` now (CI-4).

    Only enabled RSS sources are collected here. An idle source is due when it never fetched
    successfully or its poll interval (the source's own setting, default 600 s) has passed since
    the last successful fetch (a last fetch AFTER now means the clock moved backwards: the source
    is due, never parked until the clock catches up); one in backoff is due once its retry time
    has come. A source that is collecting, failed (the operator must reset it), disabled or
    without state is never due.
    """
    if not isinstance(source, NewsRssSource) or not source.enabled or state is None:
        return False
    if state.state is CollectionState.BACKOFF:
        return state.next_run_at is not None and now >= state.next_run_at
    if state.state is not CollectionState.IDLE:
        return False
    if fetch is None or fetch.last_fetch_at is None:
        return True
    if fetch.last_fetch_at > now:  # the clock moved backwards: do not wait for it to catch up
        return True
    return now >= fetch.last_fetch_at + timedelta(seconds=source.poll_interval_s)
