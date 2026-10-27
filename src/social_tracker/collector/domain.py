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
    HOST_NOT_ALLOWED = "host_not_allowed"
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
    FetchErrorKind.HOST_NOT_ALLOWED: "host not allowed",
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


DEFAULT_MAX_ITEMS_PER_SOURCE = 100_000
DEFAULT_MAX_BYTES_PER_SOURCE = 512 * 1024 * 1024
DEFAULT_MAX_BYTES_PER_FETCH = 16 * 1024 * 1024


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive whole number")
    return value


@dataclass(frozen=True)
class IngestLimits:
    """What a hostile or broken feed may cost the database. Generous by default: the point is a
    ceiling, not a budget. Over a limit the extra entries are dropped and counted (reasons
    `fetch_quota` and `source_quota`); the fetch itself still succeeds.

    - `max_bytes_per_fetch`: title, text and raw payload bytes of the items of ONE fetch;
    - `max_items_per_source` / `max_bytes_per_source`: what ONE source holds at any time
      (retention frees room as items expire).
    """

    max_items_per_source: int = DEFAULT_MAX_ITEMS_PER_SOURCE
    max_bytes_per_source: int = DEFAULT_MAX_BYTES_PER_SOURCE
    max_bytes_per_fetch: int = DEFAULT_MAX_BYTES_PER_FETCH

    def __post_init__(self) -> None:
        _positive_int(self.max_items_per_source, "max_items_per_source")
        _positive_int(self.max_bytes_per_source, "max_bytes_per_source")
        _positive_int(self.max_bytes_per_fetch, "max_bytes_per_fetch")


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
