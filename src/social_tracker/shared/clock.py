"""Time source abstraction. All timestamps are timezone-aware and normalized to UTC."""

from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from social_tracker.shared.errors import ValidationError


def ensure_utc(value: Any) -> datetime:
    """Return `value` as an aware UTC datetime; reject naive or non-datetime input."""
    if not isinstance(value, datetime):
        raise ValidationError(f"expected a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("datetime must be timezone-aware")
    return value.astimezone(UTC)


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class FixedClock:
    """Deterministic clock for tests. Time only moves forward, via `advance`."""

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise ValidationError("clock cannot move backwards")
        try:
            self._now = self._now + delta
        except OverflowError:
            raise ValidationError("clock overflow: the resulting time is out of range") from None
