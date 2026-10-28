"""Times in words for the operator pages."""

from datetime import datetime

from social_tracker.shared.clock import ensure_utc

_MINUTE, _HOUR, _DAY = 60, 3600, 86400


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}{'' if count == 1 else 's'} ago"


def ago(moment: datetime | None, now: datetime) -> str:
    """`moment` as "never", "less than a minute ago", "2 minutes ago", "3 hours ago" or
    "5 days ago". A moment in the future (a clock that moved back) counts as just now; a naive
    time is refused rather than guessed."""
    if moment is None:
        return "never"
    seconds = int((ensure_utc(now) - ensure_utc(moment)).total_seconds())
    if seconds < _MINUTE:
        return "less than a minute ago"
    if seconds < _HOUR:
        return _plural(seconds // _MINUTE, "minute")
    if seconds < _DAY:
        return _plural(seconds // _HOUR, "hour")
    return _plural(seconds // _DAY, "day")
