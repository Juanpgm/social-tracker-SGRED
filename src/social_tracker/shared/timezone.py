"""The local time zone of the operator (`SOCIAL_TRACKER_TIMEZONE`, default America/Bogota).

Everything stored, compared and decided is UTC. The zone only matters at the edge: a date a person
types into a filter is read in it (unless they give an explicit offset), and the times of the
screens and of the CLI are drawn in it with their offset, so "this morning" means their morning.
"""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError

DEFAULT_TIMEZONE = "America/Bogota"
_MAX_NAME = 64


def parse_timezone(text: str | None) -> ZoneInfo:
    """The zone named by `text` (an IANA name such as `America/Bogota`); blank is the default.
    An unknown or malformed name is a `ValidationError` that does not echo what was given."""
    name = (text or "").strip() or DEFAULT_TIMEZONE
    problem = ValidationError(
        "time zone must be an IANA name such as America/Bogota, Europe/Madrid or UTC"
    )
    if len(name) > _MAX_NAME or not name.isascii() or not name.isprintable():
        raise problem
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError, UnicodeError):
        raise problem from None


def format_local(moment: datetime, zone: ZoneInfo, *, seconds: bool = False) -> str:
    """`2026-03-01 07:00 -05` (seconds on request): the instant in the zone, with its
    abbreviation or offset. A naive time is refused rather than guessed."""
    local = ensure_utc(moment).astimezone(zone)
    pattern = "%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M"
    return f"{local.strftime(pattern)} {local.tzname()}"


def format_compact(moment: datetime, zone: ZoneInfo) -> str:
    """`2026-03-01T04:01-05`, or `...Z` at UTC: a short ISO 8601 form for a table, in the zone."""
    local = ensure_utc(moment).astimezone(zone)
    offset = local.utcoffset()
    assert offset is not None
    minutes = int(offset.total_seconds() // 60)
    if minutes == 0:
        suffix = "Z"
    else:
        sign, minutes = ("+", minutes) if minutes > 0 else ("-", -minutes)
        suffix = f"{sign}{minutes // 60:02d}" + (f":{minutes % 60:02d}" if minutes % 60 else "")
    return f"{local:%Y-%m-%dT%H:%M}{suffix}"


def format_iso(moment: datetime, zone: ZoneInfo) -> str:
    """The full ISO 8601 form (seconds and offset) of an instant in the zone."""
    return ensure_utc(moment).astimezone(zone).isoformat(timespec="seconds")
