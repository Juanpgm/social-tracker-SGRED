"""Vocabulary of the official-catalog cache (RC-6). Pure, no I/O.

Two catalogs only, both public FDSN event web services: USGS and EMSC. The Servicio Geologico
Colombiano is NOT a catalog source: it is never fetched and has no configuration here.
"""

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Final

from social_tracker.classification.catalog_match import CatalogPoint, MatchSettings
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError

CATALOG_HOSTS: Final = frozenset({"earthquake.usgs.gov", "www.seismicportal.eu"})
MAX_ERROR_CHARS: Final = 200
_ID_RE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_MIN_INTERVAL_S: Final = 60
_MAX_LOOKBACK_HOURS: Final = 24 * 30


class CatalogSource(StrEnum):
    USGS = "usgs"
    EMSC = "emsc"


def _number(value: object, low: float, high: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and low <= value <= high
    )


def _whole(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


@dataclass(frozen=True)
class CatalogEvent:
    """One earthquake of an official catalog, identified by its source and its id there. A later
    fetch of the same id replaces it (magnitudes are revised)."""

    source: CatalogSource
    external_id: str
    origin_time: datetime
    latitude: float
    longitude: float
    magnitude: float
    place: str = ""

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "source", CatalogSource(self.source))
        except ValueError:
            raise ValidationError("unknown catalog source") from None
        if not isinstance(self.external_id, str) or not _ID_RE.fullmatch(self.external_id):
            raise ValidationError("an event id is 1 to 64 letters, digits or . _ : -")
        self.to_point()  # the remaining fields are validated there

    def to_point(self) -> CatalogPoint:
        return CatalogPoint(
            self.source.value,
            self.origin_time,
            self.latitude,
            self.longitude,
            self.magnitude,
            self.place,
        )


@dataclass(frozen=True)
class CatalogSettings:
    """What the cache holds and how the check reads it.

    - the region of interest is a circle (`center_*`, `radius_km`) with events from
      `min_magnitude` up; events anywhere from `global_min_magnitude` up are kept too, for claims
      about an earthquake felt elsewhere;
    - `lookback_hours`: how far back each refresh asks; `interval_s`: how often the worker
      refreshes; `max_age_s`: a catalog whose last good refresh is older is stale (the check is
      `unavailable`); `retention_days`: events older than this are purged;
    - `time_window`, `max_magnitude_delta`, `utc_offset_hours`: how the check compares (see
      `classification.catalog_match` and `classification.claims`);
    - `max_events`: the cap sent with each request.
    """

    center_latitude: float = 3.4516
    center_longitude: float = -76.5320
    radius_km: float = 300.0
    min_magnitude: float = 2.5
    global_min_magnitude: float = 5.5
    lookback_hours: int = 48
    interval_s: int = 600
    max_age_s: int = 7200
    retention_days: int = 14
    time_window: timedelta = timedelta(hours=6)
    max_magnitude_delta: float = 1.0
    utc_offset_hours: int = -5
    max_events: int = 500

    def __post_init__(self) -> None:
        window_hours = (
            self.time_window.total_seconds() / 3600
            if isinstance(self.time_window, timedelta)
            else 0
        )
        valid = (
            _number(self.center_latitude, -90.0, 90.0)
            and _number(self.center_longitude, -180.0, 180.0)
            and _number(self.radius_km, 0.0, 20_000.0)
            and self.radius_km > 0
            and _number(self.min_magnitude, -2.0, 10.0)
            and _number(self.global_min_magnitude, -2.0, 10.0)
            and _whole(self.interval_s, _MIN_INTERVAL_S, 86_400)
            and _whole(self.max_age_s, self.interval_s, 7 * 86_400)
            and _whole(self.retention_days, 1, 365)
            and _whole(self.max_events, 1, 20_000)
            and _whole(self.utc_offset_hours, -12, 14)
            and 0 < window_hours <= 24 * 7
            and _whole(self.lookback_hours, math.ceil(2 * window_hours), _MAX_LOOKBACK_HOURS)
        )
        if not valid:
            raise ValidationError("invalid catalog settings")
        self.match_settings()  # the comparison thresholds are checked there

    def match_settings(self) -> MatchSettings:
        return MatchSettings(
            max_magnitude_delta=self.max_magnitude_delta,
            time_window=self.time_window,
            center_latitude=self.center_latitude,
            center_longitude=self.center_longitude,
            radius_km=self.radius_km,
            global_min_magnitude=self.global_min_magnitude,
        )


@dataclass(frozen=True)
class RefreshState:
    """The bookkeeping of one catalog: the last good refresh, the failures since, when to try
    again, and a claim that keeps two refreshers from fetching the same catalog at once."""

    source: CatalogSource
    last_success_at: datetime | None = None
    last_attempt_at: datetime | None = None
    consecutive_failures: int = 0
    next_attempt_at: datetime | None = None
    claimed_until: datetime | None = None
    last_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", CatalogSource(self.source))
        for name in ("last_success_at", "last_attempt_at", "next_attempt_at", "claimed_until"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, ensure_utc(value))
        if not _whole(self.consecutive_failures, 0, 1_000_000):
            raise ValidationError("the failure count must be a non-negative whole number")
        if self.last_error is not None and (
            not isinstance(self.last_error, str) or len(self.last_error) > MAX_ERROR_CHARS
        ):
            raise ValidationError("the last error is short text")
