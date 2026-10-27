"""The catalog cross-check as a pure function (RC-6): what an item claims against what the official
catalogs recorded. No I/O, no clock (the caller passes `now`), deterministic.

Outcomes (`Finding.status`)
- `consistent`: the event nearest in time to the claimed instant, within the time window and the
  region of interest (or a large event anywhere, "felt elsewhere"), differs from every claimed
  magnitude by less than `max_magnitude_delta` (1.0).
- `mismatch`: that difference is `max_magnitude_delta` or more, or no event at all lies within the
  window and the region while the cache is complete (every configured catalog was refreshed
  recently).
- `unavailable`: no event lies within the window and the cache is NOT complete, so the absence
  proves nothing. (An empty or stale cache is also `unavailable`; that decision is the caller's,
  who knows the freshness of the cache.)
- `no_claim`: nothing to check. The text claims no magnitude, talks about a date other than the
  day it was published, claims an instant in the future of `now` or older than what the cache
  covers.

Cross-source duplicates: the same earthquake in two catalogs (events close in time, place and
magnitude, from different sources) is one event, and the claim is compared with the closest of its
magnitudes, so that the disagreement of two catalogs never makes a faithful claim a mismatch.
Two events of the SAME source are never merged (an aftershock is not a duplicate).

The nearest-in-time event is compared, as the product rule says; a ranking by magnitude would hide
the case this check exists for.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from social_tracker.classification.claims import FUTURE_TOLERANCE, Claims, TimeKind
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError
from social_tracker.shared.text import has_control_chars, has_unsafe_chars

CONSISTENT: Final = "consistent"
MISMATCH: Final = "mismatch"
UNAVAILABLE: Final = "unavailable"
NO_CLAIM: Final = "no_claim"
NOT_CHECKED: Final = "not_checked"

EARTH_RADIUS_KM: Final = 6371.0
MIN_EVENT_MAGNITUDE: Final = -2.0
MAX_EVENT_MAGNITUDE: Final = 10.0
MAX_PLACE_LENGTH: Final = 200
_MAX_WINDOW: Final = timedelta(days=7)
_MAX_SOURCE_LENGTH: Final = 16
_DELTA_DIGITS: Final = 4


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _within(value: object, low: float, high: float) -> bool:
    return _is_number(value) and low <= value <= high  # type: ignore[operator]


@dataclass(frozen=True)
class CatalogPoint:
    """One event of an official catalog, as the matcher needs it."""

    source: str
    origin_time: datetime
    latitude: float
    longitude: float
    magnitude: float
    place: str = ""

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, str)
            or not 1 <= len(self.source) <= _MAX_SOURCE_LENGTH
            or not self.source.isascii()
            or not all(ch.isalnum() or ch == "_" for ch in self.source)
        ):
            raise ValidationError("an event needs a short source name")
        object.__setattr__(self, "origin_time", ensure_utc(self.origin_time))
        if not _within(self.latitude, -90.0, 90.0) or not _within(self.longitude, -180.0, 180.0):
            raise ValidationError("an event needs a valid latitude and longitude")
        if not _within(self.magnitude, MIN_EVENT_MAGNITUDE, MAX_EVENT_MAGNITUDE):
            raise ValidationError("an event magnitude is a number from -2 to 10")
        if (
            not isinstance(self.place, str)
            or len(self.place) > MAX_PLACE_LENGTH
            or has_control_chars(self.place)
            or has_unsafe_chars(self.place)
        ):
            raise ValidationError("an event place is short printable text")


@dataclass(frozen=True)
class MatchSettings:
    """Thresholds of the cross-check. The region of interest is a circle around Cali; events
    anywhere from `global_min_magnitude` up count too (an earthquake felt elsewhere)."""

    max_magnitude_delta: float = 1.0
    time_window: timedelta = timedelta(hours=6)
    center_latitude: float = 3.4516
    center_longitude: float = -76.5320
    radius_km: float = 300.0
    global_min_magnitude: float = 5.5
    duplicate_seconds: float = 120.0
    duplicate_km: float = 100.0
    duplicate_magnitude: float = 0.7

    def __post_init__(self) -> None:
        valid = (
            _within(self.max_magnitude_delta, 0.0, MAX_EVENT_MAGNITUDE)
            and self.max_magnitude_delta > 0
            and isinstance(self.time_window, timedelta)
            and timedelta(0) < self.time_window <= _MAX_WINDOW
            and _within(self.center_latitude, -90.0, 90.0)
            and _within(self.center_longitude, -180.0, 180.0)
            and _within(self.radius_km, 0.0, math.pi * EARTH_RADIUS_KM)
            and self.radius_km > 0
            and _within(self.global_min_magnitude, MIN_EVENT_MAGNITUDE, MAX_EVENT_MAGNITUDE)
            and _within(self.duplicate_seconds, 0.0, 3600.0)
            and _within(self.duplicate_km, 0.0, 1000.0)
            and _within(self.duplicate_magnitude, 0.0, MAX_EVENT_MAGNITUDE)
        )
        if not valid:
            raise ValidationError("invalid catalog match settings")


@dataclass(frozen=True)
class Finding:
    """The outcome: a status, and for a comparison the difference found, the magnitude of the
    event compared and how many seconds apart it was. No text of the item."""

    status: str
    delta_magnitude: float | None = None
    event_magnitude: float | None = None
    seconds: int | None = None


@dataclass(frozen=True)
class MergedEvent:
    """One earthquake, possibly reported by several catalogs."""

    origin_time: datetime
    latitude: float
    longitude: float
    magnitudes: tuple[float, ...]
    place: str
    sources: tuple[str, ...]


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (haversine); right across the antimeridian and over the poles."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _order(point: CatalogPoint) -> tuple:
    return (point.origin_time, point.source, point.latitude, point.longitude, point.magnitude)


def merge_duplicates(
    points: Iterable[CatalogPoint], settings: MatchSettings | None = None
) -> list[MergedEvent]:
    """Merge the events that are the same earthquake seen by different catalogs: close in time,
    place and magnitude to the first (earliest) event of the group, from a source the group does
    not have yet. The result does not depend on the order of `points`."""
    settings = settings or MatchSettings()
    groups: list[list[CatalogPoint]] = []
    for point in sorted(points, key=_order):
        for group in groups:
            anchor = group[0]
            if (
                point.source not in {p.source for p in group}
                and abs((point.origin_time - anchor.origin_time).total_seconds())
                <= settings.duplicate_seconds
                and abs(point.magnitude - anchor.magnitude) <= settings.duplicate_magnitude
                and distance_km(anchor.latitude, anchor.longitude, point.latitude, point.longitude)
                <= settings.duplicate_km
            ):
                group.append(point)
                break
        else:
            groups.append([point])
    return [
        MergedEvent(
            origin_time=group[0].origin_time,
            latitude=group[0].latitude,
            longitude=group[0].longitude,
            magnitudes=tuple(sorted(p.magnitude for p in group)),
            place=group[0].place,
            sources=tuple(sorted(p.source for p in group)),
        )
        for group in groups
    ]


def _relevant(event: MergedEvent, settings: MatchSettings) -> bool:
    """In the region of interest, or large enough to be felt elsewhere."""
    near = (
        distance_km(
            settings.center_latitude, settings.center_longitude, event.latitude, event.longitude
        )
        <= settings.radius_km
    )
    return near or max(event.magnitudes) >= settings.global_min_magnitude


def _delta(claims: Claims, event: MergedEvent) -> tuple[float, float]:
    """The smallest difference between a claimed magnitude and a magnitude of the event, with the
    magnitude that gave it."""
    return min(
        (round(claim.distance(magnitude), _DELTA_DIGITS), magnitude)
        for claim in claims.magnitudes
        for magnitude in event.magnitudes
    )


def cross_check(
    claims: Claims,
    published_at: datetime,
    points: Sequence[CatalogPoint],
    *,
    now: datetime,
    settings: MatchSettings | None = None,
    complete: bool = True,
    covered_from: datetime | None = None,
) -> Finding:
    """Compare `claims` with `points`, the cached events. `complete` says that every configured
    catalog was refreshed recently, which is what lets "no event" mean "mismatch"; `covered_from`
    is the oldest instant the cache holds events for."""
    settings = settings or MatchSettings()
    published, current = ensure_utc(published_at), ensure_utc(now)
    if not claims.magnitudes or claims.when.kind is TimeKind.HISTORIC:
        return Finding(NO_CLAIM)
    reference = claims.when.at if claims.when.at is not None else published
    window = settings.time_window
    if reference > current + FUTURE_TOLERANCE:
        return Finding(NO_CLAIM)
    if covered_from is not None and reference - window < ensure_utc(covered_from):
        return Finding(NO_CLAIM)
    pad = timedelta(seconds=settings.duplicate_seconds)
    nearby = [p for p in points if abs(p.origin_time - reference) <= window + pad]
    ranked = sorted(
        (
            (abs(event.origin_time - reference), delta, magnitude, event)
            for event in merge_duplicates(nearby, settings)
            if abs(event.origin_time - reference) <= window and _relevant(event, settings)
            for delta, magnitude in [_delta(claims, event)]
        ),
        key=lambda row: (row[0], row[1], row[3].origin_time, row[3].latitude, row[3].longitude),
    )
    if not ranked:
        return Finding(MISMATCH if complete else UNAVAILABLE)
    apart, delta, magnitude, _ = ranked[0]
    status = MISMATCH if delta >= settings.max_magnitude_delta else CONSISTENT
    return Finding(status, delta, magnitude, int(apart.total_seconds()))
