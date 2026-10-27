"""The catalog cross-check as a pure function (RC-6): what an item claims against what the official
catalogs recorded. No I/O, no clock (the caller passes `now`), deterministic.

Outcomes (`Finding.status`)
- `consistent`: every claimed magnitude differs by less than `max_magnitude_delta` (1.0) from some
  event of the pool (see below). One claimed magnitude that no event of the pool matches while
  another does is NOT consistent but `unavailable`: a text could paste a real recent magnitude
  next to an invented one (S2).
- `mismatch`: no claimed magnitude is matched by any event of the pool, or no event at all lies
  within the window and the region while the cache is complete (every configured catalog was
  refreshed recently and its answer was not cut short).
- `unavailable`: the cache cannot answer. No event lies in the window and the cache is NOT
  complete; or the claim is NEWER than the last refresh minus `ingest_grace` (breaking news the
  catalogs have not published yet: a lagging cache confirms a claim it matches but never refutes
  one, R1); or the claim names a place outside the region and nothing at or above the global
  floor was recorded, or claims less than the floor of what the cache holds (R3); or the claimed
  magnitudes are only partly matched (S2). (An empty or stale cache is also `unavailable`; that
  decision is the caller's, who knows the freshness of the cache.)
- `no_claim`: nothing to check. The text claims no magnitude (a range wider than
  `MAX_RANGE_SPAN` is not one), talks about a date other than the day it was published, claims an
  instant in the future of `now` or older than what the cache covers.

The pool of events a claim is compared with
- Events in the time window around the claim's reference (the claimed instant, else the
  publication) that are relevant to the claim: in the region of interest or at/above the global
  floor for a claim about the region (or with no place); at/above the global floor for a claim
  about a place outside it. A place is in the region when it names one of `local_places` (whole
  words), so a lesser-known town reads as outside and is checked against large events only: it
  degrades to `unavailable`, never to a false `mismatch`.
- A claim with an explicit time (relative or absolute) is compared with the events nearest that
  instant: the nearest one and every event within `duplicate_seconds` of it count as one tier,
  so two catalogs that disagree about one earthquake do not make a faithful claim a mismatch
  (R12). A claim with no time is compared with every event of the window: an aftershock minutes
  ago does not hide the mainshock hours ago (R4).

Cross-source duplicates: the same earthquake in two catalogs (events close in time, place and
magnitude, from different sources) is one event, and the claim is compared with the closest of its
magnitudes, so that the disagreement of two catalogs never makes a faithful claim a mismatch.
Two events of the SAME source are never merged (an aftershock is not a duplicate).
"""

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

from social_tracker.classification.claims import (
    FUTURE_TOLERANCE,
    MAX_RANGE_SPAN,
    Claims,
    MagnitudeClaim,
    TimeKind,
)
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
_MAX_GRACE: Final = timedelta(days=1)
_MAX_LOCAL_PLACES: Final = 200
_MAX_LOCAL_PLACE_CHARS: Final = 60
_LOCAL_PLACE = re.compile(r"[a-z]+(?: [a-z]+)*")
# Names (folded to plain lowercase) that put a claim inside the default region of interest, a
# circle of 300 km around Cali. Whole words: "california" is not "cali".
DEFAULT_LOCAL_PLACES: Final = frozenset(
    [
        "colombia",
        "cali",
        "santiago de cali",
        "valle del cauca",
        "cauca",
        "popayan",
        "buenaventura",
        "palmira",
        "tulua",
        "jamundi",
        "yumbo",
        "buga",
        "cartago",
        "choco",
        "narino",
        "pasto",
        "quindio",
        "armenia",
        "risaralda",
        "pereira",
        "caldas",
        "manizales",
        "tolima",
        "ibague",
        "huila",
        "neiva",
        "putumayo",
        "mocoa",
        "pacifico",
    ]
)


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


def _valid_places(places: object) -> bool:
    return (
        isinstance(places, frozenset)
        and len(places) <= _MAX_LOCAL_PLACES
        and all(
            isinstance(p, str)
            and len(p) <= _MAX_LOCAL_PLACE_CHARS
            and _LOCAL_PLACE.fullmatch(p) is not None
            for p in places
        )
    )


@dataclass(frozen=True)
class MatchSettings:
    """Thresholds of the cross-check. The region of interest is a circle around Cali; events
    anywhere from `global_min_magnitude` up count too (an earthquake felt elsewhere).
    `min_magnitude` is the floor of what the cache holds for the region: a claim below it cannot be
    verified. `ingest_grace` is how long a catalog may take to publish an earthquake. `local_places`
    are the place names (plain lowercase) that put a claim in the region."""

    max_magnitude_delta: float = 1.0
    time_window: timedelta = timedelta(hours=6)
    center_latitude: float = 3.4516
    center_longitude: float = -76.5320
    radius_km: float = 300.0
    global_min_magnitude: float = 5.5
    duplicate_seconds: float = 120.0
    duplicate_km: float = 100.0
    duplicate_magnitude: float = 0.7
    min_magnitude: float = 2.5
    ingest_grace: timedelta = timedelta(minutes=30)
    local_places: frozenset[str] = DEFAULT_LOCAL_PLACES

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
            and _within(self.min_magnitude, MIN_EVENT_MAGNITUDE, MAX_EVENT_MAGNITUDE)
            and isinstance(self.ingest_grace, timedelta)
            and timedelta(0) <= self.ingest_grace <= _MAX_GRACE
            and _valid_places(self.local_places)
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


def _relevant(event: MergedEvent, settings: MatchSettings, *, local: bool) -> bool:
    """For a claim about the region: in the region of interest or large enough to be felt
    elsewhere. For a claim about a place outside it: large enough for the global request."""
    if max(event.magnitudes) >= settings.global_min_magnitude:
        return True
    if not local:
        return False
    return (
        distance_km(
            settings.center_latitude, settings.center_longitude, event.latitude, event.longitude
        )
        <= settings.radius_km
    )


def _is_local(place: str | None, settings: MatchSettings) -> bool:
    """No place at all counts as the region (the item is about what its audience feels)."""
    if place is None:
        return True
    padded = f" {place} "
    return any(f" {name} " in padded for name in settings.local_places)


def _distance(claim: MagnitudeClaim, event: MergedEvent) -> tuple[float, float]:
    """How far the claim is from the closest magnitude of the event, with that magnitude."""
    return min(
        (round(claim.distance(magnitude), _DELTA_DIGITS), magnitude)
        for magnitude in event.magnitudes
    )


def _verdict(
    claimed: Sequence[MagnitudeClaim],
    pool: Sequence[tuple[timedelta, MergedEvent]],
    settings: MatchSettings,
) -> Finding:
    """The finding for the claimed magnitudes against the events of the pool (never empty)."""
    best_of_claim = []
    for claim in claimed:
        best_of_claim.append(
            min(
                (delta, apart, magnitude, event.origin_time)
                for apart, event in pool
                for delta, magnitude in [_distance(claim, event)]
            )
        )
    delta, apart, magnitude, _ = min(best_of_claim, key=lambda row: (row[0], row[1], row[3]))
    matched = sum(row[0] < settings.max_magnitude_delta for row in best_of_claim)
    if matched == 0:
        return Finding(MISMATCH, delta, magnitude, int(apart.total_seconds()))
    if matched < len(claimed):  # S2: a real number next to an invented one proves nothing
        return Finding(UNAVAILABLE)
    return Finding(CONSISTENT, delta, magnitude, int(apart.total_seconds()))


def cross_check(
    claims: Claims,
    published_at: datetime,
    points: Sequence[CatalogPoint],
    *,
    now: datetime,
    settings: MatchSettings | None = None,
    complete: bool = True,
    covered_from: datetime | None = None,
    covered_until: datetime | None = None,
) -> Finding:
    """Compare `claims` with `points`, the cached events. `complete` says that every configured
    catalog was refreshed recently and answered in full, which is what lets "no event" mean
    "mismatch"; `covered_from` is the oldest instant the cache holds events for and
    `covered_until` the time of the last refresh, after which a claim may be ahead of the cache."""
    settings = settings or MatchSettings()
    published, current = ensure_utc(published_at), ensure_utc(now)
    claimed = [m for m in claims.magnitudes if round(m.high - m.low, 2) <= MAX_RANGE_SPAN]
    if not claimed or claims.when.kind is TimeKind.HISTORIC:
        return Finding(NO_CLAIM)
    explicit = claims.when.at is not None
    reference = claims.when.at if claims.when.at is not None else published
    window = settings.time_window
    if reference > current + FUTURE_TOLERANCE:
        return Finding(NO_CLAIM)
    if covered_from is not None and reference - window < ensure_utc(covered_from):
        return Finding(NO_CLAIM)
    local = _is_local(claims.place, settings)
    floor = settings.min_magnitude if local else settings.global_min_magnitude
    if max(m.high for m in claimed) < floor:
        return Finding(UNAVAILABLE)  # the cache holds nothing that small: it cannot say
    lagging = (
        covered_until is not None and reference > ensure_utc(covered_until) - settings.ingest_grace
    )
    pad = timedelta(seconds=settings.duplicate_seconds)
    nearby = [p for p in points if abs(p.origin_time - reference) <= window + pad]
    candidates = sorted(
        (
            (abs(event.origin_time - reference), event)
            for event in merge_duplicates(nearby, settings)
            if abs(event.origin_time - reference) <= window
            and _relevant(event, settings, local=local)
        ),
        key=lambda row: (row[0], row[1].origin_time, row[1].latitude, row[1].longitude),
    )
    if not candidates:
        quiet = MISMATCH if complete and local and not lagging else UNAVAILABLE
        return Finding(quiet)
    pool = candidates
    if explicit:  # the nearest event and the ones within the duplicate distance are one tier
        pool = [row for row in candidates if row[0] <= candidates[0][0] + pad]
    finding = _verdict(claimed, pool, settings)
    if finding.status == MISMATCH and lagging:
        return Finding(UNAVAILABLE)
    return finding
