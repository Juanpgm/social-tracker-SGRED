"""Exact edges of the catalog matcher, pinned from both sides (found by a mutation sweep)."""

from datetime import UTC, datetime, timedelta

from social_tracker.classification.catalog_match import (
    CONSISTENT,
    MISMATCH,
    CatalogPoint,
    MatchSettings,
    cross_check,
    distance_km,
    merge_duplicates,
)
from social_tracker.classification.claims import Claims, MagnitudeClaim, TimeClaim, TimeKind

NOW = datetime(2026, 10, 27, 18, 0, tzinfo=UTC)
PUB = NOW - timedelta(minutes=2)
CALI = (3.4516, -76.5320)


def claim(magnitude: float) -> Claims:
    return Claims((MagnitudeClaim(magnitude, magnitude),), TimeClaim())


def point(magnitude=4.0, *, lat=CALI[0], lon=CALI[1], seconds=0.0, source="usgs") -> CatalogPoint:
    return CatalogPoint(source, PUB + timedelta(seconds=seconds), lat, lon, magnitude, "x")


def status(claims, points, **options) -> str:
    return cross_check(claims, PUB, points, now=NOW, **options).status


def test_a_small_event_exactly_on_the_radius_counts_and_one_metre_beyond_does_not():
    lat = CALI[0] + 250.0 / 111.19492664455873
    edge = distance_km(CALI[0], CALI[1], lat, CALI[1])
    small = point(4.0, lat=lat)  # below the global floor: only the region can make it count
    assert status(claim(4.0), [small], settings=MatchSettings(radius_km=edge)) == CONSISTENT
    assert status(claim(4.0), [small], settings=MatchSettings(radius_km=edge - 0.001)) == MISMATCH


def test_the_rounding_keeps_a_difference_of_exactly_one_a_mismatch():
    assert abs(3.1 - 4.1) < 1.0  # the unrounded difference is 0.9999999999999996
    assert status(claim(3.1), [point(4.1)]) == MISMATCH
    assert status(claim(4.1), [point(3.1)]) == MISMATCH


def test_two_catalogs_are_one_earthquake_up_to_and_including_the_duplicate_limits():
    base = point(6.0, source="usgs")
    other = point(6.0, seconds=120.0, source="emsc")
    assert len(merge_duplicates([base, other])) == 1
    assert len(merge_duplicates([base, point(6.0, seconds=120.001, source="emsc")])) == 2


def test_the_duplicate_magnitude_limit_is_inclusive():
    settings = MatchSettings(duplicate_magnitude=0.5)
    assert len(merge_duplicates([point(6.0), point(5.5, source="emsc")], settings)) == 1
    assert len(merge_duplicates([point(6.0), point(5.4, source="emsc")], settings)) == 2


def test_the_duplicate_distance_limit_is_inclusive():
    far = point(6.0, lat=CALI[0] + 0.5, source="emsc")
    limit = distance_km(CALI[0], CALI[1], far.latitude, far.longitude)
    assert len(merge_duplicates([point(6.0), far], MatchSettings(duplicate_km=limit))) == 1
    assert len(merge_duplicates([point(6.0), far], MatchSettings(duplicate_km=limit - 0.01))) == 2


def test_two_events_of_the_same_catalog_are_never_one_earthquake():
    first, second = point(6.0, source="usgs"), point(5.9, seconds=10, source="usgs")
    assert len(merge_duplicates([first, second])) == 2
    aftershock = point(4.0, seconds=30, source="usgs")
    assert len(merge_duplicates([first, aftershock])) == 2


def test_an_aftershock_closer_in_time_is_compared_only_when_the_claim_names_its_instant():
    main_shock = point(6.2, seconds=-600)
    aftershock = point(3.5, seconds=-5)
    # R4: with no time claim the best event of the window decides, so the aftershock hides nothing
    assert status(claim(6.2), [main_shock, aftershock]) == CONSISTENT
    assert status(claim(6.2), [main_shock]) == CONSISTENT
    at_aftershock = Claims((MagnitudeClaim(6.2, 6.2),), TimeClaim(TimeKind.RELATIVE, PUB))
    assert status(at_aftershock, [main_shock, aftershock]) == MISMATCH
