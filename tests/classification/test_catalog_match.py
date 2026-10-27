"""The pure catalog matcher: claims against the events of an official catalog (RC-6).

Boundaries are tested on both sides of every threshold: magnitude difference (0.99 / 1.0 / 1.01),
time window, radius, the global magnitude floor, the clamps of latitude and longitude.
"""

import random
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.classification.catalog_match import (
    CONSISTENT,
    MISMATCH,
    NO_CLAIM,
    UNAVAILABLE,
    CatalogPoint,
    Finding,
    MatchSettings,
    cross_check,
    distance_km,
    merge_duplicates,
)
from social_tracker.classification.claims import (
    Claims,
    MagnitudeClaim,
    TimeClaim,
    TimeKind,
    extract_claims,
)
from social_tracker.shared.errors import ValidationError

NOW = datetime(2026, 10, 27, 18, 0, tzinfo=UTC)
PUB = NOW - timedelta(minutes=2)
CALI = (3.4516, -76.5320)


def point(
    minutes: float = 0.0,
    magnitude: float = 6.0,
    *,
    lat: float = CALI[0],
    lon: float = CALI[1],
    source: str = "usgs",
    at: datetime = PUB,
) -> CatalogPoint:
    return CatalogPoint(source, at + timedelta(minutes=minutes), lat, lon, magnitude, "somewhere")


def claims(*magnitudes: float, kind: TimeKind = TimeKind.NONE, at: datetime | None = None):
    return Claims(
        tuple(MagnitudeClaim(m, m) for m in magnitudes),
        TimeClaim(kind, at),
    )


def check(claim: Claims, points, **options) -> Finding:
    options.setdefault("now", NOW)
    return cross_check(claim, PUB, points, **options)


# ---- the magnitude difference ----


@pytest.mark.parametrize(
    ("event", "status"),
    [
        (5.0, CONSISTENT),
        (5.99, CONSISTENT),
        (6.0, MISMATCH),
        (6.01, MISMATCH),
        (4.01, CONSISTENT),
        (4.0, MISMATCH),
        (3.99, MISMATCH),
    ],
)
def test_a_difference_of_one_degree_or_more_is_a_mismatch(event, status):
    assert check(claims(5.0), [point(magnitude=event)]).status == status


def test_floating_point_noise_does_not_move_the_boundary():
    assert (
        check(claims(6.2), [point(magnitude=5.2)]).status == MISMATCH
    )  # 6.2 - 5.2 = 1.0000000000000009
    assert check(claims(5.9), [point(magnitude=6.9)]).status == MISMATCH
    assert check(claims(3.3), [point(magnitude=4.2)]).status == CONSISTENT  # 0.8999999999999999


def test_the_finding_reports_the_difference_and_the_event_magnitude():
    finding = check(claims(6.2), [point(minutes=-10, magnitude=5.9)])
    assert finding == Finding(CONSISTENT, delta_magnitude=0.3, event_magnitude=5.9, seconds=600)


def test_the_threshold_is_configurable():
    strict = MatchSettings(max_magnitude_delta=0.5)
    assert check(claims(6.0), [point(magnitude=5.6)], settings=strict).status == CONSISTENT
    assert check(claims(6.0), [point(magnitude=5.5)], settings=strict).status == MISMATCH


@pytest.mark.parametrize(
    ("event", "status", "delta"),
    [(5.0, CONSISTENT, 0.5), (5.75, CONSISTENT, 0.0), (6.4, CONSISTENT, 0.4), (7.1, MISMATCH, 1.1)],
)
def test_a_range_is_as_far_from_the_event_as_its_nearest_end(event, status, delta):
    ranged = Claims((MagnitudeClaim(5.5, 6.0),), TimeClaim())
    finding = check(ranged, [point(magnitude=event)])
    assert finding.status == status
    assert finding.delta_magnitude == pytest.approx(delta)


def test_with_several_claimed_magnitudes_all_of_them_must_be_matched():
    assert check(claims(6.2, 5.8), [point(magnitude=5.9)]).status == CONSISTENT
    # S2: one claimed number matching while another does not is inconclusive, never consistent
    assert check(claims(6.2, 8.0), [point(magnitude=7.5)]).status == UNAVAILABLE
    assert check(claims(6.2, 5.8), [point(magnitude=4.0)]).status == MISMATCH
    assert check(claims(6.2, 5.8), [point(magnitude=7.4)]).status == MISMATCH


# ---- which event is compared ----


def test_the_event_nearest_in_time_is_compared_when_the_claim_names_its_instant():
    near_small = point(minutes=-1, magnitude=5.0)
    far_big = point(minutes=-30, magnitude=6.2)
    named = claims(6.2, kind=TimeKind.RELATIVE, at=PUB)
    assert check(named, [near_small, far_big]).status == MISMATCH  # the nearest is M5.0
    assert check(named, [point(minutes=-10, magnitude=6.2), point(minutes=-90)]).status == (
        CONSISTENT
    )


def test_without_a_named_instant_any_event_of_the_window_may_match():
    # R4: the nearest-in-time rule let an aftershock hide the mainshock
    near_small = point(minutes=-1, magnitude=5.0)
    far_big = point(minutes=-30, magnitude=6.2)
    assert check(claims(6.2), [near_small, far_big]).status == CONSISTENT


def test_an_equal_distance_in_time_is_broken_by_the_smaller_difference_and_is_stable():
    early = point(minutes=-5, magnitude=6.0)
    late = point(minutes=5, magnitude=5.0)
    for ordering in ([early, late], [late, early]):
        finding = check(claims(5.2), ordering)
        assert finding.event_magnitude == 5.0
        assert finding.status == CONSISTENT


def test_the_reference_time_is_the_claimed_instant_not_the_publication():
    claimed = PUB - timedelta(hours=8)
    claim = claims(6.0, kind=TimeKind.RELATIVE, at=claimed)
    old = point(minutes=0, magnitude=6.0, at=claimed)
    assert check(claim, [old]).status == CONSISTENT
    assert check(claim, [point(minutes=0, magnitude=6.0)]).status == MISMATCH  # now: out of window


def test_without_a_time_claim_the_publication_time_is_the_reference():
    assert check(claims(6.0), [point(magnitude=6.0)]).status == CONSISTENT


# ---- the time window ----


@pytest.mark.parametrize(
    ("seconds", "inside"),
    [(0, True), (6 * 3600 - 1, True), (6 * 3600, True), (6 * 3600 + 1, False)],
)
@pytest.mark.parametrize("sign", [1, -1])
def test_the_time_window_includes_its_edge_and_excludes_one_second_beyond(seconds, inside, sign):
    event = CatalogPoint("usgs", PUB + timedelta(seconds=sign * seconds), *CALI, 6.0, "x")
    finding = check(claims(6.0), [event])
    assert finding.status == (CONSISTENT if inside else MISMATCH)


def test_the_window_is_configurable():
    narrow = MatchSettings(time_window=timedelta(minutes=10))
    assert check(claims(6.0), [point(minutes=10)], settings=narrow).status == CONSISTENT
    assert check(
        claims(6.0), [point(minutes=10, at=PUB + timedelta(seconds=1))], settings=narrow
    ).status == (MISMATCH)


# ---- the radius and the global floor ----


def offset_north(km: float) -> tuple[float, float]:
    return CALI[0] + km / 111.19492664455873, CALI[1]


def test_the_radius_edge_is_included_and_a_hair_beyond_is_not():
    lat, lon = offset_north(250.0)
    edge = distance_km(CALI[0], CALI[1], lat, lon)
    event = point(magnitude=6.0, lat=lat, lon=lon)
    assert check(claims(6.0), [event], settings=MatchSettings(radius_km=edge)).status == CONSISTENT
    narrower = MatchSettings(radius_km=edge - 0.001)
    assert check(
        claims(6.0), [point(magnitude=4.0, lat=lat, lon=lon)], settings=narrower
    ).status == (MISMATCH)


@pytest.mark.parametrize(
    ("event", "status"),
    [(5.49, MISMATCH), (5.5, CONSISTENT), (5.51, CONSISTENT)],
)
def test_far_away_only_events_from_the_global_floor_up_count(event, status):
    tokyo = point(magnitude=event, lat=35.7, lon=139.7)
    assert check(claims(event), [tokyo]).status == status


def test_the_global_floor_is_configurable():
    low = MatchSettings(global_min_magnitude=4.0)
    assert check(claims(4.2), [point(magnitude=4.2, lat=35.7, lon=139.7)], settings=low).status == (
        CONSISTENT
    )


def test_distance_is_symmetric_across_the_antimeridian_and_over_the_pole():
    across = distance_km(0.0, 179.9, 0.0, -179.9)
    assert across == pytest.approx(22.24, abs=0.1)
    assert distance_km(0.0, -179.9, 0.0, 179.9) == pytest.approx(across)
    over_pole = distance_km(89.9, 0.0, 89.9, 180.0)
    assert over_pole == pytest.approx(22.24, abs=0.1)
    assert distance_km(0.0, 0.0, 0.0, 0.0) == 0.0
    assert distance_km(90.0, 0.0, -90.0, 0.0) == pytest.approx(20015.1, abs=1.0)
    assert distance_km(0.0, -180.0, 0.0, 180.0) == pytest.approx(0.0, abs=1e-6)


def test_a_region_across_the_antimeridian_matches_events_on_both_sides():
    fiji = MatchSettings(center_latitude=-17.8, center_longitude=179.9, radius_km=100.0)
    west = point(magnitude=6.0, lat=-17.8, lon=-179.9)
    assert check(claims(6.0), [west], settings=fiji).status == CONSISTENT


# ---- when there is nothing to compare ----


def test_no_event_in_the_window_is_a_mismatch_when_the_cache_is_complete():
    finding = check(claims(6.0), [])
    assert finding == Finding(MISMATCH)
    assert check(claims(6.0), [point(minutes=-400)]).status == MISMATCH


def test_no_event_in_the_window_is_unavailable_when_a_source_is_missing():
    assert check(claims(6.0), [], complete=False) == Finding(UNAVAILABLE)


def test_an_event_found_decides_even_when_a_source_is_missing():
    assert check(claims(6.0), [point()], complete=False).status == CONSISTENT
    assert check(claims(6.0), [point(magnitude=3.0)], complete=False).status == MISMATCH


def test_no_magnitude_claim_means_nothing_to_check():
    assert check(Claims(), [point()]) == Finding(NO_CLAIM)
    assert check(extract_claims("tembló fuerte en Cali", PUB), [point()]) == Finding(NO_CLAIM)


def test_a_historic_claim_is_not_checked():
    historic = Claims((MagnitudeClaim(6.0, 6.0),), TimeClaim(TimeKind.HISTORIC))
    assert check(historic, []) == Finding(NO_CLAIM)
    assert check(extract_claims("sismo de magnitud 6.0 del 12 de agosto", PUB), []) == Finding(
        NO_CLAIM
    )


def test_a_time_in_the_future_cannot_be_checked():
    soon = claims(6.0, kind=TimeKind.RELATIVE, at=NOW + timedelta(minutes=15))
    later = claims(6.0, kind=TimeKind.RELATIVE, at=NOW + timedelta(minutes=16))
    assert check(soon, []).status == MISMATCH
    assert check(later, []) == Finding(NO_CLAIM)


def test_a_publication_time_in_the_future_cannot_be_checked():
    finding = cross_check(claims(6.0), NOW + timedelta(days=2), [], now=NOW)
    assert finding == Finding(NO_CLAIM)


def test_a_time_before_what_the_cache_covers_cannot_be_checked():
    claimed = PUB - timedelta(hours=10)
    claim = claims(6.0, kind=TimeKind.RELATIVE, at=claimed)
    edge = claimed - timedelta(hours=6)
    assert check(claim, [], covered_from=edge).status == MISMATCH
    assert check(claim, [], covered_from=edge + timedelta(seconds=1)) == Finding(NO_CLAIM)


# ---- the same earthquake in two catalogs ----


def test_the_same_earthquake_in_two_catalogs_is_one_event_and_the_closest_magnitude_counts():
    usgs = point(minutes=-5, magnitude=6.1, source="usgs")
    emsc = CatalogPoint("emsc", usgs.origin_time + timedelta(seconds=20), 3.5, -76.6, 6.3, "VALLE")
    assert len(merge_duplicates([usgs, emsc])) == 1
    assert check(claims(7.2), [usgs]).status == MISMATCH
    assert check(claims(7.2), [usgs, emsc]).status == CONSISTENT  # 0.9 from the EMSC magnitude
    assert check(claims(7.2), [emsc, usgs]).status == CONSISTENT


def test_two_distinct_earthquakes_are_not_merged():
    first = point(minutes=-30, magnitude=6.0)
    soon_after_far = point(minutes=-29, magnitude=6.0, lat=35.7, lon=139.7)
    later_same_place = point(minutes=-20, magnitude=6.0)
    assert len(merge_duplicates([first, soon_after_far, later_same_place])) == 3


def test_a_merge_is_independent_of_the_order_of_the_events():
    events = [
        point(minutes=-m, magnitude=5.0 + m / 10, source="usgs" if m % 2 else "emsc")
        for m in range(1, 9)
    ]
    expected = merge_duplicates(events)
    for seed in range(5):
        shuffled = list(events)
        random.Random(seed).shuffle(shuffled)
        assert merge_duplicates(shuffled) == expected


def test_the_same_input_always_gives_the_same_finding():
    events = [point(minutes=-m, magnitude=5.0 + m / 7) for m in range(1, 30)]
    claim = extract_claims("sismo de magnitud 6.2 hace un momento", PUB)
    first = check(claim, events)
    for seed in range(5):
        shuffled = list(events)
        random.Random(seed).shuffle(shuffled)
        assert check(claim, shuffled) == first


# ---- the shapes are validated ----


@pytest.mark.parametrize(
    "changes",
    [
        {"magnitude": float("nan")},
        {"magnitude": float("inf")},
        {"magnitude": -2.1},
        {"magnitude": 10.1},
        {"magnitude": True},
        {"magnitude": "6.0"},
        {"lat": 90.1},
        {"lat": -90.1},
        {"lat": float("nan")},
        {"lon": 180.1},
        {"lon": -180.1},
        {"lon": float("inf")},
        {"source": ""},
        {"source": " "},
    ],
)
def test_an_invalid_event_is_refused(changes):
    with pytest.raises(ValidationError):
        point(**changes)


def test_a_naive_event_time_is_refused():
    with pytest.raises(ValidationError):
        CatalogPoint("usgs", datetime(2026, 10, 27, 17, 0), 0.0, 0.0, 5.0, "x")


@pytest.mark.parametrize(
    "changes",
    [
        {"lat": 90.0, "lon": 180.0},
        {"lat": -90.0, "lon": -180.0},
        {"magnitude": -2.0},
        {"magnitude": 10.0},
    ],
)
def test_the_limits_of_the_valid_ranges_are_accepted(changes):
    assert isinstance(point(**changes), CatalogPoint)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_magnitude_delta": 0.0},
        {"max_magnitude_delta": float("nan")},
        {"time_window": timedelta(0)},
        {"time_window": timedelta(days=8)},
        {"radius_km": 0.0},
        {"radius_km": 30000.0},
        {"center_latitude": 91.0},
        {"center_longitude": -181.0},
        {"global_min_magnitude": 11.0},
        {"duplicate_seconds": -1},
        {"duplicate_km": float("inf")},
    ],
)
def test_invalid_settings_are_refused(changes):
    with pytest.raises(ValidationError):
        MatchSettings(**changes)


def test_the_default_settings_are_the_product_ones():
    settings = MatchSettings()
    assert settings.max_magnitude_delta == 1.0
    assert settings.time_window == timedelta(hours=6)
    assert (settings.center_latitude, settings.center_longitude) == CALI
    assert settings.radius_km == 300.0
    assert settings.global_min_magnitude == 5.5
