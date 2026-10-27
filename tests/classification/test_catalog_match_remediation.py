"""Remediation of the matcher after the adversarial review (R1, R3, R4, R12, R13, S2).

Every test names the finding it pins and is checked from both sides of its boundary.
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
TOKYO = (35.7, 139.7)


def point(minutes=0.0, magnitude=6.0, *, where=CALI, source="usgs", at=PUB):
    return CatalogPoint(source, at + timedelta(minutes=minutes), *where, magnitude, "x")


def claims(*magnitudes, kind=TimeKind.NONE, at=None, place=None):
    return Claims(tuple(MagnitudeClaim(m, m) for m in magnitudes), TimeClaim(kind, at), place)


def check(claim, points, **options):
    options.setdefault("now", NOW)
    return cross_check(claim, PUB, points, **options)


def moment_ago(*magnitudes, place=None):
    return claims(*magnitudes, kind=TimeKind.RELATIVE, at=PUB, place=place)


# ---- R1: breaking news must not be a mismatch while the cache lags ----


def test_a_claim_newer_than_the_last_refresh_is_unavailable_not_a_mismatch_when_nothing_is_cached():
    finding = check(moment_ago(7.5), [], covered_until=NOW - timedelta(minutes=5))
    assert finding == Finding(UNAVAILABLE)


def test_a_claim_newer_than_the_last_refresh_is_unavailable_next_to_a_small_event():
    small = point(minutes=-10, magnitude=2.9)
    assert check(moment_ago(7.5), [small], covered_until=NOW).status == UNAVAILABLE


def test_a_lagging_cache_still_confirms_a_claim_it_does_match():
    recorded = point(minutes=-1, magnitude=7.4)
    assert check(moment_ago(7.5), [recorded], covered_until=NOW).status == CONSISTENT


def test_the_ingest_grace_boundary_is_exact():
    grace = MatchSettings().ingest_grace
    assert grace == timedelta(minutes=30)
    edge = PUB + grace  # the last refresh time at which the claim is exactly old enough
    assert check(moment_ago(7.5), [], covered_until=edge).status == MISMATCH
    assert check(moment_ago(7.5), [], covered_until=edge - timedelta(seconds=1)).status == (
        UNAVAILABLE
    )


def test_the_ingest_grace_is_configurable():
    none = MatchSettings(ingest_grace=timedelta(0))
    assert check(moment_ago(7.5), [], covered_until=PUB, settings=none).status == MISMATCH
    assert check(
        moment_ago(7.5), [], covered_until=PUB - timedelta(seconds=1), settings=none
    ).status == (UNAVAILABLE)
    wide = MatchSettings(ingest_grace=timedelta(hours=2))
    assert check(
        moment_ago(7.5), [], covered_until=PUB + timedelta(hours=1), settings=wide
    ).status == (UNAVAILABLE)


def test_an_old_claim_with_a_fresh_cache_is_still_a_mismatch():
    old = claims(7.5, kind=TimeKind.RELATIVE, at=PUB - timedelta(hours=3))
    assert check(old, [], covered_until=NOW).status == MISMATCH


def test_without_a_known_refresh_time_nothing_changes():
    assert check(moment_ago(7.5), []).status == MISMATCH


def test_a_claim_with_no_time_uses_the_publication_as_its_reference():
    fresh_post = claims(7.5)
    assert check(fresh_post, [], covered_until=NOW - timedelta(minutes=5)).status == UNAVAILABLE
    assert check(fresh_post, [], covered_until=PUB + timedelta(hours=1)).status == MISMATCH


@pytest.mark.parametrize("grace", [timedelta(seconds=-1), timedelta(days=1, seconds=1), 30, None])
def test_an_invalid_ingest_grace_is_refused(grace):
    with pytest.raises(ValidationError):
        MatchSettings(ingest_grace=grace)


def test_the_largest_and_smallest_ingest_graces_are_accepted():
    assert MatchSettings(ingest_grace=timedelta(0)).ingest_grace == timedelta(0)
    assert MatchSettings(ingest_grace=timedelta(days=1)).ingest_grace == timedelta(days=1)


# ---- R3: the claimed place decides what the claim is compared with ----


def test_a_claim_about_a_far_place_ignores_local_micro_events():
    claim = extract_claims("sismo de magnitud 6.5 en Japón", PUB)
    assert claim.place == "japon"
    micro = point(minutes=-30, magnitude=2.8, where=(3.9, -76.1))
    assert check(claim, [micro]) == Finding(UNAVAILABLE)


def test_a_claim_about_a_far_place_is_compared_with_the_events_of_the_global_floor():
    claim = claims(6.5, place="japon")
    assert check(claim, [point(minutes=-30, magnitude=6.4, where=TOKYO)]).status == CONSISTENT
    assert check(claim, [point(minutes=-30, magnitude=8.2, where=TOKYO)]).status == MISMATCH


def test_a_far_claim_with_no_global_event_in_the_window_is_unavailable_even_with_a_complete_cache():
    assert check(claims(6.5, place="japon"), [], complete=True) == Finding(UNAVAILABLE)


@pytest.mark.parametrize(("claimed", "status"), [(5.49, UNAVAILABLE), (5.5, MISMATCH)])
def test_a_far_claim_below_the_global_floor_cannot_be_verified(claimed, status):
    claim = claims(claimed, place="japon")
    assert check(claim, [point(magnitude=8.0, where=TOKYO)]).status == status


@pytest.mark.parametrize("place", [None, "cali", "santiago de cali", "valle del cauca", "colombia"])
def test_a_claim_about_the_region_is_compared_with_the_regional_events(place):
    micro = point(minutes=-30, magnitude=2.8, where=(3.9, -76.1))
    assert check(claims(6.5, place=place), [micro]).status == MISMATCH


@pytest.mark.parametrize(
    "place", ["japon", "california", "calima", "santiago de chile", "caucasia", "bogota"]
)
def test_a_place_outside_the_region_is_matched_by_whole_words(place):
    micro = point(minutes=-30, magnitude=2.8, where=(3.9, -76.1))
    assert check(claims(6.5, place=place), [micro]).status == UNAVAILABLE


def test_a_known_place_inside_a_longer_name_is_local():
    micro = point(minutes=-30, magnitude=2.8, where=(3.9, -76.1))
    assert check(claims(6.5, place="cali colombia"), [micro]).status == MISMATCH


@pytest.mark.parametrize(("claimed", "status"), [(2.49, UNAVAILABLE), (2.5, MISMATCH)])
def test_a_claim_below_the_regional_floor_cannot_be_verified(claimed, status):
    far_event = point(minutes=-1, magnitude=6.0, where=TOKYO)
    assert check(claims(claimed), [far_event]).status == status


def test_the_regional_floor_is_configurable():
    settings = MatchSettings(min_magnitude=1.0)
    assert check(claims(1.2), [], settings=settings).status == MISMATCH
    assert check(claims(0.9), [], settings=settings).status == UNAVAILABLE


def test_a_range_that_reaches_the_floor_can_be_verified():
    claim = Claims((MagnitudeClaim(2.0, 2.5),), TimeClaim())
    assert check(claim, []).status == MISMATCH


def test_the_local_places_are_configurable():
    settings = MatchSettings(local_places=frozenset({"lima"}))
    micro = point(minutes=-30, magnitude=2.8, where=(3.9, -76.1))
    assert check(claims(6.5, place="lima"), [micro], settings=settings).status == MISMATCH
    assert check(claims(6.5, place="cali"), [micro], settings=settings).status == UNAVAILABLE


@pytest.mark.parametrize(
    "places",
    [{"cali"}, frozenset({1}), frozenset({"Cali"}), frozenset({""}), frozenset({"x" * 61}), None],
)
def test_invalid_local_places_are_refused(places):
    with pytest.raises(ValidationError):
        MatchSettings(local_places=places)


def test_an_invalid_regional_floor_is_refused():
    for value in (11.0, -3.0, float("nan"), True):
        with pytest.raises(ValidationError):
            MatchSettings(min_magnitude=value)


# ---- R4: a claim with no time is compared with the best event in the window ----


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_an_aftershock_does_not_hide_the_mainshock(order):
    main, after = point(minutes=-180, magnitude=7.5), point(minutes=-5, magnitude=4.8)
    events = [main, after]
    random.Random(order).shuffle(events)
    finding = check(claims(7.5, place="cali"), events)
    assert finding.status == CONSISTENT
    assert finding.event_magnitude == 7.5
    assert finding.seconds == 180 * 60  # the mainshock is 3 h before the reference


def test_with_no_time_claim_the_status_is_consistent_if_any_event_in_the_window_is_close():
    events = [point(minutes=-m, magnitude=3.0 + m / 10) for m in range(1, 40)]
    assert check(claims(5.3), events).status == CONSISTENT


def test_with_no_time_claim_a_mismatch_needs_every_event_to_be_far():
    events = [point(minutes=-1, magnitude=3.0), point(minutes=-200, magnitude=3.5)]
    finding = check(claims(6.0), events)
    assert finding.status == MISMATCH
    assert finding.event_magnitude == 3.5
    assert finding.delta_magnitude == 2.5


def test_with_an_explicit_time_the_event_nearest_that_instant_is_compared():
    main, after = point(minutes=-180, magnitude=7.5), point(minutes=-5, magnitude=4.8)
    recent = moment_ago(7.5)
    assert check(recent, [main, after]).status == MISMATCH
    old = claims(7.5, kind=TimeKind.RELATIVE, at=main.origin_time)
    assert check(old, [main, after]).status == CONSISTENT


def test_the_event_outside_the_window_is_not_a_candidate_for_a_claim_with_no_time():
    main = point(minutes=-6 * 60 - 3, magnitude=7.5)  # beyond the window
    assert check(claims(7.5), [main]).status == MISMATCH


# ---- R12: events within the duplicate distance are one tier ----


@pytest.mark.parametrize("order", [0, 1, 2])
def test_two_catalogs_that_disagree_by_a_degree_do_not_make_a_faithful_claim_a_mismatch(order):
    base = PUB - timedelta(minutes=5)
    usgs = CatalogPoint("usgs", base, *CALI, 6.0, "x")
    emsc = CatalogPoint("emsc", base - timedelta(seconds=60), *CALI, 5.0, "x")
    events = [usgs, emsc]
    random.Random(order).shuffle(events)
    claim = claims(6.0, kind=TimeKind.RELATIVE, at=base - timedelta(seconds=60))
    finding = check(claim, events)
    assert finding.status == CONSISTENT
    assert finding.event_magnitude == 6.0


def test_the_tier_edge_is_the_duplicate_seconds_inclusive():
    base = PUB - timedelta(minutes=5)
    near = CatalogPoint("emsc", base, *CALI, 5.0, "x")
    claim = claims(6.0, kind=TimeKind.RELATIVE, at=base)
    inside = CatalogPoint("usgs", base + timedelta(seconds=120), *CALI, 6.0, "x")
    outside = CatalogPoint("usgs", base + timedelta(seconds=121), *CALI, 6.0, "x")
    assert check(claim, [near, inside]).status == CONSISTENT
    assert check(claim, [near, outside]).status == MISMATCH


def test_the_tier_follows_the_configured_duplicate_seconds():
    base = PUB - timedelta(minutes=5)
    near = CatalogPoint("emsc", base, *CALI, 5.0, "x")
    other = CatalogPoint("usgs", base + timedelta(seconds=30), *CALI, 6.0, "x")
    claim = claims(6.0, kind=TimeKind.RELATIVE, at=base)
    tight = MatchSettings(duplicate_seconds=29)
    assert check(claim, [near, other], settings=tight).status == MISMATCH
    assert check(claim, [near, other], settings=MatchSettings(duplicate_seconds=30)).status == (
        CONSISTENT
    )


# ---- R13: a wide range never makes a claim consistent ----


def test_a_range_wider_than_one_degree_is_not_compared():
    wide = Claims((MagnitudeClaim(4.0, 7.0),), TimeClaim())
    assert check(wide, [point(minutes=-5, magnitude=6.4)]) == Finding(NO_CLAIM)


@pytest.mark.parametrize(("high", "status"), [(5.0, CONSISTENT), (5.01, NO_CLAIM)])
def test_the_range_span_boundary_is_one_degree(high, status):
    claim = Claims((MagnitudeClaim(4.0, high),), TimeClaim())
    assert check(claim, [point(magnitude=4.5)]).status == status


def test_a_wide_range_next_to_a_single_value_leaves_only_the_value():
    claim = Claims((MagnitudeClaim(3.0, 8.0), MagnitudeClaim(4.0, 4.0)), TimeClaim())
    assert check(claim, [point(magnitude=6.0)]).status == MISMATCH


# ---- S2: a real number cannot be pasted in to buy a consistent verdict ----


def test_a_rumor_that_adds_a_real_recent_magnitude_is_not_consistent():
    claim = extract_claims("M9.0 en Cali, evacuen ya (el USGS registró M4.2)", PUB)
    assert [m.low for m in claim.magnitudes] == [9.0, 4.2]
    recorded = point(minutes=-1, magnitude=4.2)
    assert check(claim, [recorded]) == Finding(UNAVAILABLE)


def test_every_claimed_magnitude_matching_is_consistent():
    assert check(claims(4.2, 4.5), [point(magnitude=4.2)]).status == CONSISTENT


def test_no_claimed_magnitude_matching_is_a_mismatch():
    assert check(claims(9.0, 8.5), [point(magnitude=4.2)]).status == MISMATCH


@pytest.mark.parametrize(("second", "status"), [(5.19, CONSISTENT), (5.2, UNAVAILABLE)])
def test_the_unmatched_claim_boundary_is_the_delta(second, status):
    assert check(claims(4.2, second), [point(magnitude=4.2)]).status == status


def test_each_claim_may_be_matched_by_a_different_event_of_the_window():
    events = [point(minutes=-200, magnitude=7.5), point(minutes=-5, magnitude=4.8)]
    assert check(claims(7.5, 4.8), events).status == CONSISTENT
    assert check(claims(7.5, 4.8, 9.5), events).status == UNAVAILABLE


def test_the_verdict_does_not_depend_on_the_order_of_the_events():
    events = [
        point(minutes=-m, magnitude=3.0 + m / 9, source="usgs" if m % 2 else "emsc")
        for m in range(1, 25)
    ]
    for claim in (claims(5.5), claims(9.0, 4.0), claims(3.2, 3.4)):
        first = check(claim, events)
        for seed in range(6):
            shuffled = list(events)
            random.Random(seed).shuffle(shuffled)
            assert check(claim, shuffled) == first


# ---- which event the finding reports ----


@pytest.mark.parametrize("order", [0, 1, 2])
def test_among_equal_differences_the_finding_reports_the_event_nearest_in_time(order):
    events = [point(minutes=-10, magnitude=6.0), point(minutes=-5, magnitude=6.0)]
    random.Random(order).shuffle(events)
    finding = check(claims(6.0), events)
    assert (finding.status, finding.seconds) == (CONSISTENT, 300)


def test_the_finding_reports_the_closest_magnitude_before_the_nearest_time():
    far_exact = point(minutes=-60, magnitude=6.0)
    near_off = point(minutes=-1, magnitude=5.5)
    finding = check(claims(6.0), [near_off, far_exact])
    assert (finding.event_magnitude, finding.delta_magnitude, finding.seconds) == (6.0, 0.0, 3600)


def test_a_tie_in_everything_is_broken_by_the_origin_time_then_the_position():
    a = point(minutes=-5, magnitude=6.0, where=(3.0, -76.0))
    b = point(minutes=-5, magnitude=6.0, where=(3.9, -76.9), source="emsc")
    results = {check(claims(6.0), order).event_magnitude for order in ([a, b], [b, a])}
    assert results == {6.0}


def test_with_several_claims_the_finding_reports_the_best_difference_first():
    exact_far = point(minutes=-10, magnitude=6.0)
    off_near = point(minutes=-5, magnitude=5.5)
    finding = check(claims(6.0, 5.0), [exact_far, off_near])
    assert finding.status == CONSISTENT
    assert (finding.delta_magnitude, finding.event_magnitude, finding.seconds) == (0.0, 6.0, 600)


@pytest.mark.parametrize("order", [0, 1])
def test_with_several_claims_an_equal_difference_reports_the_nearest_event(order):
    far = point(minutes=-10, magnitude=6.0)
    near = point(minutes=-5, magnitude=5.0)
    events = [far, near]
    random.Random(order).shuffle(events)
    finding = check(claims(6.0, 5.0), events)
    assert (finding.delta_magnitude, finding.seconds) == (0.0, 300)
