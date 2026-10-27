"""The alert vocabulary: the status machine (AL-5), tags, evidence, the projection of members
into an alert (AL-6, AL-10) and the list filter (AL-4)."""

import itertools
import math
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.alerts.domain import (
    Alert,
    AlertFilter,
    AlertStatus,
    Evidence,
    Severity,
    alert_id_for,
    apply_status,
    project,
    tags_for,
    validate_page,
)
from social_tracker.classification.domain import Label
from social_tracker.shared.errors import InvalidTransition, ValidationError
from tests.alerts.support import T0, make_evidence, make_member

NEW, SEEN, DISMISSED = AlertStatus.NEW, AlertStatus.SEEN, AlertStatus.DISMISSED
LEGAL = {(NEW, SEEN), (NEW, DISMISSED), (SEEN, DISMISSED), (DISMISSED, SEEN)}


# ---- AL-5: the status machine ----


@pytest.mark.parametrize(("current", "target"), sorted(LEGAL))
def test_every_legal_move_changes_the_status(current, target):
    result = apply_status(current, target)
    assert (result.status, result.changed) == (target, True)


@pytest.mark.parametrize("status", list(AlertStatus))
def test_asking_for_the_status_it_has_is_a_no_op(status):
    result = apply_status(status, status)
    assert (result.status, result.changed) == (status, False)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (c, t)
        for c, t in itertools.product(AlertStatus, AlertStatus)
        if c is not t and (c, t) not in LEGAL
    ],
)
def test_every_other_move_is_refused(current, target):
    with pytest.raises(InvalidTransition):
        apply_status(current, target)


def test_there_is_no_way_back_to_new():
    for current in (SEEN, DISMISSED):
        with pytest.raises(InvalidTransition):
            apply_status(current, NEW)


@pytest.mark.parametrize("bad", ["", "NEW", "Seen", "withdrawn", "archived", None, 3, "seen "])
def test_an_unknown_status_is_a_validation_error_not_a_transition(bad):
    with pytest.raises(ValidationError):
        apply_status(NEW, bad)
    with pytest.raises(ValidationError):
        apply_status(bad, SEEN)


def test_plain_strings_are_accepted_for_known_statuses():
    assert apply_status("new", "seen").status is SEEN


# ---- tags ----


def test_tags_carry_the_topics_and_the_secretariat_attribution():
    tags = tags_for(
        ("earthquake", "secretariat"), ("official_attribution_unlinked",), "unavailable"
    )
    assert tags == ("earthquake", "secretariat", "secretariat_attribution")


def test_an_attribution_without_the_secretariat_topic_is_not_a_secretariat_attribution():
    assert tags_for(("earthquake",), ("official_attribution_unlinked",), "consistent") == (
        "earthquake",
    )


def test_the_secretariat_topic_alone_is_not_an_attribution():
    assert tags_for(("secretariat",), ("all_caps",), "unavailable") == ("secretariat",)


def test_evidence_of_a_contradiction_a_prediction_and_an_evasion_become_tags():
    tags = tags_for(
        ("earthquake",), ("prediction_specific_time", "obfuscated_text", "all_caps"), "mismatch"
    )
    assert tags == ("catalog_contradiction", "earthquake", "obfuscated_text", "prediction")


def test_tags_are_sorted_unique_and_ignore_unknown_topics():
    assert tags_for(("other_disaster", "earthquake", "earthquake", "junk"), (), "x") == (
        "earthquake",
        "other_disaster",
    )


# ---- evidence ----


def test_evidence_survives_a_round_trip_through_json():
    evidence = make_evidence(
        catalog_status="mismatch",
        settle_reason="llm",
        llm_model="st-gemma4-e4b",
        prompt_version="p-3",
        near_of="itm_other",
        provisional=True,
        truncated=True,
        source_credibility=5,
    )
    assert Evidence.from_json(evidence.to_json()) == evidence


def test_evidence_json_is_deterministic():
    assert make_evidence().to_json() == make_evidence().to_json()


@pytest.mark.parametrize(
    "bad",
    [
        {"catalog_status": "Not Valid"},
        {"catalog_status": ""},
        {"stage": ""},
        {"source_credibility": 0},
        {"source_credibility": True},
        {"provisional": "yes"},
        {"near_of": "NOT A SLUG"},
        {"llm_model": "x" * 101},
        {"signals": (("a", "b", 0.1),) * 21},
    ],
)
def test_bad_evidence_is_refused(bad):
    with pytest.raises(ValidationError):
        make_evidence(**bad)


@pytest.mark.parametrize("text", ["", "[]", "{}", "not json", '{"catalog_status": 1}', "null"])
def test_stored_evidence_that_does_not_parse_is_a_validation_error(text):
    with pytest.raises(ValidationError):
        Evidence.from_json(text)


def test_evidence_too_long_to_store_is_refused():
    big = make_evidence(signals=tuple((f"s{i}", "unintentional_rumor", 0.1) for i in range(20)))
    big.to_json()  # twenty signals fit
    huge = make_evidence(llm_model="m" * 100, prompt_version="p" * 100, settle_reason="r" * 100)
    huge.to_json()


# ---- members ----


def test_a_member_must_be_made_of_valid_parts():
    for bad in (
        {"excerpt": "x" * 501},
        {"excerpt": "has\x00nul"},
        {"rationale": ""},
        {"rationale": "r" * 301},
        {"source_name": ""},
        {"source_name": "bell\x07"},
        {"content_hash": "short"},
        {"confidence": 1.5},
        {"confidence": math.nan},
        {"severity": "urgent"},
        {"tags": ("Bad Tag",)},
        {"occurred_at": datetime(2026, 1, 1)},
        {"event_age_s": -1},
        {"active": 1},
    ):
        with pytest.raises(ValidationError):
            make_member(**bad)


def test_a_member_of_any_label_is_valid_so_a_withdrawn_verdict_can_be_kept():
    member = make_member(label=Label.NO_RISK, active=False)
    assert member.label is Label.NO_RISK


def test_the_alert_id_is_derived_from_the_first_item_only():
    assert alert_id_for("itm_one") == alert_id_for("itm_one")
    assert alert_id_for("itm_one") != alert_id_for("itm_two")
    assert alert_id_for("itm_one").startswith("alt_")
    assert len(alert_id_for("itm_one")) == 36
    with pytest.raises(ValidationError):
        alert_id_for("NOT A SLUG")


# ---- the projection ----

NOW = T0 + timedelta(hours=1)


def build(members, previous=None):
    return project("alt_x0000000000000000000000000000", members, previous=previous, now=NOW)


def test_one_member_makes_an_alert_that_shows_it():
    member = make_member("itm_a", severity=Severity.HIGH, confidence=0.75)
    alert = build([member])
    assert (alert.severity, alert.confidence, alert.item_id) == (Severity.HIGH, 0.75, "itm_a")
    assert alert.occurrences == 1
    assert alert.status is AlertStatus.NEW
    assert alert.created_at == NOW
    assert alert.withdrawn is False
    assert alert.linked_sources == (("src_news", "City News"),)


def test_the_most_severe_active_member_leads():
    low = make_member("itm_a", severity=Severity.MEDIUM, confidence=0.9)
    high = make_member("itm_b", severity=Severity.CRITICAL, confidence=0.86)
    alert = build([low, high])
    assert alert.item_id == "itm_b"
    assert alert.severity is Severity.CRITICAL
    assert alert.occurrences == 2


def test_merging_a_less_severe_duplicate_never_lowers_the_severity():
    critical = make_member("itm_a", severity=Severity.CRITICAL, confidence=0.9)
    alert = build([critical])
    merged = project(alert.id, [critical, make_member("itm_b", severity=Severity.LOW)],
                     previous=alert, now=NOW + timedelta(hours=1))  # fmt: skip
    assert merged.severity is Severity.CRITICAL
    assert merged.occurrences == 2


def test_merging_a_more_severe_duplicate_raises_it():
    medium = make_member("itm_a", severity=Severity.MEDIUM)
    alert = build([medium])
    merged = project(alert.id, [medium, make_member("itm_b", severity=Severity.HIGH)],
                     previous=alert, now=NOW)  # fmt: skip
    assert merged.severity is Severity.HIGH


def test_the_projection_does_not_depend_on_the_order_of_the_members():
    members = [make_member(f"itm_{n}", severity=Severity.HIGH, confidence=0.7) for n in "abcd"]
    first = build(members)
    assert all(build(list(p)) == first for p in itertools.permutations(members))


def test_ties_are_broken_by_confidence_then_the_earliest_then_the_id():
    a = make_member("itm_a", severity=Severity.HIGH, confidence=0.75, occurred_at=T0)
    b = make_member("itm_b", severity=Severity.HIGH, confidence=0.80, occurred_at=T0)
    c = make_member("itm_c", severity=Severity.HIGH, confidence=0.80, occurred_at=T0 - timedelta(1))
    assert build([a, b]).item_id == "itm_b"
    assert build([a, b, c]).item_id == "itm_c"
    d = make_member("itm_d", severity=Severity.HIGH, confidence=0.80, occurred_at=T0 - timedelta(1))
    assert build([c, d]).item_id == "itm_c"


def test_an_inactive_member_neither_leads_nor_counts():
    gone = make_member("itm_a", severity=Severity.CRITICAL, active=False)
    live = make_member("itm_b", severity=Severity.MEDIUM)
    alert = build([gone, live])
    assert alert.item_id == "itm_b"
    assert alert.severity is Severity.MEDIUM
    assert alert.occurrences == 1


def test_with_no_active_member_the_alert_keeps_what_it_showed():
    shown = build([make_member("itm_a", severity=Severity.HIGH, confidence=0.8)])
    after = project(
        shown.id,
        [make_member("itm_a", severity=Severity.HIGH, confidence=0.8, active=False)],
        previous=shown,
        now=NOW + timedelta(hours=2),
    )
    assert (after.severity, after.confidence, after.item_id) == (Severity.HIGH, 0.8, "itm_a")
    assert after.occurrences == 0
    assert after.updated_at == NOW + timedelta(hours=2)


def test_the_status_and_the_creation_time_survive_a_new_projection():
    first = make_member("itm_a")
    alert = build([first])
    seen = Alert(**{**alert.__dict__, "status": AlertStatus.SEEN})
    again = project(alert.id, [first, make_member("itm_b")], previous=seen, now=NOW + timedelta(1))
    assert again.status is AlertStatus.SEEN
    assert again.created_at == NOW


def test_first_and_last_seen_span_all_the_members_and_sources_are_linked():
    early = make_member("itm_a", occurred_at=T0)
    late = make_member(
        "itm_b", occurred_at=T0 + timedelta(hours=5), source_id="src_other", source_name="Other"
    )
    alert = build([late, early])
    assert (alert.first_seen_at, alert.last_seen_at) == (T0, T0 + timedelta(hours=5))
    assert alert.linked_sources == (("src_news", "City News"), ("src_other", "Other"))


def test_tags_are_the_union_of_the_active_members():
    a = make_member("itm_a", tags=("earthquake",))
    b = make_member("itm_b", tags=("prediction", "secretariat_attribution"))
    c = make_member("itm_c", tags=("obfuscated_text",), active=False)
    assert build([a, b, c]).tags == ("earthquake", "prediction", "secretariat_attribution")


def test_a_projection_needs_members():
    with pytest.raises(ValidationError):
        build([])


# ---- the filter ----


def test_an_empty_filter_is_valid_and_has_nothing_set():
    f = AlertFilter()
    assert (f.severity, f.label, f.source_id, f.status, f.tag, f.since, f.until) == (None,) * 7


def test_a_full_filter_coerces_strings():
    f = AlertFilter(
        severity="high", label="recycled_content", source_id="src_news", status="seen",
        tag="earthquake", since=T0, until=T0 + timedelta(days=1),
    )  # fmt: skip
    assert f.severity is Severity.HIGH
    assert f.label is Label.RECYCLED_CONTENT
    assert f.status is AlertStatus.SEEN


def test_the_status_filter_also_knows_withdrawn():
    assert AlertFilter(status="withdrawn").status == "withdrawn"


@pytest.mark.parametrize(
    "bad",
    [
        {"severity": "urgent"},
        {"severity": ""},
        {"label": "no_risk"},
        {"label": "made_up"},
        {"source_id": "' OR 1=1 --"},
        {"source_id": "x"},
        {"status": "archived"},
        {"status": "' OR '1'='1"},
        {"tag": "Robert'); DROP TABLE alerts;--"},
        {"tag": "a" * 33},
        {"tag": ""},
        {"since": datetime(2026, 1, 1)},
        {"until": "2026-01-01"},
    ],
)
def test_a_malformed_filter_is_refused(bad):
    with pytest.raises(ValidationError):
        AlertFilter(**bad)


def test_the_end_of_the_range_must_not_precede_its_start():
    with pytest.raises(ValidationError):
        AlertFilter(since=T0, until=T0 - timedelta(microseconds=1))
    assert AlertFilter(since=T0, until=T0).until == T0


@pytest.mark.parametrize(("limit", "offset"), [(1, 0), (50, 0), (200, 1000), (1, 1_000_000)])
def test_a_page_in_range_is_accepted(limit, offset):
    assert validate_page(limit, offset) == (limit, offset)


@pytest.mark.parametrize(
    ("limit", "offset"),
    [
        (0, 0),
        (-1, 0),
        (201, 0),
        (10, -1),
        (10, 1_000_001),
        (True, 0),
        (10, True),
        ("5", 0),
        (5, None),
    ],
)
def test_a_page_out_of_range_is_refused(limit, offset):
    with pytest.raises(ValidationError):
        validate_page(limit, offset)


def test_naive_times_in_an_alert_are_refused():
    member = make_member()
    alert = build([member])
    with pytest.raises(ValidationError):
        Alert(**{**alert.__dict__, "created_at": datetime(2026, 1, 1)})
    with pytest.raises(ValidationError):
        Alert(**{**alert.__dict__, "withdrawn_at": datetime(2026, 1, 1, tzinfo=UTC)})
