"""The alert policy (AL-1, AL-2): which verdicts alert and how severe, pinned at every boundary."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.alerts.domain import (
    AlertSettings,
    Severity,
    assess,
    event_age,
    qualifies,
    severity_for,
)
from social_tracker.classification.domain import Label
from social_tracker.shared.errors import ValidationError

SETTINGS = AlertSettings()
RISK = [Label.UNINTENTIONAL_RUMOR, Label.DELIBERATE_DISINFORMATION, Label.RECYCLED_CONTENT]
NOT_RISK = [Label.NO_RISK, Label.UNCLASSIFIED, Label.OUT_OF_SCOPE]
DAY = timedelta(hours=24)


def sev(label, confidence, *, catalog="unavailable", age=timedelta(hours=30)):
    return severity_for(label, confidence, catalog_status=catalog, event_age=age, settings=SETTINGS)


# ---- AL-1: which verdicts alert ----


@pytest.mark.parametrize("label", RISK)
@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.49, False), (0.4999, False), (0.5, True), (0.51, True), (1.0, True)],
)
def test_a_risk_label_alerts_from_the_default_threshold_up(label, confidence, expected):
    assert qualifies(label, confidence, 0.5) is expected


@pytest.mark.parametrize("label", NOT_RISK)
@pytest.mark.parametrize("confidence", [0.0, 0.5, 0.99, 1.0])
def test_a_label_that_is_no_risk_never_alerts_whatever_its_confidence(label, confidence):
    assert qualifies(label, confidence, 0.0) is False
    assert (
        assess(
            label,
            confidence,
            min_confidence=0.0,
            catalog_status="mismatch",
            event_age=timedelta(0),
            settings=SETTINGS,
        )
        is None
    )


@pytest.mark.parametrize(
    ("threshold", "confidence", "expected"),
    [
        (0.0, 0.0, True),
        (0.3, 0.29, False),
        (0.3, 0.3, True),
        (0.9, 0.89, False),
        (0.9, 0.9, True),
        (1.0, 0.99, False),
        (1.0, 1.0, True),
    ],
)
def test_the_threshold_is_configurable_and_inclusive(threshold, confidence, expected):
    assert qualifies(Label.UNINTENTIONAL_RUMOR, confidence, threshold) is expected


@pytest.mark.parametrize("bad", [True, False, None, "0.9", math.nan, math.inf, -0.1, 1.01])
def test_a_confidence_that_is_not_a_probability_is_refused(bad):
    with pytest.raises(ValidationError):
        qualifies(Label.UNINTENTIONAL_RUMOR, bad, 0.5)


@pytest.mark.parametrize("bad", [True, None, "0.5", math.nan, -0.1, 1.01])
def test_a_threshold_that_is_not_a_probability_is_refused(bad):
    with pytest.raises(ValidationError):
        qualifies(Label.UNINTENTIONAL_RUMOR, 0.9, bad)


def test_an_unknown_label_is_refused():
    with pytest.raises(ValidationError):
        qualifies("made_up", 0.9, 0.5)


# ---- AL-2: severity ----


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.5, Severity.MEDIUM),
        (0.69, Severity.MEDIUM),
        (0.6999, Severity.MEDIUM),
        (0.7, Severity.HIGH),
        (0.84, Severity.HIGH),
        (0.99, Severity.HIGH),
        (1.0, Severity.HIGH),
    ],
)
def test_a_rumor_is_medium_from_half_and_high_from_seven_tenths_and_never_critical(
    confidence, expected
):
    assert sev(Label.UNINTENTIONAL_RUMOR, confidence) is expected
    assert sev(Label.RECYCLED_CONTENT, confidence) is expected


@pytest.mark.parametrize("confidence", [0.0, 0.3, 0.49])
def test_a_qualifying_verdict_under_the_medium_floor_is_low(confidence):
    # reachable when the operator lowers the alert threshold under 0.5
    assert sev(Label.UNINTENTIONAL_RUMOR, confidence) is Severity.LOW


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.69, Severity.MEDIUM),
        (0.7, Severity.HIGH),
        (0.84, Severity.HIGH),
        (0.8499, Severity.HIGH),
        (0.85, Severity.CRITICAL),
        (0.86, Severity.CRITICAL),
        (1.0, Severity.CRITICAL),
    ],
)
def test_deliberate_disinformation_is_critical_from_85_percent(confidence, expected):
    assert sev(Label.DELIBERATE_DISINFORMATION, confidence) is expected


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (timedelta(0), Severity.CRITICAL),
        (DAY - timedelta(seconds=1), Severity.CRITICAL),
        (DAY, Severity.CRITICAL),
        (DAY + timedelta(seconds=1), Severity.HIGH),
        (timedelta(days=30), Severity.HIGH),
    ],
)
def test_a_catalog_contradiction_on_a_fresh_event_is_critical_at_the_edge_of_24_hours(
    age, expected
):
    assert sev(Label.UNINTENTIONAL_RUMOR, 0.8, catalog="mismatch", age=age) is expected


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.7, Severity.HIGH),
        (0.79, Severity.HIGH),
        (0.7999, Severity.HIGH),
        (0.8, Severity.CRITICAL),
        (0.81, Severity.CRITICAL),
    ],
)
def test_the_catalog_route_to_critical_needs_80_percent(confidence, expected):
    assert sev(Label.UNINTENTIONAL_RUMOR, confidence, catalog="mismatch", age=DAY) is expected


@pytest.mark.parametrize("status", ["consistent", "no_claim", "unavailable", "not_checked", "x"])
def test_only_a_mismatch_counts_as_a_contradiction(status):
    assert sev(Label.UNINTENTIONAL_RUMOR, 0.95, catalog=status, age=timedelta(0)) is Severity.HIGH


def test_the_thresholds_are_configurable():
    custom = AlertSettings(
        critical_disinformation=0.95,
        critical_catalog=0.9,
        high=0.8,
        medium=0.6,
        critical_window=timedelta(hours=2),
    )

    def run(label, confidence, catalog="unavailable", age=timedelta(0)):
        return severity_for(
            label, confidence, catalog_status=catalog, event_age=age, settings=custom
        )

    assert run(Label.DELIBERATE_DISINFORMATION, 0.94) is Severity.HIGH
    assert run(Label.DELIBERATE_DISINFORMATION, 0.95) is Severity.CRITICAL
    assert run(Label.UNINTENTIONAL_RUMOR, 0.79) is Severity.MEDIUM
    assert run(Label.UNINTENTIONAL_RUMOR, 0.59) is Severity.LOW
    assert run(Label.UNINTENTIONAL_RUMOR, 0.9, "mismatch", timedelta(hours=2)) is Severity.CRITICAL
    late = run(Label.UNINTENTIONAL_RUMOR, 0.9, "mismatch", timedelta(hours=2, seconds=1))
    assert late is Severity.HIGH


def test_assess_combines_both_questions():
    kwargs = {"catalog_status": "unavailable", "event_age": DAY, "settings": SETTINGS}
    assert assess(Label.UNINTENTIONAL_RUMOR, 0.49, min_confidence=0.5, **kwargs) is None
    assert assess(Label.UNINTENTIONAL_RUMOR, 0.5, min_confidence=0.5, **kwargs) is Severity.MEDIUM
    assert (
        assess(Label.DELIBERATE_DISINFORMATION, 0.85, min_confidence=0.5, **kwargs)
        is Severity.CRITICAL
    )
    # a raised threshold hides what would have been medium
    assert assess(Label.UNINTENTIONAL_RUMOR, 0.65, min_confidence=0.7, **kwargs) is None


def test_the_policy_is_deterministic():
    runs = {
        sev(Label.DELIBERATE_DISINFORMATION, 0.851, catalog="mismatch", age=DAY) for _ in range(20)
    }
    assert runs == {Severity.CRITICAL}


# ---- the age of an event ----


def test_the_age_is_measured_from_the_publication_to_the_first_sighting():
    published = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
    assert event_age(published, published + timedelta(hours=5)) == timedelta(hours=5)


def test_a_publication_date_in_the_future_counts_as_no_age_at_all():
    fetched = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
    assert event_age(fetched + timedelta(days=9), fetched) == timedelta(0)


def test_the_age_needs_aware_datetimes():
    with pytest.raises(ValidationError):
        event_age(datetime(2026, 3, 1), datetime(2026, 3, 2, tzinfo=UTC))


# ---- the settings ----


@pytest.mark.parametrize(
    "bad",
    [
        {"medium": 0.8, "high": 0.7},  # medium above high
        {"high": 1.1},
        {"medium": -0.1},
        {"critical_disinformation": 0.0},
        {"critical_catalog": 1.5},
        {"critical_window": timedelta(0)},
        {"critical_window": timedelta(days=400)},
        {"dedup_window": timedelta(seconds=-1)},
        {"retention_days": -1},
        {"retention_days": 3651},
        {"retention_days": True},
        {"retention_days": 1.5},
        {"medium": math.nan},
        {"high": "0.7"},
    ],
)
def test_incoherent_settings_are_refused(bad):
    with pytest.raises(ValidationError):
        AlertSettings(**bad)


@pytest.mark.parametrize("days", [0, 1, 180, 3650])
def test_retention_days_accept_zero_up_to_ten_years(days):
    assert AlertSettings(retention_days=days).retention_days == days


def test_the_defaults_are_the_products():
    assert SETTINGS.critical_disinformation == 0.85
    assert SETTINGS.critical_catalog == 0.8
    assert SETTINGS.high == 0.7
    assert SETTINGS.medium == 0.5
    assert SETTINGS.critical_window == DAY
    assert SETTINGS.dedup_window == DAY
    assert SETTINGS.retention_days == 180
