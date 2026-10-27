"""A3: what the age of an event can and cannot do to a severity.

The first sighting is the application's own clock; the publication date is the feed's. A date
that is missing or in the future (the collector replaced it by the fetch time and flagged it) says
nothing about the age, so the event is NOT fresh and a catalog contradiction cannot make it
critical on that basis. A backdated date can only make an event look older, which lowers critical
(through the catalog) to high, never raises anything."""

from datetime import timedelta

import pytest

from social_tracker.alerts.domain import (
    UNKNOWN_AGE,
    AlertSettings,
    Severity,
    assess,
    event_age,
)
from social_tracker.classification.domain import Label
from tests.alerts.support import T0
from tests.alerts.world import alert_world
from tests.classification.harness import PUBLISHED

SETTINGS = AlertSettings()
RUMOR = Label.UNINTENTIONAL_RUMOR


def severity(age: timedelta, *, confidence=0.85, catalog="mismatch") -> Severity | None:
    return assess(
        RUMOR,
        confidence,
        min_confidence=0.5,
        catalog_status=catalog,
        event_age=age,
        settings=SETTINGS,
    )


def test_a_known_fresh_event_with_a_contradicting_catalog_is_critical():
    assert severity(event_age(T0, T0 + timedelta(hours=1))) is Severity.CRITICAL


def test_an_unknown_date_is_never_fresh():
    assert event_age(T0, T0, date_known=False) == UNKNOWN_AGE
    assert SETTINGS.critical_window < UNKNOWN_AGE
    assert severity(event_age(T0, T0, date_known=False)) is Severity.HIGH


def test_an_unknown_date_ignores_the_publication_it_was_given():
    assert event_age(T0, T0 + timedelta(hours=100), date_known=False) == UNKNOWN_AGE


def test_a_backdated_publication_only_lowers_critical_to_high():
    fresh = severity(event_age(T0, T0 + timedelta(hours=1)))
    backdated = severity(event_age(T0 - timedelta(days=3), T0 + timedelta(hours=1)))
    assert (fresh, backdated) == (Severity.CRITICAL, Severity.HIGH)


@pytest.mark.parametrize("age_hours", [0, 1, 24])
def test_inside_the_window_the_contradiction_is_critical(age_hours):
    assert severity(timedelta(hours=age_hours)) is Severity.CRITICAL


def test_just_outside_the_window_it_is_high():
    assert severity(timedelta(hours=24, seconds=1)) is Severity.HIGH


@pytest.mark.parametrize("catalog", ["unavailable", "consistent", "no_claim"])
def test_only_a_mismatch_is_ever_critical_through_the_catalog(catalog):
    assert severity(timedelta(0), catalog=catalog) is Severity.HIGH


def test_a_backdated_date_cannot_raise_a_severity_that_nothing_else_raises():
    for hours in (0, 24, 24 * 30):
        result = severity(timedelta(hours=hours), catalog="unavailable")
        assert result is Severity.HIGH


# ---- in the service ----


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    yield world
    world.close()


def contradicted(w, name, *, published=PUBLISHED, flags=(), fetched=None):
    item = w.put(
        name,
        confidence=0.9,
        text=f"Sismo de magnitud 7 ahora mismo en Cali {name}",
        catalog="mismatch",
        published=published,
        fetched=fetched,
    )
    if flags:
        import json

        with w.db.transaction() as conn:
            conn.execute(
                "UPDATE items SET flags = ? WHERE id = ?", (json.dumps(list(flags)), item.id)
            )
    return item


def test_a_contradiction_about_a_fresh_dated_event_is_critical(w):
    contradicted(w, "a", published=PUBLISHED, fetched=PUBLISHED + timedelta(hours=1))
    w.refresh()
    assert w.only().severity is Severity.CRITICAL


@pytest.mark.parametrize("flag", ["published_at_missing", "published_at_future"])
def test_a_contradiction_about_an_event_of_unknown_age_is_only_high(w, flag):
    contradicted(w, "a", flags=(flag,))
    w.refresh()
    assert w.only().severity is Severity.HIGH


def test_other_flags_do_not_make_the_age_unknown(w):
    contradicted(w, "a", flags=("truncated",), fetched=PUBLISHED + timedelta(hours=1))
    w.refresh()
    assert w.only().severity is Severity.CRITICAL


def test_a_backdated_feed_date_lowers_the_contradiction_to_high(w):
    contradicted(
        w,
        "a",
        published=PUBLISHED - timedelta(days=5),
        fetched=PUBLISHED + timedelta(hours=1),
    )
    w.refresh()
    assert w.only().severity is Severity.HIGH


def test_the_age_is_kept_with_the_member_so_a_rebuild_does_not_change_it(w):
    contradicted(w, "a", flags=("published_at_missing",))
    w.refresh()
    w.service.rebuild()
    assert w.only().severity is Severity.HIGH
