"""Two workers and a person on one database file, after the remediation: a flood merged by two
workers, a rise racing a person's decision, a poison item failing in two workers and the purge
running next to the step. Whatever the interleaving, the invariants hold: one alert, a true count,
a trail that explains the status, and no row left without its alert."""

import contextlib
from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertEventKind, AlertFilter, AlertSettings, AlertStatus
from social_tracker.classification.domain import Label
from social_tracker.shared.errors import InvalidTransition
from tests.alerts.test_service_concurrency import run_together
from tests.alerts.world import alert_world, attach
from tests.classification.harness import PUBLISHED
from tests.llm.harness import sqlite_llm

MINUTE = timedelta(minutes=1)
START = PUBLISHED + MINUTE
SETTINGS = AlertSettings(max_members=10, max_events=10)


@pytest.fixture
def pair(tmp_path):
    first = alert_world(tmp_path, SETTINGS, name="shared.db")
    first.llm.ready()
    first.llm.ready("src_other")
    second = attach(sqlite_llm(tmp_path / "shared.db"), SETTINGS)
    yield first, second
    second.close()
    first.close()


def flood(world, count):
    original = world.put("orig", text="Mismo rumor repetido sobre temblor 0", fetched=START)
    for n in range(1, count + 1):
        world.put(
            f"c{n}",
            text=f"Mismo rumor repetido sobre temblor {n}",
            near_of=original.id,
            fetched=START + n * MINUTE,
        )
    return original


def orphans(world) -> int:
    """Rows of the alert tables that point at an alert that does not exist."""
    with world.db.snapshot() as conn:
        return sum(
            conn.execute(
                f"SELECT count(*) FROM {table} WHERE alert_id NOT IN (SELECT id FROM alerts)"
            ).fetchone()[0]
            for table in ("alert_items", "alert_events", "alert_overflow")
        )


def test_two_workers_merging_a_flood_leave_one_alert_with_a_true_count(pair):
    first, second = pair
    flood(first, 24)
    errors = run_together(first.service.refresh, second.service.refresh)
    assert errors == []
    alert = first.only()
    members = first.repo.members(alert.id)
    assert alert.occurrences == 25
    assert len(members) <= SETTINGS.max_members
    assert len(members) + alert.overflow == 25
    assert first.refresh().created == 0
    assert orphans(first) == 0


def test_two_workers_and_a_rebuild_over_a_flood_agree(pair):
    first, second = pair
    flood(first, 18)
    errors = run_together(
        first.service.refresh, second.service.rebuild, second.service.refresh, first.service.refresh
    )
    assert errors == []
    assert first.repo.count(AlertFilter()) == 1
    assert first.only().occurrences == 19
    assert first.service.rebuild().created == 0
    assert first.only().occurrences == 19


def test_a_rise_racing_a_persons_decision_leaves_a_status_the_trail_explains(pair):
    first, second = pair
    text = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    first.put("a", text=text, confidence=0.6, fetched=START)
    first.refresh()
    alert = first.only()
    first.put(
        "b",
        text=text,
        source="src_other",
        label=Label.DELIBERATE_DISINFORMATION,
        confidence=0.95,
        fetched=START + MINUTE,
    )

    def decide():
        for target in ("seen", "dismissed"):
            # the rise may come in between: `new` cannot go to the same place twice
            with contextlib.suppress(InvalidTransition):
                second.service.set_status(alert.id, target, actor="ana", reason="reviewed")

    errors = run_together(first.service.refresh, decide)
    assert errors == []
    stored = first.repo.get(alert.id)
    assert stored.severity.value == "critical"
    changes = [e for e in first.repo.events(alert.id) if e.kind is AlertEventKind.STATUS_CHANGED]
    for rise in (e for e in changes if e.actor == "system:alerts"):
        assert (rise.after, rise.before in ("seen", "dismissed")) == ("new", True)
    assert changes[-1].after == stored.status.value  # the last change explains where it ended
    if stored.status is AlertStatus.NEW:
        assert any(e.actor == "system:alerts" for e in changes)


def test_a_poison_item_failing_in_two_workers_is_counted_not_lost(pair, monkeypatch):
    first, second = pair
    bad = first.put("bad", text="rumor venenoso")
    first.put("good", text="rumor sano")
    for world in (first, second):
        real = world.service._member  # noqa: SLF001

        def guarded(alert_id, verdict, item, *args, __real=real, **kwargs):
            if item.id == bad.id:
                raise ValueError("poison")
            return __real(alert_id, verdict, item, *args, **kwargs)

        monkeypatch.setattr(world.service, "_member", guarded)
    errors = run_together(first.service.refresh, second.service.refresh)
    assert errors == []
    counts = first.service.counts()
    assert (counts.retrying, counts.parked) == (1, 0)
    assert first.repo.count(AlertFilter()) == 1


def test_the_purge_next_to_the_step_and_a_person_never_orphans_a_row(tmp_path):
    settings = AlertSettings(retention_days=0, max_members=10, max_events=10)
    first = alert_world(tmp_path, settings, name="purge.db")
    first.llm.ready()
    second = attach(sqlite_llm(tmp_path / "purge.db"), settings)
    try:
        for n in range(15):
            first.put(f"n{n}", text=f"rumor distinto numero {n} sobre el temblor", confidence=0.7)
        first.refresh()
        ids = [a.id for a in first.alerts()]

        def decide():
            for alert_id in ids[:8]:
                second.service.set_status(alert_id, "dismissed", actor="ana", reason="noise")

        errors = run_together(first.service.refresh, decide, first.service.purge_expired)
        assert errors == []
        assert orphans(first) == 0
        first.service.purge_expired()
        assert orphans(first) == 0
        assert first.repo.count(AlertFilter()) == 7  # the seven nobody looked at
    finally:
        second.close()
        first.close()
