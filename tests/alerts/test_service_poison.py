"""A13, A14, A6: the step looks only at what changed (a dirty mark), a poison item backs off and is
parked instead of being retried at every tick, a catalog or evidence change refreshes an alert, and
a moved threshold is noticed by a plain refresh."""

import dataclasses
from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertEventKind, Severity
from social_tracker.alerts.service import PARK_AFTER
from social_tracker.classification.domain import Label
from tests.alerts.world import HOUR, alert_world


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    yield world
    world.close()


def count_syncs(w, monkeypatch):
    seen: list[str] = []
    real = w.service._sync  # noqa: SLF001

    def spy(item_id, threshold):
        seen.append(item_id)
        return real(item_id, threshold)

    monkeypatch.setattr(w.service, "_sync", spy)
    return seen


def poison(w, monkeypatch, item):
    calls: list[str] = []
    real = w.service._member  # noqa: SLF001

    def broken(alert_id, verdict, it, *args, **kwargs):
        if it.id == item.id:
            calls.append(it.id)
            raise ValueError("poison")
        return real(alert_id, verdict, it, *args, **kwargs)

    monkeypatch.setattr(w.service, "_member", broken)
    return calls


# ---- the step looks only at what changed ----


def test_a_quiet_tick_looks_at_nothing(w, monkeypatch):
    for n in range(5):
        w.put(f"n{n}", text=f"rumor distinto numero {n}")
    w.refresh()
    seen = count_syncs(w, monkeypatch)
    report = w.refresh()
    assert seen == []
    assert dataclasses.astuple(report) == (0,) * 8


def test_only_the_item_whose_verdict_was_written_is_looked_at(w, monkeypatch):
    items = [w.put(f"n{n}", text=f"rumor distinto numero {n}") for n in range(5)]
    w.refresh()
    seen = count_syncs(w, monkeypatch)
    with w.db.transaction() as conn:
        conn.execute(
            "UPDATE classifications SET confidence = 0.8 WHERE item_id = ?", (items[2].id,)
        )
    report = w.refresh()
    assert seen == [items[2].id]
    assert report.updated == 1


def test_a_verdict_that_does_not_alert_is_looked_at_once_and_forgotten_by_the_step(w, monkeypatch):
    w.put("quiet", label=Label.NO_RISK, confidence=0.9, text="nada que ver")
    seen = count_syncs(w, monkeypatch)
    w.refresh()
    w.refresh()
    assert seen == []  # a no_risk verdict never even marks the item


def test_a_risk_verdict_under_the_threshold_is_looked_at_once(w, monkeypatch):
    w.put("low", confidence=0.3, text="rumor bajo")
    seen = count_syncs(w, monkeypatch)
    w.refresh()
    w.refresh()
    assert len(seen) == 1


# ---- A13: a change of evidence or catalog refreshes the alert ----


def test_a_catalog_change_with_the_same_label_and_confidence_refreshes_the_alert(w):
    item = w.put(
        "r", confidence=0.9, label=Label.UNINTENTIONAL_RUMOR, text="r1", catalog="unavailable"
    )
    w.refresh()
    assert w.only().severity is Severity.HIGH
    current = w.llm.classifications.get(item.id)
    w.llm.classifications.delete(item.id)
    w.llm.classifications.add(
        dataclasses.replace(current, catalog_status="mismatch", classified_at=w.clock.now() + HOUR)
    )
    report = w.refresh()
    alert = w.only()
    assert report.updated == 1
    assert (alert.severity, alert.evidence.catalog_status) == (Severity.CRITICAL, "mismatch")
    assert "catalog_contradiction" in alert.tags


def test_new_signals_on_the_same_verdict_refresh_the_evidence(w):
    from social_tracker.classification.domain import Signal

    item = w.put("r", confidence=0.7, text="r1")
    w.refresh()
    current = w.llm.classifications.get(item.id)
    w.llm.classifications.delete(item.id)
    w.llm.classifications.add(
        dataclasses.replace(
            current,
            signals=(*current.signals, Signal("obfuscated_text", Label.UNINTENTIONAL_RUMOR, 0.2)),
            classified_at=w.clock.now() + HOUR,
        )
    )
    w.refresh()
    assert "obfuscated_text" in w.only().tags


# ---- A14: a poison item ----


def test_a_poison_item_does_not_stall_the_others(w, monkeypatch):
    bad = w.put("bad", text="rumor venenoso")
    good = w.put("good", text="rumor sano")
    calls = poison(w, monkeypatch, bad)
    report = w.refresh()
    assert (report.created, report.errors) == (1, 1)
    assert [a.item_id for a in w.alerts()] == [good.id]
    assert calls == [bad.id]


def test_a_poison_item_waits_before_it_is_tried_again(w, monkeypatch):
    bad = w.put("bad", text="rumor venenoso")
    calls = poison(w, monkeypatch, bad)
    w.refresh()
    assert w.refresh().errors == 0  # still backing off: not even tried
    assert calls == [bad.id]
    w.clock.advance(timedelta(seconds=31))
    assert w.refresh().errors == 1
    assert len(calls) == 2


def test_a_poison_item_is_parked_after_a_few_failures_and_the_stats_say_so(w, monkeypatch):
    bad = w.put("bad", text="rumor venenoso")
    calls = poison(w, monkeypatch, bad)
    for _ in range(PARK_AFTER):
        w.refresh()
        w.clock.advance(timedelta(hours=2))
    assert len(calls) == PARK_AFTER
    counts = w.service.counts()
    assert (counts.parked, counts.retrying) == (1, 0)
    for _ in range(3):
        assert w.refresh().errors == 0
        w.clock.advance(timedelta(days=1))
    assert len(calls) == PARK_AFTER  # never again by itself


def test_a_failing_item_is_counted_as_retrying_while_it_backs_off(w, monkeypatch):
    bad = w.put("bad", text="rumor venenoso")
    poison(w, monkeypatch, bad)
    w.refresh()
    counts = w.service.counts()
    assert (counts.retrying, counts.parked) == (1, 0)


def test_a_rebuild_gives_the_parked_items_another_chance(w, monkeypatch):
    bad = w.put("bad", text="rumor venenoso")
    calls = poison(w, monkeypatch, bad)
    for _ in range(PARK_AFTER):
        w.refresh()
        w.clock.advance(timedelta(hours=2))
    report = w.service.rebuild()
    assert report.errors == 1
    assert len(calls) == PARK_AFTER + 1


def test_an_item_that_works_again_is_forgiven(w, monkeypatch):
    bad = w.put("bad", text="rumor venenoso")
    poison(w, monkeypatch, bad)
    w.refresh()
    monkeypatch.undo()
    w.clock.advance(timedelta(minutes=2))
    report = w.refresh()
    assert (report.created, report.errors) == (1, 0)
    counts = w.service.counts()
    assert (counts.retrying, counts.parked) == (0, 0)


# ---- A6: a moved threshold is noticed by a plain refresh ----


def test_a_raised_threshold_is_applied_by_a_plain_refresh(w):
    w.put("d", confidence=0.6, text="d uno")
    w.put("e", confidence=0.65, text="e dos")
    w.refresh()
    w.control.set_alert_min_confidence(0.7, actor="op", reason="stricter")
    report = w.refresh()
    assert report.withdrawn == 2
    assert w.alerts() == []
    assert len(w.alerts(status="withdrawn")) == 2


def test_a_lowered_threshold_is_applied_by_a_plain_refresh(w):
    w.control.set_alert_min_confidence(0.8, actor="op", reason="quiet")
    w.put("d", confidence=0.6, text="d uno")
    w.refresh()
    assert w.alerts() == []
    w.control.set_alert_min_confidence(0.5, actor="op", reason="normal")
    assert w.refresh().created == 1


def test_the_threshold_a_pass_used_is_remembered(w):
    w.put("d", confidence=0.6, text="d uno")
    assert w.control.liveness().synced_threshold is None
    w.refresh()
    assert w.control.liveness().synced_threshold == 0.5
    w.control.set_alert_min_confidence(0.7, actor="op", reason="stricter")
    w.refresh()
    assert w.control.liveness().synced_threshold == 0.7


def test_a_refresh_with_the_same_threshold_does_not_look_at_every_member(w, monkeypatch):
    for n in range(4):
        w.put(f"n{n}", text=f"rumor distinto numero {n}")
    w.refresh()
    seen = count_syncs(w, monkeypatch)
    w.refresh()
    assert seen == []


def test_when_the_full_pass_fails_somewhere_the_threshold_is_not_marked_as_applied(w, monkeypatch):
    bad = w.put("bad", text="rumor venenoso")
    w.put("good", text="rumor sano")
    poison(w, monkeypatch, bad)
    w.refresh()
    assert w.control.liveness().synced_threshold is None
    monkeypatch.undo()
    w.clock.advance(timedelta(minutes=2))
    w.refresh()
    assert w.control.liveness().synced_threshold == 0.5


def test_every_refresh_leaves_its_time_for_the_operator_page(w):
    assert w.control.liveness().last_alert_refresh_at is None
    w.refresh()
    first = w.control.liveness().last_alert_refresh_at
    assert first == w.clock.now()
    w.clock.advance(timedelta(minutes=3))
    w.refresh()
    assert w.control.liveness().last_alert_refresh_at == first + timedelta(minutes=3)


def test_a_status_event_is_not_lost_when_the_threshold_withdraws_a_handled_alert(w):
    w.put("d", confidence=0.6, text="d uno")
    w.refresh()
    w.service.set_status(w.only().id, "dismissed", actor="op", reason="noise")
    w.control.set_alert_min_confidence(0.9, actor="op", reason="stricter")
    w.refresh()
    kinds = [e.kind for e in w.repo.events(w.only(status="withdrawn").id)]
    assert kinds == [
        AlertEventKind.CREATED,
        AlertEventKind.STATUS_CHANGED,
        AlertEventKind.WITHDRAWN,
    ]
