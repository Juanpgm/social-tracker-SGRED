"""A1: a status the operator chose (seen, dismissed) never hides a HIGHER severity. When the
severity of an alert rises (a stronger copy joined, the model upgraded a provisional verdict, a
withdrawn alert came back stronger, bridged alerts became one) the alert goes back to `new` with an
audit entry by the system; the same or a lower severity keeps the operator's status."""

import dataclasses

import pytest

from social_tracker.alerts.domain import (
    ACTOR_SYSTEM,
    AlertEventKind,
    AlertFilter,
    AlertStatus,
    Severity,
)
from social_tracker.classification.domain import Label, Stage
from tests.alerts.world import HOUR, alert_world

RUMOR, DELIBERATE = Label.UNINTENTIONAL_RUMOR, Label.DELIBERATE_DISINFORMATION
SAME = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
OTHER = "src_other"


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    world.llm.ready(OTHER)
    yield world
    world.close()


def settle(w, item, **changes):
    """The model settles the verdict of a provisional item."""
    current = w.llm.classifications.get(item.id)
    w.clock.advance(HOUR)
    settled = dataclasses.replace(
        current,
        stage=Stage.LLM,
        needs_llm=False,
        priority=0.0,
        settled_at=w.clock.now(),
        settle_reason="llm",
        **changes,
    )
    assert w.llm.classifications.settle(settled)


def mark(w, alert, status):
    w.service.set_status(alert.id, status, actor="op", reason="checked")


def rise_events(w, alert_id):
    return [
        e
        for e in w.repo.events(alert_id)
        if e.kind is AlertEventKind.STATUS_CHANGED and e.actor == ACTOR_SYSTEM
    ]


# ---- the merge path ----


@pytest.mark.parametrize("operator", ["seen", "dismissed"])
def test_a_stronger_copy_brings_a_handled_alert_back_to_new(w, operator):
    w.put("a", text=SAME, confidence=0.6)
    w.refresh()
    mark(w, w.only(), operator)
    w.clock.advance(HOUR)
    w.put("b", text=SAME, label=DELIBERATE, confidence=0.95, source=OTHER, fetched=w.clock.now())
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.status) == (Severity.CRITICAL, AlertStatus.NEW)
    assert w.service.counts().new == 1
    (event,) = rise_events(w, alert.id)
    assert (event.before, event.after) == (operator, "new")
    assert "severity rose from medium to critical" in event.reason
    assert alert.status_changed_at == event.at


def test_the_rise_is_audited_by_the_system_not_by_a_person(w):
    w.put("a", text=SAME, confidence=0.6)
    w.refresh()
    mark(w, w.only(), "dismissed")
    w.put("b", text=SAME, label=DELIBERATE, confidence=0.95, source=OTHER)
    w.refresh()
    trail = w.repo.events(w.only().id)
    kinds = [(e.kind, e.actor) for e in trail]
    assert (AlertEventKind.STATUS_CHANGED, "op") in kinds
    assert (AlertEventKind.STATUS_CHANGED, ACTOR_SYSTEM) in kinds


def test_a_step_from_medium_to_high_resets_the_status(w):
    w.put("a", text=SAME, confidence=0.55)
    w.refresh()
    mark(w, w.only(), "dismissed")
    w.put("b", text=SAME, confidence=0.75, source=OTHER)
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.status) == (Severity.HIGH, AlertStatus.NEW)
    assert "severity rose from medium to high" in rise_events(w, alert.id)[0].reason


@pytest.mark.parametrize("second", [0.6, 0.55, 0.5])
def test_an_equal_or_lower_copy_keeps_the_operator_status(w, second):
    w.put("a", text=SAME, confidence=0.6)
    w.refresh()
    mark(w, w.only(), "dismissed")
    w.put("b", text=SAME, confidence=second, source=OTHER)
    w.refresh()
    alert = w.only()
    assert alert.status is AlertStatus.DISMISSED
    assert rise_events(w, alert.id) == []
    assert w.service.counts().new == 0


def test_a_lower_copy_of_a_high_alert_keeps_seen(w):
    w.put("a", text=SAME, confidence=0.8)
    w.refresh()
    mark(w, w.only(), "seen")
    w.put("b", text=SAME, confidence=0.55, source=OTHER)
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.status) == (Severity.HIGH, AlertStatus.SEEN)


def test_a_new_alert_that_rises_writes_no_status_entry(w):
    w.put("a", text=SAME, confidence=0.6)
    w.refresh()
    w.put("b", text=SAME, label=DELIBERATE, confidence=0.95, source=OTHER)
    w.refresh()
    alert = w.only()
    assert alert.status is AlertStatus.NEW
    assert rise_events(w, alert.id) == []


def test_two_rises_in_a_row_reset_twice(w):
    w.put("a", text=SAME, confidence=0.55)
    w.refresh()
    mark(w, w.only(), "dismissed")
    w.put("b", text=SAME, confidence=0.75, source=OTHER)
    w.refresh()
    mark(w, w.only(), "seen")
    w.put("c", text=SAME, label=DELIBERATE, confidence=0.95, source="src_news_two")
    w.llm.ready("src_news_two")
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.status) == (Severity.CRITICAL, AlertStatus.NEW)
    assert len(rise_events(w, alert.id)) == 2


# ---- the model path ----


@pytest.mark.parametrize("operator", ["seen", "dismissed"])
def test_a_model_upgrade_of_a_provisional_member_resets_the_status(w, operator):
    item = w.put("c", confidence=0.55, needs_llm=True, text="rumor unico c")
    w.refresh()
    mark(w, w.only(), operator)
    settle(w, item, label=DELIBERATE, confidence=0.92)
    report = w.refresh()
    alert = w.only()
    assert report.updated == 1
    assert (alert.severity, alert.status) == (Severity.CRITICAL, AlertStatus.NEW)
    assert w.service.counts().new == 1
    (event,) = rise_events(w, alert.id)
    assert "severity rose from medium to critical" in event.reason
    assert "verdict upgraded" in event.reason


def test_a_model_verdict_of_the_same_severity_keeps_the_status(w):
    item = w.put("c", confidence=0.55, needs_llm=True, text="rumor unico c")
    w.refresh()
    mark(w, w.only(), "dismissed")
    settle(w, item, confidence=0.6)
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.status) == (Severity.MEDIUM, AlertStatus.DISMISSED)
    assert rise_events(w, alert.id) == []


# ---- the reactivation path ----


def test_a_withdrawn_alert_that_comes_back_stronger_is_new_again(w):
    item = w.put("c", confidence=0.6, text="rumor unico c")
    w.refresh()
    mark(w, w.only(), "seen")
    current = w.llm.classifications.get(item.id)
    w.llm.classifications.delete(item.id)
    w.llm.classifications.add(dataclasses.replace(current, label=Label.NO_RISK, confidence=0.9))
    w.refresh()
    assert w.only(status="withdrawn").status is AlertStatus.SEEN
    w.llm.classifications.delete(item.id)
    w.llm.classifications.add(
        dataclasses.replace(current, label=DELIBERATE, confidence=0.95, classified_at=w.clock.now())
    )
    report = w.refresh()
    alert = w.only()
    assert report.reactivated == 1
    assert (alert.withdrawn, alert.severity, alert.status) == (
        False,
        Severity.CRITICAL,
        AlertStatus.NEW,
    )
    assert len(rise_events(w, alert.id)) == 1


def test_a_withdrawn_alert_that_comes_back_the_same_keeps_the_status(w):
    item = w.put("c", confidence=0.6, text="rumor unico c")
    w.refresh()
    mark(w, w.only(), "dismissed")
    current = w.llm.classifications.get(item.id)
    w.llm.classifications.delete(item.id)
    w.llm.classifications.add(dataclasses.replace(current, label=Label.NO_RISK, confidence=0.9))
    w.refresh()
    w.llm.classifications.delete(item.id)
    w.llm.classifications.add(dataclasses.replace(current, confidence=0.61))
    w.refresh()
    alert = w.only()
    assert (alert.withdrawn, alert.status) == (False, AlertStatus.DISMISSED)


# ---- the absorb path ----


def bridged(w, bridge_label=RUMOR, bridge_confidence=0.6, statuses=("dismissed", "dismissed")):
    a = w.put("a", text="Rumor uno sobre temblor en Cali", confidence=0.6)
    b = w.put("b", text="Otro tema distinto de inundacion en Cali", confidence=0.6)
    w.refresh()
    by_item = {x.item_id: x for x in w.alerts()}
    mark(w, by_item[a.id], statuses[0])
    mark(w, by_item[b.id], statuses[1])
    # same words as A, and it says it is a near copy of B: it bridges the two alerts
    w.put(
        "c",
        text="Rumor uno sobre temblor en Cali",
        label=bridge_label,
        confidence=bridge_confidence,
        source=OTHER,
        near_of=b.id,
    )
    w.refresh()
    return w.only()


def test_bridging_alerts_with_a_stronger_item_resets_the_status(w):
    alert = bridged(w, DELIBERATE, 0.95)
    assert (alert.severity, alert.status) == (Severity.CRITICAL, AlertStatus.NEW)
    assert len(rise_events(w, alert.id)) == 1


def test_bridging_alerts_with_an_equal_item_keeps_the_handled_status(w):
    alert = bridged(w)
    assert (alert.severity, alert.status) == (Severity.MEDIUM, AlertStatus.DISMISSED)
    assert rise_events(w, alert.id) == []


def test_bridging_a_seen_and_a_dismissed_alert_keeps_the_one_that_asks_more(w):
    alert = bridged(w, statuses=("dismissed", "seen"))
    assert alert.status is AlertStatus.SEEN


def test_the_badge_counts_only_alerts_that_ask_for_attention(w):
    w.put("a", text=SAME, confidence=0.6)
    w.put("z", text="otro rumor sin relacion alguna zzz", confidence=0.6)
    w.refresh()
    first = w.alerts()[0]
    mark(w, first, "dismissed")
    assert w.service.counts().new == 1
    w.put("b", text=first.excerpt, label=DELIBERATE, confidence=0.95, source=OTHER)
    w.refresh()
    assert w.service.counts().new == 2
    assert len(w.alerts(status="new")) == 2
    assert w.repo.count(AlertFilter(status="dismissed")) == 0
