"""A2: with a retention of 0 the purge forgets an alert as soon as it has been handled. What it
forgets is a VERDICT: the item is raised again only when its verdict changes or is upgraded, an
alert that still waits for the model is never purged, and an old GUID listed again after its item
was purged does not bring a forgotten alert back.

The exact rule: an item whose alert was forgotten stays forgotten while its verdict (label,
confidence, settled-at) equals the one it was forgotten on, for a year; any other verdict makes it
eligible again and clears the memory."""

import dataclasses

import pytest

from social_tracker.alerts.domain import AlertSettings, AlertStatus
from social_tracker.classification.domain import Label, Stage
from tests.alerts.world import HOUR, alert_world

RUMOR, DELIBERATE = Label.UNINTENTIONAL_RUMOR, Label.DELIBERATE_DISINFORMATION


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path, AlertSettings(retention_days=0))
    world.llm.ready()
    yield world
    world.close()


def settle(w, item, **changes):
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


def handle(w, status="dismissed"):
    w.service.set_status(w.only().id, status, actor="op", reason="checked")


def replace_verdict(w, item, **changes):
    current = w.llm.classifications.get(item.id)
    w.llm.classifications.delete(item.id)
    w.clock.advance(HOUR)
    w.llm.classifications.add(dataclasses.replace(current, classified_at=w.clock.now(), **changes))


# ---- an alert that still waits for the model is not purged ----


@pytest.mark.parametrize("status", ["seen", "dismissed"])
def test_a_handled_alert_that_waits_for_the_model_survives_a_retention_of_zero(w, status):
    item = w.put("s", confidence=0.6, needs_llm=True, text="rumor provisional s")
    w.refresh()
    handle(w, status)
    assert w.service.purge_expired() == 0
    assert w.repo.is_forgotten(item.id) is False
    settle(w, item, label=DELIBERATE, confidence=0.95)
    report = w.refresh()
    assert (report.updated, report.created) == (1, 0)  # the SAME alert is upgraded
    alert = w.only()
    assert alert.status is AlertStatus.NEW
    assert alert.severity.value == "critical"


def test_once_the_model_has_settled_a_handled_alert_may_go(w):
    item = w.put("s", confidence=0.6, needs_llm=True, text="rumor provisional s")
    w.refresh()
    handle(w)
    settle(w, item, confidence=0.6)
    w.refresh()
    handle(w)  # the settled verdict was not an upgrade: still dismissed
    assert w.service.purge_expired() == 1


# ---- what is forgotten is a verdict ----


def test_the_same_verdict_does_not_raise_the_forgotten_alert_again(w):
    item = w.put("f", confidence=0.7, text="rumor final f")
    w.refresh()
    handle(w)
    assert w.service.purge_expired() == 1
    assert w.refresh().created == 0
    assert w.service.rebuild().created == 0
    assert w.alerts() == []
    assert w.repo.is_forgotten(item.id)


def test_an_upgraded_verdict_after_the_forget_raises_the_alert_and_clears_the_memory(w):
    item = w.put("f", confidence=0.7, text="rumor final f")
    w.refresh()
    handle(w)
    w.service.purge_expired()
    replace_verdict(w, item, label=DELIBERATE, confidence=0.95)
    assert w.refresh().created == 1
    assert w.only().severity.value == "critical"
    assert w.repo.is_forgotten(item.id) is False


def test_a_verdict_settled_by_the_model_after_the_forget_raises_the_alert(w):
    item = w.put("f", confidence=0.7, text="rumor final f")
    w.refresh()
    handle(w)
    w.service.purge_expired()
    with w.db.transaction() as conn:  # same label and confidence, but now settled by the model
        conn.execute(
            "UPDATE classifications SET stage = 'llm', settled_at = ?, settle_reason = 'llm' "
            "WHERE item_id = ?",
            (w.clock.now().isoformat(), item.id),
        )
    assert w.refresh().created == 1


def test_a_lower_but_qualifying_verdict_is_a_changed_verdict_too(w):
    item = w.put("f", confidence=0.9, text="rumor final f")
    w.refresh()
    handle(w)
    w.service.purge_expired()
    replace_verdict(w, item, confidence=0.6)
    assert w.refresh().created == 1


def test_a_withdrawn_alert_that_requalifies_after_it_was_purged_is_raised_again(w):
    item = w.put("q", confidence=0.6, text="rumor que cambia q")
    w.refresh()
    replace_verdict(w, item, label=Label.NO_RISK, confidence=0.9)
    w.refresh()
    assert w.service.purge_expired() == 1  # withdrawn alerts go at once with a retention of 0
    assert w.repo.is_forgotten(item.id)
    replace_verdict(w, item, label=DELIBERATE, confidence=0.95)
    assert w.refresh().created == 1
    assert w.only().severity.value == "critical"


def test_a_verdict_that_never_qualified_after_the_forget_leaves_the_memory_alone(w):
    item = w.put("q", confidence=0.7, text="rumor que cambia q")
    w.refresh()
    handle(w)
    w.service.purge_expired()
    replace_verdict(w, item, label=Label.NO_RISK, confidence=0.9)
    w.refresh()
    assert w.repo.is_forgotten(item.id) is True  # flipping back to the old verdict stays quiet
    replace_verdict(w, item, label=RUMOR, confidence=0.7)
    assert w.refresh().created == 0


# ---- an old GUID listed again ----


def test_an_old_guid_listed_again_after_its_item_was_purged_does_not_bring_the_alert_back(w):
    item = w.put("g", confidence=0.7, text="rumor de un guid viejo g")
    w.refresh()
    handle(w)
    w.service.purge_expired()
    with w.db.transaction() as conn:  # the retention of items removes the item and its verdict
        conn.execute("DELETE FROM classifications WHERE item_id = ?", (item.id,))
        conn.execute("DELETE FROM items WHERE id = ?", (item.id,))
    w.service.purge_expired()
    again = w.put("g", confidence=0.7, text="rumor de un guid viejo g")  # the feed lists it again
    assert again.id == item.id
    assert w.refresh().created == 0
    assert w.alerts() == []


def test_the_memory_is_dropped_after_a_year(w):
    from datetime import timedelta

    item = w.put("g", confidence=0.7, text="rumor de un guid viejo g")
    w.refresh()
    handle(w)
    w.service.purge_expired()
    w.clock.advance(timedelta(days=366))
    w.service.purge_expired()
    assert w.repo.is_forgotten(item.id) is False
