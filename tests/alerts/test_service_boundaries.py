"""Boundaries of the second round that the mutation sweep found unpinned: which alerts' statuses a
merge compares, when the model's verdict counts as the model's, the flag and the live copy, a full
alert of one item, copies counted once, the hard cap of the trail and the tags of a copy."""

import dataclasses
from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertEventKind, AlertSettings, AlertStatus, Severity
from social_tracker.classification.domain import Label, Stage
from tests.alerts.world import HOUR, alert_world
from tests.classification.harness import PUBLISHED

RUMOR, DELIBERATE = Label.UNINTENTIONAL_RUMOR, Label.DELIBERATE_DISINFORMATION
MINUTE = timedelta(minutes=1)
START = PUBLISHED + MINUTE
OTHER = "src_other"


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    world.llm.ready(OTHER)
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


def sql(w, statement, *params):
    with w.db.transaction() as conn:
        conn.execute(statement, params)


# ---- merging alerts the person handled at different severities ----


def test_bridged_alerts_with_the_same_status_compare_with_the_highest_severity_they_had(w):
    a = w.put("a", text="Rumor uno sobre temblor en Cali", confidence=0.55)
    b = w.put("b", text="Otro tema distinto de inundacion en Cali", confidence=0.75)
    w.refresh()
    for alert in w.alerts():
        w.service.set_status(alert.id, "seen", actor="op", reason="checked")
    w.put(
        "c",
        text="Rumor uno sobre temblor en Cali",
        confidence=0.75,
        source=OTHER,
        near_of=b.id,
    )
    w.refresh()
    alert = w.only()
    assert a.id != b.id
    # the high one was already looked at as high: a high bridge is no news
    assert (alert.severity, alert.status) == (Severity.HIGH, AlertStatus.SEEN)


# ---- the model's verdict ----


def test_a_model_that_judges_again_is_the_model_even_with_the_same_stage(w):
    item = w.put("p", label=DELIBERATE, confidence=0.95, text="rumor definitivo p")
    sql(
        w,
        "UPDATE classifications SET stage = 'llm', settled_at = ?, settle_reason = 'llm' "
        "WHERE item_id = ?",
        "2026-03-01T09:00:00+00:00",
        item.id,
    )
    w.refresh()
    assert w.only().severity is Severity.CRITICAL
    sql(
        w,
        "UPDATE classifications SET label = 'unintentional_rumor', confidence = 0.55, "
        "settled_at = ? WHERE item_id = ?",
        "2026-03-01T10:00:00+00:00",
        item.id,
    )
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.downgraded_by_model) == (Severity.MEDIUM, True)


def test_a_cascade_reclassification_of_the_same_settled_verdict_is_not_the_model(w):
    item = w.put("p", label=DELIBERATE, confidence=0.95, text="rumor definitivo p")
    sql(
        w,
        "UPDATE classifications SET stage = 'llm', settled_at = ?, settle_reason = 'llm' "
        "WHERE item_id = ?",
        "2026-03-01T09:00:00+00:00",
        item.id,
    )
    w.refresh()
    sql(
        w,
        "UPDATE classifications SET label = 'unintentional_rumor', confidence = 0.55 "
        "WHERE item_id = ?",
        item.id,
    )  # same stage, same settled_at: nothing says the model spoke again
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.downgraded_by_model) == (Severity.MEDIUM, False)


def test_a_later_model_verdict_that_does_not_lower_clears_the_flag(w):
    item = w.put("p", confidence=0.75, needs_llm=True, text="rumor provisional p")
    w.refresh()
    settle(w, item, label=Label.NO_RISK, confidence=0.9)
    w.refresh()
    assert w.only().downgraded_by_model is True
    sql(
        w,
        "UPDATE classifications SET label = 'unintentional_rumor', confidence = 0.8, "
        "settled_at = ? WHERE item_id = ?",
        "2026-03-02T09:00:00+00:00",
        item.id,
    )
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.downgraded_by_model) == (Severity.HIGH, False)


def test_a_live_copy_joining_an_alert_the_model_cleared_clears_the_flag(w):
    text = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    item = w.put("p", confidence=0.75, needs_llm=True, text=text)
    w.refresh()
    settle(w, item, label=Label.NO_RISK, confidence=0.9)
    w.refresh()
    assert w.only().downgraded_by_model is True
    w.put("q", text=text, source=OTHER, confidence=0.75, fetched=w.clock.now())
    w.refresh()
    alert = w.only()
    assert alert.downgraded_by_model is False
    assert alert.occurrences == 1  # the cleared item no longer counts; the live copy does


def test_a_copy_joining_an_alert_that_is_still_live_keeps_the_flag(w):
    text = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    w.llm.ready("src_third")
    w.put("k", text=text, confidence=0.6, fetched=START)
    item = w.put(
        "p",
        text=text,
        label=DELIBERATE,
        confidence=0.95,
        needs_llm=True,
        source=OTHER,
        fetched=START,
    )
    w.refresh()
    assert w.only().severity is Severity.CRITICAL
    settle(w, item, label=RUMOR, confidence=0.55)  # lower than the provisional one, still alerts
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.downgraded_by_model, alert.occurrences) == (
        Severity.MEDIUM,
        True,
        2,
    )
    w.put("q", text=text, source="src_third", confidence=0.6, fetched=START)
    w.refresh()
    alert = w.only()
    assert (alert.downgraded_by_model, alert.occurrences) == (True, 3)


# ---- a full alert that can keep one item ----


def test_an_alert_of_one_item_counts_the_copies_it_cannot_keep(tmp_path):
    w = alert_world(tmp_path, AlertSettings(max_members=1, max_events=10))
    w.llm.ready()
    try:
        first = w.put("orig", text="Mismo rumor repetido sobre temblor 0", fetched=START)
        for n in range(1, 4):
            w.put(
                f"c{n}",
                text=f"Mismo rumor repetido sobre temblor {n}",
                near_of=first.id,
                fetched=START + n * MINUTE,
            )
        w.refresh()
        alert = w.only()
        assert (alert.occurrences, alert.overflow) == (4, 3)
        assert [m.item_id for m in w.repo.members(alert.id)] == [first.id]
        w.refresh()
        w.service.rebuild()
        assert (w.only().occurrences, w.only().overflow) == (4, 3)
    finally:
        w.close()


def test_a_stronger_copy_replaces_the_only_item_of_a_full_alert(tmp_path):
    w = alert_world(tmp_path, AlertSettings(max_members=1, max_events=10))
    w.llm.ready()
    try:
        first = w.put("orig", text="Mismo rumor repetido sobre temblor 0", fetched=START)
        strong = w.put(
            "boss",
            text="Mismo rumor repetido sobre temblor boss",
            near_of=first.id,
            label=DELIBERATE,
            confidence=0.95,
            fetched=START + MINUTE,
        )
        w.refresh()
        alert = w.only()
        assert (alert.item_id, alert.occurrences) == (strong.id, 2)
        assert [m.item_id for m in w.repo.members(alert.id)] == [strong.id]
    finally:
        w.close()


# ---- an item counted without being kept is counted once ----


def test_a_changed_verdict_of_an_item_that_was_only_counted_does_not_count_it_again(tmp_path):
    w = alert_world(tmp_path, AlertSettings(max_members=3, max_events=10))
    w.llm.ready()
    try:
        first = w.put("orig", text="Mismo rumor repetido sobre temblor 0", fetched=START)
        copies = [
            w.put(
                f"c{n}",
                text=f"Mismo rumor repetido sobre temblor {n}",
                near_of=first.id,
                fetched=START + n * MINUTE,
            )
            for n in range(1, 8)
        ]
        w.refresh()
        assert w.only().occurrences == 8
        gone = next(c for c in copies if w.repo.overflow_of(c.id) is not None)
        sql(w, "UPDATE classifications SET confidence = 0.9 WHERE item_id = ?", gone.id)
        report = w.refresh()
        assert (report.created, report.merged, report.errors) == (0, 0, 0)
        assert (len(w.alerts()), w.only().occurrences) == (1, 8)
    finally:
        w.close()


# ---- the trail has a hard cap ----


def test_even_what_matters_stops_at_twice_the_cap(tmp_path):
    w = alert_world(tmp_path, AlertSettings(max_members=10, max_events=10))
    w.llm.ready()
    try:
        item = w.put("flap", confidence=0.7, text="rumor que va y viene")
        w.refresh()
        current = w.llm.classifications.get(item.id)
        for _ in range(12):  # withdrawn, reactivated: 24 entries that are not routine
            for label in (Label.NO_RISK, RUMOR):
                w.llm.classifications.delete(item.id)
                w.clock.advance(MINUTE)
                w.llm.classifications.add(
                    dataclasses.replace(current, label=label, classified_at=w.clock.now())
                )
                w.refresh()
        alert = w.only()
        stored = w.repo.events(alert.id)
        assert len(stored) == 20
        assert alert.events_dropped >= 5
        kinds = {e.kind for e in stored}
        assert {AlertEventKind.WITHDRAWN, AlertEventKind.REACTIVATED} <= kinds
    finally:
        w.close()


# ---- the tags of a copy ----


def test_the_tags_of_a_copy_join_those_of_the_alert(w):
    from social_tracker.classification.domain import Topic

    text = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    w.put("a", text=text, topics=(Topic.EARTHQUAKE,), fetched=START)
    w.put("b", text=text, source=OTHER, topics=(Topic.SECRETARIAT,), fetched=START + MINUTE)
    w.refresh()
    assert {"earthquake", "secretariat"} <= set(w.only().tags)


# ---- the rationale of a verdict that changed ----


def test_the_rationale_of_an_updated_verdict_is_cleaned_too(w):
    item = w.put("a", confidence=0.7, text="rumor sobre sismo a")
    w.refresh()
    sql(
        w,
        "UPDATE classifications SET rationale = ?, confidence = 0.8 WHERE item_id = ?",
        "now password=hunter2hunter2 and more",
        item.id,
    )
    w.refresh()
    alert = w.only()
    (member,) = w.repo.members(alert.id)
    for text in (alert.rationale, member.rationale):
        assert "hunter2" not in text


# ---- a failure is forgiven when the item works ----


def test_the_failure_row_goes_when_the_item_works(w, monkeypatch):
    bad = w.put("bad", text="rumor venenoso")
    real = w.service._member  # noqa: SLF001

    def broken(alert_id, verdict, item, *args, **kwargs):
        if item.id == bad.id:
            raise ValueError("poison")
        return real(alert_id, verdict, item, *args, **kwargs)

    monkeypatch.setattr(w.service, "_member", broken)
    w.refresh()
    with w.db.snapshot() as conn:
        assert conn.execute("SELECT count(*) FROM alert_failures").fetchone()[0] == 1
    monkeypatch.undo()
    w.clock.advance(timedelta(minutes=2))
    assert w.refresh().created == 1
    with w.db.snapshot() as conn:
        assert conn.execute("SELECT count(*) FROM alert_failures").fetchone()[0] == 0
