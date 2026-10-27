"""A11: when the language model settles a verdict LOWER than the provisional one an alert was raised
on (or makes every member stop qualifying) the alert is NOT silently hidden: it stays in the
default list, back to `new`, flagged `downgraded_by_model`. A plain re-evaluation (a reclassified
verdict, a raised threshold) still withdraws. A10: the text a model writes is redacted before it is
stored on an alert or a member."""

import dataclasses
from types import SimpleNamespace

import pytest

from social_tracker.alerts.domain import AlertEventKind, AlertSettings, AlertStatus, Severity
from social_tracker.alerts.service import AlertService
from social_tracker.classification.domain import Label, Stage
from tests.alerts.world import HOUR, alert_world

RUMOR, DELIBERATE = Label.UNINTENTIONAL_RUMOR, Label.DELIBERATE_DISINFORMATION
SAME = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    world.llm.ready("src_other")
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


def mark(w, alert, status):
    w.service.set_status(alert.id, status, actor="op", reason="checked")


# ---- the model says no risk ----


def test_a_model_that_clears_the_only_member_does_not_hide_the_alert(w):
    item = w.put("p", confidence=0.75, needs_llm=True, text="rumor provisional p")
    w.refresh()
    settle(w, item, label=Label.NO_RISK, confidence=0.9)
    report = w.refresh()
    alert = w.only()  # in the DEFAULT list
    assert report.withdrawn == 0
    assert (alert.withdrawn, alert.downgraded_by_model, alert.status) == (
        False,
        True,
        AlertStatus.NEW,
    )
    assert w.service.counts().new == 1
    assert w.service.counts().withdrawn == 0


def test_a_dismissed_provisional_alert_the_model_clears_comes_back_as_new(w):
    item = w.put("p", confidence=0.75, needs_llm=True, text="rumor provisional p")
    w.refresh()
    mark(w, w.only(), "dismissed")
    settle(w, item, label=Label.NO_RISK, confidence=0.9)
    w.refresh()
    alert = w.only()
    assert (alert.status, alert.downgraded_by_model) == (AlertStatus.NEW, True)
    entry = [e for e in w.repo.events(alert.id) if e.kind is AlertEventKind.STATUS_CHANGED][-1]
    assert (entry.actor, entry.before, entry.after) == ("system:alerts", "dismissed", "new")
    assert "model" in entry.reason


def test_the_alert_the_model_cleared_keeps_what_it_showed_and_says_why(w):
    item = w.put("p", confidence=0.75, needs_llm=True, text="rumor provisional p")
    w.refresh()
    settle(w, item, label=Label.NO_RISK, confidence=0.9, rationale="llm: no claim to check")
    w.refresh()
    alert = w.only()
    assert alert.severity is Severity.HIGH  # what it showed
    assert "llm: no claim to check" in alert.rationale  # what the model said
    kinds = [e.kind for e in w.repo.events(alert.id)]
    assert AlertEventKind.UPDATED in kinds
    assert AlertEventKind.WITHDRAWN not in kinds


def test_a_later_pass_does_not_withdraw_the_alert_the_model_cleared(w):
    item = w.put("p", confidence=0.75, needs_llm=True, text="rumor provisional p")
    w.refresh()
    settle(w, item, label=Label.NO_RISK, confidence=0.9)
    w.refresh()
    again = w.service.rebuild()
    assert (again.withdrawn, again.errors) == (0, 0)
    alert = w.only()
    assert (alert.withdrawn, alert.downgraded_by_model) == (False, True)


# ---- the model says less ----


def test_a_model_verdict_lower_than_the_provisional_one_flags_the_alert(w):
    item = w.put("p", label=DELIBERATE, confidence=0.9, needs_llm=True, text="rumor provisional p")
    w.refresh()
    assert w.only().severity is Severity.CRITICAL
    mark(w, w.only(), "seen")
    settle(w, item, label=RUMOR, confidence=0.55)
    w.refresh()
    alert = w.only()
    assert (alert.severity, alert.status, alert.downgraded_by_model) == (
        Severity.MEDIUM,
        AlertStatus.NEW,
        True,
    )


def test_a_model_verdict_of_the_same_severity_is_not_a_downgrade(w):
    item = w.put("p", confidence=0.75, needs_llm=True, text="rumor provisional p")
    w.refresh()
    mark(w, w.only(), "seen")
    settle(w, item, confidence=0.72)
    w.refresh()
    alert = w.only()
    assert (alert.status, alert.downgraded_by_model) == (AlertStatus.SEEN, False)


def test_another_member_that_holds_the_severity_up_means_no_flag(w):
    w.put("a", text=SAME, confidence=0.8)
    item = w.put("b", text=SAME, confidence=0.75, needs_llm=True, source="src_other")
    w.refresh()
    assert w.only().occurrences == 2
    settle(w, item, label=Label.NO_RISK, confidence=0.9)
    w.refresh()
    alert = w.only()
    assert (alert.downgraded_by_model, alert.occurrences) == (False, 1)


# ---- a plain re-evaluation still withdraws ----


def test_a_reclassification_to_no_risk_by_the_cascade_still_withdraws(w):
    item = w.put("c", confidence=0.7, text="rumor determinista c")
    w.refresh()
    current = w.llm.classifications.get(item.id)
    w.llm.classifications.delete(item.id)
    w.llm.classifications.add(dataclasses.replace(current, label=Label.NO_RISK, confidence=0.9))
    report = w.refresh()
    assert report.withdrawn == 1
    assert w.alerts() == []
    gone = w.only(status="withdrawn")
    assert gone.downgraded_by_model is False


def test_a_raised_threshold_still_withdraws_even_for_a_verdict_the_model_made(w):
    item = w.put("p", confidence=0.6, needs_llm=True, text="rumor provisional p")
    w.refresh()
    settle(w, item, confidence=0.6)
    w.refresh()
    w.control.set_alert_min_confidence(0.9, actor="op", reason="stricter")
    w.service.rebuild()
    assert w.alerts() == []
    assert w.only(status="withdrawn").downgraded_by_model is False


# ---- the flag ----


def test_looking_at_the_alert_clears_the_flag(w):
    item = w.put("p", confidence=0.75, needs_llm=True, text="rumor provisional p")
    w.refresh()
    settle(w, item, label=Label.NO_RISK, confidence=0.9)
    w.refresh()
    mark(w, w.only(), "seen")
    alert = w.only()
    assert (alert.status, alert.downgraded_by_model) == (AlertStatus.SEEN, False)


# ---- retention 0 ----


def test_retention_zero_never_forgets_an_alert_the_model_downgraded(tmp_path):
    w = alert_world(tmp_path, AlertSettings(retention_days=0))
    w.llm.ready()
    try:
        item = w.put("p", confidence=0.75, needs_llm=True, text="rumor provisional p")
        w.refresh()
        settle(w, item, label=Label.NO_RISK, confidence=0.9)
        w.refresh()
        assert w.service.purge_expired() == 0
        assert w.repo.is_forgotten(item.id) is False
        alert = w.only()
        mark(w, alert, "dismissed")  # a person looked at it: now it may go
        assert w.service.purge_expired() == 1
    finally:
        w.close()


# ---- A10: what a model writes is redacted ----


def test_the_rationale_of_a_model_is_redacted_on_the_alert_and_the_member(w):
    item = w.put(
        "s",
        confidence=0.7,
        text="rumor sobre sismo s",
        rationale=(
            "see https://" + "user:pw@evil.example/x?token=abcdef123456 and password=hunter2hunter2"
        ),
    )
    w.refresh()
    alert = w.only()
    (member,) = w.repo.members(alert.id)
    for text in (alert.rationale, member.rationale):
        assert "hunter2" not in text
        assert "abcdef123456" not in text
        assert "user:pw" not in text
    assert item.id == member.item_id


def test_a_rationale_of_only_secrets_still_leaves_a_valid_alert(w):
    w.put("s", confidence=0.7, text="rumor sobre sismo s", rationale="password=hunter2hunter2")
    w.refresh()
    assert w.only().rationale


def test_a_rationale_that_grows_when_redacted_is_cut_not_refused(w):
    # 27 x "pwd=abcdef " is 297 characters; redacted, every secret becomes `[redacted]` (longer)
    w.put("s", confidence=0.7, text="rumor sobre sismo s", rationale="pwd=abcdef " * 27)
    report = w.refresh()
    assert report.errors == 0
    assert len(w.only().rationale) <= 300


def test_the_strings_of_the_evidence_are_redacted_too(w):
    judgment = SimpleNamespace(
        llm_model="gemma token=abc123def456",
        prompt_version="v1 password=hunter2hunter2",
    )
    guarded = AlertService(
        w.repo,
        w.llm.classifications,
        w.llm.items,
        w.llm.base.sources_repo,
        SimpleNamespace(get_judgment=lambda item_id: judgment),
        w.control,
        w.llm.base.uow,
        w.clock,
    )
    w.put("s", confidence=0.7, text="rumor sobre sismo s")
    guarded.refresh()
    evidence = w.only().evidence
    assert "abc123def456" not in (evidence.llm_model or "")
    assert "hunter2" not in (evidence.prompt_version or "")
