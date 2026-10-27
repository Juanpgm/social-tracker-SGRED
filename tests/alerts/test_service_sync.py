"""The alert step over stored verdicts: which alerts it raises, what they hold (AL-1 .. AL-3), that
the step is idempotent (AL-10) and how an upgrade or a downgrade of a verdict reaches the alert."""

from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertEventKind, AlertFilter, Severity
from social_tracker.classification.domain import Label, OutOfScopeReason, Signal, Stage, Topic
from social_tracker.llm.domain import PROMPT_VERSION
from tests.alerts.world import alert_world
from tests.classification.harness import PUBLISHED
from tests.llm.support import response

RUMOR, DELIBERATE, RECYCLED = (
    Label.UNINTENTIONAL_RUMOR,
    Label.DELIBERATE_DISINFORMATION,
    Label.RECYCLED_CONTENT,
)


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    yield world
    world.close()


def kinds(w, alert_id):
    return [e.kind for e in w.repo.events(alert_id)]


# ---- AL-1: which verdicts raise an alert ----


@pytest.mark.parametrize("label", [RUMOR, DELIBERATE, RECYCLED])
def test_each_risk_label_raises_an_alert(w, label):
    w.put("a", label=label, confidence=0.7)
    report = w.refresh()
    assert (report.created, report.errors) == (1, 0)
    assert w.only().label is label


@pytest.mark.parametrize("label", [Label.NO_RISK, Label.UNCLASSIFIED])
@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_a_verdict_that_is_no_risk_or_inconclusive_never_alerts(w, label, confidence):
    w.put("a", label=label, confidence=confidence)
    report = w.refresh()
    assert report.created == 0
    assert w.alerts() == []


def test_an_out_of_scope_item_never_alerts(w):
    w.put(
        "a",
        label=Label.OUT_OF_SCOPE,
        confidence=1.0,
        scope_reason=OutOfScopeReason.NO_TOPIC_MATCH,
        signals=(),
        topics=(),
    )
    assert w.refresh().created == 0
    assert w.alerts() == []


@pytest.mark.parametrize(("confidence", "alerts"), [(0.49, 0), (0.4999, 0), (0.5, 1), (0.51, 1)])
def test_the_default_threshold_is_half_inclusive(w, confidence, alerts):
    w.put("a", confidence=confidence)
    w.refresh()
    assert len(w.alerts()) == alerts


def test_a_low_confidence_verdict_under_the_threshold_never_alerts(w):
    w.put("a", confidence=0.3, low_confidence=True)
    w.refresh()
    assert w.alerts() == []


def test_the_threshold_is_the_one_the_operator_set(w):
    w.control.set_alert_min_confidence(0.7, actor="tester", reason="quieter")
    w.put("a", confidence=0.69)
    w.put("b", confidence=0.70)
    w.refresh()
    assert [a.confidence for a in w.alerts()] == [0.7]
    w.control.set_alert_min_confidence(0.0, actor="tester", reason="everything")
    assert w.refresh(full=True).created == 1
    assert len(w.alerts()) == 2


# ---- AL-2: severity through the whole step ----


@pytest.mark.parametrize(
    ("label", "confidence", "severity"),
    [
        (RUMOR, 0.5, Severity.MEDIUM),
        (RUMOR, 0.69, Severity.MEDIUM),
        (RUMOR, 0.70, Severity.HIGH),
        (DELIBERATE, 0.84, Severity.HIGH),
        (DELIBERATE, 0.85, Severity.CRITICAL),
        (RECYCLED, 0.99, Severity.HIGH),
    ],
)
def test_the_severity_follows_the_policy(w, label, confidence, severity):
    w.put("a", label=label, confidence=confidence)
    w.refresh()
    assert w.only().severity is severity


DAY = timedelta(hours=24)


@pytest.mark.parametrize(
    ("age", "confidence", "severity"),
    [
        (timedelta(hours=23, minutes=59, seconds=59), 0.8, Severity.CRITICAL),
        (DAY, 0.8, Severity.CRITICAL),
        (DAY + timedelta(seconds=1), 0.8, Severity.HIGH),
        (timedelta(minutes=5), 0.79, Severity.HIGH),
        (timedelta(minutes=5), 0.8, Severity.CRITICAL),
    ],
)
def test_a_catalog_contradiction_about_a_fresh_event_is_critical(w, age, confidence, severity):
    w.put(
        "a", confidence=confidence, catalog="mismatch", published=PUBLISHED, fetched=PUBLISHED + age
    )
    w.refresh()
    assert w.only().severity is severity


def test_a_contradiction_is_judged_by_when_the_event_was_first_seen_not_when_the_feed_says(w):
    # a feed claiming a publication date in the future does not make an old event fresh
    w.put(
        "a",
        confidence=0.8,
        catalog="mismatch",
        published=PUBLISHED + timedelta(days=30),
        fetched=PUBLISHED,
    )
    w.refresh()
    assert w.only().severity is Severity.CRITICAL  # age clamps to zero: the sighting is the clock


# ---- AL-3: what an alert holds ----


def test_an_alert_holds_everything_the_operator_needs(w):
    item = w.put(
        "a",
        label=DELIBERATE,
        confidence=0.9,
        catalog="consistent",
        title="Alerta",
        text="Mañana habrá un temblor fuerte en Cali",
        signals=(
            Signal("prediction_claim", RUMOR, 0.45),
            Signal("censorship_bait", DELIBERATE, 0.3),
        ),
        topics=(Topic.EARTHQUAKE,),
    )
    w.refresh()
    alert = w.only()
    assert alert.id.startswith("alt_")
    assert (alert.label, alert.severity, alert.confidence) == (DELIBERATE, Severity.CRITICAL, 0.9)
    assert alert.rationale == "prediction_claim"
    assert (alert.source_id, alert.source_name) == ("src_news", "City News")
    assert alert.item_id == item.id
    assert alert.excerpt == "Alerta | Mañana habrá un temblor fuerte en Cali"
    assert alert.first_seen_at == item.fetched_at == alert.last_seen_at
    assert alert.created_at == w.clock.now()
    assert alert.occurrences == 1
    assert alert.status.value == "new"
    assert alert.withdrawn is False
    evidence = alert.evidence
    assert (evidence.stage, evidence.catalog_status) == ("rules", "consistent")
    assert evidence.classifier_version == "cascade-2/topics-0"
    assert [s[0] for s in evidence.signals] == ["censorship_bait", "prediction_claim"]
    assert evidence.provisional is False
    assert (evidence.llm_model, evidence.prompt_version) == (None, None)
    assert alert.tags == ("earthquake", "prediction")


def test_the_secretariat_attribution_is_visible_and_filterable(w):
    w.put(
        "a",
        label=DELIBERATE,
        confidence=0.8,
        text="Según la Secretaría de Gestión del Riesgo, evacuen la zona ya",
        signals=(Signal("official_attribution_unlinked", DELIBERATE, 0.45),),
        topics=(Topic.SECRETARIAT, Topic.EARTHQUAKE),
    )
    w.put("b", confidence=0.6)
    w.refresh()
    flagged = w.alerts(tag="secretariat_attribution")
    assert [a.label for a in flagged] == [DELIBERATE]
    assert set(flagged[0].tags) == {"earthquake", "secretariat", "secretariat_attribution"}
    assert len(w.alerts()) == 2


def test_the_cross_check_evidence_names_the_model_when_the_model_judged(w):
    stored = w.llm.seed(1, confidence=0.55)
    w.refresh()
    assert w.only().evidence.provisional is True
    w.llm.judge.default = response(label=DELIBERATE, confidence=0.92, model="st-gemma4-e4b")
    w.llm.escalate()
    assert w.llm.verdict(stored).stage is Stage.LLM
    w.refresh()
    alert = w.only()
    assert (alert.evidence.llm_model, alert.evidence.prompt_version) == (
        "st-gemma4-e4b",
        PROMPT_VERSION,
    )
    assert (alert.evidence.stage, alert.evidence.settle_reason) == ("llm", "llm")
    assert alert.evidence.provisional is False


# ---- AL-10: idempotence ----


def test_running_the_step_again_changes_nothing(w):
    w.put("a", confidence=0.9)
    w.put("b", confidence=0.7, label=RECYCLED)
    first = w.refresh()
    snapshot = w.alerts()
    second = w.refresh()
    third = w.refresh(full=True)
    assert (first.created, second.created, third.created) == (2, 0, 0)
    assert w.alerts() == snapshot
    assert (second.errors, third.errors) == (0, 0)
    assert all(kinds(w, a.id) == [AlertEventKind.CREATED] for a in snapshot)


def test_an_item_that_is_reprocessed_does_not_get_a_second_alert(w):
    item = w.put("a", confidence=0.9)
    w.refresh()
    w.refresh()
    w.service.sync_item(item.id)  # the direct call, as a retry would
    w.service.sync_item(item.id)
    assert w.repo.count(AlertFilter()) == 1
    assert w.only().occurrences == 1


def test_the_same_inputs_make_the_same_alerts_in_two_worlds(tmp_path):
    runs = []
    for name in ("one.db", "two.db"):
        world = alert_world(tmp_path, name=name)
        world.llm.ready()
        world.put("a", confidence=0.9, label=DELIBERATE)
        world.put("b", confidence=0.55)
        world.refresh()
        runs.append([a.__dict__ for a in world.alerts()])
        world.close()
    assert runs[0] == runs[1]


def test_alerts_survive_a_restart(tmp_path):
    first = alert_world(tmp_path, name="restart.db")
    first.llm.ready()
    first.put("a", confidence=0.9)
    first.refresh()
    before = first.alerts()
    first.close()
    again = alert_world(tmp_path, name="restart.db")
    try:
        assert again.alerts() == before
        assert again.refresh().created == 0
        assert again.alerts() == before
    finally:
        again.close()


# ---- AL-10: a verdict that changes ----


def test_a_verdict_the_model_raises_updates_the_alert_it_does_not_add_one(w):
    stored = w.llm.seed(1, confidence=0.55)
    w.refresh()
    assert w.only().severity is Severity.MEDIUM
    w.llm.judge.default = response(label=DELIBERATE, confidence=0.9)
    w.llm.escalate()
    report = w.refresh()
    assert (report.created, report.updated) == (0, 1)
    alert = w.only()
    assert (alert.label, alert.severity, alert.confidence) == (DELIBERATE, Severity.CRITICAL, 0.9)
    assert alert.item_id == stored.item_id
    assert kinds(w, alert.id) == [AlertEventKind.CREATED, AlertEventKind.UPDATED]


def test_a_verdict_the_model_clears_withdraws_the_alert_instead_of_deleting_it(w):
    stored = w.llm.seed(1, confidence=0.55)
    w.refresh()
    alert_id = w.only().id
    w.llm.judge.default = response(label=Label.NO_RISK, confidence=0.95)
    w.llm.escalate()
    report = w.refresh()
    assert (report.withdrawn, report.created) == (1, 0)
    assert w.alerts() == []  # hidden from the default list...
    gone = w.only(status="withdrawn")  # ...but still there
    assert gone.id == alert_id
    assert gone.withdrawn_reason == "the verdict is now no_risk"
    assert gone.occurrences == 0
    assert (gone.label, gone.confidence) == (RUMOR, 0.55)  # it keeps what it showed
    assert w.repo.member(stored.item_id).active is False
    assert kinds(w, alert_id) == [AlertEventKind.CREATED, AlertEventKind.WITHDRAWN]


def test_a_verdict_that_falls_under_the_threshold_withdraws_the_alert(w):
    w.llm.seed(1, confidence=0.6)
    w.refresh()
    w.llm.judge.default = response(label=RUMOR, confidence=0.3)
    w.llm.escalate()
    w.refresh()
    assert w.only(status="withdrawn").withdrawn_reason == (
        "the confidence fell under the alert threshold"
    )


def test_a_withdrawn_alert_comes_back_when_the_verdict_qualifies_again(w):
    stored = w.llm.seed(1, confidence=0.55)
    w.refresh()
    alert_id = w.only().id
    w.service.set_status(alert_id, "seen", actor="tester", reason="looked")
    w.llm.judge.default = response(label=Label.NO_RISK, confidence=0.95)
    w.llm.escalate()
    w.refresh()
    assert w.alerts() == []
    with w.db.transaction() as conn:  # the verdict returns (a reclassification that agrees again)
        conn.execute(
            "UPDATE classifications SET label = 'unintentional_rumor', confidence = 0.8, "
            "stage = 'rules' WHERE item_id = ?",
            (stored.item_id,),
        )
    report = w.refresh()
    assert (report.reactivated, report.created) == (1, 0)
    back = w.only()
    assert back.id == alert_id
    assert back.withdrawn is False
    assert (back.severity, back.confidence, back.occurrences) == (Severity.HIGH, 0.8, 1)
    assert back.status.value == "seen"  # what the operator decided is kept
    assert kinds(w, alert_id) == [
        AlertEventKind.CREATED,
        AlertEventKind.STATUS_CHANGED,
        AlertEventKind.WITHDRAWN,
        AlertEventKind.REACTIVATED,
    ]


def test_a_verdict_that_nobody_changed_is_left_alone_even_when_asked_twice(w):
    w.put("a", confidence=0.9)
    w.refresh()
    before = w.only()
    assert w.refresh().updated == 0
    assert w.only() == before


# ---- the threshold moves ----


def test_a_raised_threshold_withdraws_on_a_rebuild_not_on_a_plain_refresh(w):
    w.put("a", confidence=0.6)
    w.put("b", confidence=0.9)
    w.refresh()
    w.control.set_alert_min_confidence(0.8, actor="tester", reason="quieter")
    assert w.refresh().withdrawn == 0  # a refresh follows verdicts, not the threshold
    assert len(w.alerts()) == 2
    report = w.service.rebuild()
    assert report.withdrawn == 1
    assert [a.confidence for a in w.alerts()] == [0.9]
    assert [a.confidence for a in w.alerts(status="withdrawn")] == [0.6]


def test_a_lowered_threshold_brings_the_withdrawn_back_and_adds_the_new_on_a_rebuild(w):
    w.put("a", confidence=0.6)
    w.put("b", confidence=0.4)
    w.control.set_alert_min_confidence(0.8, actor="tester", reason="quiet")
    w.service.rebuild()
    assert w.alerts() == []
    w.control.set_alert_min_confidence(0.5, actor="tester", reason="normal")
    report = w.service.rebuild()
    assert report.created == 1
    assert [a.confidence for a in w.alerts()] == [0.6]
    again = w.service.rebuild()
    assert (again.created, again.updated, again.withdrawn, again.reactivated) == (0, 0, 0, 0)


def test_a_rebuild_is_idempotent(w):
    for n in range(5):
        w.put(f"n{n}", confidence=0.5 + n / 10)
    w.service.rebuild()
    state = [a.__dict__ for a in w.alerts()]
    events = {a["id"]: kinds(w, a["id"]) for a in state}
    for _ in range(3):
        report = w.service.rebuild()
        assert (report.created, report.updated, report.withdrawn, report.reactivated) == (0,) * 4
    assert [a.__dict__ for a in w.alerts()] == state
    assert {a["id"]: kinds(w, a["id"]) for a in state} == events


# ---- text is untrusted ----


def test_a_hostile_item_makes_a_clean_bounded_excerpt(w):
    hostile = (
        "Cookie: sessionid=SECRET123; \x1b[31m"
        + chr(0x202E)
        + chr(0x200B)
        + "<script>x</script>"
        + " y" * 2000
    )
    w.put("a", confidence=0.9, title="T" * 1000, text=hostile)
    w.refresh()
    excerpt = w.only().excerpt
    assert len(excerpt) <= 500
    assert "SECRET123" not in excerpt
    assert "\x1b" not in excerpt
    assert chr(0x202E) not in excerpt
    assert chr(0x200B) not in excerpt


def test_an_item_with_nothing_but_hidden_characters_still_raises_its_alert(w):
    w.put("a", confidence=0.9, title="", text=chr(0x200B) * 50 + "x")
    w.refresh()
    assert w.only().excerpt == "x"


def test_the_source_name_snapshot_is_cleaned(w):
    w.put("a", confidence=0.9, source_name="Bad Name" + chr(0x202E))
    w.refresh()
    assert w.only().source_name == "Bad Name"


# ---- failure isolation ----


def test_one_item_that_cannot_be_processed_does_not_stop_the_others(w, monkeypatch):
    bad = w.put("a", confidence=0.9)
    w.put("b", confidence=0.9)
    real = w.repo.member

    def broken(item_id):
        if item_id == bad.id:
            raise ValueError("boom")
        return real(item_id)

    monkeypatch.setattr(w.repo, "member", broken)
    report = w.refresh()
    assert (report.created, report.errors) == (1, 1)


def test_a_storage_failure_ends_the_run_and_the_next_one_resumes(w, monkeypatch):
    from social_tracker.shared.errors import StorageError

    w.put("a", confidence=0.9)
    w.put("b", confidence=0.9)
    real = w.repo.sync_ids
    monkeypatch.setattr(w.repo, "sync_ids", lambda **kw: (_ for _ in ()).throw(StorageError("x")))
    with pytest.raises(StorageError):
        w.refresh()
    monkeypatch.setattr(w.repo, "sync_ids", real)
    assert w.refresh().created == 2
