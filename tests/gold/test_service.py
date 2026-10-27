"""The gold service: labeling texts, exporting them and measuring the cascade against them."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from social_tracker.classification.domain import Label
from social_tracker.gold.service import GoldService
from social_tracker.gold.yaml_io import load_samples
from social_tracker.llm.domain import JudgeError, JudgeErrorKind, JudgeEvidence
from social_tracker.persistence.gold_repo import SqliteGoldRepository
from social_tracker.shared.errors import ConflictError, NotFoundError, ValidationError
from tests.classification.harness import World, sqlite_world
from tests.llm.support import response

SEED = Path(__file__).parents[1] / "data" / "gold_examples.yaml"
PREDICTION = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
REPORT = "Un sismo de magnitud 4.5 se registró anoche en Jamundí sin reportes de daños ni heridos"
OFF_TOPIC = "El equipo ganó el partido de ayer en el estadio de la ciudad"


@pytest.fixture
def world(tmp_path: Path) -> Iterator[World]:
    built = sqlite_world(tmp_path / "gold.db")
    yield built
    built.close()


def make_service(world: World, **kwargs) -> GoldService:
    return GoldService(
        SqliteGoldRepository(world.db), world.topics, world.uow, world.clock, **kwargs
    )


@pytest.fixture
def service(world: World) -> GoldService:
    return make_service(world)


# ---- labeling ----


def test_a_text_is_added_with_its_label_note_and_who_added_it(service, world):
    result = service.add(
        PREDICTION,
        Label.UNINTENTIONAL_RUMOR,
        actor="ana",
        note="a chain",
        credibility=2,
        url="https://example.org/x",
    )
    assert result.added is True
    example = service.get(result.example.id)
    assert (example.label, example.note, example.credibility, example.actor) == (
        Label.UNINTENTIONAL_RUMOR,
        "a chain",
        2,
        "ana",
    )
    assert example.added_at == world.clock.now()
    assert service.count() == 1


def test_the_same_text_with_the_same_label_again_is_a_quiet_no_op(service):
    first = service.add(PREDICTION, Label.NO_RISK, actor="ana", note="first")
    again = service.add("  " + PREDICTION.replace(" ", "   ") + " ", Label.NO_RISK, actor="ben")
    assert (first.added, again.added) == (True, False)
    assert again.example == first.example  # the stored one, untouched
    assert service.count() == 1


def test_the_same_text_with_another_label_is_a_conflict_and_changes_nothing(service):
    service.add(PREDICTION, Label.NO_RISK, actor="ana")
    with pytest.raises(ConflictError) as caught:
        service.add(PREDICTION, Label.UNINTENTIONAL_RUMOR, actor="ana")
    assert "another label" in caught.value.message
    assert service.list()[0].label is Label.NO_RISK


def test_after_removing_the_text_it_can_be_labeled_again(service):
    first = service.add(PREDICTION, Label.NO_RISK, actor="ana").example
    service.remove(first.id)
    again = service.add(PREDICTION, Label.UNINTENTIONAL_RUMOR, actor="ana")
    assert again.added is True
    assert again.example.label is Label.UNINTENTIONAL_RUMOR


def test_an_unknown_example_is_not_found_to_get_or_remove(service):
    with pytest.raises(NotFoundError):
        service.get("gld_" + "0" * 16)
    with pytest.raises(NotFoundError):
        service.remove("gld_" + "0" * 16)
    with pytest.raises(NotFoundError):
        service.get("not an id")


@pytest.mark.parametrize(
    "bad",
    [
        {"text": ""},
        {"text": "   "},
        {"text": "x" * 20_001},
        {"label": "rumor"},
        {"label": None},
        {"credibility": 0},
        {"credibility": 6},
        {"note": "x" * 501},
        {"note": "bell\x07"},
        {"url": "ftp://example.org"},
        {"actor": ""},
    ],
)
def test_invalid_input_is_refused_and_nothing_is_stored(service, bad):
    fields = {"text": PREDICTION, "label": Label.NO_RISK, "actor": "ana", **bad}
    text, label = fields.pop("text"), fields.pop("label")
    with pytest.raises(ValidationError):
        service.add(text, label, **fields)
    assert service.count() == 0


def test_the_set_has_a_size_limit(world):
    small = make_service(world, max_examples=3)
    for n in range(3):
        small.add(f"texto numero {n}", Label.NO_RISK, actor="ana")
    with pytest.raises(ConflictError) as caught:
        small.add("uno mas", Label.NO_RISK, actor="ana")
    assert "full" in caught.value.message
    assert (
        small.add("texto numero 1", Label.NO_RISK, actor="ana").added is False
    )  # still idempotent


def test_the_list_filters_by_label_and_pages(service):
    for n in range(5):
        service.add(f"texto {n}", Label.NO_RISK if n % 2 else Label.UNINTENTIONAL_RUMOR, actor="a")
    assert [e.text for e in service.list(limit=2)] == ["texto 0", "texto 1"]
    assert [e.text for e in service.list(limit=2, offset=4)] == ["texto 4"]
    assert len(service.list(label=Label.NO_RISK)) == 2


# ---- export and import ----


def test_an_exported_set_is_read_back_by_the_calibration(service):
    service.add(PREDICTION, Label.UNINTENTIONAL_RUMOR, actor="ana", credibility=4)
    service.add(OFF_TOPIC, Label.OUT_OF_SCOPE, actor="ana")
    text = service.export_text()
    again = load_samples(text)
    assert [s.expected for s in again] == [(Label.UNINTENTIONAL_RUMOR,), (Label.OUT_OF_SCOPE,)]
    assert again[0].credibility == 4
    assert "do not commit" in text.splitlines()[0]
    assert service.export_text(label=Label.OUT_OF_SCOPE).count("- id:") == 1


def test_an_empty_set_exports_a_valid_empty_file(service):
    assert load_samples(service.export_text()) == ()


# ---- calibration ----


def test_the_whole_seed_is_calibrated_and_the_cascade_agrees_with_every_label(service):
    samples = load_samples(SEED.read_text(encoding="utf-8"))
    run = service.calibrate(samples, current_threshold=0.5)
    report = run.report
    assert report.total == len(samples) > 40
    assert report.correct == report.total
    assert report.accuracy == 1.0
    assert 0 < report.escalation_rate < 1
    assert (run.asked, run.answered, run.failed, run.busy) == (0, 0, 0, 0)
    assert report.suggested is not None


def test_the_local_set_is_what_is_calibrated_by_default(service):
    service.add(PREDICTION, Label.UNINTENTIONAL_RUMOR, actor="ana")
    service.add(OFF_TOPIC, Label.OUT_OF_SCOPE, actor="ana")
    service.add(REPORT, Label.UNINTENTIONAL_RUMOR, actor="ana")  # the cascade disagrees
    run = service.calibrate(service.samples(), current_threshold=0.5)
    assert (run.report.total, run.report.correct) == (3, 2)


def test_an_empty_set_is_an_empty_report(service):
    run = service.calibrate(service.samples(), current_threshold=0.5)
    assert run.report.total == 0
    assert run.report.accuracy is None


class Judge:
    """Records what it was asked and answers from a script."""

    def __init__(self, *script) -> None:
        self.script = list(script)
        self.calls: list[tuple[str, str, JudgeEvidence]] = []

    def __call__(self, item_id: str, text: str, evidence: JudgeEvidence):
        self.calls.append((item_id, text, evidence))
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return step


def seed_samples(service: GoldService):
    service.add(PREDICTION, Label.NO_RISK, actor="ana")  # escalated by the cascade
    service.add(REPORT, Label.NO_RISK, actor="ana")  # settled by the rules
    service.add(OFF_TOPIC, Label.OUT_OF_SCOPE, actor="ana")  # never escalated
    return service.samples()


def test_only_the_texts_the_cascade_would_escalate_are_sent_to_the_model(service):
    judge = Judge(response(label=Label.NO_RISK, confidence=0.9))
    run = service.calibrate(seed_samples(service), current_threshold=0.5, judge=judge)
    assert [text for _, text, _ in judge.calls] == [PREDICTION]
    assert (run.asked, run.answered, run.failed, run.busy) == (1, 1, 0, 0)
    assert run.report.llm_answered == 1
    assert run.report.correct == 3  # the model said no_risk, as the person labeled it


def test_the_answer_of_the_model_replaces_the_cascade_for_that_text(service):
    judge = Judge(response(label=Label.DELIBERATE_DISINFORMATION, confidence=0.91))
    run = service.calibrate(seed_samples(service), current_threshold=0.5, judge=judge)
    assert run.report.correct == 2
    stats = {s.label: s for s in run.report.labels}
    assert stats[Label.DELIBERATE_DISINFORMATION].predicted == 1


def test_the_model_is_given_the_evidence_of_the_rules_and_a_throwaway_id(service):
    judge = Judge(response())
    service.calibrate(seed_samples(service), current_threshold=0.5, judge=judge)
    item_id, text, evidence = judge.calls[0]
    assert item_id.startswith("gold_")
    assert text == PREDICTION
    assert evidence.prior_label is Label.UNINTENTIONAL_RUMOR
    assert evidence.catalog_status == "unavailable"
    assert [s.name for s in evidence.signals]
    assert "gld_" not in item_id  # nothing of the stored example's identity goes out


def test_without_a_judge_nothing_is_ever_asked(service):
    run = service.calibrate(seed_samples(service), current_threshold=0.5)
    assert run.asked == 0
    assert run.report.llm_answered == 0


def test_a_failing_judge_leaves_the_cascade_verdict_and_is_counted(service):
    judge = Judge(JudgeError(JudgeErrorKind.TIMEOUT, "slow"))
    run = service.calibrate(seed_samples(service), current_threshold=0.5, judge=judge)
    assert (run.asked, run.answered, run.failed) == (1, 0, 1)
    assert run.report.llm_answered == 0
    assert run.report.total == 3


def test_a_judge_that_is_busy_is_counted_apart(service):
    run = service.calibrate(seed_samples(service), current_threshold=0.5, judge=Judge(None))
    assert (run.asked, run.answered, run.failed, run.busy) == (1, 0, 0, 1)


def test_after_three_failures_in_a_row_it_stops_asking(service):
    for n in range(6):
        service.add(
            f"Mañana habrá un temblor fuerte en Cali {n}, compartan este mensaje",
            Label.NO_RISK,
            actor="a",
        )
    judge = Judge(JudgeError(JudgeErrorKind.UNREACHABLE, "down"))
    run = service.calibrate(service.samples(), current_threshold=0.5, judge=judge)
    assert len(judge.calls) == 3
    assert run.failed == 3
    assert run.report.total == 6  # the rest are scored by the cascade alone


def test_a_success_resets_the_run_of_failures(service):
    for n in range(6):
        service.add(
            f"Mañana habrá un temblor fuerte en Cali {n}, compartan este mensaje",
            Label.NO_RISK,
            actor="a",
        )
    down = JudgeError(JudgeErrorKind.UNREACHABLE, "down")
    judge = Judge(down, down, response(), down, down, response())
    run = service.calibrate(service.samples(), current_threshold=0.5, judge=judge)
    assert len(judge.calls) == 6
    assert (run.answered, run.failed) == (2, 4)


def test_calibration_writes_nothing(service, world):
    seed_samples(service)
    tables = ("items", "classifications", "llm_judgments", "control_events", "alerts")
    before = {t: world.db.query(f"SELECT count(*) FROM {t}")[0][0] for t in tables}
    service.calibrate(service.samples(), current_threshold=0.5, judge=Judge(response()))
    after = {t: world.db.query(f"SELECT count(*) FROM {t}")[0][0] for t in tables}
    assert after == before


def test_calibration_is_deterministic(service):
    samples = seed_samples(service)
    first = service.calibrate(samples, current_threshold=0.5)
    second = service.calibrate(samples, current_threshold=0.5)
    assert first == second
