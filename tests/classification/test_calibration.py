"""Calibration over a gold set (report only): the CPU cascade's prediction for a labeled text, the
per-label precision and recall, the confusion matrix, the escalation rate and the sweep of the
alert threshold."""

from pathlib import Path

import pytest
import yaml

from social_tracker.classification.calibration import (
    LABEL_ORDER,
    THRESHOLDS,
    Prediction,
    Sample,
    evaluate,
    predict,
)
from social_tracker.classification.domain import ClassifierSettings, Label, Stage
from social_tracker.shared.errors import ValidationError
from tests.classification.support import SETTINGS, default, run_rules

RUMOR, DELIBERATE, RECYCLED = (
    Label.UNINTENTIONAL_RUMOR,
    Label.DELIBERATE_DISINFORMATION,
    Label.RECYCLED_CONTENT,
)
NO_RISK, UNCLASSIFIED, OUT = Label.NO_RISK, Label.UNCLASSIFIED, Label.OUT_OF_SCOPE
DATA = Path(__file__).parents[1] / "data" / "gold_examples.yaml"


def pred(label: Label, confidence: float = 0.8, *, escalated: bool = False, llm: bool = False):
    return Prediction(
        label=label,
        confidence=confidence,
        escalated=escalated,
        stage=Stage.LLM if llm else Stage.RULES,
        rationale="r",
        answered_by_llm=llm,
    )


def sample(name: str, *expected: Label) -> Sample:
    return Sample(name, f"text {name}", tuple(expected))


def run(pairs, threshold=0.5):
    return evaluate(pairs, current_threshold=threshold)


# ---- predict: the topic filter and the rules, as the classifier runs them ----


def predict_text(text: str, **kw):
    config, matcher = default()
    return predict(
        Sample("s", text, (NO_RISK,), **kw),
        matcher=matcher,
        config=config,
        settings=SETTINGS,
        reference_year=2026,
    )


def test_a_prediction_text_is_a_rumor_that_goes_to_the_model():
    result = predict_text("Mañana habrá un temblor fuerte en Cali, compartan este mensaje")
    assert result.label is RUMOR
    assert result.escalated is True
    assert 0.4 <= result.confidence < 0.8
    assert result.stage is Stage.RULES


def test_an_off_topic_text_is_out_of_scope_and_never_escalated():
    result = predict_text("El equipo ganó el partido de ayer en el estadio de la ciudad")
    assert (result.label, result.confidence, result.escalated) == (OUT, 1.0, False)


def test_an_empty_text_is_unclassified_with_no_confidence():
    result = predict_text("  \n ")
    assert (result.label, result.confidence, result.escalated) == (UNCLASSIFIED, 0.0, False)


def test_the_prediction_is_what_the_classifier_would_say_for_every_gold_example():
    examples = yaml.safe_load(DATA.read_text(encoding="utf-8"))["examples"]
    config, matcher = default()
    checked = 0
    for example in examples:
        expect = example["expect"]
        got = predict(
            Sample(
                example["id"],
                example["text"],
                (OUT,),
                example.get("credibility", 3),
                example.get("url"),
            ),
            matcher=matcher,
            config=config,
            settings=SETTINGS,
            reference_year=2026,
        )
        if not expect["in_scope"]:
            assert got.label is OUT, example["id"]
            continue
        expected = run_rules(
            example["text"], url=example.get("url"), credibility=example.get("credibility", 3)
        )
        assert (got.label, got.confidence, got.escalated) == (
            expected.label,
            expected.confidence,
            expected.escalated,
        ), example["id"]
        checked += 1
    assert checked > 40


def test_the_same_text_always_gets_the_same_prediction():
    text = "Según la Secretaría de Gestión del Riesgo, evacuen la zona ya"
    assert len({predict_text(text) for _ in range(5)}) == 1


@pytest.mark.parametrize("credibility", [1, 3, 5])
def test_the_source_credibility_moves_the_confidence_like_in_the_classifier(credibility):
    text = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    got = predict_text(text, credibility=credibility)
    assert got.confidence == run_rules(text, credibility=credibility).confidence


def test_a_narrower_escalation_band_changes_what_is_escalated():
    config, matcher = default()
    text = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    narrow = ClassifierSettings(escalate_low=0.1, escalate_high=0.2, inconclusive_below=0.1)
    got = predict(
        Sample("s", text, (RUMOR,)),
        matcher=matcher,
        config=config,
        settings=narrow,
        reference_year=2026,
    )
    assert got.escalated is False
    assert got.label is RUMOR  # settled by the rules alone


# ---- the matrix and the per-label figures ----


def test_a_perfect_run_has_everything_on_the_diagonal():
    pairs = [(sample(str(i), label), pred(label)) for i, label in enumerate(LABEL_ORDER)]
    report = run(pairs)
    assert (report.total, report.correct, report.accuracy) == (6, 6, 1.0)
    for i in range(6):
        assert report.matrix[i][i] == 1
        assert sum(report.matrix[i]) == 1
    assert all(s.precision == 1.0 and s.recall == 1.0 for s in report.labels)


def test_the_matrix_rows_are_the_expected_label_and_the_columns_the_predicted_one():
    pairs = [
        (sample("a", RUMOR), pred(DELIBERATE)),
        (sample("b", RUMOR), pred(RUMOR)),
        (sample("c", NO_RISK), pred(RUMOR)),
    ]
    report = run(pairs)
    row = {label: i for i, label in enumerate(LABEL_ORDER)}
    assert report.matrix[row[RUMOR]][row[DELIBERATE]] == 1
    assert report.matrix[row[RUMOR]][row[RUMOR]] == 1
    assert report.matrix[row[NO_RISK]][row[RUMOR]] == 1
    assert report.correct == 1


def test_precision_and_recall_per_label():
    pairs = [
        (sample("a", RUMOR), pred(RUMOR)),
        (sample("b", RUMOR), pred(RUMOR)),
        (sample("c", RUMOR), pred(NO_RISK)),
        (sample("d", NO_RISK), pred(RUMOR)),
        (sample("e", NO_RISK), pred(NO_RISK)),
    ]
    stats = {s.label: s for s in run(pairs).labels}
    rumor = stats[RUMOR]
    assert (rumor.support, rumor.predicted, rumor.true_positives) == (3, 3, 2)
    assert rumor.precision == pytest.approx(2 / 3)
    assert rumor.recall == pytest.approx(2 / 3)
    assert rumor.f1 == pytest.approx(2 / 3)
    no_risk = stats[NO_RISK]
    assert (no_risk.support, no_risk.predicted, no_risk.true_positives) == (2, 2, 1)
    assert no_risk.precision == 0.5
    assert no_risk.recall == 0.5


def test_a_label_nobody_expected_or_predicted_has_no_figures_instead_of_a_division_by_zero():
    stats = {s.label: s for s in run([(sample("a", RUMOR), pred(RUMOR))]).labels}
    recycled = stats[RECYCLED]
    assert (recycled.support, recycled.predicted) == (0, 0)
    assert (recycled.precision, recycled.recall, recycled.f1) == (None, None, None)


def test_a_label_that_was_predicted_but_never_expected_has_zero_precision_and_no_recall():
    stats = {s.label: s for s in run([(sample("a", NO_RISK), pred(RECYCLED))]).labels}
    assert stats[RECYCLED].precision == 0.0
    assert stats[RECYCLED].recall is None
    assert stats[NO_RISK].recall == 0.0
    assert stats[NO_RISK].precision is None


def test_several_acceptable_labels_count_as_right_when_the_prediction_is_one_of_them():
    right = run([(sample("a", UNCLASSIFIED, DELIBERATE), pred(DELIBERATE))])
    assert right.correct == 1
    wrong = run([(sample("a", UNCLASSIFIED, DELIBERATE), pred(RUMOR))])
    assert wrong.correct == 0
    row = {label: i for i, label in enumerate(LABEL_ORDER)}
    assert wrong.matrix[row[UNCLASSIFIED]][row[RUMOR]] == 1  # the first one is "the" expectation


def test_an_empty_gold_set_is_an_empty_report():
    report = run([])
    assert (report.total, report.correct, report.accuracy, report.escalation_rate) == (
        0,
        0,
        None,
        None,
    )
    assert report.suggested is None
    assert all(sum(row) == 0 for row in report.matrix)


def test_the_labels_are_always_the_six_in_the_same_order():
    assert [s.label for s in run([]).labels] == list(LABEL_ORDER)
    assert len(LABEL_ORDER) == 6
    assert LABEL_ORDER[:3] == (RUMOR, DELIBERATE, RECYCLED)


# ---- escalation ----


def test_the_escalation_rate_is_the_share_of_texts_that_would_go_to_the_model():
    pairs = [
        (sample("a", RUMOR), pred(RUMOR, escalated=True)),
        (sample("b", RUMOR), pred(RUMOR, escalated=True)),
        (sample("c", NO_RISK), pred(NO_RISK)),
        (sample("d", OUT), pred(OUT, 1.0)),
    ]
    report = run(pairs)
    assert (report.escalated, report.escalation_rate) == (2, 0.5)


def test_texts_the_model_answered_are_counted_and_scored_by_its_answer():
    pairs = [
        (sample("a", NO_RISK), pred(NO_RISK, 0.9, escalated=True, llm=True)),
        (sample("b", RUMOR), pred(RUMOR, 0.6, escalated=True)),
    ]
    report = run(pairs)
    assert (report.llm_answered, report.escalated, report.correct) == (1, 2, 2)


# ---- the sweep of the alert threshold ----


def test_the_sweep_covers_30_to_95_percent_in_steps_of_five():
    assert THRESHOLDS[0] == 0.30
    assert THRESHOLDS[-1] == 0.95
    assert len(THRESHOLDS) == 14
    assert all(round(b - a, 2) == 0.05 for a, b in zip(THRESHOLDS, THRESHOLDS[1:], strict=False))


def test_a_threshold_alerts_from_its_value_inclusive():
    pairs = [(sample("a", RUMOR), pred(RUMOR, 0.55)), (sample("b", RUMOR), pred(RUMOR, 0.5))]
    rows = {r.threshold: r for r in run(pairs).sweep}
    assert rows[0.5].alerts == 2
    assert rows[0.55].alerts == 1  # 0.55 is inclusive
    assert rows[0.6].alerts == 0


def test_precision_and_recall_of_the_alerts_at_each_threshold():
    pairs = [
        (sample("a", RUMOR), pred(RUMOR, 0.9)),  # a true alert, high
        (sample("b", RUMOR), pred(RUMOR, 0.55)),  # a true alert, low
        (sample("c", NO_RISK), pred(RUMOR, 0.6)),  # a false alert
        (sample("d", RUMOR), pred(NO_RISK, 0.9)),  # a missed one
    ]
    rows = {r.threshold: r for r in run(pairs).sweep}
    low = rows[0.5]
    assert (low.alerts, low.true_alerts) == (3, 2)
    assert low.precision == pytest.approx(2 / 3)
    assert low.recall == pytest.approx(2 / 3)  # 2 of the 3 risky texts
    high = rows[0.65]
    assert (high.alerts, high.true_alerts) == (1, 1)
    assert high.precision == 1.0
    assert high.recall == pytest.approx(1 / 3)


def test_the_suggestion_is_the_threshold_with_the_best_f1_and_is_never_applied():
    pairs = [
        (sample("a", RUMOR), pred(RUMOR, 0.9)),
        (sample("b", RUMOR), pred(RUMOR, 0.8)),
        (sample("c", NO_RISK), pred(RUMOR, 0.45)),
        (sample("d", NO_RISK), pred(RUMOR, 0.4)),
    ]
    report = run(pairs, threshold=0.3)
    assert report.suggested == 0.5
    assert report.current.threshold == 0.3
    assert report.current.precision == 0.5


def test_ties_prefer_precision_and_then_the_lower_threshold():
    pairs = [(sample("a", RUMOR), pred(RUMOR, 0.9))]
    report = run(pairs)
    assert report.suggested == 0.3  # every threshold up to 0.9 is perfect: the lowest one


def test_with_no_risky_text_there_is_nothing_to_suggest():
    report = run([(sample("a", NO_RISK), pred(NO_RISK, 0.9)), (sample("b", OUT), pred(OUT, 1.0))])
    assert report.suggested is None
    assert all(r.recall is None for r in report.sweep)


def test_the_current_threshold_is_reported_even_when_it_is_not_on_the_grid():
    report = run([(sample("a", RUMOR), pred(RUMOR, 0.72))], threshold=0.72)
    assert report.current.threshold == 0.72
    assert report.current.alerts == 1
    assert 0.72 not in {r.threshold for r in report.sweep}


def test_a_recycled_or_deliberate_prediction_counts_as_an_alert_like_a_rumor():
    pairs = [
        (sample("a", DELIBERATE), pred(DELIBERATE, 0.9)),
        (sample("b", RECYCLED), pred(RECYCLED, 0.9)),
    ]
    row = {r.threshold: r for r in run(pairs).sweep}[0.5]
    assert (row.alerts, row.true_alerts) == (2, 2)


def test_the_report_is_deterministic():
    pairs = [
        (sample(str(i), RUMOR if i % 2 else NO_RISK), pred(RUMOR, 0.3 + (i % 7) / 10))
        for i in range(40)
    ]
    assert run(pairs) == run(list(pairs))


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), float("inf"), True, "0.5", None])
def test_confidences_of_the_prediction_are_validated(bad):
    with pytest.raises(ValidationError):
        pred(RUMOR, bad)


@pytest.mark.parametrize(
    "bad",
    [
        {"id": ""},
        {"id": "x" * 65},
        {"text": None},
        {"expected": ()},
        {"expected": ("rumor",)},
        {"credibility": 0},
        {"credibility": 6},
        {"credibility": True},
        {"url": 5},
    ],
)
def test_a_sample_must_be_well_formed(bad):
    fields = {"id": "s", "text": "t", "expected": (NO_RISK,), **bad}
    with pytest.raises(ValidationError):
        Sample(**fields)
