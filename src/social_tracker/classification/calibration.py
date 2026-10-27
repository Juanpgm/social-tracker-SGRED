"""Calibration of the cascade against a gold set: what it would say, how often it is right, and
where the alert threshold would sit best. Pure and deterministic, report only (nothing here
changes a setting, a verdict or a file).

`predict` is the CPU cascade for one labeled text as the classifier runs it, minus the parts that
need history or the network: the topic filter and the rules (stages 1 and 2). There is no
duplicate or recycled check (no earlier items to compare with) and the official-catalog cross-check
reads `unavailable`, so a text can only come out recycled through the signals of the rules. A text
the cascade would send to the language model is flagged `escalated`; the optional judgment of the
model (`answered_by_llm`) is added by the caller, which holds the judge.

`evaluate` turns pairs of (labeled sample, prediction) into the report: the confusion matrix (rows
are the expected label, columns the predicted one), precision, recall and F1 per label, the share
of texts that would go to the model, and the sweep of `alert_min_confidence`: for each threshold
the alerts it would raise (a risk label at or above it, inclusive), how many were right and how
many risky texts it would miss. The threshold with the best F1 (then the best precision, then the
lowest) is only SUGGESTED.

A sample may name several acceptable labels (the first is "the" expectation): a prediction that is
one of them counts as right and is placed on the diagonal.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from social_tracker.classification.domain import (
    RISK_LABELS,
    ClassifierSettings,
    Label,
    OutOfScopeReason,
    Signal,
    Stage,
    Topic,
)
from social_tracker.classification.rules import (
    evaluate_rules,
    needs_escalation,
    rationale_for_rules,
)
from social_tracker.classification.text import prepare
from social_tracker.classification.topics import TopicConfig, TopicMatcher
from social_tracker.shared.errors import ValidationError

LABEL_ORDER = (
    Label.UNINTENTIONAL_RUMOR,
    Label.DELIBERATE_DISINFORMATION,
    Label.RECYCLED_CONTENT,
    Label.NO_RISK,
    Label.UNCLASSIFIED,
    Label.OUT_OF_SCOPE,
)
THRESHOLDS = tuple(round(0.30 + 0.05 * step, 2) for step in range(14))  # 0.30 .. 0.95
MAX_SAMPLE_ID_CHARS = 64
CATALOG_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Sample:
    """A labeled text. `expected` are the acceptable labels, the first being the expectation."""

    id: str
    text: str
    expected: tuple[Label, ...]
    credibility: int = 3
    url: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, str)
            or not 1 <= len(self.id) <= MAX_SAMPLE_ID_CHARS
            or not isinstance(self.text, str)
            or not isinstance(self.expected, tuple)
            or not self.expected
            or not all(isinstance(label, Label) for label in self.expected)
            or isinstance(self.credibility, bool)
            or not isinstance(self.credibility, int)
            or not 1 <= self.credibility <= 5
            or (self.url is not None and not isinstance(self.url, str))
        ):
            raise ValidationError("a gold sample needs an id, a text, labels and a credibility")


@dataclass(frozen=True)
class Prediction:
    """What the cascade (and, when it answered, the model) says about a text."""

    label: Label
    confidence: float
    escalated: bool
    stage: Stage
    rationale: str
    topics: tuple[Topic, ...] = ()
    signals: tuple[Signal, ...] = ()
    answered_by_llm: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.label, Label)
            or isinstance(self.confidence, bool)
            or not isinstance(self.confidence, int | float)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValidationError("a prediction needs a label and a confidence from 0 to 1")


def predict(
    sample: Sample,
    *,
    matcher: TopicMatcher,
    config: TopicConfig,
    settings: ClassifierSettings,
    reference_year: int,
) -> Prediction:
    """The CPU cascade for one text: the topic filter, then the rules and the escalation rule."""
    prepared = prepare("", sample.text, max_chars=settings.max_text_chars)
    topic = matcher.evaluate_prepared(prepared)
    if not topic.in_scope:
        if topic.reason is OutOfScopeReason.EMPTY_TEXT:
            return Prediction(Label.UNCLASSIFIED, 0.0, False, Stage.TOPIC, "no words to classify")
        if prepared.truncated:  # part of the text was never read: it cannot be called off topic
            return Prediction(
                Label.UNCLASSIFIED, 0.0, True, Stage.TOPIC, "truncated text with no topic read"
            )
        return Prediction(Label.OUT_OF_SCOPE, 1.0, False, Stage.TOPIC, "out of scope")
    outcome = evaluate_rules(
        prepared,
        matcher,
        config,
        credibility=sample.credibility,
        reference_year=reference_year,
        item_url=sample.url,
        settings=settings,
        source_official=False,
        catalog_status=CATALOG_UNAVAILABLE,
    )
    escalated = needs_escalation(
        in_scope=True,
        confidence=outcome.confidence,
        inconclusive=outcome.inconclusive,
        conflicting=outcome.conflicting,
        duplicate_of_judged=False,
        settings=settings,
        label=outcome.label,
    )
    return Prediction(
        outcome.label,
        outcome.confidence,
        escalated,
        Stage.RULES,
        rationale_for_rules(outcome, topic),
        topic.topics,
        outcome.signals,
    )


# ---- the report ----


@dataclass(frozen=True)
class LabelStats:
    """The figures of one label. `precision` is None when nothing was predicted with it and
    `recall` when nothing was expected to have it."""

    label: Label
    support: int
    predicted: int
    true_positives: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class ThresholdRow:
    """What one value of `alert_min_confidence` would have done."""

    threshold: float
    alerts: int
    true_alerts: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class CalibrationReport:
    total: int
    correct: int
    accuracy: float | None
    labels: tuple[LabelStats, ...]
    matrix: tuple[tuple[int, ...], ...]
    escalated: int
    escalation_rate: float | None
    llm_answered: int
    sweep: tuple[ThresholdRow, ...]
    current: ThresholdRow
    suggested: float | None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _resolve(sample: Sample, prediction: Prediction) -> Label:
    """The label a prediction is scored against: itself when the sample accepts it, else the
    first expectation."""
    return prediction.label if prediction.label in sample.expected else sample.expected[0]


def _row(threshold: float, scored: Sequence[tuple[Label, Prediction]]) -> ThresholdRow:
    alerts = [(e, p) for e, p in scored if p.label in RISK_LABELS and p.confidence >= threshold]
    true_alerts = sum(1 for e, _ in alerts if e in RISK_LABELS)
    positives = sum(1 for e, _ in scored if e in RISK_LABELS)
    precision = _ratio(true_alerts, len(alerts))
    recall = _ratio(true_alerts, positives)
    return ThresholdRow(
        threshold, len(alerts), true_alerts, precision, recall, _f1(precision, recall)
    )


def _suggest(sweep: Sequence[ThresholdRow]) -> float | None:
    """The best F1, then the best precision, then the lowest threshold; None without a risky text
    in the set (nothing to measure the alerts against)."""
    candidates = [row for row in sweep if row.f1 is not None]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: (r.f1 or 0.0, r.precision or 0.0, -r.threshold))
    return best.threshold


def evaluate(
    pairs: Sequence[tuple[Sample, Prediction]], *, current_threshold: float
) -> CalibrationReport:
    """The report of a run. `current_threshold` is shown next to the sweep, on the grid or not."""
    index = {label: i for i, label in enumerate(LABEL_ORDER)}
    size = len(LABEL_ORDER)
    matrix = [[0] * size for _ in range(size)]
    scored: list[tuple[Label, Prediction]] = []
    for sample, prediction in pairs:
        expected = _resolve(sample, prediction)
        matrix[index[expected]][index[prediction.label]] += 1
        scored.append((expected, prediction))
    total = len(pairs)
    correct = sum(matrix[i][i] for i in range(size))
    stats = []
    for label in LABEL_ORDER:
        i = index[label]
        support = sum(matrix[i])
        predicted = sum(matrix[r][i] for r in range(size))
        precision = _ratio(matrix[i][i], predicted)
        recall = _ratio(matrix[i][i], support)
        stats.append(
            LabelStats(
                label, support, predicted, matrix[i][i], precision, recall, _f1(precision, recall)
            )
        )
    sweep = tuple(_row(t, scored) for t in THRESHOLDS)
    escalated = sum(1 for _, p in scored if p.escalated)
    return CalibrationReport(
        total=total,
        correct=correct,
        accuracy=_ratio(correct, total),
        labels=tuple(stats),
        matrix=tuple(tuple(row) for row in matrix),
        escalated=escalated,
        escalation_rate=_ratio(escalated, total),
        llm_answered=sum(1 for _, p in scored if p.answered_by_llm),
        sweep=sweep,
        current=_row(current_threshold, scored),
        suggested=_suggest(sweep),
    )
