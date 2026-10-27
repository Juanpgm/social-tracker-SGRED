"""Vocabulary of the cascade: labels, signals and the signal registry."""

from importlib.resources import files

import pytest

from social_tracker.classification.domain import (
    RISK_LABELS,
    SIGNALS,
    Label,
    Signal,
    Stage,
)
from social_tracker.shared.errors import ValidationError


def test_the_labels_are_exactly_the_specified_ones():
    assert {label.value for label in Label} == {
        "unintentional_rumor",
        "deliberate_disinformation",
        "recycled_content",
        "no_risk",
        "unclassified",
        "out_of_scope",
    }
    assert {label.value for label in RISK_LABELS} == {
        "unintentional_rumor",
        "deliberate_disinformation",
        "recycled_content",
    }


def test_the_stages_are_the_cascade_stages():
    assert [stage.value for stage in Stage] == ["topic", "rules", "dedup", "catalog", "llm"]


def test_every_default_signal_argues_for_a_label_the_rules_can_pick_with_a_valid_weight():
    for name, spec in SIGNALS.items():
        assert spec.label in (*RISK_LABELS, Label.NO_RISK), name
        assert 0.0 < spec.weight <= 1.0, name


def test_the_shipped_configuration_documents_every_signal_name():
    text = files("social_tracker.classification").joinpath("default_topics.yaml").read_text("utf-8")
    assert all(name in text for name in SIGNALS)


@pytest.mark.parametrize(
    ("name", "label", "weight"),
    [
        ("nonsense", Label.NO_RISK, 0.5),
        ("news_report", "no_risk", 0.5),
        ("news_report", Label.NO_RISK, 1.1),
        ("news_report", Label.NO_RISK, -0.1),
        ("news_report", Label.NO_RISK, True),
        ("news_report", Label.NO_RISK, "0.5"),
        ("news_report", Label.NO_RISK, None),
    ],
)
def test_an_invalid_signal_is_refused(name, label, weight):
    with pytest.raises(ValidationError):
        Signal(name, label, weight)  # type: ignore[arg-type]


def test_a_signal_weight_may_be_zero_or_one():
    assert Signal("news_report", Label.NO_RISK, 0).weight == 0
    assert Signal("news_report", Label.NO_RISK, 1.0).weight == 1.0
