"""How the catalog cross-check enters the rules (RC-6): a mismatch is evidence for risk, a
consistent claim only lowers the weight of an unlinked attribution, nothing else changes."""

from dataclasses import replace

import pytest

from social_tracker.classification.catalog_match import (
    CONSISTENT,
    MISMATCH,
    NO_CLAIM,
    NOT_CHECKED,
    UNAVAILABLE,
)
from social_tracker.classification.domain import (
    CATALOG_CONSISTENT_FACTOR,
    SIGNALS,
    ClassifierSettings,
    Label,
)
from social_tracker.classification.rules import RuleOutcome, evaluate_rules, needs_escalation
from social_tracker.classification.text import prepare
from social_tracker.classification.topics import TopicMatcher
from social_tracker.classification.yaml_io import default_topic_config

SETTINGS = ClassifierSettings()
CONFIG = default_topic_config()
MATCHER = TopicMatcher(CONFIG)
NEUTRAL = "Se sintió un temblor leve esta mañana en varios barrios de Cali, sin daños reportados"
REPORT = "El SGC reportó un sismo de magnitud 4.2 en Cali"
PREDICTION = "Mañana habrá un temblor fuerte en Cali"
ATTRIBUTION = "Según la Secretaría de Gestión del Riesgo, evacuen la zona ya"


def run(text: str, status: str | None = None, *, config=CONFIG) -> RuleOutcome:
    prepared = prepare("", text, max_chars=SETTINGS.max_text_chars)
    options = {} if status is None else {"catalog_status": status}
    return evaluate_rules(
        prepared,
        TopicMatcher(config) if config is not CONFIG else MATCHER,
        config,
        credibility=3,
        reference_year=2026,
        item_url=None,
        settings=SETTINGS,
        **options,
    )


def names(outcome: RuleOutcome) -> dict[str, float]:
    return {s.name: s.weight for s in outcome.signals}


def test_the_mismatch_signal_is_catalogued_as_a_rumor_clue_of_moderate_weight():
    spec = SIGNALS["catalog_mismatch"]
    assert spec.label is Label.UNINTENTIONAL_RUMOR
    assert spec.weight == 0.30
    assert 0 < CATALOG_CONSISTENT_FACTOR < 1


def test_a_mismatch_adds_the_signal_and_nothing_else():
    without = names(run(PREDICTION))
    with_it = names(run(PREDICTION, MISMATCH))
    assert with_it == {**without, "catalog_mismatch": 0.30}


def test_a_mismatch_raises_the_risk_evidence_and_the_queue_strength():
    plain = run(PREDICTION)
    flagged = run(PREDICTION, MISMATCH)
    assert plain.label is flagged.label is Label.UNINTENTIONAL_RUMOR
    assert flagged.confidence > plain.confidence
    assert flagged.risk_strength > plain.risk_strength


def test_a_mismatch_alone_decides_nothing_but_sends_the_item_to_review():
    plain = run(NEUTRAL)
    assert plain.label is Label.NO_RISK
    flagged = run(NEUTRAL, MISMATCH)
    assert flagged.label is Label.UNCLASSIFIED
    assert flagged.inconclusive
    assert flagged.risk_strength == pytest.approx(0.30)
    assert needs_escalation(
        in_scope=True,
        confidence=flagged.confidence,
        inconclusive=flagged.inconclusive,
        conflicting=flagged.conflicting,
        duplicate_of_judged=False,
        settings=SETTINGS,
        label=flagged.label,
    )


def test_a_mismatch_stops_a_report_from_being_settled_as_harmless():
    assert run(REPORT).label is Label.NO_RISK
    flagged = run(REPORT, MISMATCH)
    assert flagged.label is Label.UNCLASSIFIED
    assert flagged.mixed
    assert {"news_report", "catalog_mismatch"} <= set(names(flagged))


def test_a_consistent_claim_lowers_an_unlinked_attribution():
    plain = names(run(ATTRIBUTION))["official_attribution_unlinked"]
    lowered = names(run(ATTRIBUTION, CONSISTENT))["official_attribution_unlinked"]
    assert plain == 0.45
    assert lowered == pytest.approx(plain * CATALOG_CONSISTENT_FACTOR)
    assert run(ATTRIBUTION, CONSISTENT).confidence < run(ATTRIBUTION).confidence


def test_a_consistent_claim_never_creates_a_signal_or_a_label_alone():
    for text in (NEUTRAL, REPORT, PREDICTION):
        assert run(text, CONSISTENT) == run(text)
    assert "catalog_mismatch" not in names(run(ATTRIBUTION, CONSISTENT))


@pytest.mark.parametrize("status", [UNAVAILABLE, NO_CLAIM, NOT_CHECKED, "bogus", "", "MISMATCH"])
@pytest.mark.parametrize("text", [NEUTRAL, REPORT, PREDICTION, ATTRIBUTION])
def test_any_other_status_changes_nothing(status, text):
    assert run(text, status) == run(text)


def test_the_weight_can_be_calibrated_and_switched_off():
    off = replace(CONFIG, weights=(("catalog_mismatch", 0.0),))
    assert "catalog_mismatch" not in names(run(NEUTRAL, MISMATCH, config=off))
    assert run(NEUTRAL, MISMATCH, config=off).label is Label.NO_RISK
    strong = replace(CONFIG, weights=(("catalog_mismatch", 0.9),))
    outcome = run(NEUTRAL, MISMATCH, config=strong)
    assert names(outcome)["catalog_mismatch"] == 0.9
    assert outcome.label is Label.UNINTENTIONAL_RUMOR


def test_the_same_inputs_give_the_same_outcome():
    assert run(PREDICTION, MISMATCH) == run(PREDICTION, MISMATCH)
