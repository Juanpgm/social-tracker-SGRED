"""How a judgment lands, after the remediation round: guard rails on suspicious answers, a cap on
the model's confidence, settlement fields, the identity a verdict may be shared on."""

from dataclasses import replace
from datetime import timedelta

import pytest

from social_tracker.classification.domain import Label, Signal, Stage
from social_tracker.items.domain import text_identity
from social_tracker.llm.domain import (
    MAX_LLM_CONFIDENCE,
    SUSPICIOUS_NO_RISK_CAP,
    contradicts_prior,
    judged,
    left_unjudged,
    released_duplicate,
    requeued_copy,
    reused_by_duplicate,
)
from tests.llm.support import NOW, response, verdict

LATER = NOW + timedelta(minutes=5)


def _no_risk(confidence: float = 0.9):
    return response(label=Label.NO_RISK, confidence=confidence)


# ---- settlement fields ----


def test_a_judged_verdict_records_when_and_why_it_settled():
    out = judged(verdict(), response(), 0.5, LATER)
    assert (out.settled_at, out.settle_reason) == (LATER, "llm")
    assert out.needs_llm is False


def test_without_a_time_nothing_is_recorded():
    out = judged(verdict(), response(), 0.5)
    assert (out.settled_at, out.settle_reason) == (None, None)


@pytest.mark.parametrize("reason", ["invalid", "parked"])
def test_left_unjudged_records_its_reason(reason):
    out = left_unjudged(verdict(), LATER, reason)
    assert (out.settled_at, out.settle_reason) == (LATER, reason)
    assert out.low_confidence is True
    assert out.needs_llm is False


def test_a_reused_copy_records_a_duplicate_settlement():
    original = judged(verdict(1), response(), 0.5, LATER)
    copy = verdict(2, duplicate_of=original.item_id, awaiting_original=True, needs_llm=False)
    out = reused_by_duplicate(copy, original, 0.5, LATER)
    assert (out.settled_at, out.settle_reason, out.awaiting_original) == (LATER, "duplicate", False)


def test_a_released_copy_of_a_judged_original_keeps_its_confidence_flag():
    copy = verdict(2, duplicate_of="itm_00000001", awaiting_original=True, needs_llm=False)
    out = released_duplicate(copy, LATER, "parked")
    assert (out.settled_at, out.settle_reason, out.awaiting_original) == (LATER, "parked", False)
    assert out.low_confidence is True  # no verdict came: the same treatment as left_unjudged


def test_a_released_copy_of_an_invalid_original_is_low_confidence_too():
    copy = verdict(
        2,
        duplicate_of="itm_00000001",
        awaiting_original=True,
        needs_llm=False,
        low_confidence=False,
    )
    assert released_duplicate(copy, LATER, "invalid").low_confidence is True


# ---- a copy that must be judged on its own ----


def test_a_requeued_copy_is_a_normal_queued_item_linked_as_a_near_copy():
    copy = verdict(2, duplicate_of="itm_00000001", awaiting_original=True, needs_llm=False)
    out = requeued_copy(copy, near_of="itm_00000001", priority=0.4)
    assert out.duplicate_of is None
    assert out.near_of == "itm_00000001"
    assert (out.needs_llm, out.awaiting_original, out.priority) == (True, False, 0.4)
    assert (out.settled_at, out.settle_reason) == (None, None)


def test_a_requeued_copy_can_drop_the_link_when_the_original_is_gone():
    copy = verdict(2, duplicate_of="itm_00000001", awaiting_original=True, needs_llm=False)
    out = requeued_copy(copy, near_of=None, priority=0.1)
    assert (out.duplicate_of, out.near_of) == (None, None)


def test_a_requeued_copy_never_has_a_zero_priority():
    copy = verdict(2, duplicate_of="itm_00000001", awaiting_original=True, needs_llm=False)
    assert requeued_copy(copy, near_of=None, priority=0.0).priority > 0.0


# ---- the confidence of the model is capped ----


def test_a_model_cannot_be_more_than_95_percent_sure():
    assert MAX_LLM_CONFIDENCE == 0.95
    out = judged(verdict(), response(label=Label.DELIBERATE_DISINFORMATION, confidence=1.0), 0.5)
    assert out.confidence == 0.95


def test_a_confidence_under_the_cap_is_kept():
    out = judged(verdict(), response(label=Label.UNINTENTIONAL_RUMOR, confidence=0.949), 0.5)
    assert out.confidence == 0.949


def test_exactly_the_cap_is_kept():
    out = judged(verdict(), response(label=Label.UNINTENTIONAL_RUMOR, confidence=0.95), 0.5)
    assert out.confidence == 0.95


# ---- a no_risk that contradicts the earlier stages is not final ----


def _prior_with(*names: tuple[str, Label, float], **changes):
    signals = tuple(Signal(n, label, w) for n, label, w in names)
    return verdict(signals=signals, **changes)


@pytest.mark.parametrize(
    "prior",
    [
        _prior_with(("obfuscated_text", Label.UNCLASSIFIED, 0.3)),
        _prior_with(("prediction_claim", Label.UNINTENTIONAL_RUMOR, 0.6)),
        _prior_with(("official_attribution_unlinked", Label.DELIBERATE_DISINFORMATION, 0.9)),
        _prior_with(("catalog_mismatch", Label.UNINTENTIONAL_RUMOR, 0.3)),
        verdict(catalog_status="mismatch"),
        verdict(label=Label.UNINTENTIONAL_RUMOR, confidence=0.6),
        verdict(label=Label.DELIBERATE_DISINFORMATION, confidence=0.79),
    ],
)
def test_a_no_risk_against_strong_earlier_evidence_is_flagged_and_capped(prior):
    assert contradicts_prior(prior, _no_risk()) is True
    out = judged(prior, _no_risk(0.95), 0.5, LATER)
    assert out.label is Label.NO_RISK
    assert out.low_confidence is True
    assert out.confidence == SUSPICIOUS_NO_RISK_CAP == 0.7
    assert out.settle_reason == "llm"
    assert "review" in out.rationale


@pytest.mark.parametrize(
    "prior",
    [
        verdict(),  # a rumor candidate at 0.55 with one weak signal
        _prior_with(("prediction_claim", Label.UNINTENTIONAL_RUMOR, 0.59)),
        verdict(label=Label.UNINTENTIONAL_RUMOR, confidence=0.59),
        verdict(label=Label.UNCLASSIFIED, confidence=0.9),  # not a risk label
        verdict(catalog_status="no_claim"),
        _prior_with(("news_report", Label.NO_RISK, 0.9)),  # a no_risk signal is no contradiction
    ],
)
def test_a_no_risk_that_agrees_with_weak_evidence_is_final(prior):
    assert contradicts_prior(prior, _no_risk()) is False
    out = judged(prior, _no_risk(0.9), 0.5)
    assert out.confidence == 0.9
    assert out.low_confidence is False


def test_a_risk_answer_is_never_a_contradiction():
    prior = _prior_with(("obfuscated_text", Label.UNCLASSIFIED, 0.3))
    risky = response(label=Label.DELIBERATE_DISINFORMATION, confidence=0.9)
    assert contradicts_prior(prior, risky) is False
    assert judged(prior, risky, 0.5).confidence == 0.9


def test_a_suspicious_no_risk_under_the_cap_keeps_its_own_confidence():
    prior = _prior_with(("obfuscated_text", Label.UNCLASSIFIED, 0.3))
    out = judged(prior, _no_risk(0.4), 0.5)
    assert out.confidence == 0.4
    assert out.low_confidence is True


def test_an_injection_style_signal_name_counts_as_suspicious():
    # No such signal exists yet; the guard keys on the name so a future detector is covered.
    signal = Signal.__new__(Signal)
    object.__setattr__(signal, "name", "injection_attempt")
    object.__setattr__(signal, "label", Label.UNCLASSIFIED)
    object.__setattr__(signal, "weight", 0.2)
    prior = replace(verdict(), signals=())
    object.__setattr__(prior, "signals", (signal,))
    assert contradicts_prior(prior, _no_risk()) is True


def test_the_rationale_stays_within_its_limit_when_the_note_is_added():
    prior = _prior_with(("obfuscated_text", Label.UNCLASSIFIED, 0.3))
    long = response(label=Label.NO_RISK, confidence=0.9, rationale="x" * 300)
    out = judged(prior, long, 0.5)
    assert 1 <= len(out.rationale) <= 300
    assert "review" in out.rationale


# ---- the identity a verdict may be shared on (S-sec2) ----


def test_identical_text_has_the_same_identity():
    assert text_identity("Titulo", "Cuerpo del texto") == text_identity(
        "Titulo", "Cuerpo del texto"
    )


@pytest.mark.parametrize(
    "other",
    [
        ("Titulo", "Cuerpo del texto."),  # punctuation
        ("Titulo", "Cuerpo del Texto"),  # case
        ("Titulo", "Cuerpo  del texto"),  # whitespace
        ("Titulo", "Cuerpo del texto​"),  # a hidden character
        ("Titulo", "Cuerpo del tеxto"),  # a Cyrillic look-alike
        ("Titulo", "Cuerpo del texto https://sgc.gov.co/aviso"),
        ("Título", "Cuerpo del texto"),
        ("", "Titulo\nCuerpo del texto"),  # title moved into the text
    ],
)
def test_any_difference_changes_the_identity(other):
    assert text_identity("Titulo", "Cuerpo del texto") != text_identity(*other)


def test_a_url_lookalike_has_another_identity():
    a = "Aviso oficial https://sgc.gov.co/aviso"
    b = "Aviso oficial https://sgc-gov.co/aviso"
    assert text_identity("", a) != text_identity("", b)


def test_the_identity_survives_a_lone_surrogate():
    assert len(text_identity("t", "a\ud800b")) == 64


def test_the_stage_of_a_judged_verdict_is_llm():
    assert judged(verdict(), response(), 0.5).stage is Stage.LLM
