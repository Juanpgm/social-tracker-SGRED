"""The stored shapes of the cascade: a classification and an item's near-duplicate signature."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

import pytest

from social_tracker.classification.domain import (
    MAX_RATIONALE_CHARS,
    MAX_SIGNALS,
    Classification,
    ItemSignature,
    Label,
    OutOfScopeReason,
    Signal,
    Stage,
    Topic,
    signals_from_json,
    signals_to_json,
)
from social_tracker.classification.similarity import NUM_PERM
from social_tracker.shared.errors import ValidationError

NOW = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
ITEM = "itm_" + "a" * 32
OTHER = "itm_" + "b" * 32


def make(**changes) -> Classification:
    fields = {
        "item_id": ITEM,
        "label": Label.UNINTENTIONAL_RUMOR,
        "confidence": 0.6,
        "stage": Stage.RULES,
        "rationale": "rumor candidate",
        "signals": (Signal("prediction_claim", Label.UNINTENTIONAL_RUMOR, 0.45),),
        "topics": (Topic.EARTHQUAKE,),
        "classified_at": NOW,
        "classifier_version": "cascade-1",
        **changes,
    }
    return Classification(**fields)


def test_a_valid_classification_keeps_its_fields_and_defaults():
    c = make()
    assert c.label is Label.UNINTENTIONAL_RUMOR
    assert (c.low_confidence, c.needs_llm, c.truncated) == (False, False, False)
    assert (c.duplicate_of, c.recycled_of, c.scope_reason) == (None, None, None)
    assert (c.catalog_status, c.priority, c.source_credibility_used) == ("unavailable", 0.0, 3)


def test_enums_are_coerced_from_their_values_and_times_made_utc():
    plus5 = timezone(timedelta(hours=5))
    c = make(
        label="no_risk", stage="dedup", topics=("secretariat",), classified_at=NOW.astimezone(plus5)
    )
    assert (c.label, c.stage, c.topics) == (Label.NO_RISK, Stage.DEDUP, (Topic.SECRETARIAT,))
    assert c.classified_at == NOW
    assert c.classified_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("confidence", [0.0, 1.0, 0.5])
def test_the_confidence_bounds_are_inclusive(confidence):
    assert make(confidence=confidence).confidence == confidence


@pytest.mark.parametrize(
    "changes",
    [
        {"confidence": -0.01},
        {"confidence": 1.01},
        {"confidence": True},
        {"confidence": "0.5"},
        {"confidence": float("nan")},
        {"label": "made_up"},
        {"stage": "made_up"},
        {"topics": ("weather",)},
        {"rationale": ""},
        {"rationale": "x" * (MAX_RATIONALE_CHARS + 1)},
        {"rationale": "line\x00break"},
        {"item_id": "BAD ID"},
        {"duplicate_of": "BAD ID"},
        {"duplicate_of": ITEM},
        {"recycled_of": ITEM},
        {"duplicate_of": OTHER, "recycled_of": OTHER},
        {"needs_llm": True, "duplicate_of": OTHER},
        {"label": Label.OUT_OF_SCOPE, "needs_llm": True, "scope_reason": "no_topic_match"},
        {"label": Label.OUT_OF_SCOPE},  # needs its reason
        {"scope_reason": "no_topic_match"},  # a reason without the label
        {"scope_reason": "made_up", "label": Label.OUT_OF_SCOPE},
        {"low_confidence": 1},
        {"needs_llm": "yes"},
        {"truncated": None},
        {"catalog_status": "Bad Status"},
        {"catalog_status": ""},
        {"priority": 1.1},
        {"priority": -0.1},
        {"source_credibility_used": 0},
        {"source_credibility_used": 6},
        {"source_credibility_used": 3.5},
        {"source_credibility_used": True},
        {"classified_at": datetime(2026, 3, 1)},
        {"classified_at": "2026-03-01"},
        {"classifier_version": ""},
        {"classifier_version": "v" * 65},
        {
            "signals": tuple(
                Signal("all_caps", Label.UNINTENTIONAL_RUMOR, 0.1) for _ in range(MAX_SIGNALS + 1)
            )
        },
        {"signals": ("not a signal",)},
    ],
)
def test_an_invalid_classification_is_refused(changes):
    with pytest.raises(ValidationError):
        make(**changes)


def test_the_documented_limits_are_accepted_at_the_boundary():
    assert len(make(rationale="x" * MAX_RATIONALE_CHARS).rationale) == MAX_RATIONALE_CHARS
    signals = tuple(Signal("all_caps", Label.UNINTENTIONAL_RUMOR, 0.1) for _ in range(MAX_SIGNALS))
    assert len(make(signals=signals).signals) == MAX_SIGNALS
    assert make(source_credibility_used=1).source_credibility_used == 1
    assert make(source_credibility_used=5).source_credibility_used == 5


def test_an_out_of_scope_classification_carries_its_reason():
    c = make(
        label=Label.OUT_OF_SCOPE,
        scope_reason=OutOfScopeReason.CRIME_PUBLIC_ORDER,
        stage=Stage.TOPIC,
        topics=(),
        signals=(),
    )
    assert c.scope_reason is OutOfScopeReason.CRIME_PUBLIC_ORDER


def test_a_duplicate_or_a_recycled_item_references_another_item():
    assert make(duplicate_of=OTHER).duplicate_of == OTHER
    assert make(recycled_of=OTHER, label=Label.RECYCLED_CONTENT).recycled_of == OTHER


def test_signals_round_trip_through_json_in_a_fixed_order():
    signals = (
        Signal("prediction_claim", Label.UNINTENTIONAL_RUMOR, 0.45),
        Signal("all_caps", Label.UNINTENTIONAL_RUMOR, 0.15),
    )
    text = signals_to_json(signals)
    assert text == signals_to_json(signals[::-1])  # sorted by name: order independent
    assert signals_from_json(text) == tuple(sorted(signals, key=lambda s: s.name))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{}",
        "[1]",
        '[{"name":"nonsense","label":"no_risk","weight":0.5}]',
        "not json",
        '[{"name":"all_caps"}]',
    ],
)
def test_stored_signals_that_do_not_parse_are_refused(text):
    with pytest.raises(ValidationError):
        signals_from_json(text)


def test_a_signature_record_validates_its_parts():
    sig = tuple(range(NUM_PERM))
    record = ItemSignature(ITEM, "a" * 64, NOW, sig)
    assert record.minhash == sig
    for bad in (
        {"item_id": "BAD ID"},
        {"content_hash": "A" * 64},
        {"content_hash": "a" * 63},
        {"ref_at": datetime(2026, 3, 1)},
        {"minhash": sig[:-1]},
        {"minhash": tuple([2**32] + list(sig[1:]))},
    ):
        with pytest.raises(ValidationError):
            replace(record, **bad)
