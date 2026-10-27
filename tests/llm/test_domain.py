"""The pure vocabulary of the escalation: responses, sanitising, backoff, how a judgment lands."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.classification.domain import Label, Stage
from social_tracker.llm.domain import (
    FALLBACK_RATIONALE,
    JUDGE_LABELS,
    JudgeError,
    JudgeErrorKind,
    JudgeEvidence,
    JudgeRequest,
    JudgeResponse,
    backoff_delay,
    evidence_of,
    hour_start,
    judged,
    left_unjudged,
    llm_low_confidence,
    prompt_text,
    released_duplicate,
    reused_by_duplicate,
    sanitize_text,
)
from social_tracker.shared.errors import ValidationError
from tests.llm.support import evidence, request, response, verdict


def test_the_labels_a_model_may_give_exclude_unclassified_and_out_of_scope():
    assert set(JUDGE_LABELS) == {
        Label.UNINTENTIONAL_RUMOR,
        Label.DELIBERATE_DISINFORMATION,
        Label.RECYCLED_CONTENT,
        Label.NO_RISK,
    }


# ---- sanitising ----


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("plain text", "plain text"),
        ("a\x00b\x1b[31mred\x07", "a b [31mred"),  # controls become blanks, runs collapse
        ("zero​width‮bidi﻿", "zerowidthbidi"),  # format characters vanish
        ("line\nbreak\r\n\ttabs", "line break tabs"),
        ("  padded  ", "padded"),
        ("sep arator ", "separator"),
        ("lone \ud800 surrogate", "lone surrogate"),
        ("", ""),
        ("\x00\x01​", ""),
    ],
)
def test_sanitising_strips_controls_and_format_characters(raw: str, clean: str):
    assert sanitize_text(raw, 300) == clean


@pytest.mark.parametrize(("length", "kept"), [(299, 299), (300, 300), (301, 300), (5000, 300)])
def test_sanitising_caps_the_length_at_the_boundary(length: int, kept: int):
    assert len(sanitize_text("x" * length, 300)) == kept


def test_the_cap_never_leaves_a_trailing_blank():
    assert sanitize_text("ab " + "c" * 10, 3) == "ab"


# ---- the response ----


def test_a_valid_response_is_built():
    r = response()
    assert (r.label, r.confidence) == (Label.NO_RISK, 0.9)


@pytest.mark.parametrize("label", [Label.UNCLASSIFIED, Label.OUT_OF_SCOPE, "nonsense", None, 3])
def test_a_label_outside_the_four_is_refused(label: object):
    with pytest.raises(ValidationError):
        response(label=label)


@pytest.mark.parametrize(
    "confidence", [-0.01, 1.01, math.nan, math.inf, -math.inf, True, "0.5", None]
)
def test_a_confidence_that_is_not_a_number_in_range_is_refused(confidence: object):
    with pytest.raises(ValidationError):
        response(confidence=confidence)


@pytest.mark.parametrize("confidence", [0, 0.0, 0.49, 0.5, 1, 1.0])
def test_the_ends_of_the_confidence_range_are_accepted(confidence: float):
    assert response(confidence=confidence).confidence == confidence


@pytest.mark.parametrize("rationale", ["", "x" * 301, "has\x00nul", "bidi‮", 5, None])
def test_a_rationale_that_is_empty_long_or_unsafe_is_refused(rationale: object):
    with pytest.raises(ValidationError):
        response(rationale=rationale)


def test_a_rationale_of_exactly_300_characters_is_accepted():
    assert len(response(rationale="r" * 300).rationale) == 300


@pytest.mark.parametrize(
    "evidence_list",
    [("a",) * 6, ("x" * 201,), ("bad\x00",), ("",), "text", [1]],
)
def test_evidence_beyond_five_short_clean_strings_is_refused(evidence_list: object):
    with pytest.raises(ValidationError):
        response(evidence=evidence_list)


def test_five_evidence_strings_of_200_characters_are_accepted():
    assert len(response(evidence=("e" * 200,) * 5).evidence) == 5


def test_a_raw_output_is_kept_truncated_for_the_audit():
    assert len(response(raw_output="z" * 5000).raw_output) == 2048


# ---- errors ----


@pytest.mark.parametrize(
    ("kind", "item_specific"),
    [
        (JudgeErrorKind.UNREACHABLE, False),
        (JudgeErrorKind.MODEL_NOT_FOUND, False),
        (JudgeErrorKind.TIMEOUT, True),
        (JudgeErrorKind.SERVER_ERROR, True),
        (JudgeErrorKind.BAD_RESPONSE, True),
        (JudgeErrorKind.INVALID_OUTPUT, False),
    ],
)
def test_only_some_failures_count_against_the_item(kind: JudgeErrorKind, item_specific: bool):
    error = JudgeError(kind)
    assert error.item_specific is item_specific
    assert error.code == f"llm_{kind.value}"


# ---- backoff and the hour ----


@pytest.mark.parametrize(
    ("failures", "seconds"),
    [(1, 60), (2, 120), (3, 240), (4, 480), (6, 1920), (7, 3600), (8, 3600), (1000, 3600)],
)
def test_the_backoff_doubles_and_is_capped(failures: int, seconds: int):
    assert backoff_delay(failures, 60, 3600) == timedelta(seconds=seconds)


def test_the_backoff_needs_at_least_one_failure():
    with pytest.raises(ValidationError):
        backoff_delay(0, 60, 3600)


@pytest.mark.parametrize(
    ("moment", "start"),
    [
        (datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC), datetime(2026, 3, 1, 12, 0, tzinfo=UTC)),
        (
            datetime(2026, 3, 1, 12, 59, 59, 999999, tzinfo=UTC),
            datetime(2026, 3, 1, 12, tzinfo=UTC),
        ),
        (datetime(2026, 3, 1, 13, 0, 0, tzinfo=UTC), datetime(2026, 3, 1, 13, tzinfo=UTC)),
        (datetime(2026, 12, 31, 23, 59, tzinfo=UTC), datetime(2026, 12, 31, 23, tzinfo=UTC)),
    ],
)
def test_the_budget_hour_is_the_utc_hour(moment: datetime, start: datetime):
    assert hour_start(moment) == start


def test_a_naive_moment_has_no_budget_hour():
    with pytest.raises(ValidationError):
        hour_start(datetime(2026, 3, 1, 12, 0))  # noqa: DTZ001


# ---- the low-confidence rule ----


@pytest.mark.parametrize(
    ("confidence", "alert_min", "low"),
    [
        (0.49, 0.5, True),
        (0.5, 0.5, False),
        (0.51, 0.5, False),
        (0.0, 0.5, True),
        (0.55, 0.6, True),  # a stricter alert threshold applies too (RC-11)
        (0.6, 0.6, False),
        (0.49, 0.1, True),  # the 0.5 floor holds even for a lax threshold
        (0.5, 0.1, False),
    ],
)
def test_an_llm_label_under_the_floor_or_the_alert_threshold_is_low_confidence(
    confidence: float, alert_min: float, low: bool
):
    assert llm_low_confidence(confidence, alert_min) is low


# ---- how a judgment lands on the stored verdict ----


def test_a_judgment_replaces_label_confidence_stage_and_rationale_and_leaves_the_rest():
    prior = verdict(1)
    result = judged(prior, response(label=Label.DELIBERATE_DISINFORMATION, confidence=0.83), 0.5)
    assert (result.label, result.confidence, result.stage) == (
        Label.DELIBERATE_DISINFORMATION,
        0.83,
        Stage.LLM,
    )
    assert result.rationale == "the text reports a past event without a prediction"
    assert not result.needs_llm
    assert result.priority == 0.0
    assert not result.low_confidence
    for kept in (
        "item_id",
        "signals",
        "topics",
        "catalog_status",
        "truncated",
        "classified_at",
        "classifier_version",
        "source_credibility_used",
        "near_of",
        "recycled_of",
    ):
        assert getattr(result, kept) == getattr(prior, kept)


@pytest.mark.parametrize(("confidence", "low"), [(0.49, True), (0.5, False), (0.8, False)])
def test_a_judgment_under_the_floor_is_applied_but_flagged(confidence: float, low: bool):
    result = judged(verdict(), response(confidence=confidence), 0.5)
    assert result.label is Label.NO_RISK  # applied
    assert result.low_confidence is low


def test_a_prior_that_was_low_confidence_is_not_flagged_after_a_confident_judgment():
    result = judged(verdict(low_confidence=True), response(confidence=0.9), 0.5)
    assert result.low_confidence is False


def test_an_invalid_judgment_keeps_the_rules_verdict_and_clears_the_queue_flag():
    prior = verdict(label=Label.UNINTENTIONAL_RUMOR, confidence=0.55)
    result = left_unjudged(prior)
    assert (result.label, result.confidence, result.stage, result.rationale) == (
        prior.label,
        prior.confidence,
        prior.stage,
        prior.rationale,
    )
    assert not result.needs_llm
    assert result.low_confidence
    assert result.priority == 0.0


def _waiting_duplicate(n: int, original_id: str, **changes):
    return verdict(
        n,
        needs_llm=False,
        duplicate_of=original_id,
        awaiting_original=True,
        stage=Stage.DEDUP,
        priority=0.0,
        **changes,
    )


def test_a_duplicate_takes_the_judged_label_with_its_own_credibility_and_a_note():
    original = judged(verdict(1), response(label=Label.RECYCLED_CONTENT, confidence=0.7), 0.5)
    duplicate = _waiting_duplicate(2, original.item_id, source_credibility_used=5)
    result = reused_by_duplicate(duplicate, original, 0.5)
    assert (result.label, result.confidence) == (Label.RECYCLED_CONTENT, 0.7)
    assert result.stage is Stage.DEDUP  # it was not judged itself
    assert not result.awaiting_original
    assert result.duplicate_of == original.item_id
    assert result.source_credibility_used == 5
    assert original.item_id in result.rationale
    assert "reused" in result.rationale
    assert len(result.rationale) <= 300


def test_a_duplicate_of_a_low_confidence_judgment_is_flagged_low_too():
    original = judged(verdict(1), response(confidence=0.45), 0.5)
    duplicate = _waiting_duplicate(2, original.item_id)
    assert reused_by_duplicate(duplicate, original, 0.5).low_confidence


def test_a_released_duplicate_keeps_its_provisional_verdict_and_stops_waiting():
    duplicate = _waiting_duplicate(2, verdict(1).item_id)
    result = released_duplicate(duplicate)
    assert not result.awaiting_original
    assert (result.label, result.confidence, result.duplicate_of) == (
        duplicate.label,
        duplicate.confidence,
        duplicate.duplicate_of,
    )


# ---- the request ----


def test_the_evidence_of_a_verdict_carries_what_the_earlier_stages_found():
    prior = verdict(
        catalog_status="mismatch", source_credibility_used=4, near_of=verdict(9).item_id
    )
    ev = evidence_of(prior)
    assert (ev.prior_label, ev.prior_confidence, ev.prior_stage) == (
        Label.UNINTENTIONAL_RUMOR,
        0.55,
        Stage.RULES,
    )
    assert ev.catalog_status == "mismatch"
    assert ev.source_credibility == 4
    assert ev.topics == ("earthquake",)
    assert [s.name for s in ev.signals] == ["prediction_claim"]
    assert ev.near_copy is True
    assert ev.recycled_before is False


def test_the_evidence_of_a_recycled_verdict_says_so():
    ev = evidence_of(verdict(label=Label.RECYCLED_CONTENT, recycled_of=verdict(9).item_id))
    assert ev.recycled_before is True
    assert ev.near_copy is False


def test_the_request_text_is_bounded_and_flagged_deterministically():
    assert prompt_text("Title", "Body", 100) == ("Title\nBody", False)
    text = "a" * 150 + "b" * 150
    first = prompt_text("", text, 100)
    assert first == prompt_text("", text, 100)
    assert first[1] is True
    assert first[0] == "a" * 50 + "\n" + "b" * 50  # both ends are read


def test_the_request_text_boundary_is_exact():
    assert prompt_text("", "x" * 100, 100)[1] is False
    assert prompt_text("", "x" * 101, 100)[1] is True


def test_hidden_characters_never_reach_the_prompt_text():
    text, _ = prompt_text("t‮", "a\x00b​c\x1b[0m", 100)
    assert text == "t\na bc [0m"


def test_empty_text_is_allowed_in_a_request():
    assert prompt_text("", "", 100) == ("", False)


def test_the_request_checks_its_fields():
    with pytest.raises(ValidationError):
        request(item_id="X")
    with pytest.raises(ValidationError):
        request(text=None)
    with pytest.raises(ValidationError):
        request(evidence="no")
    with pytest.raises(ValidationError):
        evidence(source_credibility=6)
    with pytest.raises(ValidationError):
        evidence(catalog_status="Bad Status")
    with pytest.raises(ValidationError):
        evidence(prior_confidence=2)
    assert isinstance(request(), JudgeRequest)
    assert isinstance(evidence(), JudgeEvidence)
    assert isinstance(response(), JudgeResponse)


def test_the_fallback_rationale_is_a_valid_rationale():
    assert 1 <= len(FALLBACK_RATIONALE) <= 300
    assert sanitize_text(FALLBACK_RATIONALE, 300) == FALLBACK_RATIONALE
