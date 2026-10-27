"""Vocabulary added by the remediation round: failure levels, new settings, request retry hints,
settlement fields of a verdict."""

from datetime import UTC, datetime

import pytest

from social_tracker.classification.domain import Classification, Label, Stage
from social_tracker.llm.domain import (
    INVALID_PROBLEMS,
    JUDGE_SEED,
    JudgeError,
    JudgeErrorKind,
    JudgeRequest,
    LlmSettings,
)
from social_tracker.shared.errors import ValidationError
from tests.llm.support import NOW, evidence, request

# ---- failure levels (C1) ----


@pytest.mark.parametrize(
    ("kind", "status", "judge_level", "item_specific"),
    [
        (JudgeErrorKind.UNREACHABLE, None, True, False),
        (JudgeErrorKind.MODEL_NOT_FOUND, None, True, False),
        (JudgeErrorKind.SERVER_ERROR, 500, True, False),
        (JudgeErrorKind.SERVER_ERROR, None, True, False),
        (JudgeErrorKind.LOAD_TIMEOUT, None, True, False),
        (JudgeErrorKind.NOT_OLLAMA, None, True, False),
        (JudgeErrorKind.BAD_RESPONSE, 400, True, False),
        (JudgeErrorKind.BAD_RESPONSE, 429, True, False),
        (JudgeErrorKind.BAD_RESPONSE, None, False, True),  # a body that is not an answer
        (JudgeErrorKind.TIMEOUT, None, False, True),  # a warm call that never finished
        (JudgeErrorKind.INVALID_OUTPUT, None, False, False),  # settled on its own, not a failure
    ],
)
def test_failure_levels(kind, status, judge_level, item_specific):
    error = JudgeError(kind, status=status)
    assert error.judge_level is judge_level
    assert error.item_specific is item_specific
    assert error.refundable is judge_level


def test_a_judge_error_keeps_its_status_and_problem_out_of_the_message():
    error = JudgeError(JudgeErrorKind.INVALID_OUTPUT, "does not match", problem="schema")
    assert (error.problem, error.status, error.code) == ("schema", None, "llm_invalid_output")


def test_the_load_timeout_has_its_own_code():
    assert JudgeError(JudgeErrorKind.LOAD_TIMEOUT).code == "llm_load_timeout"


# ---- settings ----


def test_the_new_settings_have_the_documented_defaults():
    s = LlmSettings()
    assert (s.load_deadline_s, s.num_ctx, s.pass_budget_s) == (180.0, 8192, 120.0)
    assert (s.parked_ttl_s, s.source_share) == (86_400, 0.25)
    assert s.source_cap == 500  # 25 % of 2000


@pytest.mark.parametrize(
    "bad",
    [
        {"load_deadline_s": 0},
        {"load_deadline_s": -1},
        {"load_deadline_s": 1801},
        {"load_deadline_s": float("nan")},
        {"load_deadline_s": float("inf")},
        {"load_deadline_s": True},
        {"num_ctx": 2047},
        {"num_ctx": 2303, "max_output_tokens": 256},  # one token short of the minimum room
        {"num_ctx": 262_145},
        {"num_ctx": 8192.0},
        {"num_ctx": 600, "max_output_tokens": 512},  # no room left for the prompt
        {"pass_budget_s": 0},
        {"pass_budget_s": 86_401},
        {"pass_budget_s": float("nan")},
        {"parked_ttl_s": 59},
        {"parked_ttl_s": 2_592_001},
        {"parked_ttl_s": True},
        {"source_share": 0},
        {"source_share": 1.01},
        {"source_share": -0.1},
        {"source_share": float("nan")},
    ],
)
def test_bad_new_settings_are_refused(bad):
    with pytest.raises(ValidationError):
        LlmSettings(**bad)


@pytest.mark.parametrize(
    "good",
    [
        {"load_deadline_s": 0.5},
        {"load_deadline_s": 1800},
        {"num_ctx": 2304, "max_output_tokens": 256},  # exactly the minimum room
        {"num_ctx": 262_144},
        {"pass_budget_s": 86_400},
        {"parked_ttl_s": 60},
        {"parked_ttl_s": 2_592_000},
        {"source_share": 1.0},
        {"source_share": 0.0001},
    ],
)
def test_boundary_values_of_the_new_settings_are_accepted(good):
    LlmSettings(**good)


def test_a_tiny_share_still_allows_one_item_per_source():
    assert LlmSettings(queue_max=3, source_share=0.1).source_cap == 1


def test_the_cold_deadline_is_never_shorter_than_the_warm_one():
    assert LlmSettings(call_deadline_s=60, load_deadline_s=10).cold_deadline_s == 60
    assert LlmSettings(call_deadline_s=60, load_deadline_s=180).cold_deadline_s == 180


def test_the_longest_judgment_covers_a_cold_call_and_the_warm_retry():
    s = LlmSettings(call_deadline_s=60, load_deadline_s=180)
    assert s.longest_judgment_s == 240


# ---- retry hints on a request ----


def test_a_request_carries_the_retry_hint():
    first = request()
    assert (first.attempt, first.previous_problem) == (0, "")
    retry = JudgeRequest(first.item_id, first.text, False, evidence(), 1, "schema")
    assert (retry.attempt, retry.previous_problem) == (1, "schema")


@pytest.mark.parametrize(
    "bad",
    [
        {"attempt": -1},
        {"attempt": 4},
        {"attempt": True},
        {"attempt": 1, "previous_problem": "the model said hello"},
        {"attempt": 0, "previous_problem": "schema"},  # a first attempt has no problem to name
    ],
)
def test_a_bad_retry_hint_is_refused(bad):
    with pytest.raises(ValidationError):
        request(**bad)


def test_every_problem_class_is_accepted_on_a_retry():
    for problem in INVALID_PROBLEMS:
        assert request(attempt=1, previous_problem=problem).previous_problem == problem


def test_the_first_seed_is_zero():
    assert JUDGE_SEED == 0


# ---- settlement fields of a verdict ----


def _verdict(**changes):
    fields = {
        "item_id": "itm_00000001",
        "label": Label.NO_RISK,
        "confidence": 0.8,
        "stage": Stage.LLM,
        "rationale": "fine",
        "classified_at": NOW,
        "classifier_version": "cascade-2/topics-0",
    }
    fields.update(changes)
    return Classification(**fields)


def test_a_verdict_is_unsettled_by_default():
    v = _verdict()
    assert (v.settled_at, v.settle_reason) == (None, None)


@pytest.mark.parametrize("reason", ["llm", "duplicate", "invalid", "parked"])
def test_every_settle_reason_is_accepted(reason):
    v = _verdict(settled_at=NOW, settle_reason=reason)
    assert (v.settled_at, v.settle_reason) == (NOW, reason)


@pytest.mark.parametrize(
    "bad",
    [
        {"settled_at": NOW},  # a time without a reason
        {"settle_reason": "llm"},  # a reason without a time
        {"settled_at": NOW, "settle_reason": "other"},
        {"settled_at": NOW, "settle_reason": ""},
        {"settled_at": datetime(2026, 3, 1, 12, 0), "settle_reason": "llm"},  # naive
        {"settled_at": NOW, "settle_reason": "llm", "needs_llm": True},  # still in the queue
    ],
)
def test_a_bad_settlement_is_refused(bad):
    with pytest.raises(ValidationError):
        _verdict(**bad)


def test_a_settlement_time_is_normalised_to_utc():
    from datetime import timedelta, timezone

    local = datetime(2026, 3, 1, 7, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert _verdict(settled_at=local, settle_reason="llm").settled_at == NOW.astimezone(UTC)
