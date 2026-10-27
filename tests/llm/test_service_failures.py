"""When the judge cannot answer or answers badly: backoff, parking, invalid output (RC-9)."""

from datetime import timedelta

import pytest

from social_tracker.classification.domain import Label, Stage
from social_tracker.llm.domain import JudgeError, JudgeErrorKind, LlmSettings
from tests.llm.harness import error, memory_llm
from tests.llm.support import NOW, response


def invalid(raw: str = "not json") -> JudgeError:
    return JudgeError(JudgeErrorKind.INVALID_OUTPUT, "does not match the schema", raw_output=raw)


# ---- the judge is down ----


def test_a_judge_that_is_down_marks_the_item_for_a_later_retry_and_keeps_the_cpu_result():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.default = error("unreachable")
    report = w.escalate()
    assert (report.judged, report.failed, report.unreachable) == (0, 1, True)
    after = w.verdict(stored)
    assert after == stored  # label, confidence, stage, and still waiting
    state = w.escalations.get_state(stored.item_id)
    assert (state.state, state.attempts, state.item_failures) == ("escalation_failed", 1, 0)
    assert state.next_attempt_at == NOW + timedelta(seconds=60)
    assert state.last_error == "llm_unreachable"


def test_a_judge_that_is_down_stops_the_batch_after_the_first_item():
    w = memory_llm()
    w.ready()
    first = w.seed(1, priority=0.9)
    second = w.seed(2, priority=0.5)
    w.judge.default = error("unreachable")
    w.escalate()
    assert w.judge.calls == 1
    assert w.escalations.get_state(first.item_id) is not None
    assert w.escalations.get_state(second.item_id) is None


def test_the_item_is_not_retried_before_its_moment_and_is_at_that_moment():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.script.append(error("unreachable"))
    w.escalate()
    w.clock.advance(timedelta(seconds=59))
    assert w.escalate().judged == 0
    assert w.judge.calls == 1
    w.clock.advance(timedelta(seconds=1))  # exactly next_attempt_at
    assert w.escalate().judged == 1
    assert w.verdict(stored).stage is Stage.LLM


def test_a_retry_that_works_clears_the_bookkeeping():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.script.append(error("server_error"))
    w.escalate()
    assert w.escalations.get_state(stored.item_id).item_failures == 1
    w.clock.advance(timedelta(minutes=5))
    assert w.escalate().judged == 1
    assert w.escalations.get_state(stored.item_id) is None


def test_five_xx_then_ok_across_two_passes_ends_judged():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.script.extend([error("server_error"), response(label=Label.RECYCLED_CONTENT)])
    first = w.escalate()
    w.clock.advance(timedelta(hours=1))
    second = w.escalate()
    assert (first.failed, second.judged) == (1, 1)
    assert w.verdict(stored).label is Label.RECYCLED_CONTENT


@pytest.mark.parametrize(
    ("failures", "seconds"),
    [(1, 60), (2, 120), (3, 240), (4, 480), (5, 960), (6, 1920), (7, 3600), (8, 3600), (9, 3600)],
)
def test_the_backoff_doubles_with_every_failure_and_is_capped_at_an_hour(failures, seconds):
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.default = error("unreachable")
    for _ in range(failures):
        w.escalate()
        wait = w.escalations.get_state(stored.item_id).next_attempt_at
        w.clock.advance(wait - w.clock.now())
    state = w.escalations.get_state(stored.item_id)
    assert state.attempts == failures
    # the wait that was scheduled after the last failure
    scheduled = state.next_attempt_at - state.updated_at
    assert scheduled == timedelta(seconds=seconds)


def test_failures_of_a_judge_that_is_down_never_park_the_item():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.default = error("unreachable")
    for _ in range(12):
        w.escalate()
        w.clock.advance(timedelta(hours=2))
    state = w.escalations.get_state(stored.item_id)
    assert (state.state, state.item_failures) == ("escalation_failed", 0)
    assert w.verdict(stored).needs_llm is True
    w.judge.default = None
    assert w.escalate().judged == 1  # and it is judged the day the judge is back


def test_a_missing_model_is_a_problem_of_the_judge_not_of_the_item():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.default = error("model_not_found")
    for _ in range(5):
        w.escalate()
        w.clock.advance(timedelta(hours=2))
    assert w.escalations.get_state(stored.item_id).item_failures == 0
    assert w.verdict(stored).needs_llm is True


# ---- an item that makes the judge fail ----


@pytest.mark.parametrize("kind", ["server_error", "timeout", "bad_response"])
def test_an_item_that_keeps_failing_is_parked_after_the_configured_number(kind):
    w = memory_llm(LlmSettings(max_failures=3))
    w.ready()
    stored = w.seed(1)
    w.judge.default = error(kind)
    for expected in (1, 2):
        w.escalate()
        state = w.escalations.get_state(stored.item_id)
        assert (state.state, state.item_failures) == ("escalation_failed", expected)
        w.clock.advance(timedelta(hours=1))
    report = w.escalate()
    assert report.parked == 1
    state = w.escalations.get_state(stored.item_id)
    assert (state.state, state.item_failures) == ("parked", 3)
    after = w.verdict(stored)
    assert (after.label, after.confidence, after.stage) == (
        stored.label,
        stored.confidence,
        stored.stage,
    )
    assert after.needs_llm is False
    assert after.low_confidence is True
    w.clock.advance(timedelta(days=1))
    w.judge.default = None
    assert w.escalate().judged == 0  # terminal: it is not picked again
    assert w.judge.calls == 3


def test_the_limit_of_failures_is_a_setting():
    w = memory_llm(LlmSettings(max_failures=1))
    w.ready()
    stored = w.seed(1)
    w.judge.default = error("timeout")
    assert w.escalate().parked == 1
    assert w.escalations.get_state(stored.item_id).state == "parked"


def test_two_failures_in_a_row_stop_the_batch_so_a_sick_server_is_not_hammered():
    w = memory_llm()
    w.ready()
    items = [w.seed(n, priority=1 - n / 10) for n in range(4)]
    w.judge.default = error("server_error")
    report = w.escalate()
    assert (report.failed, w.judge.calls) == (2, 2)
    assert w.escalations.get_state(items[2].item_id) is None
    assert w.escalations.get_state(items[3].item_id) is None


def test_a_success_between_failures_does_not_stop_the_batch():
    w = memory_llm()
    w.ready()
    for n in range(4):
        w.seed(n, priority=1 - n / 10)
    w.judge.script.extend([error("server_error"), response(), error("server_error"), response()])
    report = w.escalate()
    assert (report.judged, report.failed, w.judge.calls) == (2, 2, 4)


def test_an_unexpected_error_of_the_judge_is_one_failure_of_that_item_not_a_crash():
    w = memory_llm()
    w.ready()
    broken = w.seed(1, priority=0.9)
    fine = w.seed(2, priority=0.5)
    w.judge.script.append(RuntimeError("bug in an adapter: ITEM-TEXT"))
    report = w.escalate()
    assert (report.failed, report.judged) == (1, 1)
    assert w.escalations.get_state(broken.item_id).last_error == "llm_unexpected"
    assert w.verdict(fine).stage is Stage.LLM


# ---- invalid output (RC-9) ----


def test_one_invalid_answer_is_retried_once_and_the_second_one_is_used():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.script.extend([invalid(), response(label=Label.RECYCLED_CONTENT, confidence=0.66)])
    report = w.escalate()
    assert (report.judged, report.invalid) == (1, 0)
    assert w.judge.calls == 2
    after = w.verdict(stored)
    assert (after.label, after.confidence, after.stage) == (Label.RECYCLED_CONTENT, 0.66, Stage.LLM)


def test_two_invalid_answers_leave_the_rules_verdict_and_contribute_nothing():
    w = memory_llm()
    w.ready()
    stored = w.seed(1, confidence=0.55)
    w.judge.script.extend([invalid("first garbage"), invalid("second garbage")])
    report = w.escalate()
    assert (report.invalid, report.judged, w.judge.calls) == (1, 0, 2)
    after = w.verdict(stored)
    assert (after.label, after.confidence, after.stage, after.rationale) == (
        stored.label,
        stored.confidence,
        stored.stage,
        stored.rationale,
    )
    assert after.needs_llm is False
    assert after.low_confidence is True
    j = w.escalations.get_judgment(stored.item_id)
    assert (j.outcome, j.label, j.confidence) == ("invalid", None, None)
    assert j.raw_output == "second garbage"  # the last answer, kept for the audit
    assert j.prior_label == "unintentional_rumor"


def test_an_invalid_item_is_not_judged_again_on_the_next_run():
    w = memory_llm()
    w.ready()
    w.seed(1)
    w.judge.script.extend([invalid(), invalid()])
    w.escalate()
    w.clock.advance(timedelta(days=1))
    assert w.escalate().judged == 0
    assert w.judge.calls == 2


def test_an_invalid_item_that_was_unclassified_stays_unclassified():
    w = memory_llm()
    w.ready()
    stored = w.seed(1, label=Label.UNCLASSIFIED, confidence=0.1, rationale="inconclusive")
    w.judge.script.extend([invalid(), invalid()])
    w.escalate()
    assert w.verdict(stored).label is Label.UNCLASSIFIED


def test_invalid_after_a_failed_first_try_is_a_failure_not_an_invalid_verdict():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.script.extend([invalid(), error("unreachable")])
    report = w.escalate()
    assert (report.invalid, report.failed) == (0, 1)
    assert w.verdict(stored).needs_llm is True


def test_a_valid_answer_ends_the_failure_streak_of_a_previous_run():
    w = memory_llm(LlmSettings(max_failures=2))
    w.ready()
    stored = w.seed(1)
    w.judge.script.append(error("timeout"))
    w.escalate()
    w.clock.advance(timedelta(hours=1))
    w.escalate()  # default response: judged, so the one failure no longer counts
    assert w.escalations.get_state(stored.item_id) is None
    assert w.verdict(stored).stage is Stage.LLM
