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
    state = w.escalations.get_state(stored.item_id)
    assert (state.attempts, state.item_failures) == (1, 0)  # a 5xx is the server's, not the item's
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
#
# A failure that MAY be the item's (a warm call that never finishes, a body that is not an answer)
# counts against it only while the judge is answering other items: `prime_success` says it is.


def broken_for(stored, kind: str):
    """A judge that fails for this one item and answers everything else."""
    return lambda request: error(kind) if request.item_id == stored.item_id else None


@pytest.mark.parametrize("kind", ["timeout", "bad_response"])
def test_an_item_that_keeps_failing_is_parked_after_the_configured_number(kind):
    w = memory_llm(LlmSettings(max_failures=3))
    w.ready()
    stored = w.seed(1)
    w.judge.default = broken_for(stored, kind)
    for expected in (1, 2):
        w.prime_success()
        w.escalate()
        state = w.escalations.get_state(stored.item_id)
        assert (state.state, state.item_failures) == ("escalation_failed", expected)
        w.clock.advance(timedelta(hours=1))
        w.escalations.forget([f"itm_{9000:08d}"])  # the success is an hour old: make a new one
    w.prime_success()
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
    assert (after.settle_reason, after.settled_at) == ("parked", w.clock.now())
    calls = w.judge.calls
    w.clock.advance(timedelta(hours=2))  # inside the time-to-live: it is not picked again
    assert w.escalate().judged == 0
    assert w.judge.calls == calls


def test_the_limit_of_failures_is_a_setting():
    w = memory_llm(LlmSettings(max_failures=1))
    w.ready()
    w.prime_success()
    stored = w.seed(1)
    w.judge.default = error("timeout")
    assert w.escalate().parked == 1
    assert w.escalations.get_state(stored.item_id).state == "parked"


def test_an_item_failure_while_the_judge_answers_nobody_is_the_judges_failure():
    w = memory_llm(LlmSettings(max_failures=1))
    w.ready()
    stored = w.seed(1)
    w.judge.default = error("timeout")
    report = w.escalate()
    assert (report.parked, report.failed, report.unreachable) == (0, 1, True)
    state = w.escalations.get_state(stored.item_id)
    assert (state.state, state.item_failures, state.attempts) == ("escalation_failed", 0, 1)
    assert w.verdict(stored).needs_llm is True


def test_an_item_failure_after_a_success_earlier_in_the_same_pass_counts_against_the_item():
    w = memory_llm(LlmSettings(max_failures=1))
    w.ready()
    good = w.seed(1, priority=0.9)
    bad = w.seed(2, priority=0.5)
    w.judge.default = broken_for(bad, "timeout")
    report = w.escalate()
    assert (report.judged, report.parked) == (1, 1)
    assert w.verdict(good).stage is Stage.LLM
    assert w.escalations.get_state(bad.item_id).state == "parked"


def test_a_success_older_than_an_hour_does_not_vouch_for_the_judge():
    w = memory_llm(LlmSettings(max_failures=1))
    w.ready()
    w.prime_success()
    w.clock.advance(timedelta(hours=1, seconds=1))
    stored = w.seed(1, classified_at=w.clock.now())
    w.judge.default = error("timeout")
    assert w.escalate().parked == 0
    assert w.escalations.get_state(stored.item_id).state == "escalation_failed"


def test_a_success_exactly_an_hour_old_still_vouches_for_the_judge():
    w = memory_llm(LlmSettings(max_failures=1))
    w.ready()
    w.prime_success()
    w.clock.advance(timedelta(hours=1))
    w.seed(1, classified_at=w.clock.now())
    w.judge.default = error("timeout")
    assert w.escalate().parked == 1


def test_a_failure_in_the_judge_kind_never_parks_however_often_it_repeats():
    for kind in ("server_error", "load_timeout", "not_ollama", "unreachable", "model_not_found"):
        w = memory_llm(LlmSettings(max_failures=1))
        w.ready()
        stored = w.seed(1)
        w.prime_success()
        w.judge.default = error(kind)
        for _ in range(4):
            w.escalate()
            w.clock.advance(timedelta(hours=2))
            w.prime_success(9000 + w.judge.calls)
        state = w.escalations.get_state(stored.item_id)
        assert (state.state, state.item_failures) == ("escalation_failed", 0), kind
        assert w.verdict(stored).needs_llm is True


def test_a_judge_level_failure_ends_the_batch_at_once():
    w = memory_llm()
    w.ready()
    items = [w.seed(n, priority=1 - n / 10) for n in range(4)]
    w.judge.default = error("server_error")
    report = w.escalate()
    assert (report.failed, w.judge.calls) == (1, 1)
    assert w.escalations.get_state(items[0].item_id) is not None
    assert w.escalations.get_state(items[1].item_id) is None


def test_two_item_failures_in_a_row_stop_the_batch_so_a_sick_server_is_not_hammered():
    w = memory_llm()
    w.ready()
    w.prime_success()
    items = [w.seed(n, priority=1 - n / 10) for n in range(4)]
    w.judge.default = error("timeout")
    report = w.escalate()
    assert (report.failed, w.judge.calls) == (2, 2)
    assert w.escalations.get_state(items[2].item_id) is None
    assert w.escalations.get_state(items[3].item_id) is None


def test_a_success_between_failures_does_not_stop_the_batch():
    w = memory_llm()
    w.ready()
    for n in range(4):
        w.seed(n, priority=1 - n / 10)
    w.judge.script.extend([response(), error("timeout"), response(), error("timeout")])
    report = w.escalate()
    assert (report.judged, report.failed, w.judge.calls) == (2, 2, 4)


def test_an_unexpected_error_of_the_judge_is_one_failure_of_that_item_not_a_crash():
    w = memory_llm()
    w.ready()
    fine = w.seed(1, priority=0.9)
    broken = w.seed(2, priority=0.5)
    w.judge.script.extend([response(), RuntimeError("bug in an adapter: ITEM-TEXT")])
    report = w.escalate()
    assert (report.failed, report.judged) == (1, 1)
    assert w.escalations.get_state(broken.item_id).last_error == "llm_unexpected"
    assert w.verdict(fine).stage is Stage.LLM


def test_an_unexpected_error_before_any_success_is_the_judges_and_ends_the_batch():
    w = memory_llm()
    w.ready()
    first = w.seed(1, priority=0.9)
    second = w.seed(2, priority=0.5)
    w.judge.script.append(RuntimeError("bug in an adapter"))
    report = w.escalate()
    assert (report.failed, report.judged, report.unreachable) == (1, 0, True)
    assert w.escalations.get_state(first.item_id).item_failures == 0
    assert w.escalations.get_state(second.item_id) is None


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
    w.prime_success()
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
    assert (after.settle_reason, after.settled_at) == ("invalid", w.clock.now())
    j = w.escalations.get_judgment(stored.item_id)
    assert (j.outcome, j.label, j.confidence) == ("invalid", None, None)
    assert j.raw_output == "second garbage"  # the last answer, kept for the audit
    assert j.prior_label == "unintentional_rumor"


def test_an_invalid_item_is_not_judged_again_on_the_next_run():
    w = memory_llm()
    w.ready()
    w.prime_success()
    w.seed(1)
    w.judge.script.extend([invalid(), invalid()])
    w.escalate()
    w.clock.advance(timedelta(days=1))
    assert w.escalate().judged == 0
    assert w.judge.calls == 2


def test_an_invalid_item_that_was_unclassified_stays_unclassified():
    w = memory_llm()
    w.ready()
    w.prime_success()
    stored = w.seed(1, label=Label.UNCLASSIFIED, confidence=0.1, rationale="inconclusive")
    w.judge.script.extend([invalid(), invalid()])
    w.escalate()
    assert w.verdict(stored).label is Label.UNCLASSIFIED


def test_two_invalid_answers_when_the_judge_answers_nobody_are_a_sick_judge_not_a_bad_item():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.script.extend([invalid(), invalid()])
    report = w.escalate()
    assert (report.invalid, report.failed, report.unreachable) == (0, 1, True)
    after = w.verdict(stored)
    assert after.needs_llm is True  # still waiting: nothing was settled
    assert w.escalations.get_judgment(stored.item_id) is None
    state = w.escalations.get_state(stored.item_id)
    assert (state.state, state.item_failures, state.last_error) == (
        "escalation_failed",
        0,
        "llm_invalid_output",
    )
    assert w.service.status().used_this_hour == 1  # the model did run twice: the call stays counted


def test_a_valid_answer_in_the_same_pass_makes_a_later_double_invalid_the_items_own():
    w = memory_llm()
    w.ready()
    good = w.seed(1, priority=0.9)
    bad = w.seed(2, priority=0.5)
    w.judge.script.extend([response(), invalid(), invalid()])
    report = w.escalate()
    assert (report.judged, report.invalid) == (1, 1)
    assert w.verdict(good).stage is Stage.LLM
    assert w.escalations.get_judgment(bad.item_id).outcome == "invalid"


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
    w.prime_success()
    stored = w.seed(1)
    w.judge.script.append(error("timeout"))
    w.escalate()
    w.clock.advance(timedelta(minutes=30))
    w.escalate()  # default response: judged, so the one failure no longer counts
    assert w.escalations.get_state(stored.item_id) is None
    assert w.verdict(stored).stage is Stage.LLM
