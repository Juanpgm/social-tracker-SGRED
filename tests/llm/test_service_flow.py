"""The escalation worker step: which items go to the judge, in which order, and what lands."""

from datetime import timedelta

import pytest

from social_tracker.classification.domain import Label, Stage
from social_tracker.llm.domain import PROMPT_VERSION, LlmSettings
from social_tracker.shared.errors import ValidationError
from tests.actors import WHO
from tests.llm.harness import ScriptedJudge, memory_llm
from tests.llm.support import NOW, response


def ids(judge: ScriptedJudge) -> list[str]:
    return [r.item_id for r in judge.requests]


def test_queued_items_are_judged_best_first_and_the_verdict_is_replaced():
    w = memory_llm()
    w.ready()
    low = w.seed(1, priority=0.2)
    high = w.seed(2, priority=0.9)
    mid = w.seed(3, priority=0.5)
    report = w.escalate()
    assert ids(w.judge) == [high.item_id, mid.item_id, low.item_id]
    assert (report.judged, report.failed, report.skipped) == (3, 0, 0)
    after = w.verdict(high)
    assert (after.label, after.confidence, after.stage) == (Label.NO_RISK, 0.9, Stage.LLM)
    assert after.rationale == "the text reports a past event without a prediction"
    assert not after.needs_llm
    assert after.priority == 0.0
    assert after.signals == high.signals  # the evidence of the earlier stages stays


def test_the_oldest_goes_first_among_equal_priorities():
    w = memory_llm()
    w.ready()
    newer = w.seed(1, priority=0.5, classified_at=NOW + timedelta(minutes=5))
    older = w.seed(2, priority=0.5)
    w.escalate()
    assert ids(w.judge) == [older.item_id, newer.item_id]


def test_a_judgment_is_recorded_with_its_model_prompt_version_and_the_prior_verdict():
    w = memory_llm()
    w.ready()
    stored = w.seed(1, confidence=0.55)
    w.judge.default = response(raw_output='{"label":"no_risk"}', evidence=("a", "b"))
    w.escalate()
    j = w.escalations.get_judgment(stored.item_id)
    assert (j.outcome, j.label, j.confidence) == ("judged", "no_risk", 0.9)
    assert (j.llm_model, j.prompt_version) == ("st-gemma4-e4b", PROMPT_VERSION)
    assert j.raw_output == '{"label":"no_risk"}'
    assert j.evidence == ("a", "b")
    assert (j.prior_label, j.prior_confidence, j.prior_stage) == (
        "unintentional_rumor",
        0.55,
        "rules",
    )
    assert j.judged_at == NOW
    assert j.reused_from is None


def test_the_request_carries_the_text_and_the_structured_evidence_of_the_earlier_stages():
    w = memory_llm()
    w.ready()
    w.seed(
        1,
        title="Alerta",
        text="Se viene un sismo",
        catalog_status="mismatch",
        source_credibility_used=4,
        near_of=None,
    )
    w.escalate()
    (req,) = w.judge.requests
    assert req.text == "Alerta\nSe viene un sismo"
    assert req.text_truncated is False
    assert req.evidence.catalog_status == "mismatch"
    assert req.evidence.source_credibility == 4
    assert req.evidence.prior_label is Label.UNINTENTIONAL_RUMOR
    assert [s.name for s in req.evidence.signals] == ["prediction_claim"]


@pytest.mark.parametrize(
    ("limit_chars", "text_len", "cut"), [(100, 99, False), (100, 100, False), (100, 101, True)]
)
def test_a_long_text_is_truncated_at_the_boundary_and_flagged(limit_chars, text_len, cut):
    w = memory_llm(LlmSettings(max_prompt_chars=limit_chars))
    w.ready()
    w.seed(1, text="x" * text_len)
    w.escalate()
    (req,) = w.judge.requests
    assert req.text_truncated is cut
    assert len(req.text) <= limit_chars + 1  # both ends and one newline


def test_a_huge_text_is_cut_deterministically_to_both_ends():
    text = "A" * 50_000 + "MIDDLE" + "Z" * 50_000
    first, second = memory_llm(), memory_llm()
    for w in (first, second):
        w.ready()
        w.seed(1, text=text)
        w.escalate()
    assert first.judge.requests[0].text == second.judge.requests[0].text
    sent = first.judge.requests[0].text
    assert sent.startswith("A")
    assert sent.endswith("Z")
    assert "MIDDLE" not in sent
    assert first.judge.requests[0].text_truncated is True


def test_an_item_with_an_empty_text_is_still_judged():
    w = memory_llm()
    w.ready()
    w.seed(1, text="", title="")
    assert w.escalate().judged == 1
    assert w.judge.requests[0].text == ""


def test_only_verdicts_that_wait_for_the_llm_are_sent():
    w = memory_llm()
    w.ready()
    waits = w.seed(1)
    w.seed(2, needs_llm=False, priority=0.0, confidence=0.85)
    w.seed(3, needs_llm=False, priority=0.0, label=Label.NO_RISK, confidence=0.8)
    w.escalate()
    assert ids(w.judge) == [waits.item_id]


def test_a_batch_is_bounded_by_the_limit_and_the_rest_waits():
    w = memory_llm()
    w.ready()
    for n in range(5):
        w.seed(n, priority=(n + 1) / 10)
    report = w.escalate(limit=2)
    assert report.judged == 2
    assert w.service.status().queued == 3
    assert w.escalate(limit=10).judged == 3


def test_the_default_batch_is_the_configured_one():
    w = memory_llm(LlmSettings(batch_limit=3))
    w.ready()
    for n in range(5):
        w.seed(n)
    assert w.escalate().judged == 3


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 2.5, "3"])
def test_a_bad_limit_is_refused_before_anything_happens(limit):
    w = memory_llm()
    w.ready()
    w.seed(1)
    with pytest.raises(ValidationError):
        w.escalate(limit=limit)
    assert w.judge.calls == 0


# ---- analysis switches ----


def test_with_global_analysis_off_nothing_is_sent():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.base.control.set_global_analysis(False, **WHO)
    report = w.escalate()
    assert report.analysis_off is True
    assert (report.judged, w.judge.calls) == (0, 0)
    assert w.verdict(stored).needs_llm is True


def test_a_source_without_analysis_is_left_alone_and_the_others_are_judged():
    w = memory_llm()
    w.ready("src_on")
    w.base.add_source("src_off", analysis=False)
    kept = w.seed(1, source="src_on")
    left = w.seed(2, source="src_off")
    report = w.escalate()
    assert ids(w.judge) == [kept.item_id]
    assert report.judged == 1
    assert w.verdict(left).needs_llm is True
    assert w.verdict(left).stage is Stage.RULES


def test_a_disabled_source_is_left_alone():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.base.registry.disable("src_news", **WHO)
    assert w.escalate().judged == 0
    assert w.verdict(stored).needs_llm is True


def test_an_item_that_is_not_classified_any_more_is_not_sent():
    w = memory_llm()
    w.ready()
    from social_tracker.items.domain import ItemStatus

    w.seed(1, status=ItemStatus.OUT_OF_SCOPE)
    assert w.escalate().judged == 0
    assert w.judge.calls == 0


# ---- the stop flag ----


def test_a_stop_requested_before_the_run_sends_nothing():
    w = memory_llm()
    w.ready()
    w.seed(1)
    w.stop.set()
    report = w.escalate()
    assert (report.judged, report.stopped, w.judge.calls) == (0, True, 0)


def test_a_stop_in_the_middle_finishes_the_item_in_flight_and_leaves_the_rest_queued():
    w = memory_llm()
    w.ready()
    first = w.seed(1, priority=0.9)
    second = w.seed(2, priority=0.5)
    w.judge.on_call = lambda request: w.stop.set()
    report = w.escalate()
    assert (report.judged, report.stopped, w.judge.calls) == (1, True, 1)
    assert w.verdict(first).stage is Stage.LLM  # the one in flight was completed and written
    assert w.verdict(second).needs_llm is True


# ---- the hourly budget ----


@pytest.mark.parametrize(
    ("cap", "judged", "capped"), [(199, 199, True), (200, 200, True), (201, 201, False)]
)
def test_the_hourly_cap_is_exact_around_200(cap: int, judged: int, capped: bool):
    w = memory_llm(LlmSettings(hourly_cap=cap, batch_limit=300))
    w.ready()
    for n in range(201):
        w.seed(n)
    report = w.escalate()
    assert report.judged == judged
    assert report.cap_reached is capped
    assert w.judge.calls == judged
    assert w.service.status().queued == 201 - judged


def test_the_budget_is_remembered_between_runs_of_the_same_hour():
    w = memory_llm(LlmSettings(hourly_cap=5, batch_limit=10))
    w.ready()
    for n in range(8):
        w.seed(n)
    assert w.escalate(limit=3).judged == 3
    second = w.escalate(limit=10)
    assert (second.judged, second.cap_reached) == (2, True)
    assert w.escalate(limit=10).judged == 0
    assert w.service.status().used_this_hour == 5


def test_the_budget_rolls_over_at_the_start_of_the_next_hour():
    w = memory_llm(LlmSettings(hourly_cap=3, batch_limit=10))
    w.ready()
    for n in range(5):
        w.seed(n)
    w.clock.advance(timedelta(minutes=59, seconds=59))  # 12:59:59
    assert w.escalate().judged == 3
    assert w.escalate().judged == 0
    w.clock.advance(timedelta(seconds=1))  # 13:00:00
    report = w.escalate()
    assert (report.judged, report.cap_reached) == (2, False)
    assert w.service.status().used_this_hour == 2


def test_an_invalid_answer_retry_counts_as_one_judgment_of_the_budget():
    from social_tracker.llm.domain import JudgeError, JudgeErrorKind

    w = memory_llm(LlmSettings(hourly_cap=2))
    w.ready()
    w.seed(1)
    w.judge.script.append(JudgeError(JudgeErrorKind.INVALID_OUTPUT))
    assert w.escalate().judged == 1
    assert w.judge.calls == 2
    assert w.service.status().used_this_hour == 1


# ---- the credibility prior ----


def test_the_credibility_of_the_source_is_recorded_and_never_moves_the_judged_confidence():
    w = memory_llm()
    w.ready()
    stored = w.seed(1, source_credibility_used=5)
    w.judge.default = response(label=Label.DELIBERATE_DISINFORMATION, confidence=0.73)
    w.escalate()
    after = w.verdict(stored)
    assert after.confidence == 0.73
    assert after.source_credibility_used == 5


@pytest.mark.parametrize(
    ("confidence", "low"), [(0.49, True), (0.5, False), (0.0, True), (1.0, False)]
)
def test_an_llm_label_under_half_is_applied_but_flagged_low_confidence(confidence, low):
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    w.judge.default = response(label=Label.NO_RISK, confidence=confidence)
    w.escalate()
    after = w.verdict(stored)
    assert after.label is Label.NO_RISK
    assert after.low_confidence is low


# ---- idempotence and dry run ----


def test_running_again_does_not_judge_again():
    w = memory_llm()
    w.ready()
    stored = w.seed(1)
    assert w.escalate().judged == 1
    again = w.escalate()
    assert (again.judged, w.judge.calls) == (0, 1)
    assert w.escalations.counts().judged == 1
    assert w.verdict(stored).stage is Stage.LLM


def test_a_dry_run_reports_what_would_be_judged_and_changes_nothing():
    w = memory_llm(LlmSettings(hourly_cap=2))
    w.ready()
    stored = [w.seed(n, priority=(n + 1) / 10) for n in range(4)]
    report = w.escalate(dry_run=True)
    assert (report.candidates, report.cap_reached, report.judged) == (2, True, 0)
    assert w.judge.calls == 0
    assert w.service.status().used_this_hour == 0
    assert all(w.verdict(s).needs_llm for s in stored)
    assert w.escalations.counts().judged == 0


def test_a_dry_run_with_an_empty_queue_has_no_candidates():
    w = memory_llm()
    w.ready()
    report = w.escalate(dry_run=True)
    assert (report.candidates, report.cap_reached) == (0, False)


def test_an_empty_queue_is_a_quiet_run():
    w = memory_llm()
    w.ready()
    report = w.escalate()
    assert (report.judged, report.failed, report.skipped, w.judge.calls) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("queued", "candidates", "capped"), [(1, 1, False), (2, 2, False), (3, 2, True)]
)
def test_a_dry_run_reports_the_cap_only_when_it_would_actually_stop_something(
    queued, candidates, capped
):
    w = memory_llm(LlmSettings(hourly_cap=2))
    w.ready()
    for n in range(queued):
        w.seed(n)
    report = w.escalate(dry_run=True)
    assert (report.candidates, report.cap_reached) == (candidates, capped)


@pytest.mark.parametrize("limit", [1, 2, 1000])
def test_the_ends_of_the_batch_range_are_accepted(limit):
    w = memory_llm()
    w.ready()
    w.seed(1)
    w.seed(2)
    assert w.escalate(limit=limit).judged == min(limit, 2)
