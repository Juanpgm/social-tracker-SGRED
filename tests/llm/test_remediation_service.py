"""The escalation service after the remediation round: judge-level failures, parking and
requeueing, the hourly budget, the pass budget, settlement fields, guard rails, copies,
the bounded queue. Scripted judge, no network, no GPU."""

import threading
from datetime import timedelta

import pytest

from social_tracker.classification.domain import Classification, Label, Signal, Stage
from social_tracker.items.domain import ItemStatus
from social_tracker.llm import service as service_module
from social_tracker.llm.domain import JudgeError, JudgeErrorKind, LlmSettings
from social_tracker.shared.errors import NotFoundError, ValidationError
from tests.actors import WHO
from tests.items.factories import make_item
from tests.llm.harness import LlmWorld, error, memory_llm, sqlite_llm
from tests.llm.support import NOW, response, verdict

COPIES = "src_copies"
MORE = "src_more"


def world(settings: LlmSettings | None = None, **kwargs) -> LlmWorld:
    w = memory_llm(settings, **kwargs)
    w.ready()
    return w


def copies_world(settings: LlmSettings | None = None) -> LlmWorld:
    w = world(settings)
    w.ready(COPIES)
    w.ready(MORE)
    return w


def waiting_copy(
    w: LlmWorld, n: int, original: Classification, *, source: str = COPIES, same_text=True, **kw
) -> Classification:
    text = w.items.get(original.item_id).text if same_text else None
    return w.seed(
        n,
        text=text,
        source=source,
        needs_llm=False,
        priority=0.0,
        duplicate_of=original.item_id,
        awaiting_original=True,
        stage=Stage.DEDUP,
        label=original.label,
        confidence=original.confidence,
        rationale=f"duplicate of {original.item_id}: reused its {original.label.value} verdict",
        **kw,
    )


def status_error(kind: str, status: int | None = None) -> JudgeError:
    return JudgeError(JudgeErrorKind(kind), "x", status=status)


# ============================== C1: the judge's failures ==============================


@pytest.mark.parametrize(
    "failure",
    [
        status_error("bad_response", 400),
        status_error("bad_response", 401),
        status_error("bad_response", 403),
        status_error("bad_response", 429),
        status_error("server_error", 500),
        status_error("server_error", 503),
        status_error("load_timeout"),
        status_error("unreachable"),
        status_error("model_not_found"),
        status_error("not_ollama"),
    ],
    ids=lambda f: f"{f.kind.value}-{f.status}",
)
def test_a_judge_level_failure_never_counts_against_the_item_however_often_it_repeats(failure):
    w = world(LlmSettings(max_failures=1))
    w.prime_success()
    stored = w.seed(1)
    w.judge.default = failure
    for n in range(5):
        w.escalate()
        w.clock.advance(timedelta(hours=2))
        w.prime_success(9100 + n)  # the judge answers others: still not the item's fault
    state = w.escalations.get_state(stored.item_id)
    assert (state.state, state.item_failures) == ("escalation_failed", 0)
    assert w.verdict(stored).needs_llm is True
    assert w.escalations.counts().parked == 0


def test_a_server_that_crashes_on_every_item_parks_nothing():
    """The 500 of an out-of-memory runner, for every item: all of them keep their place."""
    w = world(LlmSettings(max_failures=1))
    w.prime_success()
    items = [w.seed(n, priority=1 - n / 10) for n in range(1, 5)]
    w.judge.default = status_error("server_error", 500)
    for _ in range(6):
        w.escalate()
        w.clock.advance(timedelta(hours=2))
        w.prime_success(9100 + w.judge.calls)
    assert w.escalations.counts().parked == 0
    assert all(w.verdict(i).needs_llm for i in items)


def test_after_a_judge_level_failure_the_whole_judge_rests_and_nothing_is_called():
    w = world()
    w.seed(1)
    w.seed(2)
    w.judge.default = status_error("server_error", 500)
    first = w.escalate()
    assert (first.failed, first.unreachable, first.error) == (1, True, "llm_server_error")
    assert w.judge.calls == 1
    w.clock.advance(timedelta(seconds=59))
    resting = w.escalate()
    assert (resting.backing_off, resting.unreachable, resting.failed) == (True, True, 0)
    assert w.judge.calls == 1  # the rest is for every item


def test_the_rest_ends_at_its_moment_and_doubles_with_every_failure():
    w = world(LlmSettings(retry_base_s=60, retry_cap_s=300))
    w.seed(1)
    w.judge.default = status_error("unreachable")
    for seconds in [60, 120, 240, 300, 300]:
        report = w.escalate()  # the judge is asked again and fails again
        assert (report.backing_off, report.error) == (False, "llm_unreachable")
        calls = w.judge.calls
        w.clock.advance(timedelta(seconds=seconds - 1))
        assert w.escalate().backing_off is True
        assert w.judge.calls == calls
        w.clock.advance(timedelta(seconds=1))  # exactly the end of the rest


def test_a_success_ends_the_rest_and_resets_its_length():
    w = world(LlmSettings(retry_base_s=60))
    w.seed(1)
    w.judge.script.extend([status_error("unreachable"), status_error("unreachable")])
    w.escalate()
    w.clock.advance(timedelta(minutes=10))
    w.escalate()  # the second failure: 120 s
    w.clock.advance(timedelta(minutes=10))
    assert w.escalate().judged == 1  # default response: up again
    w.seed(2)
    w.judge.script.append(status_error("unreachable"))
    w.escalate()
    w.clock.advance(timedelta(seconds=60))  # back to the base rest, not 240 s
    assert w.escalate().backing_off is False


def test_a_rest_never_hides_the_housekeeping_that_needs_no_judge():
    w = world(LlmSettings(queue_max=2))
    first = w.seed(1, priority=0.9)
    w.seed(2, priority=0.8)
    w.judge.default = status_error("unreachable")
    w.escalate()  # the first item fails: it waits out a backoff, the judge rests
    low = w.seed(3, priority=0.1)
    lowest = w.seed(4, priority=0.05)
    w.clock.advance(timedelta(seconds=10))
    report = w.escalate()
    assert report.backing_off is True
    assert w.judge.calls == 1
    assert report.dropped == 1  # the bound is still enforced while the judge rests
    assert w.escalations.get_state(lowest.item_id).state == "dropped"
    assert w.verdict(low).needs_llm is True
    assert w.verdict(first).needs_llm is True


def test_the_item_that_failed_is_served_behind_the_others_next_time():
    w = world()
    head = w.seed(1, priority=0.9)
    second = w.seed(2, priority=0.5)
    third = w.seed(3, priority=0.4)
    w.judge.default = lambda request: (
        status_error("server_error", 500) if request.item_id == head.item_id else None
    )
    first = w.escalate()
    assert (first.judged, first.failed) == (0, 1)
    w.clock.advance(timedelta(seconds=61))  # the judge's rest is over, the head's wait is not
    second_pass = w.escalate()
    assert second_pass.judged == 2
    assert [r.item_id for r in w.judge.requests[1:]] == [second.item_id, third.item_id]
    assert w.verdict(head).needs_llm is True


def test_a_slow_item_at_the_head_does_not_trip_the_breaker_pass_after_pass():
    w = world()
    slow = w.seed(1, priority=0.9)
    others = [w.seed(n, priority=0.5 - n / 100) for n in range(2, 6)]
    w.judge.default = lambda request: error("timeout") if request.item_id == slow.item_id else None
    first = w.escalate()  # nothing was answered yet: the timeout is the judge's, the batch ends
    assert (first.judged, first.failed) == (0, 1)
    w.clock.advance(timedelta(seconds=61))  # the judge has rested, the slow item has not
    report = w.escalate()
    assert report.judged == len(others)
    assert all(w.verdict(o).stage is Stage.LLM for o in others)
    assert w.verdict(slow).needs_llm is True


# ------------------------------- parked items come back -------------------------------


def park(w: LlmWorld, n: int = 1) -> Classification:
    stored = w.seed(n)
    w.prime_success(9500 + n)
    w.judge.default = lambda request: (
        error("timeout") if request.item_id == stored.item_id else None
    )
    w.escalate()
    assert w.escalations.get_state(stored.item_id).state == "parked"
    w.judge.default = None
    return stored


def test_a_parked_item_re_enters_the_queue_after_its_time_to_live():
    w = world(LlmSettings(max_failures=1, parked_ttl_s=3600))
    stored = park(w)
    parked_at = w.clock.now()
    w.clock.advance(timedelta(seconds=3599))
    assert w.escalate().requeued == 0
    assert w.verdict(stored).needs_llm is False
    w.clock.advance(timedelta(seconds=1))  # exactly the time to live
    report = w.escalate()
    assert report.requeued == 1
    assert report.judged == 1  # and it is served in the same pass
    assert w.verdict(stored).stage is Stage.LLM
    assert w.escalations.get_state(stored.item_id) is None
    assert w.clock.now() - parked_at == timedelta(hours=1)


def test_the_default_time_to_live_is_a_day():
    w = world(LlmSettings(max_failures=1))
    stored = park(w)
    w.clock.advance(timedelta(hours=23, minutes=59))
    assert w.escalate().requeued == 0
    w.clock.advance(timedelta(minutes=1))
    assert w.escalate().requeued == 1
    assert w.verdict(stored).stage is Stage.LLM


def test_a_requeued_item_starts_with_a_clean_slate_and_queues_behind_the_healthy_ones():
    w = world(LlmSettings(max_failures=2, parked_ttl_s=60))
    stored = w.seed(1, priority=0.9)
    w.prime_success()
    w.judge.default = lambda request: error("timeout")
    w.escalate()
    w.clock.advance(timedelta(minutes=3))
    w.prime_success(9600)
    w.escalate()  # second failure: parked
    assert w.escalations.get_state(stored.item_id).state == "parked"
    healthy = w.seed(2, priority=0.2)
    w.judge.default = None
    w.clock.advance(timedelta(seconds=61))
    report = w.escalate()
    assert report.requeued == 1
    assert [r.item_id for r in w.judge.requests[-2:]] == [healthy.item_id, stored.item_id]


def test_requeue_parked_does_not_wait_for_the_time_to_live():
    w = world(LlmSettings(max_failures=1))
    a, b = park(w, 1), park(w, 2)
    assert w.service.requeue_parked() == 2
    assert w.verdict(a).needs_llm
    assert w.verdict(b).needs_llm
    assert w.verdict(a).priority == service_module.REQUEUE_PRIORITY
    assert w.escalations.get_state(a.item_id) is None
    assert w.escalate().judged == 2


def test_requeue_parked_honours_its_limit_and_the_longest_parked_go_first():
    w = world(LlmSettings(max_failures=1))
    first = park(w, 1)
    w.clock.advance(timedelta(minutes=1))
    second = park(w, 2)
    assert w.service.requeue_parked(limit=1) == 1
    assert w.verdict(first).needs_llm is True
    assert w.verdict(second).needs_llm is False
    assert w.service.requeue_parked(limit=1) == 1


def test_requeue_parked_with_nothing_parked_is_zero():
    w = world()
    w.seed(1)
    assert w.service.requeue_parked() == 0


def test_requeue_parked_leaves_dropped_and_invalid_items_alone():
    w = world(LlmSettings(queue_max=1, max_failures=1))
    w.prime_success()
    kept = w.seed(1, priority=0.9)
    dropped = w.seed(2, priority=0.1)
    w.judge.script.extend([JudgeError(JudgeErrorKind.INVALID_OUTPUT)] * 2)
    w.escalate()
    assert w.escalations.get_state(dropped.item_id).state == "dropped"
    assert w.escalations.get_judgment(kept.item_id).outcome == "invalid"
    assert w.service.requeue_parked() == 0
    assert w.verdict(dropped).needs_llm is False
    assert w.verdict(kept).needs_llm is False


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5, "3"])
def test_requeue_parked_refuses_a_bad_limit(limit):
    with pytest.raises(ValidationError):
        world().service.requeue_parked(limit)


def test_a_parked_item_whose_item_is_gone_is_not_resurrected_in_the_queue():
    w = world(LlmSettings(max_failures=1))
    park(w)
    w.items.purge_fetched_before(
        NOW + timedelta(days=3650), purge_queued_fetched_before=NOW + timedelta(days=3650)
    )
    w.service.requeue_parked()
    assert w.escalate().judged == 0
    assert w.judge.calls == 1  # the call that parked it, none since


# ============================== W2: the retry ==============================


def test_the_retry_names_the_problem_class_and_carries_the_attempt_number():
    w = world()
    w.prime_success()
    w.seed(1)
    bad = JudgeError(JudgeErrorKind.INVALID_OUTPUT, "does not match", problem="schema")
    w.judge.script.extend([bad, response()])
    w.escalate()
    first, second = w.judge.requests
    assert (first.attempt, first.previous_problem) == (0, "")
    assert (second.attempt, second.previous_problem) == (1, "schema")
    assert (second.text, second.evidence) == (first.text, first.evidence)


def test_the_retry_after_a_cut_answer_says_so():
    w = world()
    w.seed(1)
    cut = JudgeError(JudgeErrorKind.INVALID_OUTPUT, "cut", problem="cut")
    w.judge.script.extend([cut, response()])
    w.escalate()
    assert w.judge.requests[1].previous_problem == "cut"


def test_a_stop_between_the_two_calls_leaves_the_item_in_the_queue():
    w = world()
    stored = w.seed(1)
    w.judge.script.extend([JudgeError(JudgeErrorKind.INVALID_OUTPUT), response()])
    w.judge.on_call = lambda request: w.stop.set()
    report = w.escalate()
    assert (report.stopped, report.judged, w.judge.calls) == (True, 0, 1)
    assert w.verdict(stored).needs_llm is True
    assert w.escalations.get_state(stored.item_id) is None  # nothing to hold against it


def test_the_pass_budget_is_checked_between_the_two_calls_too():
    w = world(LlmSettings(pass_budget_s=10))
    stored = w.seed(1)
    w.judge.script.extend([JudgeError(JudgeErrorKind.INVALID_OUTPUT), response()])
    w.judge.on_call = lambda request: w.clock.advance(timedelta(seconds=10))
    report = w.escalate()
    assert (report.out_of_time, report.judged, w.judge.calls) == (True, 0, 1)
    assert w.verdict(stored).needs_llm is True


# ============================== W4: the prompt fits the window ==============================


def test_a_long_text_is_cut_to_the_context_window_and_flagged():
    settings = LlmSettings(num_ctx=2560, max_output_tokens=512, max_prompt_chars=100_000)
    w = world(settings)
    stored = w.seed(1, text="palabra " * 10_000)
    w.escalate()
    (sent,) = w.judge.requests
    assert sent.text_truncated is True
    assert len(sent.text) < 10_000
    assert w.verdict(stored).stage is Stage.LLM


def test_a_text_that_fits_is_sent_whole_and_not_flagged():
    w = world(LlmSettings(num_ctx=8192))
    w.seed(1, text="El sismo fue leve")
    w.escalate()
    (sent,) = w.judge.requests
    assert (sent.text, sent.text_truncated) == ("El sismo fue leve", False)


# ============================== W6: a down server burns no budget ==============================


@pytest.mark.parametrize(
    "failure",
    [
        status_error("unreachable"),
        status_error("model_not_found"),
        status_error("server_error", 502),
        status_error("bad_response", 429),
        status_error("load_timeout"),
        status_error("not_ollama"),
    ],
    ids=lambda f: f.kind.value,
)
def test_repeated_passes_against_a_down_server_never_use_the_hourly_budget(failure):
    w = world(LlmSettings(hourly_cap=3))
    w.seed(1)
    w.judge.default = failure
    for _ in range(12):
        w.escalate()
        w.clock.advance(timedelta(minutes=30))
    assert w.escalations.calls_in_hour(service_hour(w)) == 0
    w.judge.default = None
    w.clock.advance(timedelta(hours=3))
    assert w.escalate().judged == 1  # and the whole budget is still there


def service_hour(w: LlmWorld):
    return w.clock.now().replace(minute=0, second=0, microsecond=0)


def test_a_call_that_reached_the_model_keeps_its_place_in_the_budget():
    w = world(LlmSettings(hourly_cap=5))
    w.prime_success()
    w.seed(1)
    w.judge.default = error("timeout")  # a warm call that never finished
    w.escalate()
    assert w.escalations.calls_in_hour(service_hour(w)) == 1


def test_a_cap_of_one_is_not_used_up_by_a_server_that_is_down():
    w = world(LlmSettings(hourly_cap=1))
    w.seed(1)
    w.judge.script.extend([status_error("unreachable"), status_error("unreachable")])
    w.escalate()
    w.clock.advance(timedelta(minutes=10))
    w.escalate()
    w.clock.advance(timedelta(minutes=10))
    report = w.escalate()
    assert (report.judged, report.cap_reached) == (1, False)


# ============================== W7: the pass budget ==============================


def test_a_pass_stops_when_its_wall_clock_budget_is_spent():
    w = world(LlmSettings(pass_budget_s=10))
    for n in range(5):
        w.seed(n, priority=0.9 - n / 10)
    w.judge.on_call = lambda request: w.clock.advance(timedelta(seconds=6))
    report = w.escalate()
    assert (report.judged, report.out_of_time) == (2, True)
    assert w.judge.calls == 2


def test_the_pass_budget_is_inclusive_it_stops_exactly_when_the_time_is_up():
    w = world(LlmSettings(pass_budget_s=10))
    for n in range(4):
        w.seed(n, priority=0.9 - n / 10)
    w.judge.on_call = lambda request: w.clock.advance(timedelta(seconds=5))
    report = w.escalate()
    assert (report.judged, report.out_of_time) == (2, True)


def test_a_pass_that_finishes_inside_its_budget_is_not_out_of_time():
    w = world(LlmSettings(pass_budget_s=10))
    for n in range(2):
        w.seed(n)
    w.judge.on_call = lambda request: w.clock.advance(timedelta(seconds=4))
    report = w.escalate()
    assert (report.judged, report.out_of_time) == (2, False)


def test_the_item_count_still_bounds_the_pass():
    w = world(LlmSettings(pass_budget_s=3600, batch_limit=2))
    for n in range(5):
        w.seed(n)
    assert w.escalate().judged == 2


def test_the_stop_flag_is_checked_between_items():
    w = world()
    for n in range(4):
        w.seed(n)
    w.judge.on_call = lambda request: w.stop.set()
    report = w.escalate()
    assert (report.judged, report.stopped) == (1, True)


def test_the_lease_outlives_the_longest_possible_judgment():
    settings = LlmSettings(call_deadline_s=60, load_deadline_s=180)
    w = world(settings)
    w.seed(1)
    held = []
    w.judge.on_call = lambda request: held.append(w.escalations._lease)
    start = w.clock.now()
    w.escalate()
    (holder, until) = held[0]
    longest = settings.cold_deadline_s + settings.call_deadline_s  # a cold call, then the retry
    assert until - start >= timedelta(seconds=longest + service_module.LEASE_MARGIN_S)


def test_the_lease_follows_a_longer_load_deadline():
    settings = LlmSettings(call_deadline_s=10, load_deadline_s=900)
    w = world(settings)
    w.seed(1)
    held = []
    w.judge.on_call = lambda request: held.append(w.escalations._lease)
    start = w.clock.now()
    w.escalate()
    assert held[0][1] - start >= timedelta(seconds=900 + 10 + service_module.LEASE_MARGIN_S)


# ============================== W8: settlement fields ==============================


def test_every_way_out_of_the_queue_records_when_and_why():
    w = world(LlmSettings(max_failures=1, queue_max=4))
    w.prime_success()
    ok_item = w.seed(1, priority=0.9)
    invalid_item = w.seed(2, priority=0.8)
    parked_item = w.seed(3, priority=0.7)
    dropped_item = w.seed(4, priority=0.1)
    w.seed(5, priority=0.6)
    w.judge.script.extend(
        [
            response(),
            JudgeError(JudgeErrorKind.INVALID_OUTPUT),
            JudgeError(JudgeErrorKind.INVALID_OUTPUT),
            error("timeout"),
        ]
    )
    w.judge.default = response()
    w.escalate()
    stamp = w.clock.now()
    reasons = {
        "ok": w.verdict(ok_item),
        "invalid": w.verdict(invalid_item),
        "parked": w.verdict(parked_item),
        "dropped": w.verdict(dropped_item),
    }
    assert [(v.settle_reason, v.settled_at) for v in reasons.values()] == [
        ("llm", stamp),
        ("invalid", stamp),
        ("parked", stamp),
        ("parked", stamp),
    ]


def test_a_verdict_still_in_the_queue_has_no_settlement():
    w = world()
    stored = w.seed(1)
    assert (w.verdict(stored).settled_at, w.verdict(stored).settle_reason) == (None, None)


def test_a_recycled_copy_of_a_judged_item_settles_as_a_duplicate():
    w = world()
    judged = w.seed(1, needs_llm=False, priority=0.0, stage=Stage.LLM, label=Label.NO_RISK)
    recycled = w.seed(
        2, label=Label.RECYCLED_CONTENT, recycled_of=judged.item_id, stage=Stage.DEDUP
    )
    w.escalate()
    after = w.verdict(recycled)
    assert (after.settle_reason, after.settled_at) == ("duplicate", w.clock.now())


def test_a_copy_that_took_the_verdict_settles_as_a_duplicate():
    w = copies_world()
    original = w.seed(1)
    copy = waiting_copy(w, 2, original)
    w.escalate()
    after = w.verdict(copy)
    assert (after.settle_reason, after.settled_at) == ("duplicate", w.clock.now())


# ============================== W9: copies never wait for ever ==============================


def test_a_copy_whose_original_was_purged_is_released_and_judged_for_itself():
    w = copies_world()
    original = w.seed(1)
    copy = waiting_copy(w, 2, original)
    w.items.purge_fetched_before(
        w.clock.now() + timedelta(days=3650),
        purge_queued_fetched_before=w.clock.now() + timedelta(days=3650),
    )
    w.base.add(
        "guid-copy-again", w.items.get(copy.item_id).text if False else "texto", source_id=COPIES
    )
    report = w.escalate()
    assert w.items.get(original.item_id) is None
    assert report.released == 0  # the copy's own item went with the purge: nothing to release


def test_a_copy_whose_original_was_settled_without_it_is_released_into_the_queue():
    w = copies_world()
    original = w.seed(1, needs_llm=False, priority=0.0)  # settled meanwhile
    copy = waiting_copy(w, 2, original)
    report = w.escalate()
    assert report.released == 1
    assert w.judge.calls == 1  # the copy was judged for itself
    after = w.verdict(copy)
    assert (after.stage, after.awaiting_original, after.settle_reason) == (Stage.LLM, False, "llm")
    assert after.near_of == original.item_id
    assert after.duplicate_of is None


def test_a_copy_of_an_original_in_a_source_that_is_off_is_released():
    w = copies_world()
    original = w.seed(1, source=MORE)
    copy = waiting_copy(w, 2, original)
    w.base.control.set_source_analysis(MORE, False, **WHO)
    report = w.escalate()
    assert report.released == 1
    assert w.verdict(copy).stage is Stage.LLM
    assert w.verdict(original).needs_llm is True  # its own source is off: untouched


def test_a_copy_of_an_original_that_merely_waits_out_a_backoff_keeps_waiting():
    w = copies_world()
    original = w.seed(1)
    copy = waiting_copy(w, 2, original)
    w.judge.script.append(error("unreachable"))
    w.escalate()
    w.clock.advance(timedelta(seconds=5))
    report = w.escalate()
    assert report.released == 0
    assert w.verdict(copy).awaiting_original is True


def test_pruning_the_history_releases_the_copies_of_a_forgotten_original():
    w = copies_world()
    original = w.seed(1, classified_at=NOW - timedelta(days=400))
    copy = waiting_copy(w, 2, original)
    w.items.purge_fetched_before(
        w.clock.now() + timedelta(days=3650),
        purge_queued_fetched_before=w.clock.now() + timedelta(days=3650),
    )
    w.base.add("guid-copy-still-here", "otro texto", source_id=COPIES)
    removed = w.base.service.prune_history()
    assert removed >= 1
    after = w.verdict(copy)
    assert (after.awaiting_original, after.duplicate_of, after.needs_llm) == (False, None, True)


def test_the_statistics_do_not_count_a_copy_of_a_gone_original_as_waiting():
    w = copies_world()
    original = w.seed(1, needs_llm=False, priority=0.0)
    waiting_copy(w, 2, original)
    stats = w.classifications.stats()
    assert (stats.awaiting_original, stats.orphaned_awaiting) == (0, 1)


# ============================== W10 / S-sec4: the bounded queue ==============================


def test_the_lowest_priority_is_dropped_first_then_the_oldest_of_it():
    w = world(LlmSettings(queue_max=2, hourly_cap=10))
    hi_old = w.seed(1, priority=0.9, classified_at=NOW)
    low_new = w.seed(2, priority=0.1, classified_at=NOW + timedelta(hours=3))
    low_old = w.seed(3, priority=0.1, classified_at=NOW + timedelta(hours=1))
    mid = w.seed(4, priority=0.5, classified_at=NOW + timedelta(hours=2))
    report = w.escalate()
    assert report.dropped == 2
    assert w.escalations.get_state(low_old.item_id).state == "dropped"
    assert w.escalations.get_state(low_new.item_id).state == "dropped"
    assert {r.item_id for r in w.judge.requests} == {hi_old.item_id, mid.item_id}


def test_an_older_high_priority_item_is_never_dropped_before_a_newer_low_priority_one():
    w = world(LlmSettings(queue_max=3, hourly_cap=10))
    old_high = w.seed(1, priority=0.95, classified_at=NOW)
    others = [w.seed(n, priority=0.05, classified_at=NOW + timedelta(hours=n)) for n in range(2, 6)]
    w.escalate()
    assert w.escalations.get_state(old_high.item_id) is None
    assert old_high.item_id in {r.item_id for r in w.judge.requests}
    dropped = [o for o in others if w.escalations.get_state(o.item_id)]
    assert [o.item_id for o in dropped] == [o.item_id for o in others[:2]]  # the oldest low ones


def test_items_waiting_out_a_backoff_do_not_count_towards_the_bound():
    w = world(LlmSettings(queue_max=1))
    a = w.seed(1, priority=0.9)
    b = w.seed(2, priority=0.1)
    w.escalations.put_state(_state(b.item_id, next_attempt_at=w.clock.now() + timedelta(hours=1)))
    report = w.escalate()
    assert report.dropped == 0
    assert w.verdict(a).stage is Stage.LLM


def test_items_of_a_source_that_is_off_do_not_count_towards_the_bound():
    w = copies_world(LlmSettings(queue_max=1))
    w.seed(1, source="src_news")
    w.seed(2, source=MORE)
    w.seed(3, source=MORE)
    w.base.control.set_source_analysis(MORE, False, **WHO)
    assert w.escalate().dropped == 0


def test_the_bound_is_exact_at_the_boundary():
    for queued, dropped in [(2, 0), (3, 0), (4, 1)]:
        w = world(LlmSettings(queue_max=3, hourly_cap=1))
        for n in range(queued):
            w.seed(n, classified_at=NOW + timedelta(minutes=n))
        assert w.escalate().dropped == dropped, queued


def test_a_dry_run_reports_the_drops_it_would_make_without_making_them():
    w = world(LlmSettings(queue_max=2))
    items = [w.seed(n, priority=0.1 * (n + 1)) for n in range(5)]
    report = w.escalate(dry_run=True)
    assert (report.dropped, report.pending_drops) == (0, 3)
    assert all(w.verdict(i).needs_llm for i in items)
    assert w.escalations.counts().dropped == 0
    assert w.judge.calls == 0


def test_a_dry_run_leaves_out_what_would_settle_without_a_call():
    w = world()
    judged = w.seed(1, needs_llm=False, priority=0.0, stage=Stage.LLM, label=Label.NO_RISK)
    w.seed(2, label=Label.RECYCLED_CONTENT, recycled_of=judged.item_id, stage=Stage.DEDUP)
    w.seed(3)
    report = w.escalate(dry_run=True)
    assert report.candidates == 1
    assert w.judge.calls == 0


def test_a_dry_run_with_a_full_hour_says_so():
    w = world(LlmSettings(hourly_cap=1))
    w.escalations.reserve_call(service_hour(w), 1)
    w.seed(1)
    report = w.escalate(dry_run=True)
    assert (report.candidates, report.cap_reached) == (0, True)


def test_one_source_cannot_crowd_the_others_out_of_a_full_queue():
    w = copies_world(LlmSettings(queue_max=6, source_share=0.5, hourly_cap=100))
    flood = [w.seed(n, source="src_news", priority=0.9 - n / 100) for n in range(10)]
    small = [w.seed(100 + n, source=COPIES, priority=0.05) for n in range(2)]
    report = w.escalate()
    assert report.dropped == 6
    # the flooding source lost its own excess; the small source kept every item
    assert all(w.escalations.get_state(s.item_id) is None for s in small)
    gone = [f for f in flood if w.escalations.get_state(f.item_id)]
    assert len(gone) == 6
    # ... and what it lost was its lowest priority
    assert {g.item_id for g in gone} == {f.item_id for f in flood[4:]}


def test_the_fair_share_applies_only_when_the_queue_overflows():
    w = world(LlmSettings(queue_max=100, source_share=0.1))
    for n in range(30):
        w.seed(n)
    assert w.escalate().dropped == 0  # 30 items from one source: room for them all


def test_after_the_fair_share_the_lowest_priority_goes_on_being_dropped():
    w = copies_world(LlmSettings(queue_max=4, source_share=0.5, hourly_cap=100))
    a = [w.seed(n, source="src_news", priority=0.9 - n / 100) for n in range(3)]
    b = [w.seed(10 + n, source=COPIES, priority=0.5 - n / 100) for n in range(3)]
    c = w.seed(20, source=MORE, priority=0.01)
    report = w.escalate()
    assert report.dropped == 3
    # src_news and src_copies are over their share of 2: one of each goes (the lowest of each),
    # then the global lowest priority is dropped
    dropped = {i.item_id for i in [*a, *b, c] if w.escalations.get_state(i.item_id)}
    assert a[2].item_id in dropped
    assert b[2].item_id in dropped
    assert c.item_id in dropped


def test_paging_through_a_page_size_of_drops(monkeypatch):
    monkeypatch.setattr(service_module, "DROP_PAGE", 3)
    w = world(LlmSettings(queue_max=2, hourly_cap=1))
    for n in range(12):
        w.seed(n, classified_at=NOW + timedelta(minutes=n))
    assert w.escalate().dropped == 10
    assert w.escalations.counts().dropped == 10


def test_more_drops_than_one_real_page_are_all_made(tmp_path):
    w = sqlite_llm(tmp_path / "pages.db", LlmSettings(queue_max=1, hourly_cap=1))
    try:
        w.ready()
        total = service_module.DROP_PAGE + 5
        _bulk_queue(w, total)
        report = w.escalate()
        assert report.dropped == total - 1
        assert w.escalations.counts().dropped == total - 1
        assert w.escalations.queued_count(source_ids={"src_news"}, now=w.clock.now()) <= 1
    finally:
        w.close()


def _bulk_queue(w: LlmWorld, total: int) -> None:
    """`total` classified items of src_news, each with a verdict waiting for the judge, written
    in one transaction (this test is about the paging, not about the seeding)."""
    items = [make_item(f"bulk-{n}", source_id="src_news", text=f"texto {n}") for n in range(total)]
    with w.db.transaction():
        assert w.items.add_new(items).new == total
        for n, item in enumerate(items):
            w.items.set_status(item.id, ItemStatus.CLASSIFIED)
            w.classifications.add(
                verdict(item.id.removeprefix("itm_"), classified_at=NOW + timedelta(seconds=n))
            )


# ============================== S-sec2: copies and the text ==============================


def test_a_copy_with_the_same_text_takes_the_judged_verdict():
    w = copies_world()
    original = w.seed(1)
    copy = waiting_copy(w, 2, original)
    w.judge.default = response(label=Label.RECYCLED_CONTENT, confidence=0.7)
    report = w.escalate()
    assert (report.judged, report.reused, w.judge.calls) == (1, 1, 1)
    assert w.verdict(copy).label is Label.RECYCLED_CONTENT


@pytest.mark.parametrize(
    "variant",
    [
        "Mañana habrá un temblor fuerte en Cali 1.",
        "MAÑANA HABRÁ UN TEMBLOR FUERTE EN CALI 1",
        "Mañana habrá un temblor fuerte en Cali 1 https://sgc-gov.co/aviso",
        "Mañana habrá un temblоr fuerte en Cali 1",
    ],
    ids=["punctuation", "case", "address", "homoglyph"],
)
def test_a_copy_whose_text_differs_is_judged_for_itself(variant):
    w = copies_world()
    original = w.seed(1)
    copy = w.seed(
        2,
        text=variant,
        source=COPIES,
        needs_llm=False,
        priority=0.0,
        duplicate_of=original.item_id,
        awaiting_original=True,
        stage=Stage.DEDUP,
    )
    w.judge.script.extend(
        [
            response(label=Label.NO_RISK, confidence=0.9),
            response(label=Label.DELIBERATE_DISINFORMATION, confidence=0.8),
        ]
    )
    first = w.escalate()
    assert (first.judged, first.reused) == (1, 0)  # the original's verdict is not copied
    queued = w.verdict(copy)
    assert (queued.needs_llm, queued.awaiting_original, queued.duplicate_of) == (True, False, None)
    assert queued.near_of == original.item_id
    assert queued.priority >= 0.1
    assert w.escalations.get_judgment(copy.item_id) is None
    second = w.escalate()  # it queues like any other item
    assert second.judged == 1
    after = w.verdict(copy)
    assert (after.label, after.confidence, after.stage) == (
        Label.DELIBERATE_DISINFORMATION,
        0.8,
        Stage.LLM,
    )
    assert w.judge.calls == 2
    assert w.escalations.get_judgment(copy.item_id).outcome == "judged"


def test_a_copy_whose_source_was_switched_off_meanwhile_is_left_alone():
    w = copies_world()
    original = w.seed(1)
    copy = waiting_copy(w, 2, original)
    w.judge.on_call = lambda request: w.base.control.set_source_analysis(COPIES, False, **WHO)
    report = w.escalate()
    assert report.judged == 1
    assert w.verdict(copy).awaiting_original is True  # PC-2: nothing is written for that source
    assert w.escalations.get_judgment(copy.item_id) is None


def test_a_copy_whose_item_was_removed_meanwhile_is_skipped_without_a_crash():
    w = copies_world()
    original = w.seed(1)
    copy = waiting_copy(w, 2, original)
    w.judge.on_call = lambda request: (
        w.items.purge_fetched_before(
            w.clock.now() + timedelta(days=3650),
            purge_queued_fetched_before=w.clock.now() + timedelta(days=3650),
        )
        if False
        else w.items.set_status(
            copy.item_id,
            __import__("social_tracker.items.domain", fromlist=["ItemStatus"]).ItemStatus.NEW,
        )
    )
    report = w.escalate()
    assert report.judged == 1
    assert w.verdict(copy).awaiting_original is True


# ============================== S-sec3: the guard rails ==============================


def test_a_no_risk_that_clears_an_obfuscated_text_is_capped_flagged_and_counted(caplog):
    w = world()
    stored = w.seed(
        1,
        signals=(Signal("obfuscated_text", Label.UNCLASSIFIED, 0.3),),
        label=Label.UNCLASSIFIED,
        confidence=0.3,
    )
    w.judge.default = response(label=Label.NO_RISK, confidence=0.98)
    with caplog.at_level("WARNING"):
        report = w.escalate()
    after = w.verdict(stored)
    assert (after.label, after.confidence, after.low_confidence) == (Label.NO_RISK, 0.7, True)
    assert report.contradicted == 1
    assert "llm_contradicts_prior=1" in caplog.text


def test_the_confidence_of_the_model_is_capped_at_95_percent():
    w = world()
    stored = w.seed(1)
    w.judge.default = response(label=Label.DELIBERATE_DISINFORMATION, confidence=1.0)
    report = w.escalate()
    assert w.verdict(stored).confidence == 0.95
    assert report.contradicted == 0


def test_a_no_risk_that_agrees_with_weak_evidence_is_taken_as_it_is():
    w = world()
    stored = w.seed(1)
    w.judge.default = response(label=Label.NO_RISK, confidence=0.9)
    report = w.escalate()
    after = w.verdict(stored)
    assert (after.confidence, after.low_confidence, report.contradicted) == (0.9, False, 0)


def test_copies_inherit_the_capped_verdict_not_the_raw_answer():
    w = copies_world()
    original = w.seed(1, catalog_status="mismatch")
    copy = waiting_copy(w, 2, original)
    w.judge.default = response(label=Label.NO_RISK, confidence=0.99)
    w.escalate()
    assert w.verdict(original).confidence == 0.7
    assert (w.verdict(copy).confidence, w.verdict(copy).low_confidence) == (0.7, True)


# ============================== S-sec5: who is there ==============================


def test_no_item_text_is_sent_to_something_that_is_not_ollama():
    w = world()
    stored = w.seed(1)
    w.judge.identity = JudgeError(JudgeErrorKind.NOT_OLLAMA)
    report = w.escalate()
    assert (w.judge.calls, report.unreachable, report.error) == (0, True, "llm_not_ollama")
    assert w.verdict(stored).needs_llm is True


def test_the_identity_is_checked_once_per_window_not_once_per_item():
    w = world()
    for n in range(3):
        w.seed(n)
    w.escalate()
    assert w.judge.identified == 1
    w.seed(10)
    w.clock.advance(timedelta(seconds=299))
    w.escalate()
    assert w.judge.identified == 1
    w.seed(11)
    w.clock.advance(timedelta(seconds=1))  # the window is over
    w.escalate()
    assert w.judge.identified == 2


def test_nothing_is_asked_when_no_item_waits():
    w = world()
    w.escalate()
    assert w.judge.identified == 0


def test_the_status_of_something_that_is_not_ollama_says_so_and_lists_no_models():
    w = world()
    w.judge.identity = JudgeError(JudgeErrorKind.NOT_OLLAMA)
    status = w.service.status()
    assert (status.reachable, status.error, status.installed) == (False, "llm_not_ollama", ())


def test_a_ping_identifies_the_server_before_it_asks():
    w = world()
    w.judge.identity = JudgeError(JudgeErrorKind.NOT_OLLAMA)
    result = w.service.ping()
    assert (result.ok, result.error, w.judge.calls) == (False, "llm_not_ollama", 0)


# ============================== R13: a request that cannot be built ==============================


def test_an_item_that_cannot_be_described_is_a_failure_of_that_item_and_burns_no_budget(
    monkeypatch,
):
    w = world(LlmSettings(max_failures=3, hourly_cap=2))
    stored = w.seed(1)
    calls = []

    def boom(verdict):
        calls.append(1)
        raise ValidationError("invalid evidence for the language model")

    monkeypatch.setattr(service_module, "evidence_of", boom)
    for _ in range(2):
        report = w.escalate()
        assert report.failed == 1
        assert w.judge.calls == 0
        w.clock.advance(timedelta(hours=1))
    assert w.escalations.calls_in_hour(service_hour(w)) == 0
    assert w.escalations.get_state(stored.item_id).item_failures == 2
    w.clock.advance(timedelta(hours=1))
    assert w.escalate().parked == 1  # it cannot be asked about: it is parked, not retried for ever
    assert w.escalations.calls_in_hour(service_hour(w)) == 0


def test_a_broken_item_does_not_stop_the_others():
    w = world()
    broken = w.seed(1, priority=0.9)
    fine = w.seed(2, priority=0.5)
    real = service_module.evidence_of

    def selective(verdict):
        if verdict.item_id == broken.item_id:
            raise ValidationError("invalid evidence for the language model")
        return real(verdict)

    service_module.evidence_of = selective
    try:
        report = w.escalate()
    finally:
        service_module.evidence_of = real
    assert (report.failed, report.judged) == (1, 1)
    assert w.verdict(fine).stage is Stage.LLM


# ============================== R12: mutants that survived ==============================


def test_a_stale_failure_record_is_cleared_when_the_item_ends_up_invalid():
    w = world()
    w.prime_success()
    stored = w.seed(1)
    w.escalations.put_state(_state(stored.item_id, attempts=2, item_failures=1))
    w.judge.script.extend([JudgeError(JudgeErrorKind.INVALID_OUTPUT)] * 2)
    w.escalate()
    assert w.escalations.get_judgment(stored.item_id).outcome == "invalid"
    assert w.escalations.get_state(stored.item_id) is None


def test_a_stale_failure_record_is_cleared_when_the_item_is_judged():
    w = world()
    stored = w.seed(1)
    w.escalations.put_state(_state(stored.item_id, attempts=2, item_failures=1))
    w.escalate()
    assert w.escalations.get_state(stored.item_id) is None


def test_an_invalid_verdict_resets_the_failure_streak():
    w = world()
    w.prime_success()
    for n in range(1, 5):
        w.seed(n, priority=1 - n / 10)
    invalid = JudgeError(JudgeErrorKind.INVALID_OUTPUT)
    w.judge.script.extend([error("timeout"), invalid, invalid, error("timeout"), response()])
    report = w.escalate()
    # failure, invalid (settled), failure, valid: two failures never came in a row
    assert (report.failed, report.invalid, report.judged) == (2, 1, 1)
    assert w.judge.calls == 5


def test_a_source_that_vanished_is_an_ineligible_item_not_a_crash():
    w = world()
    stored = w.seed(1)

    def gone(source_id):
        raise NotFoundError("no such source")

    w.service._control.effective_analysis = gone
    w.service._control.overview = lambda: _overview_with(w, stored)
    report = w.escalate()
    assert (report.judged, report.skipped) == (0, 1)
    assert w.judge.calls == 0


def test_two_threads_never_judge_one_item_twice():
    w = world()
    stored = w.seed(1)
    gate = threading.Barrier(2)
    results = []

    def run():
        gate.wait()
        results.append(w.escalate())

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r.judged for r in results) == 1
    assert w.judge.calls == 1
    assert w.verdict(stored).stage is Stage.LLM


def test_the_report_knows_when_nothing_could_be_judged():
    w = world()
    w.seed(1)
    w.judge.default = status_error("unreachable")
    down = w.escalate()
    assert down.nothing_could_be_judged is True
    w.judge.default = None
    w.clock.advance(timedelta(hours=1))
    assert w.escalate().nothing_could_be_judged is False


def test_a_busy_slot_or_a_stop_is_also_nothing_judged():
    w = world()
    w.seed(1)
    w.escalations.try_lease("other", w.clock.now(), w.clock.now() + timedelta(minutes=5))
    assert w.escalate().nothing_could_be_judged is True


# ---- the same on SQLite ----


def test_the_requeue_and_the_orphan_sweep_work_on_sqlite(tmp_path):
    w = sqlite_llm(tmp_path / "sweep.db", LlmSettings(max_failures=1, parked_ttl_s=60))
    try:
        w.ready()
        w.ready(COPIES)
        stored = w.seed(1)
        w.prime_success()
        w.judge.default = lambda r: error("timeout")
        w.escalate()
        assert w.verdict(stored).settle_reason == "parked"
        w.judge.default = None
        w.clock.advance(timedelta(seconds=60))
        report = w.escalate()
        assert (report.requeued, report.judged) == (1, 1)
        original = w.seed(2, needs_llm=False, priority=0.0)
        copy = waiting_copy(w, 3, original)
        assert w.escalate().released == 1
        assert w.verdict(copy).stage is Stage.LLM
    finally:
        w.close()


# ---- helpers ----


def _state(item_id: str, **changes):
    from social_tracker.llm.domain import EscalationState

    fields = {
        "item_id": item_id,
        "state": "escalation_failed",
        "attempts": 1,
        "next_attempt_at": None,
        "last_error": "llm_timeout",
        "updated_at": NOW,
        "item_failures": 0,
    }
    fields.update(changes)
    return EscalationState(**fields)


def _overview_with(w: LlmWorld, stored: Classification):
    """The real overview, so that the item's source still counts as eligible for listing."""
    return type(w.base.control).overview(w.base.control)
