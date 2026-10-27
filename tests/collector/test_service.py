"""CollectorService.collect_once on the fakes and on SQLite (CI-1, CI-3, CI-5, CI-6, CI-11)."""

import threading
from datetime import timedelta

import pytest

from social_tracker.collector.domain import (
    MAX_RETRY_AFTER_S,
    FetchError,
    FetchErrorKind,
    FetchResponse,
)
from social_tracker.collector.service import CollectOutcome, CollectResult
from social_tracker.control.domain import MAX_CONSECUTIVE_FAILURES, CollectionState, FetchStatus
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import NotFoundError, StorageError, ValidationError
from tests.actors import WHO
from tests.collector.harness import (
    FEED,
    FEED_URL,
    START,
    FakeFetcher,
    World,
    memory_world,
    rss,
)
from tests.sources.factories import payload


def collect(world: World, source_id: str = "src_news") -> CollectResult:
    return world.collector.collect_once(source_id)


# ---- the happy path ----


def test_a_collection_fetches_the_feed_and_stores_normalized_items(world):
    world.ready()
    result = collect(world)
    assert result == CollectResult("src_news", CollectOutcome.COLLECTED, new=2, duplicate=0)
    assert [r.url for r in world.fetcher.requests] == [FEED_URL]
    stored = {i.external_id: i for i in world.items.list()}
    assert set(stored) == {"a", "b"}
    first = stored["a"]
    assert (first.source_id, first.source_name) == ("src_news", "Name of src_news")
    assert first.title == "Sismo en Cali"
    assert first.fetched_at == START
    assert first.url == "https://news.example.org/a"


def test_the_source_goes_collecting_then_back_to_idle_with_audited_transitions(world):
    world.ready()
    collect(world)
    state = world.state()
    assert (state.state, state.consecutive_failures) == (CollectionState.IDLE, 0)
    assert world.kinds()[-2:] == ["collection_started", "collection_succeeded"]
    for kind in ("collection_started", "collection_succeeded"):
        (detail,) = world.details(kind)
        assert detail["actor"] == "system:worker"


def test_the_last_fetch_is_recorded_and_the_last_error_is_empty(world):
    world.ready()
    collect(world)
    assert world.control_repo.get_fetch_status("src_news") == FetchStatus("src_news", START, None)


def test_re_ingesting_the_same_feed_reports_zero_new(world):
    world.ready()
    collect(world)
    world.clock.advance(timedelta(minutes=11))
    again = collect(world)
    assert (again.outcome, again.new, again.duplicate) == (CollectOutcome.COLLECTED, 0, 2)
    assert world.items.count() == 2


def test_the_same_content_under_another_url_and_id_is_stored_once(world):
    world.ready()
    world.fetcher.then(rss(("one", "Same story", "Same text"), ("two", "Same story", "Same text")))
    result = collect(world)
    assert (result.new, result.duplicate) == (1, 1)
    world.clock.advance(timedelta(minutes=11))
    world.fetcher.then(rss(("three", "Same story", "Same text")))
    assert collect(world).new == 0
    assert world.items.count() == 1


def test_an_empty_feed_is_a_successful_collection(world):
    world.ready()
    world.fetcher.then(rss())
    result = collect(world)
    assert (result.outcome, result.new, result.duplicate) == (CollectOutcome.COLLECTED, 0, 0)
    assert world.state().state is CollectionState.IDLE
    assert world.control_repo.get_fetch_status("src_news").last_fetch_at == START


def test_discarded_entries_are_counted_in_the_result(world):
    world.ready()
    world.fetcher.then(rss(("a", "Real", "text"), ("b", "  ", "<p>&nbsp;</p>"), ("c", "", "")))
    result = collect(world)
    assert (result.new, result.discarded) == (1, {"empty": 2})


def test_html_and_scripts_in_a_description_are_stored_as_inert_text(world):
    world.ready()
    world.fetcher.then(
        rss(("a", "Alert", "&lt;p&gt;Hola&lt;/p&gt;&lt;script&gt;steal()&lt;/script&gt;"))
    )
    collect(world)
    (item,) = world.items.list()
    assert item.text == "Hola"
    assert item.raw_payload["description"] == "<p>Hola</p><script>steal()</script>"


def test_a_collected_source_keeps_its_items_after_it_is_deleted(world):
    world.ready()
    collect(world)
    world.registry.delete("src_news", **WHO)
    assert world.items.count("src_news") == 2
    assert {i.source_name for i in world.items.list()} == {"Name of src_news"}
    assert world.control_repo.get_fetch_status("src_news") is None


# ---- when nothing may be fetched (CI-11, PC-2) ----


def test_a_disabled_source_is_not_fetched(world):
    world.analysis_on()
    world.add_source(enable=False)
    result = collect(world)
    assert result.outcome is CollectOutcome.SKIPPED_DISABLED
    assert world.fetcher.requests == []
    assert world.items.count() == 0
    assert world.kinds() == ["source_added"]


def test_nothing_is_fetched_while_global_analysis_is_off(world):
    world.add_source()
    result = collect(world)
    assert result.outcome is CollectOutcome.SKIPPED_ANALYSIS_OFF
    assert world.fetcher.requests == []
    assert world.state().state is CollectionState.IDLE
    assert "collection_started" not in world.kinds()


def test_turning_global_analysis_on_makes_the_next_collection_run(world):
    world.add_source()
    assert collect(world).outcome is CollectOutcome.SKIPPED_ANALYSIS_OFF
    world.analysis_on()
    assert collect(world).outcome is CollectOutcome.COLLECTED


def test_the_per_source_analysis_flag_does_not_gate_collection(world):
    world.ready()
    assert world.control.source_analysis_enabled("src_news") is False
    assert collect(world).outcome is CollectOutcome.COLLECTED


def test_only_rss_sources_are_collected(world):
    world.analysis_on()
    world.registry.add(payload("live_stream", id="src_live", name="Live"), **WHO)
    world.registry.enable("src_live", **WHO)
    assert collect(world, "src_live").outcome is CollectOutcome.SKIPPED_UNSUPPORTED
    assert world.fetcher.requests == []


def test_an_unknown_source_is_not_found(world):
    world.analysis_on()
    with pytest.raises(NotFoundError):
        collect(world, "src_missing")


@pytest.mark.parametrize("bad", ["", "Bad Id", "x", None])
def test_an_invalid_source_id_is_refused_without_touching_anything(world, bad):
    world.analysis_on()
    with pytest.raises(ValidationError):
        collect(world, bad)
    assert world.fetcher.requests == []


# ---- failures: backoff, no partial items (CI-5) ----


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FetchError(FetchErrorKind.TIMEOUT), "timeout"),
        (FetchError(FetchErrorKind.DNS), "name lookup failed"),
        (FetchError(FetchErrorKind.CONNECT), "connection failed"),
        (FetchError.http(500), "http 500"),
        (FetchError.http(404), "http 404"),
        (FetchError(FetchErrorKind.TOO_LARGE), "response too large"),
        (FetchError(FetchErrorKind.BLOCKED_ADDRESS), "address not allowed"),
    ],
)
def test_a_failed_fetch_is_recorded_and_the_source_backs_off(world, failure, message):
    world.ready()
    world.fetcher.then(failure)
    result = collect(world)
    assert (result.outcome, result.error) == (CollectOutcome.FAILED, message)
    state = world.state()
    assert (state.state, state.consecutive_failures) == (CollectionState.BACKOFF, 1)
    assert state.next_run_at == START + timedelta(seconds=60)
    assert world.items.count() == 0
    assert world.control_repo.get_fetch_status("src_news") == FetchStatus("src_news", None, message)
    (detail,) = world.details("collection_failed")
    assert detail["reason"] == message


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not a feed",
        b"<html><body>An error page</body></html>",
        rss(("a", "t", "b"))[:-30],
        b"<!DOCTYPE r [<!ENTITY x SYSTEM 'file:///etc/passwd'>]><rss><channel><item><title>&x;</title></item></channel></rss>",
        b"\x00\x01\xff\xfe",
    ],
    ids=["empty", "text", "html", "truncated", "xxe", "binary"],
)
def test_an_unusable_document_is_a_failure_and_stores_no_items(world, body):
    world.ready()
    world.fetcher.then(body)
    result = collect(world)
    assert result.outcome is CollectOutcome.FAILED
    assert result.error.startswith("invalid feed: ")
    assert world.items.count() == 0
    assert world.state().state is CollectionState.BACKOFF


def test_a_feed_truncated_after_good_entries_stores_nothing_not_the_good_ones(world):
    world.ready()
    good = rss(("a", "Good one", "text a"), ("b", "Good two", "text b"))
    world.fetcher.then(good[: good.index(b"</channel>")])
    assert collect(world).outcome is CollectOutcome.FAILED
    assert world.items.count() == 0


def test_the_backoff_doubles_with_each_consecutive_failure(world):
    world.ready()
    world.fetcher.then(FetchError(FetchErrorKind.TIMEOUT))
    delays = []
    for _ in range(MAX_CONSECUTIVE_FAILURES - 1):
        before = world.clock.now()
        assert collect(world).outcome is CollectOutcome.FAILED
        delays.append((world.state().next_run_at - before).total_seconds())
        world.wait_out_backoff()
    assert delays == [60, 120, 240, 480]


def test_a_source_that_keeps_failing_ends_failed_after_five_consecutive_failures(world):
    world.ready()
    world.fetcher.then(FetchError.http(404))
    for attempt in range(MAX_CONSECUTIVE_FAILURES):
        assert collect(world).outcome is CollectOutcome.FAILED
        if attempt < MAX_CONSECUTIVE_FAILURES - 1:
            world.wait_out_backoff()
    state = world.state()
    assert (state.state, state.consecutive_failures) == (
        CollectionState.FAILED,
        MAX_CONSECUTIVE_FAILURES,
    )
    assert len(world.fetcher.requests) == MAX_CONSECUTIVE_FAILURES
    world.clock.advance(timedelta(days=30))
    assert collect(world).outcome is CollectOutcome.SKIPPED_NOT_ELIGIBLE
    assert len(world.fetcher.requests) == MAX_CONSECUTIVE_FAILURES  # a failed source is left alone
    assert world.control_repo.get_fetch_status("src_news").last_error == "http 404"


def test_an_operator_reset_brings_a_failed_source_back(world):
    world.ready()
    world.fetcher.then(FetchError.http(404))
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        collect(world)
        if world.state().state is CollectionState.BACKOFF:
            world.wait_out_backoff()
    world.control.reset_failed("src_news", **WHO)
    world.fetcher.then(FEED)
    assert collect(world).outcome is CollectOutcome.COLLECTED


def test_a_500_followed_by_a_200_stores_each_item_once(world):
    world.ready()
    world.fetcher.then(FetchError.http(500), FEED)
    assert collect(world).outcome is CollectOutcome.FAILED
    assert world.items.count() == 0
    world.wait_out_backoff()
    assert collect(world).outcome is CollectOutcome.COLLECTED
    assert world.items.count() == 2
    state = world.state()
    assert (state.state, state.consecutive_failures) == (CollectionState.IDLE, 0)
    status = world.control_repo.get_fetch_status("src_news")
    assert (status.last_fetch_at, status.last_error) == (world.clock.now(), None)
    assert world.kinds().count("collection_retried") == 1


def test_a_source_in_backoff_is_not_fetched_before_its_retry_time(world):
    world.ready()
    world.fetcher.then(FetchError(FetchErrorKind.TIMEOUT))
    collect(world)
    world.clock.advance(timedelta(seconds=59))
    assert collect(world).outcome is CollectOutcome.SKIPPED_NOT_ELIGIBLE
    assert len(world.fetcher.requests) == 1
    world.clock.advance(timedelta(seconds=1))
    assert collect(world).outcome is CollectOutcome.FAILED
    assert len(world.fetcher.requests) == 2


@pytest.mark.parametrize("status", [401, 403, 429])
def test_refusals_are_failures_with_a_clear_reason(world, status):
    world.ready()
    world.fetcher.then(FetchError.http(status))
    result = collect(world)
    assert result.outcome is CollectOutcome.FAILED
    assert result.error.startswith(f"http {status} (")
    assert world.state().state is CollectionState.BACKOFF
    assert world.control_repo.get_fetch_status("src_news").last_error == result.error


@pytest.mark.parametrize(
    ("retry_after", "expected_s"),
    [
        (None, 60),
        (0.0, 60),
        (30.0, 60),
        (60.0, 60),
        (120.0, 120),
        (3000.0, 3000),
        (3600.0, 3600),
        (3601.0, 3601),  # an explicit Retry-After may exceed the exponential cap...
        (99_999.0, 99_999),
        (MAX_RETRY_AFTER_S, MAX_RETRY_AFTER_S),
        (MAX_RETRY_AFTER_S + 1.0, MAX_RETRY_AFTER_S),  # ...up to MAX_RETRY_AFTER_S, no further
        (10.0**12, MAX_RETRY_AFTER_S),
    ],
)
def test_retry_after_lengthens_the_wait_up_to_its_own_maximum(world, retry_after, expected_s):
    world.ready()
    world.fetcher.then(FetchError.http(429, retry_after_s=retry_after))
    collect(world)
    assert (world.state().next_run_at - START).total_seconds() == expected_s


def test_a_retry_after_beyond_its_maximum_is_clamped_and_the_clamp_is_logged(world, caplog):
    world.ready()
    world.fetcher.then(FetchError.http(429, retry_after_s=MAX_RETRY_AFTER_S * 2.0))
    with caplog.at_level("INFO", logger="social_tracker.collector"):
        collect(world)
    assert any("clamped" in record.getMessage() for record in caplog.records)
    assert "src_news" in caplog.text


def test_a_retry_after_within_its_maximum_logs_no_clamp(world, caplog):
    world.ready()
    world.fetcher.then(FetchError.http(429, retry_after_s=MAX_RETRY_AFTER_S - 1.0))
    with caplog.at_level("INFO", logger="social_tracker.collector"):
        collect(world)
    assert "clamped" not in caplog.text


def test_retry_after_is_compared_with_the_exponential_delay_of_the_next_failure(world):
    world.ready()
    world.fetcher.then(FetchError.http(503))
    collect(world)
    world.wait_out_backoff()
    world.fetcher.then(FetchError.http(429, retry_after_s=90.0))  # less than the 120 s backoff
    now = world.clock.now()
    collect(world)
    assert world.state().next_run_at == now + timedelta(seconds=120)
    world.wait_out_backoff()
    now = world.clock.now()
    world.fetcher.then(FetchError.http(429, retry_after_s=500.0))  # more than the 240 s backoff
    collect(world)
    assert world.state().next_run_at == now + timedelta(seconds=500)


def test_jitter_is_added_to_the_wait(tmp_path):
    jittered = memory_world(
        FakeFetcher(FetchError(FetchErrorKind.TIMEOUT)), jitter=lambda base: 7.0
    )
    jittered.ready()
    jittered.collector.collect_once("src_news")
    assert (jittered.state().next_run_at - START).total_seconds() == 67


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0])
def test_a_misbehaving_jitter_source_cannot_break_the_state_machine(bad):
    broken = memory_world(FakeFetcher(FetchError(FetchErrorKind.TIMEOUT)), jitter=lambda base: bad)
    broken.ready()
    assert broken.collector.collect_once("src_news").outcome is CollectOutcome.FAILED
    assert (broken.state().next_run_at - START).total_seconds() == 60


def test_an_unexpected_error_in_the_fetcher_is_a_failure_with_no_leaked_text(world):
    world.ready()
    world.fetcher.then(RuntimeError("boom at https://news.example.org/rss.xml?token=hunter2"))
    result = collect(world)
    assert result.outcome is CollectOutcome.FAILED
    assert result.error == "unexpected error (RuntimeError)"
    assert "hunter2" not in repr(world.events()) + repr(world.control_repo.list_fetch_statuses())
    assert world.state().state is CollectionState.BACKOFF


# ---- conditional requests ----


def test_validators_from_one_fetch_are_sent_with_the_next(world):
    world.ready()
    world.fetcher.then(
        FetchResponse(FEED, etag='"v1"', last_modified="Sun, 01 Mar 2026 07:00:00 GMT")
    )
    collect(world)
    world.clock.advance(timedelta(minutes=11))
    collect(world)
    first, second = world.fetcher.requests
    assert (first.etag, first.last_modified) == (None, None)
    assert (second.etag, second.last_modified) == ('"v1"', "Sun, 01 Mar 2026 07:00:00 GMT")


def test_not_modified_is_a_success_without_items_and_keeps_the_validators(world):
    world.ready()
    world.fetcher.then(FetchResponse(FEED, etag='"v1"'), FetchResponse(b"", not_modified=True))
    collect(world)
    world.clock.advance(timedelta(minutes=11))
    result = collect(world)
    assert (result.outcome, result.new, result.duplicate) == (CollectOutcome.NOT_MODIFIED, 0, 0)
    status = world.control_repo.get_fetch_status("src_news")
    assert status.etag == '"v1"'
    assert status.last_fetch_at == world.clock.now()
    assert world.state().state is CollectionState.IDLE
    assert world.items.count() == 2


# ---- the source changes while it is being fetched ----


def test_a_source_disabled_during_the_fetch_discards_what_was_fetched(world):
    world.ready()

    def disable_then_answer(request):
        world.registry.disable("src_news", **WHO)
        return FEED

    world.fetcher.then(disable_then_answer)
    result = collect(world)
    assert result.outcome is CollectOutcome.DISCARDED
    assert world.items.count() == 0
    assert world.state().state is CollectionState.DISABLED
    assert "collection_succeeded" not in world.kinds()
    assert world.control_repo.get_fetch_status("src_news") is None


def test_a_source_disabled_during_a_failing_fetch_is_not_marked_failed(world):
    world.ready()

    def disable_then_fail(request):
        world.registry.disable("src_news", **WHO)
        return FetchError(FetchErrorKind.TIMEOUT)

    world.fetcher.then(disable_then_fail)
    assert collect(world).outcome is CollectOutcome.DISCARDED
    assert "collection_failed" not in world.kinds()
    assert world.state().state is CollectionState.DISABLED


def test_a_source_deleted_during_the_fetch_stores_nothing(world):
    world.ready()

    def delete_then_answer(request):
        world.registry.delete("src_news", **WHO)
        return FEED

    world.fetcher.then(delete_then_answer)
    assert collect(world).outcome is CollectOutcome.DISCARDED
    assert world.items.count() == 0
    assert world.control_repo.get_fetch_status("src_news") is None


def test_a_source_disabled_and_enabled_again_during_the_fetch_discards_the_stale_unit(world):
    world.ready()

    def bounce(request):
        world.registry.disable("src_news", **WHO)
        world.registry.enable("src_news", **WHO)
        return FEED

    world.fetcher.then(bounce)
    assert collect(world).outcome is CollectOutcome.DISCARDED
    assert world.items.count() == 0
    assert world.state().state is CollectionState.IDLE


def test_global_analysis_turned_off_during_the_fetch_does_not_waste_the_unit(world):
    world.ready()

    def switch_off(request):
        world.control.set_global_analysis(False, **WHO)
        return FEED

    world.fetcher.then(switch_off)
    assert collect(world).outcome is CollectOutcome.COLLECTED
    assert world.items.count() == 2
    assert collect(world).outcome is CollectOutcome.SKIPPED_ANALYSIS_OFF


# ---- all or nothing ----


class FailingItems:
    """Wraps an item repository; `add_new` stores its items and THEN fails, like a crash would."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def add_new(self, items, **limits):
        self._inner.add_new(items, **limits)
        raise StorageError("simulated failure after the write")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_if_storing_fails_no_item_and_no_success_is_left_behind(world):
    world.ready()
    real = world.items
    world.collector._items = FailingItems(real)  # type: ignore[attr-defined]
    result = collect(world)
    assert result.outcome is CollectOutcome.FAILED
    assert result.error == "storage error"
    assert real.count() == 0
    assert "collection_succeeded" not in world.kinds()
    assert world.state().state is CollectionState.BACKOFF
    world.collector._items = real  # type: ignore[attr-defined]
    world.wait_out_backoff()
    assert collect(world).outcome is CollectOutcome.COLLECTED
    assert real.count() == 2


# ---- concurrency ----


def test_two_collectors_on_one_source_produce_exactly_one_effective_collection(world):
    world.ready()
    inside, release = threading.Event(), threading.Event()

    def hold(request):
        inside.set()
        assert release.wait(timeout=10)
        return FEED

    world.fetcher.then(hold)
    results: list[CollectResult] = []
    first = threading.Thread(target=lambda: results.append(collect(world)))
    first.start()
    assert inside.wait(timeout=10)
    second = collect(world)  # the source is already collecting
    release.set()
    first.join(timeout=10)
    assert second.outcome is CollectOutcome.SKIPPED_NOT_ELIGIBLE
    assert [r.outcome for r in results] == [CollectOutcome.COLLECTED]
    assert len(world.fetcher.requests) == 1
    assert world.items.count() == 2


# ---- interrupted collections ----


def test_a_collection_interrupted_by_a_crash_is_recovered_as_a_failure(world):
    world.ready()
    world.control.start_collection("src_news", actor="system:worker", reason="crashed run")
    recovered = world.collector.recover_interrupted(force=True)
    assert recovered == ["src_news"]
    state = world.state()
    assert (state.state, state.consecutive_failures) == (CollectionState.BACKOFF, 1)
    assert world.control_repo.get_fetch_status("src_news").last_error == "interrupted collection"
    assert world.collector.recover_interrupted(force=True) == []


def test_recovery_leaves_idle_disabled_and_backing_off_sources_alone(world):
    world.ready()
    world.add_source("src_idle")
    world.add_source("src_off", enable=False)
    world.add_source("src_back")
    world.control.start_collection("src_back", actor="system:worker", reason="x")
    world.control.fail_collection("src_back", actor="system:worker", reason="x")
    assert world.collector.recover_interrupted() == []
    assert world.state("src_idle").state is CollectionState.IDLE
    assert world.state("src_back").consecutive_failures == 1


def claim(world, source_id: str = "src_news") -> None:
    world.control.start_collection(source_id, actor="system:worker", reason="in flight")


def timeout_s(world) -> float:
    return world.collector.claim_timeout_s


def test_a_collection_that_just_started_is_not_taken_for_a_dead_one(world):
    world.ready()
    claim(world)
    assert world.collector.recover_interrupted() == []
    assert world.state().state is CollectionState.COLLECTING


@pytest.mark.parametrize(
    ("age_s", "recovered"), [(-1.0, False), (0.0, False), (0.001, True), (1.0, True)]
)
def test_a_claim_is_recovered_only_once_it_is_older_than_the_claim_timeout(world, age_s, recovered):
    world.ready()
    claim(world)
    world.clock.advance(timedelta(seconds=timeout_s(world) + age_s))
    assert (world.collector.recover_interrupted() != []) is recovered
    assert (world.state().state is CollectionState.BACKOFF) is recovered


def test_the_claim_timeout_leaves_room_for_the_whole_fetch_deadline():
    from social_tracker.collector.http_fetcher import FetchLimits
    from social_tracker.collector.service import DEFAULT_CLAIM_TIMEOUT_S

    assert FetchLimits().deadline_s + 30 < DEFAULT_CLAIM_TIMEOUT_S


def test_force_recovers_even_a_claim_that_started_a_moment_ago(world):
    world.ready()
    claim(world)
    assert world.collector.recover_interrupted(force=True) == ["src_news"]


def test_the_age_of_a_claim_counts_from_its_latest_start_not_an_earlier_one(world):
    world.ready()
    claim(world)
    world.control.fail_collection("src_news", actor="system:worker", reason="first try failed")
    world.clock.advance(timedelta(seconds=timeout_s(world) * 10))  # the first start is ancient
    world.control.retry_collection("src_news", actor="system:worker", reason="retry")
    claim(world)  # a fresh claim
    assert world.collector.recover_interrupted() == []
    assert world.state().state is CollectionState.COLLECTING


def test_a_claim_with_a_start_time_in_the_future_is_treated_as_stale(world):
    world.ready()
    claim(world)
    # the clock moved backwards since the claim (a FixedClock cannot, so swap the collector's)
    world.collector._clock = FixedClock(world.clock.now() - timedelta(hours=3))
    assert world.collector.recover_interrupted() == ["src_news"]


def test_a_collecting_source_without_any_start_event_is_recovered(world):
    from social_tracker.control.domain import SourceState

    world.ready()
    world.control_repo.save_source_state(SourceState("src_news", CollectionState.COLLECTING))
    assert world.collector.recover_interrupted() == ["src_news"]


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "5"])
def test_an_invalid_claim_timeout_is_refused(world, bad):
    from social_tracker.collector.service import CollectorService

    with pytest.raises(ValueError, match="claim_timeout_s"):
        CollectorService(
            world.sources_repo,
            world.control,
            world.control_repo,
            world.items,
            world.fetcher,
            world.uow,
            world.clock,
            claim_timeout_s=bad,
        )


# ---- due sources ----


def test_due_sources_follow_the_poll_interval_and_the_backoff(world):
    world.ready()
    assert world.collector.due_sources() == ["src_news"]
    collect(world)
    assert world.collector.due_sources() == []
    world.clock.advance(timedelta(seconds=599))
    assert world.collector.due_sources() == []
    world.clock.advance(timedelta(seconds=1))
    assert world.collector.due_sources() == ["src_news"]


def test_due_sources_are_sorted_and_skip_disabled_and_failed_ones(world):
    world.ready("src_b")
    world.add_source("src_a")
    world.add_source("src_off", enable=False)
    assert world.collector.due_sources() == ["src_a", "src_b"]
    world.fetcher.then(FetchError.http(404))
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        collect(world, "src_b")
        if world.state("src_b").state is CollectionState.BACKOFF:
            world.wait_out_backoff("src_b")
    assert world.collector.due_sources() == ["src_a"]
