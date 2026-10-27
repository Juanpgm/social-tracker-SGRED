"""The worker: startup reset, scheduling pass, graceful stop and isolation of failures."""

import logging
import threading
from datetime import timedelta

import pytest

from social_tracker.collector.domain import FetchError, FetchErrorKind
from social_tracker.collector.service import CollectOutcome
from social_tracker.collector.worker import Worker
from social_tracker.control.domain import CollectionState
from social_tracker.shared.errors import StorageError
from tests.collector.harness import FEED, START, FakeFetcher, World, memory_world, rss


class Rig:
    """A world plus a worker whose sleeper advances the clock instead of waiting."""

    def __init__(self, world: World, **options) -> None:
        self.world = world
        self.stop = threading.Event()
        self.sleeps: list[float] = []
        self.on_sleep = lambda: None
        options.setdefault("tick_s", 5.0)
        self.worker = Worker(
            world.collector,
            world.control,
            world.clock,
            self.stop,
            sleeper=self.sleep,
            **options,
        )

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.world.clock.advance(timedelta(seconds=seconds))
        self.on_sleep()


@pytest.fixture
def rig() -> Rig:
    return Rig(memory_world(FakeFetcher()))


# ---- startup (PC-2, PC-10) ----


def test_starting_resets_global_analysis_to_off_with_an_audited_event(rig):
    rig.world.analysis_on()
    rig.worker.start()
    assert rig.world.control.global_analysis_enabled() is False
    events = [e for e in rig.world.control_repo.list_control_events() if e.source_id is None]
    changes = [e for e in events if e.kind == "global_analysis_changed"]
    assert len(changes) == 2  # on (the operator), then off (the worker)
    detail = rig.world.details("global_analysis_changed", None)[-1]
    assert (detail["actor"], detail["reason"]) == ("system:worker", "worker startup")
    assert (detail["from"], detail["to"]) == ("active", "paused")


def test_starting_when_analysis_is_already_off_writes_no_event(rig):
    rig.worker.start()
    assert rig.world.control_repo.list_control_events() == []


def test_starting_without_the_reset_leaves_analysis_on_and_writes_no_event(rig):
    rig.world.analysis_on()
    before = rig.world.control_repo.list_control_events()
    rig.worker.start(reset_analysis=False)
    assert rig.world.control.global_analysis_enabled() is True
    assert rig.world.control_repo.list_control_events() == before
    assert "worker startup" not in [e.detail for e in before]


def test_starting_without_the_reset_leaves_analysis_off_when_it_is_off(rig):
    rig.worker.start(reset_analysis=False)
    assert rig.world.control.global_analysis_enabled() is False
    assert rig.world.control_repo.list_control_events() == []


def test_starting_without_the_reset_still_recovers_an_interrupted_collection(rig):
    rig.world.ready()
    rig.world.control.start_collection("src_news", actor="system:worker", reason="before crash")
    rig.worker.start(reset_analysis=False)
    assert rig.world.state().state is CollectionState.BACKOFF


def test_starting_recovers_a_collection_a_crashed_worker_left_running(rig):
    rig.world.ready()
    rig.world.control.start_collection("src_news", actor="system:worker", reason="before crash")
    rig.worker.start()
    assert rig.world.state().state is CollectionState.BACKOFF


def test_a_pass_right_after_the_reset_collects_nothing_until_the_operator_turns_analysis_on(rig):
    rig.world.ready()
    rig.worker.start()
    summary = rig.worker.run_pass()
    assert (summary.global_off, summary.results) == (True, ())
    assert rig.world.fetcher.requests == []
    rig.world.analysis_on()
    summary = rig.worker.run_pass()
    assert summary.global_off is False
    assert [r.outcome for r in summary.results] == [CollectOutcome.COLLECTED]


# ---- scheduling ----


def test_a_pass_collects_the_due_sources_one_after_the_other_in_id_order(rig):
    rig.world.ready("src_b")
    rig.world.add_source("src_a")
    active, peak, order = [0], [0], []

    def watch(request):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        order.append(request.url)
        active[0] -= 1
        return FEED

    rig.world.fetcher.then(watch)
    summary = rig.worker.run_pass()
    assert [r.source_id for r in summary.results] == ["src_a", "src_b"]
    assert peak[0] == 1


def test_sources_are_collected_again_only_after_their_interval(rig):
    rig.world.ready(poll_interval_s=300)
    assert len(rig.worker.run_pass().results) == 1
    rig.world.clock.advance(timedelta(seconds=299))
    assert rig.worker.run_pass().results == ()
    rig.world.clock.advance(timedelta(seconds=1))
    assert len(rig.worker.run_pass().results) == 1
    assert len(rig.world.fetcher.requests) == 2


def test_a_failing_source_does_not_hold_back_the_others_and_is_retried_after_its_backoff(rig):
    rig.world.ready("src_a")
    rig.world.add_source("src_b")
    rig.world.fetcher.then(
        lambda r: (
            FetchError(FetchErrorKind.TIMEOUT) if len(rig.world.fetcher.requests) == 1 else FEED
        )
    )
    summary = rig.worker.run_pass()
    assert [r.outcome for r in summary.results] == [CollectOutcome.FAILED, CollectOutcome.COLLECTED]
    assert rig.worker.run_pass().results == ()  # src_a is backing off, src_b is not due
    rig.world.wait_out_backoff("src_a")
    assert [r.source_id for r in rig.worker.run_pass().results] == ["src_a"]


def test_disabled_and_failed_sources_are_left_alone(rig):
    rig.world.ready("src_ok")
    rig.world.add_source("src_off", enable=False)
    summary = rig.worker.run_pass()
    assert [r.source_id for r in summary.results] == ["src_ok"]


# ---- the loop and the graceful stop (PC-6) ----


def test_the_loop_sleeps_one_tick_between_passes_until_stopped(rig):
    rig.world.ready()
    rig.on_sleep = lambda: rig.stop.set() if len(rig.sleeps) == 3 else None
    rig.worker.run()
    assert rig.sleeps == [5.0, 5.0, 5.0]


def test_once_runs_a_single_pass_without_sleeping(rig):
    rig.world.ready()
    rig.world.analysis_on()  # the reset at startup turns it off again
    summary = rig.worker.run(once=True)
    assert rig.sleeps == []
    assert summary.global_off is True
    assert rig.world.fetcher.requests == []


def test_once_without_the_reset_collects_what_is_due_while_analysis_is_on(rig):
    rig.world.ready()
    rig.world.analysis_on()
    summary = rig.worker.run(once=True, reset_analysis=False)
    assert summary.global_off is False
    assert [r.outcome for r in summary.results] == [CollectOutcome.COLLECTED]
    assert rig.world.items.count("src_news") == 2
    assert rig.world.control.global_analysis_enabled() is True  # and it stays on afterwards
    assert rig.sleeps == []


def test_once_without_the_reset_and_with_analysis_off_collects_nothing(rig):
    rig.world.add_source("src_news")  # enabled, but global analysis was never turned on
    summary = rig.worker.run(once=True, reset_analysis=False)
    assert (summary.global_off, summary.results) == (True, ())
    assert rig.world.fetcher.requests == []


def test_the_reset_is_the_default_of_run(rig):
    rig.world.ready()
    rig.world.analysis_on()
    summary = rig.worker.run(once=True)
    assert summary.global_off is True
    assert rig.world.fetcher.requests == []


def test_a_stop_requested_before_the_loop_runs_no_pass(rig):
    rig.world.ready()
    rig.stop.set()
    assert rig.worker.run() is None
    assert rig.sleeps == []


def test_a_stop_during_a_collection_lets_that_unit_finish_and_starts_no_other(rig):
    rig.world.ready("src_a")
    rig.world.add_source("src_b")
    rig.world.analysis_on()

    def stop_while_fetching(request):
        rig.stop.set()
        return FEED

    rig.world.fetcher.then(stop_while_fetching)
    summary = rig.worker.run_pass()
    assert [r.source_id for r in summary.results] == ["src_a"]
    assert rig.world.items.count("src_a") == 2  # the unit that was running was completed
    assert rig.world.state("src_a").state is CollectionState.IDLE
    assert rig.world.state("src_b").state is CollectionState.IDLE  # never started
    assert len(rig.world.fetcher.requests) == 1


def test_the_loop_ends_without_another_sleep_once_a_stop_arrives_during_a_pass(rig):
    rig.world.ready()

    def stop_while_fetching(request):
        rig.stop.set()
        return FEED

    rig.world.fetcher.then(stop_while_fetching)
    rig.on_sleep = (
        rig.world.analysis_on
    )  # the operator switches analysis on after the startup reset
    rig.worker.run()
    assert rig.sleeps == [5.0]  # one idle tick while analysis was off, none after the stop
    assert rig.world.items.count() == 2


def test_request_stop_sets_the_flag(rig):
    assert rig.worker.stopped is False
    rig.worker.request_stop()
    assert rig.worker.stopped is True
    assert rig.stop.is_set()


# ---- isolation of failures ----


def test_an_unexpected_error_in_one_source_is_logged_and_the_pass_goes_on(rig, caplog):
    rig.world.ready("src_a")
    rig.world.add_source("src_b")
    real = rig.world.collector.collect_once

    def flaky(source_id):
        if source_id == "src_a":
            raise RuntimeError("bug with https://secret.example/?token=hunter2")
        return real(source_id)

    rig.world.collector.collect_once = flaky  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO, logger="social_tracker"):
        summary = rig.worker.run_pass()
    assert [r.source_id for r in summary.results] == ["src_b"]
    assert summary.errors == 1
    assert "hunter2" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_a_locked_database_costs_one_pass_not_the_worker(rig, caplog):
    rig.world.ready()
    real = rig.world.collector.due_sources
    calls = []

    def locked():
        calls.append(1)
        if len(calls) == 1:
            raise StorageError("database error: SQLITE_BUSY")
        return real()

    rig.world.collector.due_sources = locked  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="social_tracker"):
        first = rig.worker.run_pass()
        second = rig.worker.run_pass()
    assert (first.errors, first.results) == (1, ())
    assert [r.outcome for r in second.results] == [CollectOutcome.COLLECTED]
    assert "storage_error" in caplog.text


# ---- retention ----


def test_the_retention_purge_runs_on_the_first_pass_and_then_at_its_interval():
    world = memory_world(FakeFetcher())
    calls = []
    rig = Rig(world, purge=lambda: calls.append(world.clock.now()), purge_every_s=3600.0)
    rig.worker.run_pass()
    assert calls == [START]
    world.clock.advance(timedelta(seconds=3599))
    rig.worker.run_pass()
    assert len(calls) == 1
    world.clock.advance(timedelta(seconds=1))
    rig.worker.run_pass()
    assert len(calls) == 2


def test_a_failing_purge_is_logged_and_does_not_stop_collection(caplog):
    world = memory_world(FakeFetcher())
    world.ready()

    def broken():
        raise StorageError("database error")

    rig = Rig(world, purge=broken)
    with caplog.at_level(logging.WARNING, logger="social_tracker"):
        summary = rig.worker.run_pass()
    assert [r.outcome for r in summary.results] == [CollectOutcome.COLLECTED]
    assert "purge" in caplog.text


# ---- logging ----


def test_logs_name_sources_and_counts_but_never_urls_or_item_text(rig, caplog):
    rig.world.ready()
    rig.world.fetcher.then(rss(("a", "Secret headline", "Secret body text")))
    with caplog.at_level(logging.INFO, logger="social_tracker"):
        rig.worker.run_pass()
    assert "src_news" in caplog.text
    assert "1 new" in caplog.text
    for forbidden in ("news.example.org", "Secret headline", "Secret body"):
        assert forbidden not in caplog.text


def test_failures_are_logged_with_their_fixed_reason(rig, caplog):
    rig.world.ready()
    rig.world.fetcher.then(FetchError.http(503))
    with caplog.at_level(logging.INFO, logger="social_tracker"):
        rig.worker.run_pass()
    assert "http 503" in caplog.text
    assert "news.example.org" not in caplog.text


def test_invalid_configuration_is_refused():
    world = memory_world(FakeFetcher())
    for tick in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tick_s"):
            Worker(world.collector, world.control, world.clock, threading.Event(), tick_s=tick)
