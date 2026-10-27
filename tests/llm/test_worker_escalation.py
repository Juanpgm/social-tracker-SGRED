"""The worker escalates after the cascade, only while global analysis is on."""

import logging
import threading

import pytest

from social_tracker.collector.worker import Worker
from social_tracker.shared.errors import StorageError
from tests.collector.harness import FakeFetcher, World, memory_world


class Recorder:
    def __init__(self, result=None, error=None) -> None:
        self.calls = 0
        self.result, self.error = result, error

    def __call__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def make(world: World, escalate, *, classify=None, catalog=None, stop=None) -> Worker:
    return Worker(
        world.collector,
        world.control,
        world.clock,
        stop or threading.Event(),
        sleeper=lambda seconds: None,
        classify=classify,
        catalog=catalog,
        escalate=escalate,
    )


@pytest.fixture
def world() -> World:
    return memory_world(FakeFetcher())


def test_a_pass_escalates_after_collecting_classifying_and_refreshing_the_catalogs(world):
    world.ready()
    order = []
    original = world.collector.collect_once
    world.collector.collect_once = lambda source_id: (order.append("collect"), original(source_id))[
        1
    ]
    worker = make(
        world,
        lambda: (order.append("escalate"), "judged")[1],
        classify=lambda: (order.append("classify"), "report")[1],
        catalog=lambda: (order.append("catalog"), "refreshed")[1],
    )
    summary = worker.run_pass()
    # the slow, optional stage comes last: it holds nothing else back
    assert order == ["collect", "classify", "catalog", "escalate"]
    assert summary.escalation == "judged"
    assert summary.classification == "report"
    assert summary.catalog == "refreshed"


def test_nothing_is_escalated_while_global_analysis_is_off(world):
    world.add_source()
    recorder = Recorder("x")
    summary = make(world, recorder).run_pass()
    assert (summary.global_off, recorder.calls, summary.escalation) == (True, 0, None)


def test_the_hook_runs_on_every_pass(world):
    world.analysis_on()
    recorder = Recorder("x")
    worker = make(world, recorder)
    worker.run_pass()
    worker.run_pass()
    assert recorder.calls == 2


def test_without_the_hook_a_pass_is_what_it_was(world):
    world.ready()
    summary = make(world, None).run_pass()
    assert summary.escalation is None


@pytest.mark.parametrize(
    "error", [StorageError("database error: SQLITE_BUSY"), RuntimeError("boom"), ValueError("x")]
)
def test_a_failing_escalation_costs_that_step_only(world, error, caplog):
    world.ready()
    classify, catalog = Recorder("report"), Recorder("refreshed")
    with caplog.at_level(logging.DEBUG):
        summary = make(world, Recorder(error=error), classify=classify, catalog=catalog).run_pass()
    assert (summary.escalation, summary.classification, summary.catalog) == (
        None,
        "report",
        "refreshed",
    )
    assert summary.errors == 1
    assert "boom" not in caplog.text


def test_a_failing_classification_does_not_stop_the_escalation_of_what_is_queued(world):
    world.ready()
    escalate = Recorder("judged")
    summary = make(world, escalate, classify=Recorder(error=RuntimeError("boom"))).run_pass()
    assert escalate.calls == 1
    assert summary.escalation == "judged"


def test_a_stop_requested_during_collection_skips_the_escalation(world):
    world.ready()
    stop = threading.Event()
    original = world.collector.collect_once

    def collect_then_stop(source_id):
        result = original(source_id)
        stop.set()
        return result

    world.collector.collect_once = collect_then_stop
    recorder = Recorder("x")
    summary = make(world, recorder, stop=stop).run_pass()
    assert (recorder.calls, summary.escalation) == (0, None)


def test_a_stop_requested_during_the_catalog_refresh_skips_the_escalation(world):
    world.ready()
    stop = threading.Event()
    recorder = Recorder("x")

    def refresh():
        stop.set()
        return "refreshed"

    summary = make(world, recorder, catalog=refresh, stop=stop).run_pass()
    assert (recorder.calls, summary.escalation, summary.catalog) == (0, None, "refreshed")


def test_the_log_line_names_counts_and_never_text(world, caplog):
    world.ready()

    class Report:
        judged, invalid, reused, skipped, failed, parked, dropped = 3, 1, 2, 0, 1, 0, 0

    with caplog.at_level(logging.INFO):
        make(world, lambda: Report()).run_pass()
    assert "judged 3" in caplog.text
    assert "invalid 1" in caplog.text
