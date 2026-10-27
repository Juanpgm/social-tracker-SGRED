"""The worker runs the classification cascade after each collection pass (CPU only)."""

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


def make(world: World, classify, stop=None) -> Worker:
    return Worker(
        world.collector,
        world.control,
        world.clock,
        stop or threading.Event(),
        sleeper=lambda seconds: None,
        classify=classify,
    )


@pytest.fixture
def world() -> World:
    return memory_world(FakeFetcher())


def test_a_pass_with_analysis_on_collects_and_then_classifies_once(world):
    world.ready()
    order = []
    original = world.collector.collect_once
    world.collector.collect_once = lambda source_id: (order.append("collect"), original(source_id))[
        1
    ]
    recorder = Recorder(result="report")
    worker = make(world, lambda: (order.append("classify"), recorder())[1])
    summary = worker.run_pass()
    assert order == ["collect", "classify"]
    assert recorder.calls == 1
    assert summary.classification == "report"
    assert len(summary.results) == 1


def test_a_pass_with_nothing_due_still_classifies_what_is_pending(world):
    world.analysis_on()  # no source at all: nothing to collect
    recorder = Recorder()
    make(world, recorder).run_pass()
    assert recorder.calls == 1


def test_nothing_is_classified_while_global_analysis_is_off(world):
    world.add_source()
    recorder = Recorder()
    summary = make(world, recorder).run_pass()
    assert (summary.global_off, recorder.calls, summary.classification) == (True, 0, None)


def test_without_a_classifier_a_pass_is_what_it_was(world):
    world.ready()
    summary = Worker(
        world.collector, world.control, world.clock, threading.Event(), sleeper=lambda s: None
    ).run_pass()
    assert summary.classification is None


def test_a_stop_requested_during_collection_skips_the_classification(world):
    world.ready()
    stop = threading.Event()
    original = world.collector.collect_once

    def collect_then_stop(source_id):
        result = original(source_id)
        stop.set()
        return result

    world.collector.collect_once = collect_then_stop
    recorder = Recorder()
    make(world, recorder, stop).run_pass()
    assert recorder.calls == 0


@pytest.mark.parametrize(
    "error",
    [StorageError("database error: SQLITE_BUSY"), RuntimeError("boom")],
    ids=["domain", "bug"],
)
def test_a_failing_classification_costs_that_step_and_nothing_more(world, error, caplog):
    world.ready()
    recorder = Recorder(error=error)
    worker = make(world, recorder)
    with caplog.at_level(logging.INFO, logger="social_tracker"):
        first = worker.run_pass()
        second = worker.run_pass()
    assert recorder.calls == 2  # the next pass tries again
    assert (first.errors, first.classification) == (1, None)
    assert second.errors == 1
    assert "boom" not in caplog.text  # a message never reaches the log, only the kind of failure


def test_the_classification_is_logged_with_counts_only(world, caplog):
    world.ready()

    class Report:
        classified, out_of_scope, skipped, errors, needs_llm, eligible_sources = 3, 2, 1, 0, 2, 1

    with caplog.at_level(logging.INFO, logger="social_tracker.worker"):
        make(world, Recorder(result=Report())).run_pass()
    assert "classified 3, out of scope 2, skipped 1, errors 0" in caplog.text
