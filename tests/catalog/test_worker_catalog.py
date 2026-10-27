"""The worker refreshes the official catalogs each pass, only while global analysis is on."""

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


def make(world: World, catalog, classify=None, stop=None) -> Worker:
    return Worker(
        world.collector,
        world.control,
        world.clock,
        stop or threading.Event(),
        sleeper=lambda seconds: None,
        classify=classify,
        catalog=catalog,
    )


@pytest.fixture
def world() -> World:
    return memory_world(FakeFetcher())


def test_a_pass_refreshes_the_catalogs_after_collecting_and_classifying(world):
    world.ready()
    order = []
    original = world.collector.collect_once
    world.collector.collect_once = lambda source_id: (order.append("collect"), original(source_id))[
        1
    ]
    worker = make(
        world,
        lambda: (order.append("catalog"), "refreshed")[1],
        classify=lambda: (order.append("classify"), "report")[1],
    )
    summary = worker.run_pass()
    # R14: a slow catalog refresh must never hold the classification back
    assert order == ["collect", "classify", "catalog"]
    assert summary.catalog == "refreshed"
    assert summary.classification == "report"


def test_nothing_is_fetched_while_global_analysis_is_off(world):
    world.add_source()
    recorder = Recorder("x")
    summary = make(world, recorder).run_pass()
    assert (summary.global_off, recorder.calls, summary.catalog) == (True, 0, None)


def test_the_hook_runs_on_every_pass_and_leaves_the_interval_to_the_refresher(world):
    world.analysis_on()
    recorder = Recorder("x")
    worker = make(world, recorder)
    worker.run_pass()
    worker.run_pass()
    assert recorder.calls == 2


def test_without_a_catalog_hook_a_pass_is_what_it_was(world):
    world.ready()
    summary = make(world, None).run_pass()
    assert summary.catalog is None


@pytest.mark.parametrize(
    "error", [StorageError("database error: SQLITE_BUSY"), RuntimeError("boom"), ValueError("x")]
)
def test_a_failing_refresh_costs_that_step_and_classification_still_runs(world, error, caplog):
    world.ready()
    classify = Recorder("report")
    with caplog.at_level(logging.ERROR):
        summary = make(world, Recorder(error=error), classify=classify).run_pass()
    assert summary.catalog is None
    assert summary.classification == "report"
    assert classify.calls == 1
    assert "boom" not in caplog.text


def test_a_stop_requested_during_collection_skips_the_refresh(world):
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
    assert recorder.calls == 0
    assert summary.catalog is None


def test_a_failing_classification_does_not_prevent_the_refresh(world):
    world.ready()
    refresh = Recorder("refreshed")
    summary = make(world, refresh, classify=Recorder(error=RuntimeError("boom"))).run_pass()
    assert refresh.calls == 1
    assert summary.catalog == "refreshed"
    assert summary.errors == 1


def test_a_stop_requested_during_classification_skips_the_refresh(world):
    world.ready()
    stop = threading.Event()
    refresh = Recorder("x")

    def classify():
        stop.set()
        return "report"

    summary = make(world, refresh, classify=classify, stop=stop).run_pass()
    assert refresh.calls == 0
    assert (summary.classification, summary.catalog) == ("report", None)


def test_the_classification_is_already_done_when_the_refresh_starts(world):
    world.ready()
    seen = []

    def refresh():
        seen.append(classify.calls)
        return "x"

    classify = Recorder("report")
    make(world, refresh, classify=classify).run_pass()
    assert seen == [1]
