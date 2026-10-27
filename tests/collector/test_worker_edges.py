"""Worker details: recovery on every pass, error classes in the log, and the real wait."""

import logging
import threading
import time

from social_tracker.collector.worker import Worker
from social_tracker.control.domain import CollectionState
from social_tracker.shared.errors import NotFoundError
from tests.collector.harness import FakeFetcher, memory_world
from tests.collector.test_worker import Rig


def test_every_pass_recovers_a_collection_that_a_crash_left_running():
    rig = Rig(memory_world(FakeFetcher()))
    rig.world.ready()
    rig.world.control.start_collection("src_news", actor="system:worker", reason="crashed run")
    rig.worker.run_pass()
    assert rig.world.state().state is CollectionState.BACKOFF


def test_a_source_that_vanishes_between_planning_and_collecting_is_logged_by_its_code(caplog):
    rig = Rig(memory_world(FakeFetcher()))
    rig.world.ready()

    def gone(source_id):
        raise NotFoundError("source not found")

    rig.world.collector.collect_once = gone  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO, logger="social_tracker"):
        summary = rig.worker.run_pass()
    assert (summary.results, summary.errors) == ((), 1)
    assert "not_found" in caplog.text


def test_by_default_the_worker_waits_on_the_stop_flag_between_passes():
    world = memory_world(FakeFetcher())
    stop = threading.Event()
    worker = Worker(world.collector, world.control, world.clock, stop, tick_s=30.0)
    passes = []
    real_pass = worker.run_pass

    def counting_pass():
        passes.append(1)
        return real_pass()

    worker.run_pass = counting_pass  # type: ignore[method-assign]
    timer = threading.Timer(0.3, stop.set)
    timer.start()
    started = time.monotonic()
    try:
        worker.run()
    finally:
        timer.cancel()
    assert len(passes) == 1  # one pass, then a 30 s wait that the stop flag cut short
    assert 0.2 < time.monotonic() - started < 10
