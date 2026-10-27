"""A5: every pass of the worker leaves its time in the settings store, with analysis on or off,
so that an operator page can tell a working system from a silent one."""

from datetime import timedelta

from social_tracker.shared.errors import StorageError
from tests.collector.harness import FakeFetcher, memory_world
from tests.collector.test_worker import Rig


def test_a_pass_with_analysis_off_still_records_that_the_worker_is_alive():
    rig = Rig(memory_world(FakeFetcher()))
    assert rig.world.control.liveness().last_pass_at is None
    summary = rig.worker.run_pass()
    assert summary.global_off is True
    assert rig.world.control.liveness().last_pass_at == rig.world.clock.now()


def test_a_pass_with_analysis_on_records_it_and_every_pass_moves_it_forward():
    rig = Rig(memory_world(FakeFetcher()))
    rig.world.control.set_global_analysis(True, actor="tester", reason="go")
    rig.worker.run_pass()
    first = rig.world.control.liveness().last_pass_at
    rig.world.clock.advance(timedelta(seconds=5))
    rig.worker.run_pass()
    assert rig.world.control.liveness().last_pass_at == first + timedelta(seconds=5)


def test_a_failure_to_record_never_stops_the_worker(monkeypatch):
    rig = Rig(memory_world(FakeFetcher()))

    def broken():
        raise StorageError("disk full")

    monkeypatch.setattr(rig.world.control, "record_worker_pass", broken)
    summary = rig.worker.run_pass()  # does not raise
    assert summary.global_off is True


def test_a_pass_skipped_by_a_storage_error_is_still_recorded(monkeypatch):
    rig = Rig(memory_world(FakeFetcher()))
    rig.world.control.set_global_analysis(True, actor="tester", reason="go")

    def broken():
        raise StorageError("locked")

    monkeypatch.setattr(rig.world.collector, "due_sources", broken)
    summary = rig.worker.run_pass()
    assert summary.errors == 1
    # the worker did run: the pass is recorded even though it found the store unavailable
    assert rig.world.control.liveness().last_pass_at == rig.world.clock.now()
