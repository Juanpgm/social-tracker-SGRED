"""Collector behaviour that only SQLite can show: a locked database and several connections."""

import sqlite3
import threading
from datetime import timedelta

import pytest

from social_tracker.collector.service import CollectOutcome
from social_tracker.control.domain import CollectionState
from social_tracker.persistence.db import Database
from social_tracker.shared.errors import StorageError
from tests.collector.harness import FEED, FakeFetcher, sqlite_world


@pytest.fixture
def locked_world(tmp_path):
    world = sqlite_world(tmp_path / "locked.db", FakeFetcher(), busy_timeout_ms=50)
    world.ready()
    yield world
    world.close()


def blocker(world) -> sqlite3.Connection:
    connection = sqlite3.connect(world.db.path, isolation_level=None)
    connection.execute("BEGIN IMMEDIATE")
    return connection


def test_a_database_locked_at_the_start_changes_nothing_and_recovers(locked_world):
    other = blocker(locked_world)
    try:
        with pytest.raises(StorageError):
            locked_world.collector.collect_once("src_news")
    finally:
        other.execute("ROLLBACK")
        other.close()
    assert locked_world.fetcher.requests == []
    assert locked_world.state().state is CollectionState.IDLE
    assert locked_world.collector.collect_once("src_news").outcome is CollectOutcome.COLLECTED


def test_a_database_locked_while_fetching_stores_nothing_and_the_leftover_is_recovered(
    locked_world,
):
    holder: list[sqlite3.Connection] = []

    def lock_then_answer(request):
        holder.append(blocker(locked_world))
        return FEED

    locked_world.fetcher.then(lock_then_answer)
    try:
        with pytest.raises(StorageError):  # neither the items nor the failure could be written
            locked_world.collector.collect_once("src_news")
    finally:
        holder[0].execute("ROLLBACK")
        holder[0].close()
    assert locked_world.items.count() == 0
    assert (
        locked_world.state().state is CollectionState.COLLECTING
    )  # stuck, as a crash would leave it
    assert locked_world.collector.recover_interrupted() == ["src_news"]
    assert locked_world.state().state is CollectionState.BACKOFF
    locked_world.wait_out_backoff()
    locked_world.fetcher.then(FEED)
    assert locked_world.collector.collect_once("src_news").outcome is CollectOutcome.COLLECTED
    assert locked_world.items.count() == 2


def test_two_processes_collecting_the_same_source_fetch_once(tmp_path):
    path = tmp_path / "shared.db"
    first_fetcher, second_fetcher = FakeFetcher(), FakeFetcher()
    first = sqlite_world(path, first_fetcher)
    first.ready()
    second = sqlite_world(path, second_fetcher)
    inside, release = threading.Event(), threading.Event()

    def hold(request):
        inside.set()
        assert release.wait(timeout=10)
        return FEED

    first_fetcher.then(hold)
    results = []
    worker = threading.Thread(
        target=lambda: results.append(first.collector.collect_once("src_news"))
    )
    try:
        worker.start()
        assert inside.wait(timeout=10)
        skipped = second.collector.collect_once("src_news")
        release.set()
        worker.join(timeout=10)
        assert skipped.outcome is CollectOutcome.SKIPPED_NOT_ELIGIBLE
        assert second_fetcher.requests == []
        assert [r.outcome for r in results] == [CollectOutcome.COLLECTED]
        assert second.items.count() == 2
    finally:
        release.set()
        first.close()
        second.close()


def test_collected_items_and_the_fetch_status_survive_a_restart(tmp_path):
    path = tmp_path / "restart.db"
    world = sqlite_world(path, FakeFetcher())
    world.ready()
    world.collector.collect_once("src_news")
    world.close()
    with Database(path) as reopened:
        assert reopened.query("SELECT count(*) FROM items")[0][0] == 2
        row = reopened.query("SELECT last_fetch_at, last_error FROM source_fetch_status")[0]
        assert row["last_error"] is None
        assert row["last_fetch_at"] is not None


def test_the_clock_decides_the_fetch_time_stored(tmp_path):
    world = sqlite_world(tmp_path / "t.db", FakeFetcher())
    try:
        world.ready()
        world.clock.advance(timedelta(hours=3))
        world.collector.collect_once("src_news")
        assert {i.fetched_at for i in world.items.list()} == {world.clock.now()}
    finally:
        world.close()
