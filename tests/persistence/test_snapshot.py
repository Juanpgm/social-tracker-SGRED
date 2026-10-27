"""`Database.snapshot()` / `SqliteUnitOfWork.snapshot()`: a consistent read view that never takes
the write lock, so readers coexist with writers (WAL) and see one point in time."""

import sqlite3
import threading
from pathlib import Path

import pytest

from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.errors import StorageError
from tests.shared.uow_contract import Env, UnitOfWorkContract


class TestSqliteUnitOfWorkContract(UnitOfWorkContract):
    """The unit-of-work contract, which only the in-memory fake ran before, on real SQLite."""

    @pytest.fixture(autouse=True)
    def _databases(self, tmp_path):
        self._root, self._opened = tmp_path, []
        yield
        for db in self._opened:
            db.close()

    def make_env(self) -> Env:
        db = Database(self._root / f"uow-{len(self._opened)}.db")
        self._opened.append(db)
        return Env(SqliteUnitOfWork(db), SqliteSourceRepository(db), SqliteControlRepository(db))


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "snap.db"


@pytest.fixture
def db(path: Path):
    with Database(path) as database:
        with database.transaction() as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
        yield database


def values(database: Database) -> list[int]:
    return [row[0] for row in database.query("SELECT x FROM t ORDER BY x")]


def test_every_read_in_a_snapshot_sees_the_same_point_in_time(db: Database, path: Path):
    with Database(path) as writer, db.snapshot():
        assert values(db) == [1]
        with writer.transaction() as conn:
            conn.execute("INSERT INTO t VALUES (2)")  # committed while our snapshot is open
        assert values(writer) == [1, 2]
        assert values(db) == [1]  # the snapshot does not move
    assert values(db) == [1, 2]  # and the next read sees it


def test_a_snapshot_does_not_block_writers_in_other_connections(db: Database, path: Path):
    with Database(path, busy_timeout_ms=100) as writer, db.snapshot():
        values(db)
        with writer.transaction() as conn:
            conn.execute("INSERT INTO t VALUES (3)")


def test_a_snapshot_reads_while_another_connection_holds_the_write_lock(db: Database, path: Path):
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("INSERT INTO t VALUES (9)")
    try:
        with db.snapshot():
            assert values(db) == [1]  # the uncommitted row is invisible, and nothing waits
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def test_a_write_inside_a_snapshot_is_refused_and_the_snapshot_survives(db: Database):
    with db.snapshot():
        with pytest.raises(StorageError) as caught, db.transaction():
            pytest.fail("the write transaction must not start")
        assert "snapshot" in caught.value.message
        assert values(db) == [1]
    assert values(db) == [1]


def test_the_connection_is_back_to_autocommit_after_a_snapshot_even_after_an_error(
    db: Database,
):
    with db.snapshot() as conn:
        assert conn.in_transaction
    assert not conn.in_transaction
    with pytest.raises(RuntimeError), db.snapshot():
        raise RuntimeError("boom")
    assert not conn.in_transaction
    with db.transaction() as inner:  # writing works again
        inner.execute("INSERT INTO t VALUES (2)")
    assert values(db) == [1, 2]


def test_a_snapshot_inside_a_write_transaction_joins_it(db: Database):
    with db.transaction() as conn:
        conn.execute("INSERT INTO t VALUES (2)")
        with db.snapshot():
            assert values(db) == [1, 2]  # it sees the transaction's own uncommitted write
        assert conn.in_transaction  # leaving the snapshot did not end the transaction
        conn.execute("INSERT INTO t VALUES (3)")
    assert values(db) == [1, 2, 3]


def test_nested_snapshots_share_the_outer_one(db: Database, path: Path):
    with Database(path) as writer, db.snapshot() as outer:
        values(db)
        with db.snapshot() as inner:
            assert inner is outer
            with writer.transaction() as conn:
                conn.execute("INSERT INTO t VALUES (2)")
        assert outer.in_transaction  # the inner exit did not end the outer snapshot
        assert values(db) == [1]


def test_a_snapshot_on_a_closed_database_is_a_storage_error(path: Path):
    database = Database(path)
    database.close()
    with pytest.raises(StorageError), database.snapshot():
        pytest.fail("must not start")


def test_another_thread_waits_for_the_snapshot_to_end(db: Database):
    inside, release = threading.Event(), threading.Event()
    order: list[str] = []

    def owner() -> None:
        with db.snapshot():
            inside.set()
            release.wait(5)
            order.append("owner-done")

    def other() -> None:
        inside.wait(5)
        with db.transaction() as conn:
            order.append("other-entered")
            conn.execute("INSERT INTO t VALUES (2)")

    first, second = threading.Thread(target=owner), threading.Thread(target=other)
    first.start()
    second.start()
    assert inside.wait(5)
    second.join(0.3)
    assert second.is_alive()
    release.set()
    first.join(5)
    second.join(5)
    assert order == ["owner-done", "other-entered"]
