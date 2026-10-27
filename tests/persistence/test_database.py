"""Database module: connection setup, transactions, error mapping and the migration runner."""

import hashlib
import os
import re
import sqlite3
import stat
import threading
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from social_tracker.persistence import db as db_module
from social_tracker.persistence.db import (
    Database,
    Migration,
    format_utc,
    load_migrations,
    parse_utc,
)
from social_tracker.shared.errors import ConflictError, StorageError

COUNTER = Migration(
    1,
    "counter",
    "CREATE TABLE counter (id INTEGER PRIMARY KEY, n INTEGER NOT NULL);"
    "INSERT INTO counter VALUES (1, 0);",
)
DEFERRED_FK = Migration(
    1,
    "fk",
    "CREATE TABLE parent (id INTEGER PRIMARY KEY);"
    "CREATE TABLE child (id INTEGER PRIMARY KEY, "
    "parent_id INTEGER REFERENCES parent (id) DEFERRABLE INITIALLY DEFERRED);",
)
UNIQUE_T = Migration(
    1,
    "t",
    "CREATE TABLE t (x TEXT UNIQUE, y INTEGER CHECK (y > 0), z INTEGER REFERENCES p(id))"
    "; CREATE TABLE p (id INTEGER PRIMARY KEY);",
)


@pytest.fixture
def opened():
    """Track every Database a test opens so Windows can delete the files afterwards."""
    dbs: list[Database] = []

    def open_(path, **kwargs) -> Database:
        db = Database(path, **kwargs)
        dbs.append(db)
        return db

    yield open_
    for db in dbs:
        db.close()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions(db: Database) -> list[int]:
    return [r["version"] for r in db.query("SELECT version FROM schema_migrations ORDER BY 1")]


def raw_versions(path: Path) -> list[int]:
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY 1")]
    finally:
        conn.close()


def write_then_raise(db: Database, error: BaseException) -> None:
    with db.transaction() as conn:
        conn.execute("UPDATE counter SET n = 7")
        raise error


def insert_good_then_bad(db: Database) -> None:
    with db.transaction() as conn:
        conn.execute("INSERT INTO t (x, y) VALUES ('ok', 1)")
        conn.execute("INSERT INTO t (x, y) VALUES ('bad', -1)")


# ---- opening ----


def test_open_creates_missing_parent_directories_and_the_file(tmp_path, opened):
    path = tmp_path / "a" / "b" / "c" / "app.db"
    opened(path)
    assert path.is_file()


def test_accepts_str_paths(tmp_path, opened):
    db = opened(str(tmp_path / "s.db"))
    assert db.path == tmp_path / "s.db"


def test_pragmas_are_applied(tmp_path, opened):
    db = opened(tmp_path / "p.db")
    pragma = {
        name: db.query(f"PRAGMA {name}")[0][0]
        for name in ("journal_mode", "busy_timeout", "foreign_keys", "synchronous")
    }
    assert pragma == {
        "journal_mode": "wal",
        "busy_timeout": 5000,
        "foreign_keys": 1,
        "synchronous": 1,  # NORMAL
    }


def test_busy_timeout_is_configurable(tmp_path, opened):
    db = opened(tmp_path / "p.db", busy_timeout_ms=123)
    assert db.query("PRAGMA busy_timeout")[0][0] == 123


@pytest.mark.parametrize("bad", [-1, 0, True, "5", None, 1.5])
def test_invalid_busy_timeout_is_rejected(tmp_path, bad):
    with pytest.raises(StorageError):
        Database(tmp_path / "p.db", busy_timeout_ms=bad)


def test_foreign_keys_are_really_enforced(tmp_path, opened):
    db = opened(tmp_path / "fk.db", migrations=[UNIQUE_T])
    with pytest.raises(StorageError), db.transaction() as conn:
        conn.execute("INSERT INTO t (x, y, z) VALUES ('a', 1, 999)")


def test_zero_byte_file_is_treated_as_an_empty_database(tmp_path, opened):
    path = tmp_path / "empty.db"
    path.write_bytes(b"")
    db = opened(path)
    assert versions(db) == [1]


def test_path_that_is_a_directory_is_a_storage_error(tmp_path):
    with pytest.raises(StorageError) as info:
        Database(tmp_path)
    assert info.value.code == "storage_error"


def test_parent_that_is_a_file_is_a_storage_error(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    with pytest.raises(StorageError) as info:
        Database(blocker / "app.db")
    assert info.value.code == "storage_error"
    assert blocker.read_text() == "x"


@pytest.mark.skipif(os.name == "nt", reason="directory permissions are not enforced on Windows")
def test_read_only_directory_is_a_storage_error(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running with privileges that ignore directory permissions")
        with pytest.raises(StorageError) as info:
            Database(locked / "app.db")
        assert info.value.code == "storage_error"
    finally:
        locked.chmod(stat.S_IRWXU)


@pytest.mark.parametrize("content", [os.urandom(4096), b"not a database at all", b"\x00" * 7])
def test_corrupted_file_is_a_storage_error_and_is_never_modified(tmp_path, content):
    path = tmp_path / "corrupt.db"
    path.write_bytes(content)
    before = sha(path)
    with pytest.raises(StorageError) as info:
        Database(path)
    assert info.value.code == "storage_error"
    assert sha(path) == before
    assert path.read_bytes() == content


def test_error_messages_do_not_echo_file_contents_or_paths(tmp_path):
    path = tmp_path / "secret-location" / "corrupt.db"
    path.parent.mkdir()
    path.write_bytes(b"TOP-SECRET-PAYLOAD " * 50)
    with pytest.raises(StorageError) as info:
        Database(path)
    text = str(info.value)
    assert "TOP-SECRET" not in text
    assert "secret-location" not in text


def test_corrupt_page_inside_a_valid_database_is_a_storage_error(tmp_path, opened):
    path = tmp_path / "damaged.db"
    db = opened(path)
    db.close()
    data = bytearray(path.read_bytes())
    data[0:16] = b"X" * 16  # destroy the magic header
    path.write_bytes(bytes(data))
    with pytest.raises(StorageError):
        Database(path)


def _hold_a_read_lock_on_a_rollback_journal_file(path: Path) -> sqlite3.Connection:
    """Leaves the file in rollback-journal mode with a reader that blocks the switch to WAL."""
    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    holder.execute("CREATE TABLE seed (x)")
    holder.execute("BEGIN")
    holder.execute("SELECT * FROM seed").fetchall()
    return holder


def test_switching_to_wal_waits_for_a_short_lived_reader(tmp_path, opened):
    path = tmp_path / "wal.db"
    holder = _hold_a_read_lock_on_a_rollback_journal_file(path)
    timer = threading.Timer(0.3, lambda: (holder.execute("COMMIT"), holder.close()))
    timer.start()
    try:
        db = opened(path)
    finally:
        timer.join(timeout=5)
    assert db.query("PRAGMA journal_mode")[0][0] == "wal"


def test_switching_to_wal_gives_up_after_the_busy_timeout(tmp_path):
    path = tmp_path / "wal.db"
    holder = _hold_a_read_lock_on_a_rollback_journal_file(path)
    try:
        started = time.monotonic()
        with pytest.raises(StorageError) as info:
            Database(path, busy_timeout_ms=300)
        elapsed = time.monotonic() - started
    finally:
        holder.execute("COMMIT")
        holder.close()
    assert "SQLITE_BUSY" in str(info.value)
    assert 0.25 <= elapsed < 5  # it waited for the timeout instead of failing at once


def test_a_database_that_cannot_use_wal_is_rejected_and_not_leaked(tmp_path, monkeypatch):
    real_connect = sqlite3.connect

    class NoWal:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql, *args):
            if sql.startswith("PRAGMA journal_mode"):
                return self._conn.execute("SELECT 'delete'")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: NoWal(real_connect(*a, **k)))
    path = tmp_path / "nowal.db"
    with pytest.raises(StorageError) as info:
        Database(path)
    assert "write-ahead" in str(info.value)
    monkeypatch.undo()
    path.unlink()  # the rejected connection was closed


class _FaultyConnection:
    """Wraps a real connection: records close() and raises `error` on matching statements."""

    def __init__(self, conn: sqlite3.Connection, fail_on: str, error: BaseException) -> None:
        self._conn = conn
        self._fail_on = fail_on
        self._error = error
        self.closed = False

    def execute(self, sql, *args):
        if self._fail_on in sql:
            raise self._error
        return self._conn.execute(sql, *args)

    def close(self) -> None:
        self.closed = True
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.mark.parametrize(
    ("fail_on", "error", "expected"),
    [
        ("PRAGMA synchronous", sqlite3.OperationalError("disk I/O error"), StorageError),
        ("PRAGMA foreign_keys", sqlite3.DatabaseError("malformed"), StorageError),
        (
            "CREATE TABLE IF NOT EXISTS schema_migrations",
            sqlite3.OperationalError("x"),
            StorageError,
        ),
        ("PRAGMA synchronous", KeyboardInterrupt(), KeyboardInterrupt),
        ("CREATE TABLE IF NOT EXISTS schema_migrations", MemoryError(), MemoryError),
    ],
    ids=[
        "sqlite-error-in-pragma",
        "sqlite-error-late",
        "sqlite-error-in-runner",
        "interrupt",
        "oom",
    ],
)
def test_any_failure_while_opening_closes_the_connection(
    tmp_path, monkeypatch, fail_on, error, expected
):
    real_connect = sqlite3.connect
    created: list[_FaultyConnection] = []

    def connect(*args, **kwargs):
        faulty = _FaultyConnection(real_connect(*args, **kwargs), fail_on, error)
        created.append(faulty)
        return faulty

    monkeypatch.setattr(sqlite3, "connect", connect)
    with pytest.raises(expected):
        Database(tmp_path / "leak.db")
    assert len(created) == 1
    assert created[0].closed, "the connection leaked when opening failed"


# ---- lifecycle ----


def test_close_is_idempotent(tmp_path):
    db = Database(tmp_path / "c.db")
    db.close()
    db.close()


def test_use_after_close_is_a_storage_error(tmp_path):
    db = Database(tmp_path / "c.db")
    db.close()
    with pytest.raises(StorageError) as info:
        db.query("SELECT 1")
    assert info.value.code == "storage_error"
    with pytest.raises(StorageError), db.transaction():
        pytest.fail("the body must not run on a closed database")


def test_context_manager_closes(tmp_path):
    with Database(tmp_path / "c.db") as db:
        assert db.query("SELECT 1")[0][0] == 1
    with pytest.raises(StorageError):
        db.query("SELECT 1")


def test_closing_releases_the_file(tmp_path):
    path = tmp_path / "c.db"
    Database(path).close()
    path.unlink()  # would fail on Windows if the handle were still open
    assert not path.exists()


def _wal_size(path: Path) -> int:
    wal = Path(str(path) + "-wal")
    return wal.stat().st_size if wal.exists() else 0


def test_close_truncates_the_wal_even_when_another_connection_is_still_open(tmp_path):
    path = tmp_path / "w.db"
    db = Database(path, migrations=[LOG])
    with db.transaction() as conn:
        for n in range(50):
            conn.execute("INSERT INTO log (msg) VALUES (?)", (f"row {n}",))
    bystander = sqlite3.connect(path)  # keeps the WAL alive: SQLite only cleans up on last close
    try:
        bystander.execute("SELECT count(*) FROM log").fetchone()
        assert _wal_size(path) > 0
        db.close()
        assert _wal_size(path) == 0
        assert bystander.execute("SELECT count(*) FROM log").fetchone()[0] == 50
    finally:
        bystander.close()


def test_close_does_not_fail_when_the_checkpoint_cannot_run(tmp_path):
    db = Database(tmp_path / "w.db", migrations=[LOG])
    faulty = _FaultyConnection(db._conn, "wal_checkpoint", sqlite3.OperationalError("locked"))
    db._conn = faulty  # type: ignore[assignment]
    db.close()
    assert faulty.closed
    with pytest.raises(StorageError):
        db.query("SELECT 1")
    db.close()  # still idempotent


def test_close_does_not_fail_while_another_connection_blocks_the_checkpoint(tmp_path):
    path = tmp_path / "w.db"
    db = Database(path, migrations=[LOG])
    with db.transaction() as conn:
        conn.execute("INSERT INTO log (msg) VALUES ('a')")
    reader = sqlite3.connect(path, isolation_level=None)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM log").fetchone()  # holds a read snapshot
        with db.transaction() as conn:
            conn.execute("INSERT INTO log (msg) VALUES ('b')")
        db.close()  # TRUNCATE cannot finish: it must give up quietly
    finally:
        reader.close()
    with Database(path, migrations=[LOG]) as again:
        assert logged(again) == ["a", "b"]


def test_closing_inside_an_open_transaction_rolls_it_back_and_reports_it(tmp_path):
    path = tmp_path / "w.db"
    db = Database(path, migrations=[LOG])
    with pytest.raises(StorageError) as info, db.transaction() as conn:  # noqa: PT012 - the body is the scenario
        conn.execute("INSERT INTO log (msg) VALUES ('lost')")
        db.close()
    assert "closed" in str(info.value)
    with Database(path, migrations=[LOG]) as again:
        assert logged(again) == []


def test_two_instances_on_one_file_see_each_others_writes(tmp_path, opened):
    first = opened(tmp_path / "w.db", migrations=[COUNTER])
    second = opened(tmp_path / "w.db", migrations=[COUNTER])
    with first.transaction() as conn:
        conn.execute("UPDATE counter SET n = 41")
    assert second.query("SELECT n FROM counter")[0]["n"] == 41
    with second.transaction() as conn:
        conn.execute("UPDATE counter SET n = 42")
    assert first.query("SELECT n FROM counter")[0]["n"] == 42


def test_query_rows_are_addressable_by_name_and_params_are_bound(tmp_path, opened):
    db = opened(tmp_path / "q.db", migrations=[COUNTER])
    rows = db.query("SELECT n AS total FROM counter WHERE id = ?", (1,))
    assert rows[0]["total"] == 0
    assert db.query("SELECT n FROM counter WHERE id = ?", (2,)) == []


def test_query_with_unbindable_parameter_is_a_storage_error(tmp_path, opened):
    db = opened(tmp_path / "q.db", migrations=[COUNTER])
    with pytest.raises(StorageError):
        db.query("SELECT n FROM counter WHERE id = ?", ([1],))


def test_query_with_broken_sql_is_a_storage_error(tmp_path, opened):
    db = opened(tmp_path / "q.db")
    with pytest.raises(StorageError) as info:
        db.query("SELEC nothing")
    assert "SELEC" not in str(info.value)


# ---- transactions ----


def test_transaction_commits_on_success(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[COUNTER])
    with db.transaction() as conn:
        conn.execute("UPDATE counter SET n = 7")
    assert db.query("SELECT n FROM counter")[0]["n"] == 7


def test_transaction_rolls_back_on_exception_and_reraises_the_same_one(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[COUNTER])
    boom = RuntimeError("boom")
    with pytest.raises(RuntimeError) as info:
        write_then_raise(db, boom)
    assert info.value is boom
    assert db.query("SELECT n FROM counter")[0]["n"] == 0


def test_transaction_rolls_back_on_keyboard_interrupt(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[COUNTER])
    with pytest.raises(KeyboardInterrupt):
        write_then_raise(db, KeyboardInterrupt())
    assert db.query("SELECT n FROM counter")[0]["n"] == 0
    with db.transaction() as conn:  # the lock was released and no transaction is left open
        conn.execute("UPDATE counter SET n = 1")


def test_unique_violation_in_a_transaction_is_a_conflict_error(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[UNIQUE_T])
    with db.transaction() as conn:
        conn.execute("INSERT INTO t (x, y) VALUES ('a', 1)")
    with pytest.raises(ConflictError) as info, db.transaction() as conn:
        conn.execute("INSERT INTO t (x, y) VALUES ('a', 2)")
    assert info.value.code == "conflict"
    assert "'a'" not in str(info.value)
    assert [r["y"] for r in db.query("SELECT y FROM t")] == [1]


def test_check_violation_in_a_transaction_is_a_storage_error_not_a_conflict(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[UNIQUE_T])
    with pytest.raises(StorageError) as info, db.transaction() as conn:
        conn.execute("INSERT INTO t (x, y) VALUES ('a', -5)")
    assert not isinstance(info.value, ConflictError)
    assert info.value.code == "storage_error"
    assert db.query("SELECT count(*) FROM t")[0][0] == 0


def test_a_failed_statement_rolls_back_the_whole_transaction(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[UNIQUE_T])
    with pytest.raises(StorageError):
        insert_good_then_bad(db)
    assert db.query("SELECT count(*) FROM t")[0][0] == 0


# ---- nested (joined) transactions ----

LOG = Migration(
    1,
    "log",
    "CREATE TABLE log (id INTEGER PRIMARY KEY, msg TEXT UNIQUE);"
    "CREATE TABLE guarded (x INTEGER);"
    "CREATE TRIGGER guarded_abort BEFORE INSERT ON guarded WHEN NEW.x < 0 "
    "BEGIN SELECT RAISE(ROLLBACK, 'whole transaction rolled back'); END;",
)


def logged(db: Database) -> list[str]:
    return [r["msg"] for r in db.query("SELECT msg FROM log ORDER BY id")]


def test_a_nested_transaction_joins_the_outer_one_and_commits_with_it(tmp_path, opened):
    db = opened(tmp_path / "j.db", migrations=[LOG])
    with db.transaction() as outer:
        outer.execute("INSERT INTO log (msg) VALUES ('outer')")
        with db.transaction() as inner:
            assert inner is outer  # one connection, one transaction
            inner.execute("INSERT INTO log (msg) VALUES ('inner')")
        assert db.query("SELECT count(*) FROM log")[0][0] == 2  # visible to the outer code
    assert logged(db) == ["outer", "inner"]


def test_rolling_back_the_outer_transaction_undoes_the_joined_inner_one(tmp_path, opened):
    db = opened(tmp_path / "j.db", migrations=[LOG])
    with pytest.raises(RuntimeError), db.transaction() as outer:  # noqa: PT012 - the body is the scenario
        outer.execute("INSERT INTO log (msg) VALUES ('outer')")
        with db.transaction() as inner:
            inner.execute("INSERT INTO log (msg) VALUES ('inner')")
        raise RuntimeError("late failure")
    assert logged(db) == []


def test_an_inner_failure_caught_by_the_outer_code_keeps_only_the_outer_writes(tmp_path, opened):
    db = opened(tmp_path / "j.db", migrations=[LOG])
    with db.transaction() as outer:
        outer.execute("INSERT INTO log (msg) VALUES ('outer')")
        with pytest.raises(ConflictError), db.transaction() as inner:  # noqa: PT012 - the body is the scenario
            inner.execute("INSERT INTO log (msg) VALUES ('partial')")
            inner.execute("INSERT INTO log (msg) VALUES ('outer')")  # unique violation
        outer.execute("INSERT INTO log (msg) VALUES ('after')")
    assert logged(db) == ["outer", "after"]


def test_an_inner_failure_that_is_not_caught_rolls_everything_back(tmp_path, opened):
    db = opened(tmp_path / "j.db", migrations=[LOG])
    boom = RuntimeError("boom")
    with pytest.raises(RuntimeError) as info, db.transaction() as outer:  # noqa: PT012 - the body is the scenario
        outer.execute("INSERT INTO log (msg) VALUES ('outer')")
        with db.transaction() as inner:
            inner.execute("INSERT INTO log (msg) VALUES ('inner')")
            raise boom
    assert info.value is boom
    assert logged(db) == []


def test_a_keyboard_interrupt_in_a_nested_transaction_undoes_only_the_inner_part(tmp_path, opened):
    db = opened(tmp_path / "j.db", migrations=[LOG])
    with db.transaction() as outer:
        outer.execute("INSERT INTO log (msg) VALUES ('outer')")
        with pytest.raises(KeyboardInterrupt), db.transaction() as inner:  # noqa: PT012 - the body is the scenario
            inner.execute("INSERT INTO log (msg) VALUES ('inner')")
            raise KeyboardInterrupt
    assert logged(db) == ["outer"]


def test_nesting_works_several_levels_deep_and_for_siblings(tmp_path, opened):
    db = opened(tmp_path / "j.db", migrations=[LOG])
    with db.transaction() as conn:
        with db.transaction():
            conn.execute("INSERT INTO log (msg) VALUES ('a')")
            with db.transaction():
                conn.execute("INSERT INTO log (msg) VALUES ('b')")
                with pytest.raises(RuntimeError), db.transaction():  # noqa: PT012 - the body is the scenario
                    conn.execute("INSERT INTO log (msg) VALUES ('lost')")
                    raise RuntimeError
        with db.transaction():  # a sibling reuses the savepoint name after the first was released
            conn.execute("INSERT INTO log (msg) VALUES ('c')")
    assert logged(db) == ["a", "b", "c"]


def test_a_failure_that_makes_sqlite_roll_back_everything_surfaces_at_the_outer_commit(
    tmp_path, opened
):
    db = opened(tmp_path / "j.db", migrations=[LOG])
    with pytest.raises(StorageError), db.transaction() as outer:  # noqa: PT012 - the body is the scenario
        outer.execute("INSERT INTO log (msg) VALUES ('outer')")
        with pytest.raises(StorageError), db.transaction() as inner:
            inner.execute("INSERT INTO guarded VALUES (-1)")  # RAISE(ROLLBACK) in the trigger
    assert logged(db) == []
    with db.transaction() as conn:  # the connection is usable and not stuck in a transaction
        conn.execute("INSERT INTO log (msg) VALUES ('fresh')")
    assert logged(db) == ["fresh"]


def test_depth_bookkeeping_resets_after_exceptions(tmp_path, opened):
    path = tmp_path / "j.db"
    db = opened(path, migrations=[LOG])
    other = opened(path, migrations=[LOG], busy_timeout_ms=100)
    for _ in range(3):
        with pytest.raises(RuntimeError), db.transaction(), db.transaction():
            raise RuntimeError
    # A new top-level transaction is a real BEGIN IMMEDIATE: it excludes other connections.
    with db.transaction() as conn:  # no write yet: only BEGIN IMMEDIATE already holds the lock
        with pytest.raises(StorageError) as info, other.transaction():
            pytest.fail("the other connection must not get the write lock")
        assert "SQLITE_BUSY" in str(info.value)
        conn.execute("INSERT INTO log (msg) VALUES ('mine')")
    assert logged(db) == ["mine"]


def test_a_failed_outer_commit_resets_the_depth_too(tmp_path, opened):
    db = opened(tmp_path / "j.db", migrations=[DEFERRED_FK])
    with pytest.raises(StorageError), db.transaction() as conn:
        conn.execute("INSERT INTO child (parent_id) VALUES (99)")  # violated at COMMIT
    assert db.query("SELECT count(*) FROM child")[0][0] == 0
    with db.transaction() as conn:  # a plain top-level transaction again, not a savepoint
        conn.execute("INSERT INTO parent (id) VALUES (1)")
        assert conn.in_transaction
    assert db.query("SELECT count(*) FROM parent")[0][0] == 1


def test_another_thread_does_not_join_a_transaction_it_does_not_own(tmp_path, opened):
    db = opened(tmp_path / "j.db", migrations=[LOG])
    holding, finished = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def other_thread():
        try:
            holding.wait(timeout=5)
            with db.transaction() as conn:  # must wait for the owner, not join it
                conn.execute("INSERT INTO log (msg) VALUES ('other')")
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=other_thread)
    thread.start()
    with pytest.raises(RuntimeError), db.transaction() as conn:  # noqa: PT012 - the body is the scenario
        conn.execute("INSERT INTO log (msg) VALUES ('owner')")
        holding.set()
        assert not finished.wait(timeout=0.3)  # blocked while the owner's transaction is open
        raise RuntimeError("owner rolls back")
    thread.join(timeout=10)
    assert errors == []
    # The other thread's own transaction committed; the owner's rollback did not touch it.
    assert logged(db) == ["other"]


def test_nested_use_after_close_is_a_storage_error(tmp_path):
    db = Database(tmp_path / "j.db", migrations=[LOG])
    with pytest.raises(StorageError), db.transaction():  # noqa: PT012 - the body is the scenario
        db.close()
        with db.transaction():
            pytest.fail("the body must not run on a closed database")


def test_a_reader_thread_never_sees_an_uncommitted_row_that_is_later_rolled_back(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[LOG])
    written, reader_done = threading.Event(), threading.Event()
    seen: list[list[str]] = []

    def reader():
        written.wait(timeout=5)
        seen.append(logged(db))  # shares the connection: it must queue behind the writer
        reader_done.set()

    thread = threading.Thread(target=reader)
    thread.start()
    with pytest.raises(RuntimeError), db.transaction() as conn:  # noqa: PT012 - the body is the scenario
        conn.execute("INSERT INTO log (msg) VALUES ('dirty')")
        written.set()
        assert not reader_done.wait(timeout=0.3), "the reader ran during the open transaction"
        raise RuntimeError("roll back")
    thread.join(timeout=10)
    assert seen == [[]]


def test_a_reader_thread_sees_the_committed_state_once_the_writer_finishes(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[LOG])
    written = threading.Event()
    seen: list[list[str]] = []

    def reader():
        written.wait(timeout=5)
        seen.append(logged(db))

    thread = threading.Thread(target=reader)
    thread.start()
    with db.transaction() as conn:
        conn.execute("INSERT INTO log (msg) VALUES ('kept')")
        written.set()
        time.sleep(0.1)
    thread.join(timeout=10)
    assert seen == [["kept"]]


def test_a_failed_commit_leaves_the_connection_usable_and_outside_a_transaction(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[DEFERRED_FK])
    with pytest.raises(StorageError) as info, db.transaction() as conn:
        conn.execute("INSERT INTO child (parent_id) VALUES (99)")  # only COMMIT notices
    assert not isinstance(info.value, ConflictError)
    assert db.query("SELECT count(*) FROM child")[0][0] == 0  # rolled back, not half-open
    with db.transaction() as conn:  # BEGIN would fail inside a leftover transaction
        conn.execute("INSERT INTO parent (id) VALUES (1)")
        conn.execute("INSERT INTO child (parent_id) VALUES (1)")
    assert db.query("SELECT count(*) FROM child")[0][0] == 1


def test_threads_sharing_one_instance_never_lose_updates(tmp_path, opened):
    db = opened(tmp_path / "t.db", migrations=[COUNTER])
    threads_n, rounds = 8, 25
    barrier = threading.Barrier(threads_n)
    errors: list[BaseException] = []

    def worker():
        barrier.wait(timeout=5)
        try:
            for _ in range(rounds):
                with db.transaction() as conn:
                    current = conn.execute("SELECT n FROM counter").fetchone()[0]
                    conn.execute("UPDATE counter SET n = ?", (current + 1,))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    assert db.query("SELECT n FROM counter")[0]["n"] == threads_n * rounds


def test_separate_instances_on_one_file_serialize_through_sqlite(tmp_path, opened):
    path = tmp_path / "t.db"
    dbs = [opened(path, migrations=[COUNTER]) for _ in range(4)]
    rounds = 15
    barrier = threading.Barrier(len(dbs))
    errors: list[BaseException] = []

    def worker(db: Database):
        barrier.wait(timeout=5)
        try:
            for _ in range(rounds):
                with db.transaction() as conn:
                    current = conn.execute("SELECT n FROM counter").fetchone()[0]
                    conn.execute("UPDATE counter SET n = ?", (current + 1,))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(db,)) for db in dbs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    assert dbs[0].query("SELECT n FROM counter")[0]["n"] == len(dbs) * rounds


def test_a_write_blocked_beyond_the_busy_timeout_is_a_storage_error(tmp_path, opened):
    path = tmp_path / "busy.db"
    db = opened(path, migrations=[COUNTER], busy_timeout_ms=50)
    blocker = sqlite3.connect(path, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        with pytest.raises(StorageError) as info, db.transaction() as conn:
            conn.execute("UPDATE counter SET n = 1")
        assert time.monotonic() - started < 5
        assert info.value.code == "storage_error"
        assert "SQLITE_BUSY" in str(info.value)
        assert not isinstance(info.value, ConflictError)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    with db.transaction() as conn:  # recovers once the other writer is gone
        conn.execute("UPDATE counter SET n = 2")
    assert db.query("SELECT n FROM counter")[0]["n"] == 2


def test_a_write_waits_for_a_short_lived_competing_writer(tmp_path, opened):
    path = tmp_path / "busy.db"
    db = opened(path, migrations=[COUNTER])  # default busy timeout: 5 s
    blocker = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")
    timer = threading.Timer(0.3, lambda: (blocker.execute("COMMIT"), blocker.close()))
    timer.start()
    try:
        with db.transaction() as conn:
            conn.execute("UPDATE counter SET n = 3")
    finally:
        timer.join(timeout=5)
    assert db.query("SELECT n FROM counter")[0]["n"] == 3


# ---- migrations: runner ----


def test_fresh_database_gets_every_packaged_migration_recorded(tmp_path, opened):
    db = opened(tmp_path / "m.db")
    assert versions(db) == [m.version for m in load_migrations()]
    assert versions(db)[0] == 1
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sources", "source_state", "settings", "control_events", "schema_migrations"} <= tables


def test_applied_at_is_recorded_as_utc_text(tmp_path, opened):
    db = opened(tmp_path / "m.db")
    applied = db.query("SELECT applied_at FROM schema_migrations")[0]["applied_at"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00", applied)
    assert abs(parse_utc(applied) - datetime.now(UTC)) < timedelta(minutes=1)


def test_reopening_is_a_no_op(tmp_path):
    path = tmp_path / "m.db"
    first = Database(path)
    rows = first.query("SELECT version, applied_at FROM schema_migrations")
    snapshot = [tuple(r) for r in rows]
    first.close()
    time.sleep(0.01)
    second = Database(path)
    try:
        again = [
            tuple(r) for r in second.query("SELECT version, applied_at FROM schema_migrations")
        ]
    finally:
        second.close()
    assert again == snapshot


def test_only_missing_migrations_are_applied_on_upgrade(tmp_path, opened):
    path = tmp_path / "m.db"
    opened(path, migrations=[COUNTER]).close()
    second = Migration(2, "more", "CREATE TABLE extra (id INTEGER)")
    db = opened(path, migrations=[COUNTER, second])
    assert versions(db) == [1, 2]
    assert db.query("SELECT n FROM counter")[0]["n"] == 0  # migration 1 did not re-run


def test_failing_migration_rolls_back_and_is_not_recorded(tmp_path):
    path = tmp_path / "m.db"
    bad = Migration(2, "bad", "CREATE TABLE half (id INTEGER); INSERT INTO missing VALUES (1);")
    with pytest.raises(StorageError) as info:
        Database(path, migrations=[COUNTER, bad])
    assert info.value.code == "storage_error"
    assert "missing" not in str(info.value)
    assert raw_versions(path) == [1]  # migration 1 stays, migration 2 left no trace
    conn = sqlite3.connect(path)
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()
    assert "half" not in names


def test_a_fixed_migration_can_be_applied_after_a_failure(tmp_path, opened):
    path = tmp_path / "m.db"
    bad = Migration(2, "bad", "CREATE TABLE half (id INTEGER); SELECT * FROM missing;")
    with pytest.raises(StorageError):
        Database(path, migrations=[COUNTER, bad])
    good = Migration(2, "good", "CREATE TABLE half (id INTEGER)")
    db = opened(path, migrations=[COUNTER, good])
    assert versions(db) == [1, 2]


TX_CONTROL_SQL = [
    "COMMIT",
    "commit;",
    "BEGIN",
    "begin transaction;",
    "BEGIN IMMEDIATE",
    "END",
    "END TRANSACTION",
    "ROLLBACK",
    "ROLLBACK TO sp",
    "SAVEPOINT sp",
    "RELEASE sp",
    "RELEASE SAVEPOINT sp",
    "-- a comment\nCOMMIT",
    "/* a comment */ COMMIT",
    "/* a ; comment */\n  begin",
    "CREATE TABLE a (x); COMMIT; BEGIN; CREATE TABLE b (x);",
    "CREATE TABLE a (x); COMMIT",  # no trailing semicolon
    "CREATE TABLE a (x);\nCOMMIT;\nBEGIN;\nCREATE TABLE b (x);\nSELECT * FROM nope;",
    ("CREATE TABLE a (x);CREATE TRIGGER trg AFTER INSERT ON a BEGIN SELECT 1; END;COMMIT;"),
]


@pytest.mark.parametrize("sql", TX_CONTROL_SQL)
def test_a_migration_that_manages_transactions_is_rejected_before_anything_runs(tmp_path, sql):
    path = tmp_path / "m.db"
    with pytest.raises(StorageError) as info:
        Database(path, migrations=[Migration(1, "sneaky", sql)])
    assert info.value.code == "storage_error"
    assert "transaction" in str(info.value)
    assert not path.exists()  # rejected up front: the file was never opened


TX_LOOKALIKE_SQL = {
    "trigger body": (
        "CREATE TABLE a (x);"
        "CREATE TRIGGER trg AFTER INSERT ON a BEGIN SELECT 1; UPDATE a SET x = 2; END;"
    ),
    "string literal": "CREATE TABLE a (x); INSERT INTO a VALUES ('1; COMMIT; BEGIN; SAVEPOINT x');",
    "line comment": "CREATE TABLE a (x); -- COMMIT;\nINSERT INTO a VALUES (1);",
    "block comment": "CREATE TABLE a (x); /* ROLLBACK; */ INSERT INTO a VALUES (1);",
    "quoted identifiers": 'CREATE TABLE "commit" ("begin", "end", "release");',
    "trailing comment only": "CREATE TABLE a (x); -- done",
}


@pytest.mark.parametrize("sql", list(TX_LOOKALIKE_SQL.values()), ids=list(TX_LOOKALIKE_SQL))
def test_statements_that_merely_mention_transaction_keywords_are_accepted(tmp_path, opened, sql):
    db = opened(tmp_path / "m.db", migrations=[Migration(1, "ok", sql)])
    assert versions(db) == [1]


def test_a_trigger_migration_is_applied_and_the_trigger_works(tmp_path, opened):
    sql = (
        "CREATE TABLE a (x INTEGER); CREATE TABLE log (n INTEGER);"
        "CREATE TRIGGER trg AFTER INSERT ON a BEGIN INSERT INTO log VALUES (NEW.x); END;"
    )
    db = opened(tmp_path / "m.db", migrations=[Migration(1, "trg", sql)])
    with db.transaction() as conn:
        conn.execute("INSERT INTO a VALUES (7)")
    assert [r["n"] for r in db.query("SELECT n FROM log")] == [7]


def test_the_runner_still_notices_a_migration_that_ended_the_transaction(tmp_path, monkeypatch):
    # Defence in depth: even if static detection were fooled, a migration that left the
    # transaction must not be recorded as applied.
    monkeypatch.setattr(db_module, "_has_transaction_control", lambda sql: False)
    path = tmp_path / "m.db"
    with pytest.raises(StorageError):
        Database(path, migrations=[Migration(1, "sneaky", "CREATE TABLE t (x); COMMIT;")])
    assert raw_versions(path) == []


@pytest.mark.parametrize(
    "numbers",
    [[1, 3], [2], [1, 1], [1, 2, 4], [0], [-1], [1, 2, 2], [2, 1]],
    ids=str,
)
def test_gaps_duplicates_and_bad_numbering_are_rejected_before_touching_the_file(tmp_path, numbers):
    path = tmp_path / "m.db"
    migrations = [Migration(n, f"m{n}", "SELECT 1") for n in numbers]
    with pytest.raises(StorageError) as info:
        Database(path, migrations=migrations)
    assert info.value.code == "storage_error"
    assert not path.exists()


def test_no_migrations_at_all_is_a_storage_error(tmp_path):
    with pytest.raises(StorageError):
        Database(tmp_path / "m.db", migrations=[])


def test_database_newer_than_the_known_migrations_is_a_storage_error(tmp_path, opened):
    path = tmp_path / "m.db"
    opened(path, migrations=[COUNTER]).close()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO schema_migrations VALUES (99, '2026-01-01T00:00:00+00:00', 'x')")
    conn.commit()
    conn.close()
    with pytest.raises(StorageError) as info:
        Database(path, migrations=[COUNTER])
    assert info.value.code == "storage_error"
    assert "newer version" in str(info.value)


def test_database_with_a_hole_in_the_applied_versions_is_a_storage_error(tmp_path):
    path = tmp_path / "m.db"
    Database(path, migrations=[COUNTER]).close()
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO schema_migrations VALUES (3, '2026-01-01T00:00:00+00:00', 'x')")
    conn.commit()
    conn.close()
    known = [COUNTER, Migration(2, "b", "SELECT 1"), Migration(3, "c", "SELECT 1")]
    with pytest.raises(StorageError) as info:
        Database(path, migrations=known)
    assert "contiguous" in str(info.value)


def test_a_failed_open_leaves_no_open_handle_behind(tmp_path):
    path = tmp_path / "m.db"
    with pytest.raises(StorageError):
        Database(path, migrations=[Migration(1, "bad", "SELECT * FROM nope")])
    path.unlink()  # fails on Windows when the connection leaked
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


# ---- migrations: checksums ----


def _stored_checksums(path: Path) -> dict[int, str]:
    conn = sqlite3.connect(path)
    try:
        return dict(conn.execute("SELECT version, checksum FROM schema_migrations"))
    finally:
        conn.close()


def test_the_checksum_is_the_sha256_of_the_sql_with_newlines_normalised():
    assert Migration(1, "a", "SELECT 1\n").checksum == hashlib.sha256(b"SELECT 1\n").hexdigest()
    lf = Migration(1, "a", "SELECT 1;\nSELECT 2;\n")
    assert Migration(1, "a", "SELECT 1;\r\nSELECT 2;\r\n").checksum == lf.checksum
    assert Migration(1, "a", "SELECT 1;\rSELECT 2;\r").checksum == lf.checksum
    assert Migration(1, "a", "SELECT 1;\nSELECT 3;\n").checksum != lf.checksum
    assert Migration(1, "a", "SELECT 1;\nSELECT 2; \n").checksum != lf.checksum


def test_each_applied_migration_is_recorded_with_its_checksum(tmp_path, opened):
    second = Migration(2, "more", "CREATE TABLE extra (id INTEGER)")
    opened(tmp_path / "m.db", migrations=[COUNTER, second])
    assert _stored_checksums(tmp_path / "m.db") == {
        1: COUNTER.checksum,
        2: second.checksum,
    }


def test_the_checksum_column_is_mandatory(tmp_path, opened):
    db = opened(tmp_path / "m.db", migrations=[COUNTER])
    columns = {r["name"]: r for r in db.query("PRAGMA table_info(schema_migrations)")}
    assert columns["checksum"]["notnull"] == 1
    assert columns["checksum"]["type"] == "TEXT"


def test_editing_an_applied_migration_is_detected_on_open(tmp_path, opened):
    path = tmp_path / "m.db"
    opened(path, migrations=[COUNTER]).close()
    edited = Migration(1, "counter", COUNTER.sql + "CREATE TABLE sneaky (x);")
    with pytest.raises(StorageError) as info:
        Database(path, migrations=[edited])
    assert info.value.code == "migration_checksum_mismatch"
    assert "sneaky" not in str(info.value)
    assert "CREATE" not in str(info.value)


def test_a_line_ending_only_change_is_not_drift(tmp_path, opened):
    path = tmp_path / "m.db"
    lf = Migration(1, "t", "CREATE TABLE t (a);\nCREATE TABLE u (b);\n")
    opened(path, migrations=[lf]).close()
    crlf = Migration(1, "t", lf.sql.replace("\n", "\r\n"))
    assert versions(opened(path, migrations=[crlf])) == [1]


def test_drift_in_an_old_migration_blocks_pending_ones(tmp_path):
    path = tmp_path / "m.db"
    Database(path, migrations=[COUNTER]).close()
    edited = Migration(1, "counter", COUNTER.sql + " ")
    pending = Migration(2, "more", "CREATE TABLE extra (id INTEGER)")
    with pytest.raises(StorageError) as info:
        Database(path, migrations=[edited, pending])
    assert info.value.code == "migration_checksum_mismatch"
    assert raw_versions(path) == [1]  # the pending migration was not applied


def test_drift_in_a_later_migration_is_detected_too(tmp_path):
    path = tmp_path / "m.db"
    second = Migration(2, "more", "CREATE TABLE extra (id INTEGER)")
    Database(path, migrations=[COUNTER, second]).close()
    edited = Migration(2, "more", "CREATE TABLE extra (id INTEGER, y TEXT)")
    with pytest.raises(StorageError) as info:
        Database(path, migrations=[COUNTER, edited])
    assert info.value.code == "migration_checksum_mismatch"


def test_an_applied_version_without_a_migration_file_is_a_storage_error(tmp_path):
    path = tmp_path / "m.db"
    second = Migration(2, "more", "CREATE TABLE extra (id INTEGER)")
    Database(path, migrations=[COUNTER, second]).close()
    with pytest.raises(StorageError) as info:
        Database(path, migrations=[COUNTER])  # the file for version 2 is gone
    assert "newer version" in str(info.value)


def test_a_schema_migrations_table_without_checksums_is_rejected_cleanly(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE schema_migrations "
        "(version INTEGER NOT NULL PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO schema_migrations VALUES (1, '2026-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()
    with pytest.raises(StorageError) as info:
        Database(path, migrations=[COUNTER])
    assert "unsupported" in str(info.value)
    assert info.value.code == "storage_error"


# ---- migrations: discovery ----


def _write_migrations(directory: Path, files: dict[str, str]) -> Path:
    directory.mkdir()
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")
    return directory


def test_packaged_migrations_start_at_one_and_have_no_gaps():
    found = load_migrations()
    assert [m.version for m in found] == list(range(1, len(found) + 1))
    assert found[0].name == "init"
    assert "CREATE TABLE sources" in found[0].sql


def test_discovery_sorts_numerically_and_ignores_non_sql_files(tmp_path):
    directory = _write_migrations(
        tmp_path / "m",
        {
            "0002_second.sql": "SELECT 2",
            "0001_first.sql": "SELECT 1",
            "0010_tenth.sql": "SELECT 10",
            "README.md": "notes",
            "__init__.py": "",
        },
    )
    # Gap between 2 and 10 is rejected by the loader as well.
    with pytest.raises(StorageError):
        load_migrations(directory)
    (directory / "0010_tenth.sql").unlink()
    found = load_migrations(directory)
    assert [(m.version, m.name, m.sql) for m in found] == [
        (1, "first", "SELECT 1"),
        (2, "second", "SELECT 2"),
    ]


@pytest.mark.parametrize(
    "bad_name",
    [
        "init.sql",
        "1_init.sql",
        "0001.sql",
        "0001_Init.sql",
        "0001-init.sql",
        "00001_x.sql",
        "0001_in it.sql",
        "0001_init.SQL",
    ],
)
def test_misnamed_sql_files_are_rejected(tmp_path, bad_name):
    directory = _write_migrations(tmp_path / "m", {"0001_ok.sql": "SELECT 1", bad_name: "SELECT 2"})
    with pytest.raises(StorageError):
        load_migrations(directory)


def test_numbers_must_be_zero_padded_to_four_digits(tmp_path):
    # Valid on their own if the padding rule were loosened (versions 1 and 2), so no other file
    # in the directory can mask the failure.
    directory = _write_migrations(tmp_path / "m", {"1_a.sql": "SELECT 1", "2_b.sql": "SELECT 2"})
    with pytest.raises(StorageError):
        load_migrations(directory)


def test_duplicate_numbers_in_a_directory_are_rejected(tmp_path):
    directory = _write_migrations(
        tmp_path / "m", {"0001_a.sql": "SELECT 1", "0001_b.sql": "SELECT 2"}
    )
    with pytest.raises(StorageError):
        load_migrations(directory)


def test_empty_directory_and_empty_sql_file_are_rejected(tmp_path):
    with pytest.raises(StorageError):
        load_migrations(_write_migrations(tmp_path / "none", {}))
    with pytest.raises(StorageError):
        load_migrations(_write_migrations(tmp_path / "blank", {"0001_blank.sql": "  \n"}))


def test_a_non_utf8_migration_file_is_a_storage_error(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    (directory / "0001_bad.sql").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(StorageError):
        load_migrations(directory)


class _ReversedListing:
    """A directory whose listing comes back in descending order, as some filesystems may."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def iterdir(self):
        return iter(sorted(self._directory.iterdir(), reverse=True))


def test_discovery_does_not_depend_on_the_order_the_directory_is_listed(tmp_path):
    directory = _write_migrations(
        tmp_path / "m", {f"{n:04d}_step.sql": f"SELECT {n}" for n in (1, 2, 3, 4, 5)}
    )
    found = load_migrations(_ReversedListing(directory))  # type: ignore[arg-type]
    assert [m.version for m in found] == [1, 2, 3, 4, 5]


def test_a_loser_of_the_migration_race_leaves_no_transaction_open(tmp_path):
    path = tmp_path / "race.db"
    # Prepare an empty WAL database whose migration table exists but records nothing yet.
    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute(
        "CREATE TABLE schema_migrations "
        "(version INTEGER NOT NULL PRIMARY KEY, applied_at TEXT NOT NULL, checksum TEXT NOT NULL)"
    )
    holder.execute("BEGIN IMMEDIATE")  # blocks the racers right where they must take the lock
    dbs: list[Database] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def racer():
        try:
            db = Database(path, migrations=[COUNTER], busy_timeout_ms=3000)
            with lock:
                dbs.append(db)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=racer) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.4)  # let every racer read "nothing applied" and queue on the write lock
    # The winner applies the migration by hand while the racers wait.
    holder.execute("CREATE TABLE counter (id INTEGER PRIMARY KEY, n INTEGER NOT NULL)")
    holder.execute("INSERT INTO counter VALUES (1, 0)")
    holder.execute(
        "INSERT INTO schema_migrations VALUES (1, '2026-01-01T00:00:00+00:00', ?)",
        (COUNTER.checksum,),
    )
    holder.execute("COMMIT")
    holder.close()
    for t in threads:
        t.join(timeout=30)
    try:
        assert errors == []
        assert len(dbs) == 3
        for db in dbs:  # a losing racer must not still hold the write lock
            with db.transaction() as conn:
                conn.execute("UPDATE counter SET n = n + 1")
        assert dbs[0].query("SELECT n FROM counter")[0]["n"] == 3
    finally:
        for db in dbs:
            db.close()


def test_many_instances_opening_a_new_file_at_once_all_succeed(tmp_path):
    path = tmp_path / "race.db"
    count = 6
    barrier = threading.Barrier(count)
    opened_dbs: list[Database] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker():
        barrier.wait(timeout=5)
        try:
            db = Database(path)
            with lock:
                opened_dbs.append(db)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    try:
        assert errors == []
        assert len(opened_dbs) == count
        assert versions(opened_dbs[0]) == [m.version for m in load_migrations()]
    finally:
        for db in opened_dbs:
            db.close()


def test_database_can_run_discovered_migrations(tmp_path, opened):
    directory = _write_migrations(
        tmp_path / "m", {"0001_a.sql": "CREATE TABLE a (x)", "0002_b.sql": "CREATE TABLE b (y)"}
    )
    db = opened(tmp_path / "x.db", migrations=load_migrations(directory))
    assert versions(db) == [1, 2]


# ---- timestamps ----


def test_utc_round_trip_keeps_microseconds():
    moment = datetime(2026, 5, 6, 7, 8, 9, 123456, tzinfo=UTC)
    text = format_utc(moment)
    assert text == "2026-05-06T07:08:09.123456+00:00"
    assert parse_utc(text) == moment


def test_parse_utc_converts_other_offsets_to_utc():
    parsed = parse_utc("2026-01-01T14:00:00+02:00")
    assert parsed == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert parsed.utcoffset() == timedelta(0)


def test_format_utc_converts_other_offsets():
    plus_two = datetime(2026, 1, 1, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    assert format_utc(plus_two) == "2026-01-01T12:00:00+00:00"


@pytest.mark.parametrize(
    "bad", [None, 5, "", "garbage", "2026-01-01T12:00:00", "2026-13-01T00:00:00+00:00"]
)
def test_parse_utc_rejects_malformed_or_naive_values(bad):
    with pytest.raises(StorageError):
        parse_utc(bad)


def test_parse_utc_message_does_not_echo_the_value():
    with pytest.raises(StorageError) as info:
        parse_utc("SECRET-VALUE")
    assert "SECRET-VALUE" not in str(info.value)
