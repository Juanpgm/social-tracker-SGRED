"""SqliteControlRepository: the shared contract plus everything specific to SQLite."""

import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from social_tracker.control.domain import (
    MAX_CONSECUTIVE_FAILURES,
    SETTING_GLOBAL_ANALYSIS_ENABLED,
    CollectionState,
    ControlEvent,
    SourceState,
)
from social_tracker.control.ports import ControlRepository
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.shared.errors import StorageError, ValidationError
from tests.control.repository_contract import ControlRepositoryContract, event
from tests.fakes.control_repository import InMemoryControlRepository

AT = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
TS = "2026-03-01T08:00:00+00:00"


class TestSqliteControlRepositoryContract(ControlRepositoryContract):
    """Every contract test, against a fresh database file per repository.

    `ALLOWED_EXTRA_PUBLIC` stays empty on purpose: the repository has no `close()`. The
    `Database` owns the connection and its lifecycle, so closing is not part of this surface.
    """

    @pytest.fixture(autouse=True)
    def _databases(self, tmp_path: Path):
        self._root = tmp_path
        self._opened: list[Database] = []
        yield
        for db in self._opened:
            db.close()

    def make_repo(self) -> ControlRepository:
        db = Database(self._root / f"control-{len(self._opened)}.db")
        self._opened.append(db)
        return SqliteControlRepository(db)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "app.db")
    yield database
    database.close()


@pytest.fixture
def repo(db) -> SqliteControlRepository:
    return SqliteControlRepository(db)


def insert_raw(db: Database, table: str, **row: Any) -> None:
    """Insert bypassing the repository and the CHECK constraints."""
    db.query("PRAGMA ignore_check_constraints=ON")
    try:
        with db.transaction() as conn:
            marks = ", ".join("?" for _ in row)
            conn.execute(
                f"INSERT INTO {table} ({', '.join(row)}) VALUES ({marks})", list(row.values())
            )
    finally:
        db.query("PRAGMA ignore_check_constraints=OFF")


def test_it_satisfies_the_port(repo):
    assert isinstance(repo, ControlRepository)


# ---- settings ----


def test_fresh_database_defaults_global_analysis_to_false_without_a_stored_row(repo, db):
    assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"
    assert db.query("SELECT count(*) FROM settings")[0][0] == 0


def test_a_stored_value_overrides_the_default_and_a_write_is_an_upsert(repo, db):
    repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true")
    repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true")
    repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "false")
    repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true")
    assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "true"
    assert db.query("SELECT count(*) FROM settings")[0][0] == 1


INVALID_SETTING_CALLS = [
    ("set", "", "v"),
    ("set", "   ", "v"),
    ("set", "k" * 101, "v"),
    ("set", 5, "v"),
    ("set", None, "v"),
    ("set", "k_one", None),
    ("set", "k_one", 5),
    ("set", "k_one", "v" * 1001),
    ("set", SETTING_GLOBAL_ANALYSIS_ENABLED, "TRUE"),
    ("set", SETTING_GLOBAL_ANALYSIS_ENABLED, ""),
    ("set", SETTING_GLOBAL_ANALYSIS_ENABLED, "1"),
    ("set", SETTING_GLOBAL_ANALYSIS_ENABLED, " true"),
    ("get", "", None),
    ("get", "k" * 101, None),
    ("get", None, None),
    ("get", 7, None),
]


@pytest.mark.parametrize(("op", "key", "value"), INVALID_SETTING_CALLS)
def test_invalid_settings_fail_exactly_like_the_reference_fake(tmp_path, op, key, value):
    def run(repo: Any) -> type[BaseException] | None:
        try:
            repo.set_setting(key, value) if op == "set" else repo.get_setting(key)
        except Exception as exc:
            return type(exc)
        return None

    with Database(tmp_path / "parity.db") as database:
        sqlite_outcome = run(SqliteControlRepository(database))
    assert sqlite_outcome is ValidationError
    assert run(InMemoryControlRepository()) is sqlite_outcome


def test_a_rejected_write_leaves_the_store_untouched(repo, db):
    repo.set_setting("k_one", "keep")
    with pytest.raises(ValidationError):
        repo.set_setting("k_one", "v" * 1001)
    with pytest.raises(ValidationError):
        repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "maybe")
    assert repo.get_setting("k_one") == "keep"
    assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"
    assert db.query("SELECT count(*) FROM settings")[0][0] == 1


def test_sql_metacharacters_in_settings_are_data(repo, db):
    nasty = "x'); DROP TABLE settings; --"
    repo.set_setting(nasty, nasty)
    assert repo.get_setting(nasty) == nasty
    assert repo.get_setting("x' OR '1'='1") is None
    assert db.query("SELECT name FROM sqlite_master WHERE name = 'settings'") != []


def test_a_leading_nul_in_a_key_is_rejected_by_the_schema_not_stored(repo):
    # SQLite's length() stops at the first NUL, so the CHECK sees an empty key.
    with pytest.raises(StorageError):
        repo.set_setting("\x00key", "v")
    assert repo.get_setting("ab\x00cd") is None


def test_stored_setting_that_breaks_the_rules_is_a_storage_error(repo, db):
    insert_raw(db, "settings", key=SETTING_GLOBAL_ANALYSIS_ENABLED, value="TRUE")
    insert_raw(db, "settings", key="long_one", value="v" * 5000)
    for key in (SETTING_GLOBAL_ANALYSIS_ENABLED, "long_one"):
        with pytest.raises(StorageError) as info:
            repo.get_setting(key)
        assert info.value.code == "storage_error"
    assert "TRUE" not in str(info.value)


# ---- source state ----


@pytest.mark.parametrize(
    "state",
    [
        SourceState("src_a", CollectionState.DISABLED),
        SourceState("src_a", CollectionState.IDLE),
        SourceState("src_a", CollectionState.COLLECTING, 4),
        SourceState("src_a", CollectionState.BACKOFF, 1, AT + timedelta(seconds=60)),
        SourceState("src_a", CollectionState.BACKOFF, 4, AT.replace(microsecond=999999)),
        SourceState("src_a", CollectionState.FAILED, MAX_CONSECUTIVE_FAILURES),
    ],
    ids=lambda s: f"{s.state.value}-{s.consecutive_failures}",
)
def test_every_state_shape_round_trips(repo, state):
    repo.save_source_state(state)
    got = repo.get_source_state("src_a")
    assert got == state
    assert got.state is state.state  # type: ignore[union-attr]


def test_the_state_is_stored_with_its_wire_value_and_utc_text(repo, db):
    repo.save_source_state(
        SourceState("src_a", CollectionState.BACKOFF, 2, AT + timedelta(seconds=120))
    )
    row = db.query("SELECT * FROM source_state")[0]
    assert tuple(row) == ("src_a", "backoff", 2, "2026-03-01T08:02:00+00:00")


def test_saving_replaces_in_place_without_duplicating(repo, db):
    for state in (CollectionState.IDLE, CollectionState.COLLECTING, CollectionState.DISABLED):
        repo.save_source_state(SourceState("src_a", state))
    assert db.query("SELECT count(*) FROM source_state")[0][0] == 1
    assert repo.get_source_state("src_a").state is CollectionState.DISABLED  # type: ignore[union-attr]


def test_leaving_backoff_clears_the_stored_schedule(repo):
    repo.save_source_state(SourceState("src_a", CollectionState.BACKOFF, 1, AT))
    repo.save_source_state(SourceState("src_a", CollectionState.IDLE, 1))
    assert repo.get_source_state("src_a").next_run_at is None  # type: ignore[union-attr]


def test_non_state_arguments_are_rejected(repo):
    for bad in (None, {"source_id": "src_a"}, "idle", 3):
        with pytest.raises(ValidationError):
            repo.save_source_state(bad)  # type: ignore[arg-type]


def test_unbindable_state_lookup_is_a_storage_error(repo):
    with pytest.raises(StorageError):
        repo.get_source_state(["src_a"])  # type: ignore[arg-type]


STATE_ROWS = {
    "unknown state": {"state": "paused"},
    "state in the wrong case": {"state": "IDLE"},
    "failures negative": {"consecutive_failures": -1},
    "failures above the maximum": {"consecutive_failures": MAX_CONSECUTIVE_FAILURES + 1},
    "failures not a number": {"consecutive_failures": "many"},
    "failed with too few failures": {"state": "failed", "consecutive_failures": 2},
    "idle with the maximum failures": {"consecutive_failures": MAX_CONSECUTIVE_FAILURES},
    "disabled with failures": {"state": "disabled", "consecutive_failures": 1},
    "backoff without a schedule": {"state": "backoff", "consecutive_failures": 1},
    "schedule outside backoff": {"next_run_at": TS},
    "garbage schedule": {"state": "backoff", "consecutive_failures": 1, "next_run_at": "soon"},
    "naive schedule": {
        "state": "backoff",
        "consecutive_failures": 1,
        "next_run_at": "2026-03-01T08:00:00",
    },
    "invalid id": {"source_id": "Bad Id"},
}


@pytest.mark.parametrize("overrides", STATE_ROWS.values(), ids=STATE_ROWS.keys())
def test_a_malformed_stored_state_is_a_storage_error(repo, db, overrides):
    row = {
        "source_id": "src_a",
        "state": "idle",
        "consecutive_failures": 0,
        "next_run_at": None,
        **overrides,
    }
    insert_raw(db, "source_state", **row)
    with pytest.raises(StorageError) as info:
        repo.get_source_state(row["source_id"])
    assert info.value.code == "storage_error"


# ---- control events ----


def test_events_are_stored_with_utc_text_and_null_source_for_global_events(repo, db):
    repo.append_control_event(ControlEvent(at=AT, kind="paused", detail="why"))
    repo.append_control_event(ControlEvent(at=AT, kind="enabled", source_id="src_a"))
    rows = [tuple(r) for r in db.query("SELECT at, kind, source_id, detail FROM control_events")]
    assert rows == [(TS, "paused", None, "why"), (TS, "enabled", "src_a", "")]


def test_raw_update_and_delete_of_events_fail_at_the_database_level(repo, db):
    repo.append_control_event(event("keep_me", "src_a"))
    for sql in (
        "UPDATE control_events SET kind = 'tampered'",
        "UPDATE control_events SET source_id = NULL",
        "DELETE FROM control_events",
        "DELETE FROM control_events WHERE id = 1",
        "INSERT OR REPLACE INTO control_events (id, at, kind, detail) "
        "VALUES (1, '2026-03-01T08:00:00+00:00', 'x', '')",
    ):
        with pytest.raises(StorageError) as info, db.transaction() as conn:
            conn.execute(sql)
        assert info.value.code == "storage_error"
    assert [e.kind for e in repo.list_control_events()] == ["keep_me"]


def test_events_come_back_in_insertion_order_not_timestamp_order(repo):
    appended = [event("late", minutes=50), event("early", minutes=0), event("mid", minutes=20)]
    for e in appended:
        repo.append_control_event(e)
    assert repo.list_control_events() == appended


def test_source_filter_is_exact(repo):
    repo.append_control_event(event("g_kind"))
    repo.append_control_event(event("a_kind", "src_a"))
    repo.append_control_event(event("b_kind", "src_ab"))
    assert [e.kind for e in repo.list_control_events("src_a")] == ["a_kind"]
    assert repo.list_control_events("") == []
    assert repo.list_control_events("src_unknown") == []
    assert repo.list_control_events("src_%") == []  # LIKE wildcards are not patterns here
    assert len(repo.list_control_events(None)) == 3


def test_sql_metacharacters_in_event_fields_are_data(repo, db):
    nasty = "x'); DROP TABLE control_events; --"
    repo.append_control_event(ControlEvent(at=AT, kind=nasty, detail=nasty))
    assert [(e.kind, e.detail) for e in repo.list_control_events()] == [(nasty, nasty)]
    assert repo.list_control_events(nasty) == []
    assert db.query("SELECT name FROM sqlite_master WHERE name = 'control_events'") != []


def test_event_boundaries_round_trip(repo):
    biggest = ControlEvent(at=AT, kind="k" * 64, source_id="s" * 64, detail="d" * 2000)
    repo.append_control_event(biggest)
    assert repo.list_control_events() == [biggest]


def test_non_events_are_rejected_without_touching_the_log(repo, db):
    for bad in (None, {"kind": "x"}, "paused"):
        with pytest.raises(ValidationError):
            repo.append_control_event(bad)  # type: ignore[arg-type]
    assert db.query("SELECT count(*) FROM control_events")[0][0] == 0


EVENT_ROWS = {
    "garbage timestamp": {"at": "yesterday"},
    "naive timestamp": {"at": "2026-03-01T08:00:00"},
    "empty kind": {"kind": ""},
    "oversized kind": {"kind": "k" * 65},
    "invalid source id": {"source_id": "Bad Id"},
    "oversized detail": {"detail": "d" * 2001},
}


@pytest.mark.parametrize("overrides", EVENT_ROWS.values(), ids=EVENT_ROWS.keys())
def test_a_malformed_stored_event_is_a_storage_error(repo, db, overrides):
    repo.append_control_event(event("good_one"))
    row = {"at": TS, "kind": "bad_one", "source_id": None, "detail": "", **overrides}
    insert_raw(db, "control_events", **row)
    with pytest.raises(StorageError) as info:
        repo.list_control_events()
    assert info.value.code == "storage_error"


def test_a_5000_event_log_appends_and_reads_back_in_order_quickly(repo):
    total = 5000
    started = time.monotonic()
    appended = [
        ControlEvent(
            at=AT + timedelta(seconds=(i * 7919) % 5000),
            kind=f"kind_{i}",
            source_id=("src_even" if i % 2 == 0 else "src_odd") if i % 5 else None,
        )
        for i in range(total)
    ]
    for e in appended:
        repo.append_control_event(e)
    listed = repo.list_control_events()
    elapsed = time.monotonic() - started
    assert listed == appended
    assert repo.list_control_events("src_even") == [
        e for e in appended if e.source_id == "src_even"
    ]
    assert elapsed < 30, f"5000 appends + reads took {elapsed:.1f}s"


# ---- concurrency and busy handling ----


@pytest.mark.parametrize("shared_instance", [True, False], ids=["one-database", "two-databases"])
def test_two_threads_toggling_the_same_source_stay_consistent(tmp_path, shared_instance):
    path = tmp_path / "toggle.db"
    first = Database(path)
    second = first if shared_instance else Database(path)
    repos = [SqliteControlRepository(first), SqliteControlRepository(second)]
    rounds = 40
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []
    finals: dict[int, CollectionState] = {}

    def worker(index: int):
        barrier.wait(timeout=5)
        try:
            for step in range(rounds):
                state = CollectionState.IDLE if step % 2 == 0 else CollectionState.DISABLED
                repos[index].save_source_state(SourceState("src_a", state))
                repos[index].append_control_event(event(f"toggle_{index}_{step}", "src_a"))
                finals[index] = state
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert errors == []
        events = repos[0].list_control_events("src_a")
        assert len(events) == 2 * rounds
        for index in range(2):  # each writer's events stay in its own order
            mine = [e.kind for e in events if e.kind.startswith(f"toggle_{index}_")]
            assert mine == [f"toggle_{index}_{step}" for step in range(rounds)]
        assert repos[0].get_source_state("src_a").state is CollectionState.DISABLED  # type: ignore[union-attr]
    finally:
        first.close()
        second.close()


def test_a_writer_blocked_beyond_the_busy_timeout_gets_a_storage_error(tmp_path):
    path = tmp_path / "busy.db"
    database = Database(path, busy_timeout_ms=50)
    repo = SqliteControlRepository(database)
    blocker = sqlite3.connect(path, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        for action in (
            lambda: repo.set_setting("k_one", "v"),
            lambda: repo.save_source_state(SourceState("src_a", CollectionState.IDLE)),
            lambda: repo.append_control_event(event()),
        ):
            with pytest.raises(StorageError) as info:
                action()
            assert info.value.code == "storage_error"
        assert time.monotonic() - started < 5
        # WAL readers are not blocked by the competing writer.
        assert repo.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"
        assert repo.list_control_events() == []
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    repo.set_setting("k_one", "v")  # recovers once the lock is gone
    assert repo.get_setting("k_one") == "v"
    database.close()


# ---- persistence across reopen ----


def test_everything_survives_closing_and_reopening_the_database_file(tmp_path):
    path = tmp_path / "durable.db"
    backoff = SourceState("src_b", CollectionState.BACKOFF, 3, AT + timedelta(minutes=5))
    events = [event("first", "src_a", 0), event("second", None, 3), event("third", "src_b", 1)]
    first = Database(path)
    repo = SqliteControlRepository(first)
    repo.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true")
    repo.set_setting("note", "café")
    repo.save_source_state(SourceState("src_a", CollectionState.IDLE))
    repo.save_source_state(backoff)
    for e in events:
        repo.append_control_event(e)
    first.close()

    with Database(path) as second:
        reopened = SqliteControlRepository(second)
        assert reopened.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "true"
        assert reopened.get_setting("note") == "café"
        assert reopened.get_source_state("src_a") == SourceState("src_a", CollectionState.IDLE)
        assert reopened.get_source_state("src_b") == backoff
        assert reopened.list_control_events() == events
        reopened.append_control_event(event("fourth"))
        assert [e.kind for e in reopened.list_control_events()][-1] == "fourth"


def test_a_closed_database_surfaces_as_a_storage_error(tmp_path):
    database = Database(tmp_path / "closed.db")
    repo = SqliteControlRepository(database)
    database.close()
    for action in (
        lambda: repo.get_setting("k_one"),
        lambda: repo.set_setting("k_one", "v"),
        lambda: repo.get_source_state("src_a"),
        lambda: repo.save_source_state(SourceState("src_a", CollectionState.IDLE)),
        lambda: repo.append_control_event(event()),
        lambda: repo.list_control_events(),
    ):
        with pytest.raises(StorageError):
            action()
