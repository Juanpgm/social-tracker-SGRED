"""Migration 0001: the schema itself enforces the invariants the domain relies on.

These tests talk to SQLite directly (no repository, no runner) so that every constraint is
proven at the database level, independently of the Python code that normally guards it.
"""

import sqlite3
from typing import Any

import pytest

from social_tracker.control.domain import MAX_CONSECUTIVE_FAILURES
from tests.persistence.schema_helpers import TS, raw_connection

SOURCE: dict[str, Any] = {
    "id": "src_one",
    "name": "Some Source",
    "normalized_name": "some source",
    "type": "news_rss",
    "mode": "recorded",
    "credibility": 3,
    "enabled": 0,
    "secret_ref": None,
    "config_json": '{"feed_url": "https://example.org/rss.xml"}',
    "created_at": TS,
    "updated_at": TS,
}


@pytest.fixture
def conn():
    connection = raw_connection()
    yield connection
    connection.close()


def insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    columns = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", list(row.values()))


def insert_source(conn: sqlite3.Connection, **overrides: Any) -> None:
    insert(conn, "sources", {**SOURCE, **overrides})


def insert_state(conn: sqlite3.Connection, **overrides: Any) -> None:
    row = {"source_id": "src_one", "state": "idle", "consecutive_failures": 0, "next_run_at": None}
    insert(conn, "source_state", {**row, **overrides})


def insert_event(conn: sqlite3.Connection, **overrides: Any) -> None:
    row = {"at": TS, "kind": "paused", "source_id": None, "detail": ""}
    insert(conn, "control_events", {**row, **overrides})


def rejected(conn: sqlite3.Connection, action, *fragments: str) -> None:
    with pytest.raises(sqlite3.IntegrityError) as info:
        action(conn)
    for fragment in fragments:
        assert fragment in str(info.value)


def count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_expected_tables_exist_and_nothing_else(conn):
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert names == {
        "sources",
        "source_state",
        "settings",
        "control_events",
        "items",
        "source_fetch_status",
        "items_fts",  # the full-text index and its four shadow tables
        "items_fts_data",
        "items_fts_idx",
        "items_fts_docsize",
        "items_fts_config",
    }


def test_normalized_name_has_a_unique_index(conn):
    insert_source(conn)
    rejected(
        conn,
        lambda c: insert_source(c, id="src_two", name="Other"),
        "UNIQUE constraint failed: sources.normalized_name",
    )


def test_control_events_by_source_are_served_by_an_index(conn):
    plan = " ".join(
        str(row[3])
        for row in conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM control_events WHERE source_id = 'src_one'"
        )
    )
    assert "USING INDEX" in plan.upper() or "USING COVERING INDEX" in plan.upper()
    assert "SCAN" not in plan.upper().replace("SEARCH", "")


@pytest.mark.parametrize("table", ["source_state", "control_events"])
def test_derived_tables_carry_no_foreign_key_to_sources(conn, table):
    # SR-9: deleting a source must never cascade into the data derived from it.
    assert conn.execute(f"PRAGMA foreign_key_list({table})").fetchall() == []


def test_duplicate_id_is_rejected(conn):
    insert_source(conn)
    rejected(
        conn,
        lambda c: insert_source(c, name="Other", normalized_name="other"),
        "UNIQUE constraint failed: sources.id",
    )


# ---- sources ----


@pytest.mark.parametrize("good", ["abc", "a" * 64, "src_1", "0_9", "___"])
def test_slug_boundaries_accepted(conn, good):
    insert_source(conn, id=good)
    assert count(conn, "sources") == 1


@pytest.mark.parametrize(
    "bad",
    ["", "ab", "a" * 65, "Src_one", "src-one", "src one", "src_one\n", "src_ñne", "src.one"],
)
def test_slug_violations_rejected(conn, bad):
    rejected(conn, lambda c: insert_source(c, id=bad), "CHECK constraint failed")


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "podcast"},
        {"type": ""},
        {"mode": "live"},
        {"mode": "streaming"},  # news_rss must be recorded
        {"type": "live_stream", "mode": "recorded", "config_json": "{}"},
        {"credibility": 0},
        {"credibility": 6},
        {"credibility": 3.5},
        {"credibility": "high"},
        {"credibility": None},
        {"enabled": 2},
        {"enabled": -1},
        {"enabled": None},
        {"name": ""},
        {"name": "n" * 101},
        {"normalized_name": ""},
        {"config_json": "not json"},
        {"config_json": "[1, 2]"},
        {"config_json": '"text"'},
        {"config_json": "null"},
        {"config_json": None},
        {"created_at": "2026-01-01T12:00:00"},
        {"created_at": "2026-01-01T12:00:00+02:00"},
        {"updated_at": "yesterday"},
        {"created_at": None},
        {"type": "instagram", "secret_ref": None, "config_json": "{}"},
    ],
    ids=lambda o: ",".join(f"{k}={v!r}" for k, v in o.items())[:60],
)
def test_source_check_constraints(conn, overrides):
    rejected(conn, lambda c: insert_source(c, **overrides), "constraint failed")
    assert count(conn, "sources") == 0


def test_every_type_mode_pair_the_domain_defines_is_accepted(conn):
    pairs = [
        ("news_rss", "recorded"),
        ("instagram", "recorded"),
        ("live_stream", "streaming"),
        ("recorded_audio", "recorded"),
    ]
    for index, (type_, mode) in enumerate(pairs):
        insert_source(
            conn,
            id=f"src_{index}",
            normalized_name=f"n{index}",
            type=type_,
            mode=mode,
            secret_ref="ig_main",
        )
    assert count(conn, "sources") == 4


def test_sql_metacharacters_in_text_columns_are_just_data(conn):
    nasty = "x'); DROP TABLE sources;--"
    insert_source(conn, name=nasty, normalized_name=nasty)
    assert conn.execute("SELECT name FROM sources").fetchone()[0] == nasty


# ---- source_state ----


def test_state_accepts_every_domain_state(conn):
    insert_state(conn, source_id="src_aaa", state="disabled")
    insert_state(conn, source_id="src_bbb", state="idle")
    insert_state(conn, source_id="src_ccc", state="collecting", consecutive_failures=2)
    insert_state(conn, source_id="src_ddd", state="backoff", consecutive_failures=1, next_run_at=TS)
    insert_state(
        conn,
        source_id="src_eee",
        state="failed",
        consecutive_failures=MAX_CONSECUTIVE_FAILURES,
    )
    assert count(conn, "source_state") == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"state": "paused"},
        {"state": "IDLE"},
        {"state": ""},
        {"state": None},
        {"consecutive_failures": -1},
        {"consecutive_failures": MAX_CONSECUTIVE_FAILURES + 1},
        {"consecutive_failures": 1.5},
        {"consecutive_failures": None},
        {"source_id": "no"},
        {"source_id": "Bad_Id"},
        {"state": "failed", "consecutive_failures": MAX_CONSECUTIVE_FAILURES - 1},
        {"state": "idle", "consecutive_failures": MAX_CONSECUTIVE_FAILURES},
        {"state": "disabled", "consecutive_failures": 1},
        {"state": "backoff", "consecutive_failures": 1, "next_run_at": None},
        {"state": "idle", "next_run_at": TS},
        {"next_run_at": "tomorrow", "state": "backoff", "consecutive_failures": 1},
    ],
    ids=lambda o: ",".join(f"{k}={v!r}" for k, v in o.items())[:60],
)
def test_state_check_constraints(conn, overrides):
    rejected(conn, lambda c: insert_state(c, **overrides), "constraint failed")
    assert count(conn, "source_state") == 0


def test_state_primary_key_is_the_source_id(conn):
    insert_state(conn)
    rejected(conn, lambda c: insert_state(c), "UNIQUE constraint failed: source_state.source_id")


def test_state_has_no_foreign_key_to_sources(conn):
    insert_state(conn, source_id="src_ghost")  # state may exist without a registered source
    assert conn.execute("PRAGMA foreign_key_list(source_state)").fetchall() == []


# ---- settings ----


def test_setting_boundaries_are_accepted(conn):
    conn.execute("INSERT INTO settings (key, value) VALUES ('k', 'v')")
    conn.execute("INSERT INTO settings (key, value) VALUES (?, '')", ("k" * 100,))
    conn.execute("INSERT INTO settings (key, value) VALUES ('big', ?)", ("v" * 1000,))
    assert count(conn, "settings") == 3


@pytest.mark.parametrize(
    ("key", "value"), [("", "v"), ("k" * 101, "v"), ("big2", "v" * 1001), ("nul", None)]
)
def test_setting_violations_rejected(conn, key, value):
    rejected(
        conn,
        lambda c: c.execute("INSERT INTO settings (key, value) VALUES (?, ?)", (key, value)),
        "constraint failed",
    )


def test_setting_key_is_unique(conn):
    conn.execute("INSERT INTO settings (key, value) VALUES ('k', 'v')")
    rejected(
        conn,
        lambda c: c.execute("INSERT INTO settings (key, value) VALUES ('k', 'again')"),
        "UNIQUE constraint failed: settings.key",
    )


@pytest.mark.parametrize("value", ["TRUE", "1", "", "yes", "False"])
def test_global_analysis_flag_is_checked_in_the_database_too(conn, value):
    rejected(
        conn,
        lambda c: c.execute(
            "INSERT INTO settings (key, value) VALUES ('global_analysis_enabled', ?)", (value,)
        ),
        "constraint failed",
    )


@pytest.mark.parametrize("value", ["true", "false"])
def test_global_analysis_flag_accepts_true_and_false(conn, value):
    conn.execute(
        "INSERT INTO settings (key, value) VALUES ('global_analysis_enabled', ?)", (value,)
    )
    assert count(conn, "settings") == 1


# ---- control_events ----


def test_events_get_increasing_ids(conn):
    insert_event(conn, kind="first")
    insert_event(conn, kind="second", source_id="src_one")
    rows = conn.execute("SELECT id, kind FROM control_events ORDER BY id").fetchall()
    assert rows == [(1, "first"), (2, "second")]


def test_event_ids_are_never_reused_even_if_the_guards_are_removed(conn):
    insert_event(conn, kind="first")
    insert_event(conn, kind="second")
    conn.execute("DROP TRIGGER control_events_no_delete")
    conn.execute("DELETE FROM control_events WHERE id = 2")
    insert_event(conn, kind="third")
    assert conn.execute("SELECT id FROM control_events WHERE kind = 'third'").fetchone()[0] == 3


def test_events_cannot_be_updated(conn):
    insert_event(conn)
    rejected(conn, lambda c: c.execute("UPDATE control_events SET kind = 'x'"), "append-only")
    rejected(conn, lambda c: c.execute("UPDATE control_events SET id = 99"), "append-only")
    assert conn.execute("SELECT kind FROM control_events").fetchone()[0] == "paused"


def test_events_cannot_be_deleted(conn):
    insert_event(conn)
    rejected(conn, lambda c: c.execute("DELETE FROM control_events"), "append-only")
    rejected(conn, lambda c: c.execute("DELETE FROM control_events WHERE id = 1"), "append-only")
    assert count(conn, "control_events") == 1


def test_replace_cannot_overwrite_an_event(conn):
    insert_event(conn)
    rejected(
        conn,
        lambda c: c.execute(
            "INSERT OR REPLACE INTO control_events (id, at, kind, detail) VALUES (1, ?, 'x', '')",
            (TS,),
        ),
        "append-only",
    )
    assert conn.execute("SELECT kind FROM control_events").fetchone()[0] == "paused"


def test_explicit_ids_may_only_grow(conn):
    insert_event(conn, kind="first")
    rejected(conn, lambda c: insert_event(c, id=1, kind="dup"), "append-only")
    insert_event(conn, id=5, kind="jump")
    rejected(conn, lambda c: insert_event(c, id=3, kind="late"), "append-only")
    rejected(conn, lambda c: insert_event(c, id=5, kind="again"), "append-only")
    assert [r[0] for r in conn.execute("SELECT kind FROM control_events ORDER BY id")] == [
        "first",
        "jump",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": ""},
        {"kind": "k" * 65},
        {"kind": None},
        {"detail": "d" * 2001},
        {"detail": None},
        {"at": "2026-01-01 12:00:00"},
        {"at": None},
        {"source_id": ""},
        {"source_id": "Bad"},
    ],
    ids=lambda o: ",".join(f"{k}={v!r}" for k, v in o.items())[:60],
)
def test_event_check_constraints(conn, overrides):
    rejected(conn, lambda c: insert_event(c, **overrides), "constraint failed")
    assert count(conn, "control_events") == 0


def test_event_boundaries_accepted(conn):
    insert_event(conn, kind="k" * 64, detail="d" * 2000, source_id="a" * 64)
    assert count(conn, "control_events") == 1
