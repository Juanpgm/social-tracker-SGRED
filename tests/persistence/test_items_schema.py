"""Migration 0002: the schema itself enforces the invariants the item domain relies on.

Like `test_schema`, these tests talk to SQLite directly so each constraint is proven at the
database level, independently of the Python code that normally guards it.
"""

import sqlite3
from typing import Any

import pytest

from tests.persistence.schema_helpers import TS, raw_connection

ITEM: dict[str, Any] = {
    "id": "itm_one",
    "source_id": "src_news",
    "source_name": "City News",
    "kind": "article",
    "external_id": "guid-1",
    "parent_id": None,
    "url": "https://example.org/a",
    "title": "Title",
    "author": None,
    "text": "Body",
    "language": None,
    "published_at": TS,
    "fetched_at": TS,
    "content_hash": "a" * 64,
    "flags": "[]",
    "status": "new",
    "raw_payload": "{}",
}


@pytest.fixture
def conn():
    connection = raw_connection()
    yield connection
    connection.close()


def insert_item(conn: sqlite3.Connection, **overrides: Any) -> None:
    row = {**ITEM, **overrides}
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO items ({', '.join(row)}) VALUES ({marks})", list(row.values()))


def test_a_valid_item_is_accepted(conn):
    insert_item(conn)
    assert conn.execute("SELECT count(*) FROM items").fetchone()[0] == 1


REJECTED = {
    "id too short": {"id": "ab"},
    "id uppercase": {"id": "Itm_One"},
    "source id not a slug": {"source_id": "Bad Id"},
    "empty source name": {"source_name": ""},
    "long source name": {"source_name": "n" * 101},
    "unknown kind": {"kind": "video"},
    "empty external id": {"external_id": ""},
    "long external id": {"external_id": "e" * 513},
    "parent id not a slug": {"parent_id": "Bad Parent"},
    "url with another scheme": {"url": "javascript:alert(1)"},
    "long url": {"url": "https://example.org/" + "a" * 2100},
    "long title": {"title": "t" * 1001},
    "null title": {"title": None},
    "text over the byte limit": {"text": "é" * 51_201},
    "null text": {"text": None},
    "naive published_at": {"published_at": "2026-01-01T12:00:00"},
    "non-utc published_at": {"published_at": "2026-01-01T12:00:00+02:00"},
    "garbage fetched_at": {"fetched_at": "yesterday"},
    "short hash": {"content_hash": "abc"},
    "uppercase hash": {"content_hash": "A" * 64},
    "flags not json": {"flags": "[oops"},
    "flags an object": {"flags": "{}"},
    "status with spaces": {"status": "not valid"},
    "empty status": {"status": ""},
    "payload not json": {"raw_payload": "{oops"},
    "payload an array": {"raw_payload": "[]"},
    "payload over the size cap": {"raw_payload": '{"d": "' + "x" * 262_144 + '"}'},
}


@pytest.mark.parametrize("overrides", REJECTED.values(), ids=REJECTED.keys())
def test_invalid_rows_are_refused_by_the_schema(conn, overrides):
    with pytest.raises(sqlite3.IntegrityError):
        insert_item(conn, **overrides)


def test_boundaries_just_inside_the_limits_are_accepted(conn):
    insert_item(
        conn,
        text="a" * 102_400,
        title="t" * 1000,
        external_id="e" * 512,
        raw_payload='{"d": "' + "x" * (262_144 - 9) + '"}',
    )


def test_the_same_source_and_external_id_cannot_be_stored_twice(conn):
    insert_item(conn)
    with pytest.raises(sqlite3.IntegrityError):
        insert_item(conn, id="itm_two")
    insert_item(conn, id="itm_three", source_id="src_other")  # another source: fine


def test_the_item_id_is_unique(conn):
    insert_item(conn)
    with pytest.raises(sqlite3.IntegrityError):
        insert_item(conn, external_id="guid-2")


def test_there_is_no_foreign_key_to_sources(conn):
    # SR-9: nothing ties an item to a source row, so deleting a source cannot remove items.
    assert conn.execute("PRAGMA foreign_key_list(items)").fetchall() == []
    insert_item(conn)  # no such source exists, and that is fine


def test_the_planned_indexes_exist(conn):
    names = {row[1] for row in conn.execute("PRAGMA index_list(items)")}
    assert {
        "ux_items_source_external",
        "ix_items_content_hash",
        "ix_items_published_at",
        "ix_items_fetched_at",
        "ix_items_parent_id",
    } <= names


def test_a_fetch_status_row_is_checked(conn):
    ok = ("src_news", TS, "http 503", "etag-1", "Wed, 01 Jan 2026 00:00:00 GMT")
    conn.execute("INSERT INTO source_fetch_status VALUES (?, ?, ?, ?, ?)", ok)
    for bad in (
        ("Bad Id", TS, None, None, None),
        ("src_two", "yesterday", None, None, None),
        ("src_two", None, "", None, None),
        ("src_two", None, "e" * 301, None, None),
        ("src_two", None, None, "e" * 513, None),
    ):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO source_fetch_status VALUES (?, ?, ?, ?, ?)", bad)


# ---- tombstones (migration 0003) ----


@pytest.mark.parametrize(
    "row",
    [
        ("ab", "guid"),
        ("Bad Id", "guid"),
        ("src_news", ""),
        ("src_news", "g" * 513),
    ],
    ids=["short source", "bad source", "empty external id", "long external id"],
)
def test_an_invalid_tombstone_is_refused_by_the_schema(conn, row):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO item_tombstones (source_id, external_id) VALUES (?, ?)", row)


def test_a_tombstone_is_unique_per_source_and_external_id(conn):
    sql = "INSERT INTO item_tombstones (source_id, external_id) VALUES (?, ?)"
    conn.execute(sql, ("src_news", "guid"))
    conn.execute(sql, ("src_other", "guid"))
    conn.execute(sql, ("src_news", "other"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, ("src_news", "guid"))


def test_boundary_tombstones_are_accepted(conn):
    sql = "INSERT INTO item_tombstones (source_id, external_id) VALUES (?, ?)"
    conn.execute(sql, ("abc", "g"))
    conn.execute(sql, ("s" * 64, "g" * 512))
    assert conn.execute("SELECT count(*) FROM item_tombstones").fetchone()[0] == 2
