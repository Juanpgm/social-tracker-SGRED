"""Helpers shared by the persistence tests."""

import sqlite3
from importlib.resources import files

TS = "2026-01-01T12:00:00+00:00"
MIGRATION_FILES = (
    "0001_init.sql",
    "0002_items_and_fetch_status.sql",
    "0003_item_tombstones.sql",
)


def migration_sql(name: str = "0001_init.sql") -> str:
    return (files("social_tracker.persistence") / "migrations" / name).read_text(encoding="utf-8")


def raw_connection() -> sqlite3.Connection:
    """In-memory connection with every migration applied directly, with no runner involved."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    for name in MIGRATION_FILES:
        conn.executescript(migration_sql(name))
    return conn
