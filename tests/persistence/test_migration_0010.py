"""Migration 0010 (the alert store) on a database that already holds data: nothing may be lost or
changed, and the migration must be safe to meet twice, in a race and after a crash."""

import re
import threading
from pathlib import Path

import pytest

from social_tracker.persistence.db import Database, Migration, load_migrations
from social_tracker.shared.errors import StorageError
from tests.persistence.test_migration_0005 import COLUMNS, legacy_database

NEW_TABLES = {"alerts", "alert_items", "alert_events", "alert_forgotten"}


def tables(db: Database) -> set[str]:
    return {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_the_tenth_migration_is_the_alert_store():
    migrations = load_migrations()
    assert migrations[9].version == 10
    assert migrations[9].name == "alerts"


def test_a_database_at_the_previous_version_keeps_everything_when_migrated(tmp_path: Path):
    path = tmp_path / "old.db"
    legacy_database(path)  # items, verdicts and signatures, as the early schema left them
    old = Database(path, migrations=load_migrations()[:9])
    before = {
        "classifications": [
            tuple(r) for r in old.query(f"SELECT seq, {COLUMNS} FROM classifications ORDER BY seq")
        ],
        "items": [tuple(r) for r in old.query("SELECT * FROM items ORDER BY seq")],
        "signatures": [tuple(r) for r in old.query("SELECT * FROM item_signatures")],
    }
    assert NEW_TABLES.isdisjoint(tables(old))
    old.close()
    migrated = Database(path)
    try:
        assert tables(migrated) >= NEW_TABLES
        assert before["classifications"] == [
            tuple(r)
            for r in migrated.query(f"SELECT seq, {COLUMNS} FROM classifications ORDER BY seq")
        ]
        assert before["items"] == [
            tuple(r) for r in migrated.query("SELECT * FROM items ORDER BY seq")
        ]
        assert before["signatures"] == [
            tuple(r) for r in migrated.query("SELECT * FROM item_signatures")
        ]
        for table in NEW_TABLES:
            assert migrated.query(f"SELECT count(*) FROM {table}")[0][0] == 0
        versions = [
            r[0] for r in migrated.query("SELECT version FROM schema_migrations ORDER BY 1")
        ]
        assert versions == list(range(1, len(load_migrations()) + 1))
    finally:
        migrated.close()


def test_opening_twice_applies_it_once_and_keeps_its_data(tmp_path: Path):
    path = tmp_path / "twice.db"
    first = Database(path)
    with first.transaction() as conn:
        conn.execute(
            "INSERT INTO alert_forgotten (item_id, forgotten_at) VALUES ('itm_a', ?)",
            ("2026-03-01T12:00:00+00:00",),
        )
    first.close()
    second = Database(path)
    try:
        assert second.query("SELECT item_id FROM alert_forgotten")[0][0] == "itm_a"
        assert second.query("SELECT count(*) FROM schema_migrations")[0][0] == len(
            load_migrations()
        )
    finally:
        second.close()


def test_two_processes_opening_a_database_at_the_previous_version_apply_it_once(tmp_path: Path):
    path = tmp_path / "race.db"
    Database(path, migrations=load_migrations()[:9]).close()
    errors: list[BaseException] = []
    opened: list[Database] = []

    def open_it() -> None:
        try:
            opened.append(Database(path))
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=open_it) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert errors == []
        assert opened[0].query("SELECT count(*) FROM schema_migrations")[0][0] == len(
            load_migrations()
        )
    finally:
        for db in opened:
            db.close()


def test_a_failing_migration_leaves_the_database_at_the_previous_version(tmp_path: Path):
    path = tmp_path / "crash.db"
    Database(path, migrations=load_migrations()[:9]).close()
    broken = Migration(
        10, "alerts", "CREATE TABLE alerts (id TEXT); CREATE TABLE alerts (id TEXT);"
    )
    with pytest.raises(StorageError):
        Database(path, migrations=[*load_migrations()[:9], broken])
    again = Database(path, migrations=load_migrations()[:9])  # nothing half-applied
    try:
        assert NEW_TABLES.isdisjoint(tables(again))
    finally:
        again.close()
    fixed = Database(path)
    try:
        assert tables(fixed) >= NEW_TABLES
    finally:
        fixed.close()


def test_the_migration_only_creates_things():
    code = re.sub(r"--[^\n]*", "", load_migrations()[9].sql)
    statements = [part.strip() for part in code.split(";") if part.strip()]
    assert statements
    assert all(statement.upper().startswith("CREATE ") for statement in statements)


def test_the_foreign_keys_cascade_in_the_new_tables(tmp_path: Path):
    db = Database(tmp_path / "fk.db")
    try:
        stamp = "2026-03-01T12:00:00+00:00"
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO alerts (id, label, severity, confidence, rationale, source_id, "
                "source_name, item_id, excerpt, tags, evidence, linked_sources, status, "
                "occurrences, first_seen_at, last_seen_at, created_at, updated_at) VALUES "
                "('alt_a', 'recycled_content', 'low', 0.5, 'r', 'src_a', 'A', 'itm_a', '', '[]', "
                "'{}', '[]', 'new', 1, ?, ?, ?, ?)",
                (stamp, stamp, stamp, stamp),
            )
            conn.execute(
                "INSERT INTO alert_events (alert_id, at, kind, actor, reason, from_value, "
                "to_value) VALUES ('alt_a', ?, 'created', 'x', 'y', '', '')",
                (stamp,),
            )
        with db.transaction() as conn:
            conn.execute("DELETE FROM alerts WHERE id = 'alt_a'")
        assert db.query("SELECT count(*) FROM alert_events")[0][0] == 0
        with pytest.raises(StorageError), db.transaction() as conn:
            conn.execute(
                "INSERT INTO alert_events (alert_id, at, kind, actor, reason, from_value, "
                "to_value) VALUES ('alt_zz', ?, 'created', 'x', 'y', '', '')",
                (stamp,),
            )
    finally:
        db.close()
