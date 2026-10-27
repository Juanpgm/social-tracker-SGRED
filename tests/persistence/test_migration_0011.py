"""Migration 0011 (the local gold set) on a database that already holds data."""

import re
from pathlib import Path

import pytest

from social_tracker.persistence.db import Database, load_migrations
from social_tracker.shared.errors import StorageError
from tests.persistence.test_migration_0005 import COLUMNS, legacy_database


def test_the_eleventh_migration_is_the_gold_set():
    migration = load_migrations()[10]
    assert (migration.version, migration.name) == (11, "gold")


def test_a_database_at_the_previous_version_keeps_everything_when_migrated(tmp_path: Path):
    path = tmp_path / "old.db"
    legacy_database(path)
    old = Database(path, migrations=load_migrations()[:10])
    before = [tuple(r) for r in old.query(f"SELECT seq, {COLUMNS} FROM classifications")]
    items = [tuple(r) for r in old.query("SELECT * FROM items ORDER BY seq")]
    assert not old.query("SELECT 1 FROM sqlite_master WHERE name = 'gold_examples'")
    old.close()
    migrated = Database(path)
    try:
        assert migrated.query("SELECT count(*) FROM gold_examples")[0][0] == 0
        assert before == [
            tuple(r) for r in migrated.query(f"SELECT seq, {COLUMNS} FROM classifications")
        ]
        assert items == [tuple(r) for r in migrated.query("SELECT * FROM items ORDER BY seq")]
        versions = [
            r[0] for r in migrated.query("SELECT version FROM schema_migrations ORDER BY 1")
        ]
        assert versions == list(range(1, len(load_migrations()) + 1))
    finally:
        migrated.close()


def test_the_migration_only_creates_things():
    code = re.sub(r"--[^\n]*", "", load_migrations()[10].sql)
    statements = [part.strip() for part in code.split(";") if part.strip()]
    assert statements
    assert all(statement.upper().startswith("CREATE ") for statement in statements)


def test_opening_twice_keeps_the_examples(tmp_path: Path):
    path = tmp_path / "twice.db"
    first = Database(path)
    with first.transaction() as conn:
        conn.execute(
            "INSERT INTO gold_examples (id, text, label, note, credibility, added_at, actor) "
            "VALUES ('gld_0123456789abcdef', 'hola', 'no_risk', '', 3, "
            "'2026-03-01T12:00:00+00:00', 'ana')"
        )
    first.close()
    second = Database(path)
    try:
        assert second.query("SELECT text FROM gold_examples")[0][0] == "hola"
    finally:
        second.close()


@pytest.mark.parametrize(
    "values",
    [
        "('gld_short', 'x', 'no_risk', '', 3, '2026-03-01T12:00:00+00:00', 'a')",
        "('gld_0123456789abcdef', '', 'no_risk', '', 3, '2026-03-01T12:00:00+00:00', 'a')",
        "('gld_0123456789abcdef', 'x', 'bogus', '', 3, '2026-03-01T12:00:00+00:00', 'a')",
        "('gld_0123456789abcdef', 'x', 'no_risk', '', 0, '2026-03-01T12:00:00+00:00', 'a')",
        "('gld_0123456789abcdef', 'x', 'no_risk', '', 3, 'yesterday', 'a')",
        "('gld_0123456789abcdef', 'x', 'no_risk', '', 3, '2026-03-01T12:00:00+00:00', '')",
    ],
)
def test_the_table_refuses_nonsense(tmp_path: Path, values: str):
    db = Database(tmp_path / "x.db")
    try:
        with pytest.raises(StorageError), db.transaction() as conn:
            conn.execute(
                "INSERT INTO gold_examples (id, text, label, note, credibility, added_at, actor) "
                f"VALUES {values}"
            )
    finally:
        db.close()
