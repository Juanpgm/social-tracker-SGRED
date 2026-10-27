"""Migration 0012 (the alert remediation: audit flags, overflow, failures, dirty marks, forgotten
signatures) on a database that already holds data."""

from pathlib import Path

import pytest

from social_tracker.persistence.db import Database, load_migrations
from social_tracker.shared.errors import StorageError
from tests.persistence.test_migration_0005 import COLUMNS, legacy_database

NEW_TABLES = {"alert_overflow", "alert_failures", "alert_dirty"}


def names(db: Database, kind: str) -> set[str]:
    return {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = ?", (kind,))}


def columns(db: Database, table: str) -> set[str]:
    return {r[1] for r in db.query(f"PRAGMA table_info({table})")}


def test_the_twelfth_migration_is_the_alert_remediation():
    migration = load_migrations()[11]
    assert (migration.version, migration.name) == (12, "alert_remediation")


def test_a_database_at_the_previous_version_keeps_everything_when_migrated(tmp_path: Path):
    path = tmp_path / "old.db"
    legacy_database(path)
    old = Database(path, migrations=load_migrations()[:11])
    before = [tuple(r) for r in old.query(f"SELECT seq, {COLUMNS} FROM classifications")]
    items = [tuple(r) for r in old.query("SELECT * FROM items ORDER BY seq")]
    assert NEW_TABLES.isdisjoint(names(old, "table"))
    old.close()
    migrated = Database(path)
    try:
        assert names(migrated, "table") >= NEW_TABLES
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


def test_the_new_columns_exist_with_safe_defaults(tmp_path: Path):
    db = Database(tmp_path / "cols.db")
    try:
        assert {"downgraded_by_model", "overflow", "events_dropped"} <= columns(db, "alerts")
        assert "from_absorbed" in columns(db, "alert_events")
        assert {"label", "confidence", "settled_at"} <= columns(db, "alert_forgotten")
    finally:
        db.close()


def test_every_existing_verdict_that_could_matter_is_marked_for_the_first_pass(tmp_path: Path):
    """A database upgraded from before this migration has no dirty marks: the migration marks the
    verdicts of risk labels (and every member's) so that the first pass looks at them once."""
    path = tmp_path / "old.db"
    legacy_database(path)
    Database(path, migrations=load_migrations()[:11]).close()
    migrated = Database(path)
    try:
        risky = migrated.query(
            "SELECT count(*) FROM classifications WHERE label IN "
            "('unintentional_rumor','deliberate_disinformation','recycled_content')"
        )[0][0]
        marked = migrated.query("SELECT count(*) FROM alert_dirty")[0][0]
        assert marked == risky
    finally:
        migrated.close()


def test_a_new_or_changed_risk_verdict_marks_the_item_and_a_quiet_one_does_not(tmp_path: Path):
    from datetime import UTC, datetime

    from social_tracker.classification.domain import Classification, Label, Stage
    from social_tracker.persistence.classification_repo import SqliteClassificationRepository

    db = Database(tmp_path / "dirty.db")
    try:
        repo = SqliteClassificationRepository(db)
        now = datetime(2026, 3, 1, 12, tzinfo=UTC)

        def verdict(item_id, label, confidence):
            return Classification(
                item_id=item_id,
                label=label,
                confidence=confidence,
                stage=Stage.RULES,
                rationale="x",
                classified_at=now,
                classifier_version="cascade-2/topics-0",
            )

        assert repo.add(verdict("itm_quiet", Label.NO_RISK, 0.9))
        assert repo.add(verdict("itm_risky", Label.UNINTENTIONAL_RUMOR, 0.6))
        marked = {r[0] for r in db.query("SELECT item_id FROM alert_dirty")}
        assert marked == {"itm_risky"}
        with db.transaction() as conn:
            conn.execute("DELETE FROM alert_dirty")
            conn.execute("UPDATE classifications SET confidence = 0.7 WHERE item_id = 'itm_risky'")
            conn.execute("UPDATE classifications SET confidence = 0.8 WHERE item_id = 'itm_quiet'")
        assert {r[0] for r in db.query("SELECT item_id FROM alert_dirty")} == {"itm_risky"}
    finally:
        db.close()


def test_opening_twice_applies_it_once(tmp_path: Path):
    path = tmp_path / "twice.db"
    Database(path).close()
    again = Database(path)
    try:
        assert again.query("SELECT count(*) FROM schema_migrations")[0][0] == len(load_migrations())
    finally:
        again.close()


def test_a_failing_migration_leaves_the_database_at_the_previous_version(tmp_path: Path):
    from social_tracker.persistence.db import Migration

    path = tmp_path / "crash.db"
    Database(path, migrations=load_migrations()[:11]).close()
    broken = Migration(
        12, "alert_remediation", "CREATE TABLE alert_dirty (a TEXT); CREATE TABLE alert_dirty (a);"
    )
    with pytest.raises(StorageError):
        Database(path, migrations=[*load_migrations()[:11], broken])
    again = Database(path, migrations=load_migrations()[:11])
    try:
        assert NEW_TABLES.isdisjoint(names(again, "table"))
    finally:
        again.close()
