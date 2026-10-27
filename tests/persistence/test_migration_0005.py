"""Migration 0005 on a database that already holds data: nothing may be lost or changed."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from social_tracker.classification.domain import Label
from social_tracker.persistence.classification_repo import (
    SqliteClassificationRepository,
    SqliteSignatureRepository,
)
from social_tracker.persistence.db import Database, load_migrations
from social_tracker.shared.errors import StorageError

COLUMNS = (
    "item_id, label, confidence, stage, rationale, signals, topics, low_confidence, needs_llm, "
    "duplicate_of, recycled_of, truncated, catalog_status, priority, scope_reason, "
    "source_credibility_used, classified_at, classifier_version"
)


def legacy_database(path: Path) -> None:
    """A database as 0004 left it, with a verdict of each kind and a signature with its bands."""
    db = Database(path, migrations=load_migrations()[:4])
    try:
        with db.transaction() as conn:
            rows = [
                ("itm_old1", "unintentional_rumor", 1, 1, None, None, None),
                ("itm_old2", "out_of_scope", 0, 0, None, None, "no_topic_match"),
                ("itm_old3", "no_risk", 0, 0, "itm_old1", None, None),
                ("itm_old4", "recycled_content", 0, 0, None, "itm_old1", None),
            ]
            for item_id, label, low, llm, dup, rec, reason in rows:
                values = (
                    item_id, label, 0.55, "rules", f"legacy {item_id}", "[]", '["earthquake"]',
                    low, llm, dup, rec, 0, "unavailable", 0.31, reason, 3,
                    "2026-03-01T08:00:00+00:00", "cascade-2/topics-0",
                )  # fmt: skip
                conn.execute(
                    f"INSERT INTO classifications ({COLUMNS}) VALUES ({', '.join('?' * 18)})",
                    values,
                )
            conn.execute(
                "INSERT INTO item_signatures (item_id, content_hash, ref_at, minhash) "
                "VALUES ('itm_old1', ?, '2026-03-01T08:00:00+00:00', ?)",
                ("a" * 64, bytes(range(256)) + bytes(64)),
            )
            conn.executemany(
                "INSERT INTO item_signature_bands (item_id, band, key) VALUES ('itm_old1', ?, ?)",
                [(band, band * 1000) for band in range(16)],
            )
            conn.execute(
                "INSERT INTO items (id, source_id, source_name, kind, external_id, title, text, "
                "published_at, fetched_at, content_hash, flags, status, raw_payload) VALUES "
                "('itm_old1', 'src_one', 'One', 'article', 'e1', 't', 'x', "
                "'2026-03-01T08:00:00+00:00', '2026-03-01T08:00:00+00:00', ?, '[]', 'classified', "
                "'{}')",
                ("b" * 64,),
            )
    finally:
        db.close()


@pytest.fixture
def migrated(tmp_path: Path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    before = Database(path, migrations=load_migrations()[:4])
    snapshot = [tuple(r) for r in before.query(f"SELECT seq, {COLUMNS} FROM classifications")]
    before.close()
    db = Database(path)
    yield db, snapshot
    db.close()


def test_every_stored_verdict_survives_the_rebuild_unchanged(migrated):
    db, snapshot = migrated
    after = [tuple(r) for r in db.query(f"SELECT seq, {COLUMNS} FROM classifications ORDER BY seq")]
    assert after == sorted(snapshot)
    assert len(after) == 4
    extra = db.query("SELECT near_of, awaiting_original FROM classifications")
    assert {(r[0], r[1]) for r in extra} == {(None, 0)}


def test_the_verdicts_read_back_through_the_repository(migrated):
    db, _ = migrated
    repo = SqliteClassificationRepository(db)
    first = repo.get("itm_old1")
    assert (first.label, first.needs_llm, first.near_of, first.awaiting_original) == (
        Label.UNINTENTIONAL_RUMOR,
        True,
        None,
        False,
    )
    assert repo.get("itm_old4").recycled_of == "itm_old1"
    assert repo.stats().total == 4


def test_the_signature_and_its_bands_survive_and_gain_empty_digests(migrated):
    db, _ = migrated
    signature = SqliteSignatureRepository(db).get("itm_old1")
    assert (signature.text_hash, signature.digits_hash, signature.n_tokens) == (None, None, 0)
    assert signature.content_hash == "a" * 64
    assert db.query("SELECT count(*) FROM item_signature_bands")[0][0] == 16


def test_an_old_signature_still_matches_by_identical_content_hash(migrated):
    db, _ = migrated
    found = SqliteSignatureRepository(db).find_exact(
        item_id="itm_new1", content_hash="a" * 64, text_hash="c" * 64
    )
    assert [s.item_id for s in found] == ["itm_old1"]


def test_the_indexes_and_the_new_tables_exist_after_the_rebuild(migrated):
    db, _ = migrated
    names = {
        r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type IN ('index','table')")
    }
    assert {
        "ix_classifications_label",
        "ix_classifications_queue",
        "ix_classifications_awaiting",
        "ix_classifications_classified_at",
        "ix_item_signatures_text_hash",
        "classification_failures",
    } <= names
    assert "classifications_rebuilt" not in names
    plan = " ".join(
        str(r["detail"])
        for r in db.query(
            "EXPLAIN QUERY PLAN SELECT item_id FROM classifications WHERE needs_llm = 1 "
            "ORDER BY priority DESC, classified_at, item_id"
        )
    )
    assert "ix_classifications_queue" in plan


def test_the_new_constraints_apply_to_rows_after_the_rebuild(migrated):
    db, _ = migrated
    repo = SqliteClassificationRepository(db)
    assert repo.note_failure("itm_old1", datetime.now(UTC)) == 1
    with pytest.raises(StorageError), db.transaction() as conn:  # noqa: PT012
        conn.execute("UPDATE classifications SET awaiting_original = 1 WHERE item_id = 'itm_old2'")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE items SET status = 'classification_failed' WHERE id = 'itm_old1'"
        )  # the status needs no schema change
    assert db.query("SELECT status FROM items")[0][0] == "classification_failed"


def test_the_migration_is_recorded_once_and_reopening_changes_nothing(tmp_path):
    path = tmp_path / "again.db"
    legacy_database(path)
    for _ in range(2):
        db = Database(path)
        try:
            versions = [r[0] for r in db.query("SELECT version FROM schema_migrations ORDER BY 1")]
            assert versions == list(range(1, len(load_migrations()) + 1))
            assert db.query("SELECT count(*) FROM classifications")[0][0] == 4
        finally:
            db.close()


def test_the_rebuild_does_not_touch_the_collected_items(migrated):
    db, _ = migrated
    assert db.query("SELECT count(*) FROM items")[0][0] == 1
    assert db.query("SELECT count(*) FROM items_fts WHERE items_fts MATCH 'x'")[0][0] == 1
