"""The SQLite classification ports: the shared contracts plus what is specific to SQLite."""

import sqlite3
import time
from datetime import timedelta
from pathlib import Path

import pytest

from social_tracker.classification.domain import Label
from social_tracker.classification.similarity import band_keys
from social_tracker.classification.similarity import signature as make_minhash
from social_tracker.persistence.classification_repo import (
    SqliteClassificationRepository,
    SqliteSignatureRepository,
    SqliteTopicConfigRepository,
)
from social_tracker.persistence.db import Database
from social_tracker.shared.errors import ConflictError, StorageError
from tests.classification.repository_contracts import (
    NOW,
    ClassificationRepositoryContract,
    SignatureRepositoryContract,
    TopicConfigRepositoryContract,
    item,
    signature,
    stored,
    verdict,
)


class _OnDisk:
    @pytest.fixture(autouse=True)
    def _databases(self, tmp_path: Path):
        self._root = tmp_path
        self._opened: list[Database] = []
        yield
        for db in self._opened:
            db.close()

    def database(self) -> Database:
        db = Database(self._root / f"classification-{len(self._opened)}.db")
        self._opened.append(db)
        return db


class TestSqliteClassificationRepositoryContract(_OnDisk, ClassificationRepositoryContract):
    def make_repo(self):
        return SqliteClassificationRepository(self.database())


class TestSqliteSignatureRepositoryContract(_OnDisk, SignatureRepositoryContract):
    def make_repo(self):
        return SqliteSignatureRepository(self.database())


class TestSqliteTopicConfigRepositoryContract(_OnDisk, TopicConfigRepositoryContract):
    def make_repo(self):
        return SqliteTopicConfigRepository(self.database())


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "app.db")
    yield database
    database.close()


# ---- tampered rows are a storage error, never a half-valid object ----


def test_a_tampered_verdict_is_a_storage_error(db):
    repo = SqliteClassificationRepository(db)
    repo.add(verdict(1))
    with db.transaction() as conn:
        conn.execute('UPDATE classifications SET signals = \'[{"name":"nope"}]\'')
    with pytest.raises(StorageError):
        repo.get(item(1))


def test_a_tampered_signature_is_a_storage_error(db):
    repo = SqliteSignatureRepository(db)
    repo.add(signature(1))
    with db.transaction() as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute("UPDATE item_signatures SET minhash = x'0102'")
        conn.execute("PRAGMA ignore_check_constraints = OFF")
    with pytest.raises(StorageError):
        repo.get(item(1))


def test_a_tampered_configuration_is_a_storage_error(db):
    repo = SqliteTopicConfigRepository(db)
    repo.append(stored())
    with db.transaction() as conn:
        conn.execute("UPDATE topic_config SET config = '{\"version\": 7}'")
    with pytest.raises(StorageError):
        repo.latest()


# ---- the near-duplicate lookup never scans ----


def _plan(db: Database, repo: SqliteSignatureRepository, probe) -> list[str]:
    from social_tracker.persistence import classification_repo as module

    params: list = []
    for band, key in enumerate(band_keys(probe.minhash)):
        params.extend((band, key))
    params.extend((16, probe.content_hash, probe.item_id, 10))
    rows = db.query("EXPLAIN QUERY PLAN " + module._CANDIDATES, params)
    return [str(row["detail"]) for row in rows]


def test_the_candidate_lookup_uses_the_indexes_and_scans_no_signature_table(db):
    repo = SqliteSignatureRepository(db)
    plan = _plan(db, repo, signature(1))
    scans = [line for line in plan if line.startswith("SCAN") and "USING" not in line]
    assert [line for line in scans if "item_signature" in line] == [], plan
    assert any("ix_item_signature_bands_lookup" in line for line in plan), plan
    assert any("ix_item_signatures_hash" in line for line in plan), plan


def _noisy(n: int) -> tuple[int, ...]:
    return make_minhash(tuple(f"word{(n * 7 + k) % 5000}" for k in range(40)))


def test_a_lookup_costs_the_same_with_one_signature_as_with_five_thousand(db):
    repo = SqliteSignatureRepository(db)
    probe = signature(7, minhash=_noisy(10**6))
    repo.add(signature(1, minhash=_noisy(1)))
    statements: list[str] = []

    def lookup() -> float:
        started = time.perf_counter()
        repo.find_candidates(
            item_id=probe.item_id,
            content_hash=probe.content_hash,
            band_keys=band_keys(probe.minhash),
            limit=50,
        )
        return time.perf_counter() - started

    one = min(lookup() for _ in range(5))
    conn = db._connection()
    conn.set_trace_callback(statements.append)
    lookup()
    few = len(statements)
    conn.set_trace_callback(None)

    with db.transaction() as c:
        for n in range(2, 5001):
            minhash = _noisy(n)
            c.execute(
                "INSERT INTO item_signatures (item_id, content_hash, ref_at, minhash) "
                "VALUES (?, ?, ?, ?)",
                (
                    item(n),
                    f"{n:064x}",
                    NOW.isoformat(),
                    b"".join(v.to_bytes(4, "big") for v in minhash),
                ),
            )
            c.executemany(
                "INSERT INTO item_signature_bands (item_id, band, key) VALUES (?, ?, ?)",
                [(item(n), band, key) for band, key in enumerate(band_keys(minhash))],
            )
    statements.clear()
    conn.set_trace_callback(statements.append)
    many_time = min(lookup() for _ in range(5))
    lookup_statements = len(statements) // 5
    conn.set_trace_callback(None)

    assert lookup_statements == few  # the same number of statements: nothing is looped per row
    assert many_time < max(one * 50, 0.05)  # an index lookup, not a scan of 5,000 rows


# ---- retention ----


def test_pruning_removes_the_bands_with_the_signature(db):
    repo = SqliteSignatureRepository(db)
    repo.add(signature(1, days=-10))
    repo.add(signature(2, days=0))
    assert db.query("SELECT count(*) FROM item_signature_bands")[0][0] == 32
    repo.prune_before(NOW - timedelta(days=1))
    assert db.query("SELECT count(*) FROM item_signature_bands")[0][0] == 16
    assert db.query("SELECT DISTINCT item_id FROM item_signature_bands")[0][0] == item(2)


def test_a_signature_and_its_bands_are_written_together_or_not_at_all(db):
    repo = SqliteSignatureRepository(db)
    with pytest.raises(RuntimeError), db.transaction():  # noqa: PT012
        repo.add(signature(1))
        raise RuntimeError
    assert repo.get(item(1)) is None
    assert db.query("SELECT count(*) FROM item_signature_bands")[0][0] == 0


# ---- concurrency ----


def test_two_connections_racing_to_classify_one_item_store_one_verdict(tmp_path):
    first, second = Database(tmp_path / "race.db"), Database(tmp_path / "race.db")
    try:
        a, b = SqliteClassificationRepository(first), SqliteClassificationRepository(second)
        results = [
            a.add(verdict(1, label=Label.UNINTENTIONAL_RUMOR)),
            b.add(verdict(1, label=Label.NO_RISK, confidence=0.1)),
        ]
        assert sorted(results) == [False, True]
        assert a.get(item(1)).label is Label.UNINTENTIONAL_RUMOR
        assert a.stats().total == 1
    finally:
        first.close()
        second.close()


def test_a_locked_database_surfaces_as_a_storage_error(tmp_path):
    path = tmp_path / "locked.db"
    holder = Database(path)
    waiter = Database(path, busy_timeout_ms=50)
    try:
        repo = SqliteClassificationRepository(waiter)
        with holder.transaction(), pytest.raises(StorageError):
            repo.add(verdict(1))
        assert repo.add(verdict(1)) is True  # released: it works again
    finally:
        holder.close()
        waiter.close()


# ---- the schema is the last line of defence ----


def raw(db: Database, sql: str, params=()) -> None:
    with db.transaction() as conn:
        conn.execute(sql, params)


BASE = (
    "itm_one", "no_risk", 0.5, "rules", "r", "[]", "[]", 0, 0, None, None, None, 0, 0,
    "unavailable", 0.0, None, 3, "2026-03-01T08:00:00+00:00", "v1",
)  # fmt: skip
COLUMNS = (
    "item_id, label, confidence, stage, rationale, signals, topics, low_confidence, needs_llm, "
    "duplicate_of, recycled_of, near_of, awaiting_original, truncated, catalog_status, priority, "
    "scope_reason, source_credibility_used, classified_at, classifier_version"
)


def insert_row(db: Database, **changes):
    names = COLUMNS.split(", ")
    row = dict(zip(names, BASE, strict=True)) | changes
    marks = ", ".join("?" for _ in row)
    raw(db, f"INSERT INTO classifications ({', '.join(row)}) VALUES ({marks})", list(row.values()))


def test_a_valid_row_is_accepted(db):
    insert_row(db)
    assert db.query("SELECT count(*) FROM classifications")[0][0] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"item_id": "ab"},
        {"item_id": "Itm_One"},
        {"label": "other"},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"stage": "magic"},
        {"rationale": ""},
        {"rationale": "x" * 301},
        {"signals": "{}"},
        {"signals": "not json"},
        {"signals": "[" + ",".join(["0"] * 3000) + "]"},
        {"topics": "{}"},
        {"low_confidence": 2},
        {"needs_llm": 2},
        {"truncated": -1},
        {"duplicate_of": "Bad Id"},
        {"duplicate_of": "itm_one"},
        {"recycled_of": "itm_one"},
        {"duplicate_of": "itm_two", "recycled_of": "itm_two"},
        {"duplicate_of": "itm_two", "near_of": "itm_three"},
        {"recycled_of": "itm_two", "near_of": "itm_three"},
        {"near_of": "itm_one"},
        {"near_of": "Bad Id"},
        {"awaiting_original": 2},
        {"awaiting_original": 1},  # only a duplicate can wait for its original
        {"awaiting_original": 1, "recycled_of": "itm_two"},
        {"scope_reason": "english", "label": "out_of_scope"},
        {"duplicate_of": "itm_two", "needs_llm": 1},
        {"catalog_status": "Bad"},
        {"catalog_status": ""},
        {"priority": 1.5},
        {"scope_reason": "bored"},
        {"scope_reason": "no_topic_match"},  # a reason without the out_of_scope label
        {"label": "out_of_scope"},  # the label without a reason
        {"label": "out_of_scope", "scope_reason": "no_topic_match", "needs_llm": 1},
        {"source_credibility_used": 0},
        {"source_credibility_used": 6},
        {"classified_at": "yesterday"},
        {"classifier_version": ""},
        {"classifier_version": "v" * 65},
    ],
)
def test_the_schema_rejects_an_invalid_row(db, changes):
    with pytest.raises(StorageError):
        insert_row(db, **changes)
    assert db.query("SELECT count(*) FROM classifications")[0][0] == 0


def test_the_schema_accepts_the_boundary_values(db):
    insert_row(db, item_id="itm_a", confidence=0.0, priority=0.0, source_credibility_used=1)
    insert_row(db, item_id="itm_b", confidence=1.0, priority=1.0, source_credibility_used=5)
    insert_row(db, item_id="itm_c", rationale="x" * 300)
    insert_row(db, item_id="itm_d", label="out_of_scope", scope_reason="empty_text")
    insert_row(db, item_id="itm_e", label="out_of_scope", scope_reason="non_spanish")
    insert_row(db, item_id="itm_f", near_of="itm_x", needs_llm=1)
    insert_row(db, item_id="itm_g", duplicate_of="itm_x", awaiting_original=1)


def test_the_schema_keeps_one_verdict_per_item(db):
    insert_row(db)
    with pytest.raises(ConflictError):
        insert_row(db, label="unclassified")


@pytest.mark.parametrize(
    "changes",
    [
        {"item_id": "ab"},
        {"content_hash": "A" * 64},
        {"content_hash": "a" * 63},
        {"ref_at": "tomorrow"},
        {"minhash": b"\x00" * 319},
        {"minhash": b"\x00" * 321},
        {"minhash": "text" * 80},
    ],
)
def test_the_schema_rejects_an_invalid_signature_row(db, changes):
    row = {
        "item_id": "itm_one",
        "content_hash": "a" * 64,
        "ref_at": NOW.isoformat(),
        "minhash": b"\x00" * 320,
    }
    row.update(changes)
    with pytest.raises(StorageError):
        raw(
            db,
            "INSERT INTO item_signatures (item_id, content_hash, ref_at, minhash) "
            "VALUES (?, ?, ?, ?)",
            list(row.values()),
        )


def test_the_schema_rejects_a_band_out_of_range_and_an_orphan_band(db):
    with pytest.raises(StorageError):
        raw(db, "INSERT INTO item_signature_bands (item_id, band, key) VALUES ('itm_one', 0, 1)")
    SqliteSignatureRepository(db).add(signature(1))
    with pytest.raises(StorageError):
        raw(
            db,
            "INSERT INTO item_signature_bands (item_id, band, key) VALUES (?, 16, 1)",
            (item(1),),
        )


@pytest.mark.parametrize(
    "row",
    [
        (0, "{}", "2026-03-01T08:00:00+00:00", "a", "r"),
        (1, "[]", "2026-03-01T08:00:00+00:00", "a", "r"),
        (1, "nope", "2026-03-01T08:00:00+00:00", "a", "r"),
        (1, "{}", "never", "a", "r"),
        (1, "{}", "2026-03-01T08:00:00+00:00", "", "r"),
        (1, "{}", "2026-03-01T08:00:00+00:00", "a" * 101, "r"),
        (1, "{}", "2026-03-01T08:00:00+00:00", "a", ""),
        (1, "{}", "2026-03-01T08:00:00+00:00", "a", "r" * 501),
    ],
)
def test_the_schema_rejects_an_invalid_configuration_row(db, row):
    with pytest.raises(StorageError):
        raw(
            db,
            "INSERT INTO topic_config (version, config, created_at, actor, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            row,
        )


def test_the_pending_items_index_exists_and_serves_the_queue(db):
    plan = db.query(
        "EXPLAIN QUERY PLAN SELECT * FROM items WHERE status = 'new' "
        "ORDER BY published_at, id LIMIT 10"
    )
    text = " ".join(str(r["detail"]) for r in plan)
    assert "ix_items_status_published" in text
    assert "TEMP B-TREE" not in text  # already in order


def test_the_migration_is_recorded(db):
    versions = [r[0] for r in db.query("SELECT version FROM schema_migrations ORDER BY 1")]
    assert versions == [1, 2, 3, 4, 5, 6, 7, 8]
    tables = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"classifications", "item_signatures", "item_signature_bands", "topic_config"} <= tables
    assert sqlite3.sqlite_version_info >= (3, 35)
