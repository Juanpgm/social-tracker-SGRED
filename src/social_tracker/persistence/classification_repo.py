"""SQLite implementations of the classification ports (migrations 0004 and 0005).

Validation lives in the domain (`Classification`, `ItemSignature`, `StoredTopicConfig`) and is
applied here on every call, exactly like the in-memory references. Every row read back is rebuilt
through the domain, so a tampered row is a `StorageError`, never a half-valid object.

Idempotence is decided by the database: `add` is `INSERT ... ON CONFLICT DO NOTHING` on the
unique item id (never `OR IGNORE`, which would also swallow a CHECK violation), so two racing
classifiers store one verdict between them and the loser is told so.

The near-duplicate lookups never scan: the exact lookup comes from the `content_hash` and
`text_hash` indexes (it is never limited by the candidate cap of the bands), the candidates from
the `(band, key)` index ranked by the number of bands they share, and are only then joined to
their signatures.
"""

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from social_tracker.classification.domain import (
    Classification,
    ClassificationStats,
    ItemSignature,
    signals_from_json,
    signals_to_json,
)
from social_tracker.classification.similarity import BANDS, band_keys, pack, unpack
from social_tracker.classification.topics import StoredTopicConfig, TopicConfig
from social_tracker.items.domain import validate_limit
from social_tracker.persistence.db import Database, format_utc, map_sqlite_error, parse_utc
from social_tracker.shared.errors import DomainError, StorageError, ValidationError

_COLUMNS = (
    "item_id, label, confidence, stage, rationale, signals, topics, low_confidence, needs_llm, "
    "duplicate_of, recycled_of, near_of, awaiting_original, truncated, catalog_status, priority, "
    "scope_reason, source_credibility_used, classified_at, classifier_version, "
    "settled_at, settle_reason"
)
_INSERT_VERDICT = (
    f"INSERT INTO classifications ({_COLUMNS}) VALUES ({', '.join('?' * 22)}) "
    "ON CONFLICT (item_id) DO NOTHING"
)
_SETTLE = (
    "UPDATE classifications SET label = ?, confidence = ?, stage = ?, rationale = ?, "
    "low_confidence = ?, needs_llm = ?, awaiting_original = ?, priority = ?, "
    "duplicate_of = ?, near_of = ?, settled_at = ?, settle_reason = ? "
    "WHERE item_id = ? AND (needs_llm = 1 OR awaiting_original = 1)"
)
_REOPEN = (
    "UPDATE classifications SET needs_llm = 1, priority = ?, settled_at = NULL, "
    "settle_reason = NULL WHERE item_id = ? AND needs_llm = 0 AND settle_reason = 'parked'"
)
# A waiting copy is "live" while its original still waits for the judge (`needs_llm`).
_LIVE_ORIGINAL = (
    "EXISTS (SELECT 1 FROM classifications o WHERE o.item_id = classifications.duplicate_of "
    "AND o.needs_llm = 1)"
)
_COUNTS = (
    "SELECT count(*), COALESCE(SUM(needs_llm), 0), COALESCE(SUM(low_confidence), 0), "
    "COALESCE(SUM(duplicate_of IS NOT NULL), 0), COALESCE(SUM(recycled_of IS NOT NULL), 0), "
    "COALESCE(SUM(truncated), 0), COALESCE(SUM(near_of IS NOT NULL), 0), "
    f"COALESCE(SUM(awaiting_original AND {_LIVE_ORIGINAL}), 0), "
    f"COALESCE(SUM(awaiting_original AND NOT {_LIVE_ORIGINAL}), 0) FROM classifications"
)
# The copies that waited for an original that is about to be forgotten become normal queued items:
# no link, the priority the original had (never under 0.1).
_RELEASE_COPIES = (
    "UPDATE classifications SET awaiting_original = 0, duplicate_of = NULL, needs_llm = 1, "
    "priority = MAX(0.1, COALESCE((SELECT o.priority FROM classifications o "
    "WHERE o.item_id = classifications.duplicate_of), 0.1)) "
    "WHERE awaiting_original = 1 AND duplicate_of IN ({marks})"
)
_SIGNATURE_COLUMNS = "item_id, content_hash, ref_at, minhash, text_hash, digits_hash, n_tokens"
_CANDIDATES = (
    "SELECT s.item_id, s.content_hash, s.ref_at, s.minhash, s.text_hash, s.digits_hash, "
    "s.n_tokens FROM ("
    "SELECT item_id, count(*) AS shared FROM item_signature_bands WHERE "
    + " OR ".join(["(band = ? AND key = ?)"] * BANDS)
    + " GROUP BY item_id "
    "UNION SELECT item_id, ? FROM item_signatures WHERE content_hash = ? "
    ") c JOIN item_signatures s ON s.item_id = c.item_id WHERE s.item_id <> ? "
    "GROUP BY s.item_id ORDER BY max(c.shared) DESC, s.ref_at, s.item_id LIMIT ?"
)


def _release_copies(conn: sqlite3.Connection, ids: Sequence[str]) -> None:
    """Before verdicts are forgotten: nobody keeps waiting for an original that is gone."""
    marks = ", ".join("?" * len(ids))
    conn.execute(_RELEASE_COPIES.format(marks=marks), list(ids))


def _forget_escalation(conn: sqlite3.Connection, ids: Sequence[str]) -> None:
    """The judgments and the bookkeeping of the LLM stage go with the verdict they belong to, so
    an item sent back to the classifier is judged afresh (migration 0008)."""
    marks = ", ".join("?" * len(ids))
    conn.execute(f"DELETE FROM llm_judgments WHERE item_id IN ({marks})", list(ids))
    conn.execute(f"DELETE FROM llm_escalations WHERE item_id IN ({marks})", list(ids))


def _verdict_from_row(row: sqlite3.Row) -> Classification:
    try:
        return Classification(
            item_id=row["item_id"],
            label=row["label"],
            confidence=row["confidence"],
            stage=row["stage"],
            rationale=row["rationale"],
            signals=signals_from_json(row["signals"]),
            topics=tuple(json.loads(row["topics"])),
            low_confidence=bool(row["low_confidence"]),
            needs_llm=bool(row["needs_llm"]),
            duplicate_of=row["duplicate_of"],
            recycled_of=row["recycled_of"],
            near_of=row["near_of"],
            awaiting_original=bool(row["awaiting_original"]),
            truncated=bool(row["truncated"]),
            catalog_status=row["catalog_status"],
            priority=row["priority"],
            scope_reason=row["scope_reason"],
            source_credibility_used=row["source_credibility_used"],
            classified_at=parse_utc(row["classified_at"]),
            classifier_version=row["classifier_version"],
            settled_at=parse_utc(row["settled_at"]) if row["settled_at"] is not None else None,
            settle_reason=row["settle_reason"],
        )
    except (DomainError, ValueError, TypeError):
        raise StorageError("stored classification is invalid") from None


def _verdict_params(c: Classification) -> tuple[Any, ...]:
    return (
        c.item_id,
        c.label.value,
        c.confidence,
        c.stage.value,
        c.rationale,
        signals_to_json(c.signals),
        json.dumps([t.value for t in c.topics]),
        int(c.low_confidence),
        int(c.needs_llm),
        c.duplicate_of,
        c.recycled_of,
        c.near_of,
        int(c.awaiting_original),
        int(c.truncated),
        c.catalog_status,
        c.priority,
        c.scope_reason.value if c.scope_reason is not None else None,
        c.source_credibility_used,
        format_utc(c.classified_at),
        c.classifier_version,
        format_utc(c.settled_at) if c.settled_at is not None else None,
        c.settle_reason,
    )


class SqliteClassificationRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, classification: Classification) -> bool:
        if not isinstance(classification, Classification):
            raise ValidationError("a classification is required")
        with self._db.transaction() as conn:
            return conn.execute(_INSERT_VERDICT, _verdict_params(classification)).rowcount > 0

    def get(self, item_id: str) -> Classification | None:
        rows = self._db.query(
            f"SELECT {_COLUMNS} FROM classifications WHERE item_id = ?", (item_id,)
        )
        return _verdict_from_row(rows[0]) if rows else None

    def stats(self) -> ClassificationStats:
        try:
            with self._db.snapshot() as conn:
                counts = conn.execute(_COUNTS).fetchone()
                labels = conn.execute(
                    "SELECT label, count(*) FROM classifications GROUP BY label ORDER BY label"
                ).fetchall()
                reasons = conn.execute(
                    "SELECT scope_reason, count(*) FROM classifications "
                    "WHERE scope_reason IS NOT NULL GROUP BY scope_reason ORDER BY scope_reason"
                ).fetchall()
        except sqlite3.Error as exc:
            raise map_sqlite_error(exc) from None
        total, needs_llm, low, duplicates, recycled, truncated, near, awaiting, orphaned = (
            int(n) for n in counts
        )
        return ClassificationStats(
            total=total,
            by_label=tuple((str(name), int(n)) for name, n in labels),
            out_of_scope_by_reason=tuple((str(name), int(n)) for name, n in reasons),
            needs_llm=needs_llm,
            low_confidence=low,
            duplicates=duplicates,
            recycled=recycled,
            truncated=truncated,
            near_duplicates=near,
            awaiting_original=awaiting,
            orphaned_awaiting=orphaned,
        )

    def settle(self, classification: Classification) -> bool:
        if not isinstance(classification, Classification):
            raise ValidationError("a classification is required")
        c = classification
        with self._db.transaction() as conn:
            return (
                conn.execute(
                    _SETTLE,
                    (
                        c.label.value,
                        c.confidence,
                        c.stage.value,
                        c.rationale,
                        int(c.low_confidence),
                        int(c.needs_llm),
                        int(c.awaiting_original),
                        c.priority,
                        c.duplicate_of,
                        c.near_of,
                        format_utc(c.settled_at) if c.settled_at is not None else None,
                        c.settle_reason,
                        c.item_id,
                    ),
                ).rowcount
                > 0
            )

    def reopen(self, item_id: str, priority: float) -> bool:
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int | float)
            or not 0.0 <= priority <= 1.0
        ):
            raise ValidationError("a priority is a number from 0 to 1")
        with self._db.transaction() as conn:
            return conn.execute(_REOPEN, (priority, item_id)).rowcount > 0

    def awaiting(self, original_id: str) -> list[Classification]:
        rows = self._db.query(
            f"SELECT {_COLUMNS} FROM classifications "
            "WHERE awaiting_original = 1 AND duplicate_of = ? ORDER BY item_id",
            (original_id,),
        )
        return [_verdict_from_row(row) for row in rows]

    def delete(self, item_id: str) -> bool:
        with self._db.transaction() as conn:
            _release_copies(conn, [item_id])
            _forget_escalation(conn, [item_id])
            conn.execute("DELETE FROM classification_failures WHERE item_id = ?", (item_id,))
            removed = conn.execute("DELETE FROM classifications WHERE item_id = ?", (item_id,))
            return removed.rowcount > 0

    def older_than(self, cutoff: datetime, limit: int, offset: int = 0) -> list[str]:
        validate_limit(limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValidationError("offset must be a whole number from 0")
        rows = self._db.query(
            "SELECT item_id FROM classifications WHERE classified_at < ? "
            "ORDER BY classified_at, item_id LIMIT ? OFFSET ?",
            (format_utc(cutoff), limit, offset),
        )
        return [str(row[0]) for row in rows]

    def delete_many(self, item_ids: Sequence[str]) -> int:
        ids = list(item_ids)
        if not ids:
            return 0
        marks = ", ".join("?" * len(ids))
        with self._db.transaction() as conn:
            _release_copies(conn, ids)
            _forget_escalation(conn, ids)
            conn.execute(f"DELETE FROM classification_failures WHERE item_id IN ({marks})", ids)
            removed = conn.execute(f"DELETE FROM classifications WHERE item_id IN ({marks})", ids)
            return removed.rowcount

    def note_failure(self, item_id: str, at: datetime) -> int:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO classification_failures (item_id, failures, last_at) VALUES (?, 1, ?) "
                "ON CONFLICT (item_id) DO UPDATE SET failures = MIN(failures + 1, 1000), "
                "last_at = excluded.last_at",
                (item_id, format_utc(at)),
            )
            row = conn.execute(
                "SELECT failures FROM classification_failures WHERE item_id = ?", (item_id,)
            ).fetchone()
            return int(row[0])

    def clear_failure(self, item_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM classification_failures WHERE item_id = ?", (item_id,))


def _signature_from_row(row: sqlite3.Row) -> ItemSignature:
    try:
        return ItemSignature(
            item_id=row["item_id"],
            content_hash=row["content_hash"],
            ref_at=parse_utc(row["ref_at"]),
            minhash=unpack(row["minhash"]),
            text_hash=row["text_hash"],
            digits_hash=row["digits_hash"],
            n_tokens=row["n_tokens"],
        )
    except (DomainError, ValueError, TypeError):
        raise StorageError("stored signature is invalid") from None


def _band_keys(keys: Sequence[int]) -> tuple[int, ...]:
    ok = (
        isinstance(keys, tuple | list)
        and len(keys) == BANDS
        and all(
            isinstance(k, int) and not isinstance(k, bool) and -(2**63) <= k < 2**63 for k in keys
        )
    )
    if not ok:
        raise ValidationError("one 64-bit key per band is required")
    return tuple(keys)


class SqliteSignatureRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, signature: ItemSignature) -> bool:
        if not isinstance(signature, ItemSignature):
            raise ValidationError("a signature is required")
        keys = band_keys(signature.minhash)
        with self._db.transaction() as conn:
            stored = conn.execute(
                f"INSERT INTO item_signatures ({_SIGNATURE_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (item_id) DO NOTHING",
                (
                    signature.item_id,
                    signature.content_hash,
                    format_utc(signature.ref_at),
                    pack(signature.minhash),
                    signature.text_hash,
                    signature.digits_hash,
                    signature.n_tokens,
                ),
            ).rowcount
            if not stored:
                return False
            conn.executemany(
                "INSERT INTO item_signature_bands (item_id, band, key) VALUES (?, ?, ?)",
                [(signature.item_id, band, key) for band, key in enumerate(keys)],
            )
            return True

    def get(self, item_id: str) -> ItemSignature | None:
        rows = self._db.query(
            f"SELECT {_SIGNATURE_COLUMNS} FROM item_signatures WHERE item_id = ?", (item_id,)
        )
        return _signature_from_row(rows[0]) if rows else None

    def find_candidates(
        self, *, item_id: str, content_hash: str, band_keys: Sequence[int], limit: int
    ) -> list[ItemSignature]:
        validate_limit(limit)
        keys = _band_keys(band_keys)
        params: list[Any] = []
        for band, key in enumerate(keys):
            params.extend((band, key))
        params.extend((BANDS, content_hash, item_id, limit))
        return [_signature_from_row(row) for row in self._db.query(_CANDIDATES, params)]

    def find_exact(
        self,
        *,
        item_id: str,
        content_hash: str,
        text_hash: str | None,
        ref_from: datetime | None = None,
        ref_to: datetime | None = None,
        limit: int = 100,
    ) -> list[ItemSignature]:
        validate_limit(limit)
        same = "(content_hash = ?" + (" OR text_hash = ?" if text_hash else "") + ")"
        params: list[Any] = [content_hash] + ([text_hash] if text_hash else [])
        sql = f"SELECT {_SIGNATURE_COLUMNS} FROM item_signatures WHERE {same} AND item_id <> ?"
        params.append(item_id)
        if ref_from is not None:
            sql += " AND ref_at >= ?"
            params.append(format_utc(ref_from))
        if ref_to is not None:
            sql += " AND ref_at <= ?"
            params.append(format_utc(ref_to))
        rows = self._db.query(f"{sql} ORDER BY ref_at, item_id LIMIT ?", [*params, limit])
        return [_signature_from_row(row) for row in rows]

    def prune_before(self, cutoff: datetime) -> int:
        text = format_utc(cutoff)  # raises ValidationError for a naive or non-datetime value
        with self._db.transaction() as conn:
            return conn.execute("DELETE FROM item_signatures WHERE ref_at < ?", (text,)).rowcount


def _stored_config_from_row(row: sqlite3.Row) -> StoredTopicConfig:
    try:
        return StoredTopicConfig(
            config=TopicConfig.from_json(row["config"]),
            created_at=parse_utc(row["created_at"]),
            actor=row["actor"],
            reason=row["reason"],
            version=row["version"],
        )
    except (DomainError, ValueError, TypeError):
        raise StorageError("stored topic configuration is invalid") from None


class SqliteTopicConfigRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def latest(self) -> StoredTopicConfig | None:
        rows = self._db.query(
            "SELECT version, config, created_at, actor, reason FROM topic_config "
            "ORDER BY version DESC LIMIT 1"
        )
        return _stored_config_from_row(rows[0]) if rows else None

    def append(self, stored: StoredTopicConfig) -> int:
        if not isinstance(stored, StoredTopicConfig):
            raise ValidationError("a stored configuration is required")
        text = stored.config.to_json()
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO topic_config (version, config, created_at, actor, reason) VALUES "
                "((SELECT COALESCE(MAX(version), 0) + 1 FROM topic_config), ?, ?, ?, ?)",
                (text, format_utc(stored.created_at), stored.actor, stored.reason),
            )
            return int(cursor.lastrowid or 0)

    def versions(self, limit: int) -> list[StoredTopicConfig]:
        validate_limit(limit)
        rows = self._db.query(
            "SELECT version, config, created_at, actor, reason FROM topic_config "
            "ORDER BY version DESC LIMIT ?",
            (limit,),
        )
        return [_stored_config_from_row(row) for row in rows]
