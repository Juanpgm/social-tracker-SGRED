"""SQLite implementations of the classification ports (migration 0004).

Validation lives in the domain (`Classification`, `ItemSignature`, `StoredTopicConfig`) and is
applied here on every call, exactly like the in-memory references. Every row read back is rebuilt
through the domain, so a tampered row is a `StorageError`, never a half-valid object.

Idempotence is decided by the database: `add` is `INSERT ... ON CONFLICT DO NOTHING` on the
unique item id (never `OR IGNORE`, which would also swallow a CHECK violation), so two racing
classifiers store one verdict between them and the loser is told so.

The near-duplicate lookup never scans: candidates come from the `(band, key)` index and the
`content_hash` index, and are only then joined to their signatures.
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
    "duplicate_of, recycled_of, truncated, catalog_status, priority, scope_reason, "
    "source_credibility_used, classified_at, classifier_version"
)
_INSERT_VERDICT = (
    f"INSERT INTO classifications ({_COLUMNS}) VALUES ({', '.join('?' * 18)}) "
    "ON CONFLICT (item_id) DO NOTHING"
)
_COUNTS = (
    "SELECT count(*), COALESCE(SUM(needs_llm), 0), COALESCE(SUM(low_confidence), 0), "
    "COALESCE(SUM(duplicate_of IS NOT NULL), 0), COALESCE(SUM(recycled_of IS NOT NULL), 0), "
    "COALESCE(SUM(truncated), 0) FROM classifications"
)
_CANDIDATES = (
    "SELECT s.item_id, s.content_hash, s.ref_at, s.minhash FROM ("
    "SELECT item_id FROM item_signatures WHERE content_hash = ? "
    "UNION SELECT item_id FROM item_signature_bands WHERE "
    + " OR ".join(["(band = ? AND key = ?)"] * BANDS)
    + ") c JOIN item_signatures s ON s.item_id = c.item_id WHERE s.item_id <> ? "
    "ORDER BY s.ref_at, s.item_id LIMIT ?"
)


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
            truncated=bool(row["truncated"]),
            catalog_status=row["catalog_status"],
            priority=row["priority"],
            scope_reason=row["scope_reason"],
            source_credibility_used=row["source_credibility_used"],
            classified_at=parse_utc(row["classified_at"]),
            classifier_version=row["classifier_version"],
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
        int(c.truncated),
        c.catalog_status,
        c.priority,
        c.scope_reason.value if c.scope_reason is not None else None,
        c.source_credibility_used,
        format_utc(c.classified_at),
        c.classifier_version,
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
        total, needs_llm, low, duplicates, recycled, truncated = (int(n) for n in counts)
        return ClassificationStats(
            total=total,
            by_label=tuple((str(name), int(n)) for name, n in labels),
            out_of_scope_by_reason=tuple((str(name), int(n)) for name, n in reasons),
            needs_llm=needs_llm,
            low_confidence=low,
            duplicates=duplicates,
            recycled=recycled,
            truncated=truncated,
        )


def _signature_from_row(row: sqlite3.Row) -> ItemSignature:
    try:
        return ItemSignature(
            item_id=row["item_id"],
            content_hash=row["content_hash"],
            ref_at=parse_utc(row["ref_at"]),
            minhash=unpack(row["minhash"]),
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
                "INSERT INTO item_signatures (item_id, content_hash, ref_at, minhash) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (item_id) DO NOTHING",
                (
                    signature.item_id,
                    signature.content_hash,
                    format_utc(signature.ref_at),
                    pack(signature.minhash),
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
            "SELECT item_id, content_hash, ref_at, minhash FROM item_signatures WHERE item_id = ?",
            (item_id,),
        )
        return _signature_from_row(rows[0]) if rows else None

    def find_candidates(
        self, *, item_id: str, content_hash: str, band_keys: Sequence[int], limit: int
    ) -> list[ItemSignature]:
        validate_limit(limit)
        keys = _band_keys(band_keys)
        params: list[Any] = [content_hash]
        for band, key in enumerate(keys):
            params.extend((band, key))
        params.extend((item_id, limit))
        return [_signature_from_row(row) for row in self._db.query(_CANDIDATES, params)]

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
