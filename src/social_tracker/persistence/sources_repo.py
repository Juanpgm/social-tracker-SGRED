"""SQLite implementation of `SourceRepository`.

Common columns are real columns; the per-type fields live in `config_json`. Reading a row goes
back through `parse_source`, so a row that was tampered with (or written by a buggy version)
surfaces as `StorageError` and never as a half-valid `Source`.
"""

import json
import sqlite3
from typing import Any

from social_tracker.persistence.db import Database, format_utc
from social_tracker.shared.errors import (
    NotFoundError,
    StorageError,
    ValidationError,
)
from social_tracker.sources.domain import Source, SourceBase, normalize_name, parse_source

# Fields that have their own column; everything else a source carries goes into `config_json`.
_COLUMN_FIELDS = frozenset(
    {
        "id",
        "name",
        "credibility",
        "enabled",
        "secret_ref",
        "type",
        "mode",
        "created_at",
        "updated_at",
    }
)
_COLUMNS = (
    "id, name, normalized_name, type, mode, credibility, enabled, secret_ref, config_json, "
    "created_at, updated_at"
)
_NOT_FOUND = "source not found"
_FLAGS = {0: False, 1: True}  # anything else stays as is and fails strict validation


def _require_source(source: Any) -> Source:
    if not isinstance(source, SourceBase):
        raise ValidationError("expected a Source")
    return source  # type: ignore[return-value]


def _to_params(source: Source) -> tuple[Any, ...]:
    config = {k: v for k, v in source.model_dump().items() if k not in _COLUMN_FIELDS}
    return (
        source.id,
        source.name,
        source.normalized_name,
        source.type,
        source.mode,
        source.credibility,
        int(source.enabled),
        source.secret_ref,
        json.dumps(config, sort_keys=True),
        format_utc(source.created_at),
        format_utc(source.updated_at),
    )


def _from_row(row: sqlite3.Row) -> Source:
    try:
        config = json.loads(row["config_json"])
        if not isinstance(config, dict) or _COLUMN_FIELDS & config.keys():
            raise ValueError("config_json must hold only type-specific fields")
        return parse_source(
            {
                **config,
                "id": row["id"],
                "name": row["name"],
                "type": row["type"],
                "mode": row["mode"],
                "credibility": row["credibility"],
                "enabled": _FLAGS.get(row["enabled"], row["enabled"]),
                "secret_ref": row["secret_ref"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    except (ValueError, TypeError, RecursionError, ValidationError):
        raise StorageError("stored source row is invalid") from None


class SqliteSourceRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, source: Source) -> None:
        params = _to_params(_require_source(source))
        with self._db.transaction() as conn:
            conn.execute(
                f"INSERT INTO sources ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params,
            )

    def update(self, source: Source) -> None:
        (source_id, *rest) = _to_params(_require_source(source))
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE sources SET name = ?, normalized_name = ?, type = ?, mode = ?, "
                "credibility = ?, enabled = ?, secret_ref = ?, config_json = ?, "
                "created_at = ?, updated_at = ? WHERE id = ?",
                (*rest, source_id),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(_NOT_FOUND)

    def get(self, source_id: str) -> Source:
        rows = self._db.query(f"SELECT {_COLUMNS} FROM sources WHERE id = ?", (source_id,))
        if not rows:
            raise NotFoundError(_NOT_FOUND)
        return _from_row(rows[0])

    def find_by_name_ci(self, name: str) -> Source | None:
        rows = self._db.query(
            f"SELECT {_COLUMNS} FROM sources WHERE normalized_name = ?", (normalize_name(name),)
        )
        return _from_row(rows[0]) if rows else None

    def delete(self, source_id: str) -> None:
        with self._db.transaction() as conn:
            cursor = conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            if cursor.rowcount == 0:
                raise NotFoundError(_NOT_FOUND)

    def list(self) -> "list[Source]":
        rows = self._db.query(f"SELECT {_COLUMNS} FROM sources ORDER BY id")
        return [_from_row(row) for row in rows]
