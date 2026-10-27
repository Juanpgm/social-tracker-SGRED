"""SQLite implementation of `ItemRepository`.

Validation lives in the domain (`Item`, `search_terms`, `validate_limit`) and is applied here on
every call, exactly like the in-memory reference. Every row read back is rebuilt through `Item`,
so a tampered row is a `StorageError`, never a half-valid object.

Dedup is decided inside the write transaction (`BEGIN IMMEDIATE`, so concurrent writers queue):
each item is inserted only if no row of the same source has its external id or its content hash,
which makes a batch idempotent and a repeated or racing ingest store nothing twice.

User search text never reaches SQL as syntax: `search_terms` reduces it to plain words and each
word is bound inside a double-quoted FTS5 string, so `AND`, `NEAR`, `*`, `:` and quotes are data.
"""

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from social_tracker.items.domain import (
    DEFAULT_LIST_LIMIT,
    AddReport,
    Item,
    ItemStatus,
    raw_json,
    search_terms,
    validate_limit,
)
from social_tracker.persistence.db import Database, format_utc, parse_utc
from social_tracker.shared.errors import DomainError, StorageError, ValidationError

_COLUMNS = (
    "id, source_id, source_name, kind, external_id, parent_id, url, title, author, text, "
    "language, published_at, fetched_at, content_hash, flags, status, raw_payload"
)
_QUALIFIED = ", ".join(f"items.{name.strip()}" for name in _COLUMNS.split(","))
_SELECT = f"SELECT {_COLUMNS} FROM items"
_INSERT_IF_NEW = (
    f"INSERT INTO items ({_COLUMNS}) "
    "SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? "
    "WHERE NOT EXISTS (SELECT 1 FROM items WHERE source_id = ? "
    "AND (external_id = ? OR content_hash = ?))"
)
_NEWEST_FIRST = " ORDER BY published_at DESC, id DESC LIMIT ?"


def _stored_flags(text: str) -> tuple[str, ...]:
    flags = json.loads(text)
    if not isinstance(flags, list):
        raise TypeError("flags must be a JSON array")
    return tuple(flags)


def _item_from_row(row: sqlite3.Row) -> Item:
    try:
        return Item(
            id=row["id"],
            source_id=row["source_id"],
            source_name=row["source_name"],
            kind=row["kind"],
            external_id=row["external_id"],
            parent_id=row["parent_id"],
            url=row["url"],
            title=row["title"],
            author=row["author"],
            text=row["text"],
            language=row["language"],
            published_at=parse_utc(row["published_at"]),
            fetched_at=parse_utc(row["fetched_at"]),
            content_hash=row["content_hash"],
            flags=_stored_flags(row["flags"]),
            status=row["status"],
            raw_payload=json.loads(row["raw_payload"]),
        )
    except (DomainError, ValueError, TypeError):
        raise StorageError("stored item is invalid") from None


def _insert_params(item: Item) -> tuple[Any, ...]:
    return (
        item.id,
        item.source_id,
        item.source_name,
        item.kind.value,
        item.external_id,
        item.parent_id,
        item.url,
        item.title,
        item.author,
        item.text,
        item.language,
        format_utc(item.published_at),
        format_utc(item.fetched_at),
        item.content_hash,
        json.dumps(list(item.flags)),
        item.status.value,
        raw_json(item.raw_payload),
        item.source_id,
        item.external_id,
        item.content_hash,
    )


class SqliteItemRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add_new(self, items: Sequence[Item]) -> AddReport:
        if not all(isinstance(item, Item) for item in items):
            raise ValidationError("items must be Item objects")
        new = 0
        with self._db.transaction() as conn:
            for item in items:
                new += conn.execute(_INSERT_IF_NEW, _insert_params(item)).rowcount
        return AddReport(new=new, duplicate=len(items) - new)

    def get(self, item_id: str) -> Item | None:
        rows = self._db.query(f"{_SELECT} WHERE id = ?", (item_id,))
        return _item_from_row(rows[0]) if rows else None

    def count(self, source_id: str | None = None) -> int:
        if source_id is None:
            rows = self._db.query("SELECT count(*) FROM items")
        else:
            rows = self._db.query("SELECT count(*) FROM items WHERE source_id = ?", (source_id,))
        return int(rows[0][0])

    def search(self, query: str, *, limit: int = DEFAULT_LIST_LIMIT) -> list[Item]:
        validate_limit(limit)
        terms = search_terms(query)
        if not terms:
            return []
        match = " ".join(f'"{term}"' for term in terms)
        rows = self._db.query(
            f"SELECT {_QUALIFIED} FROM items "
            "JOIN items_fts ON items_fts.rowid = items.seq WHERE items_fts MATCH ?"
            " ORDER BY items.published_at DESC, items.id DESC LIMIT ?",
            (match, limit),
        )
        return [_item_from_row(row) for row in rows]

    def set_status(self, item_id: str, status: ItemStatus) -> bool:
        try:
            stage = ItemStatus(status)
        except ValueError:
            raise ValidationError("unknown item status") from None
        with self._db.transaction() as conn:
            return (
                conn.execute(
                    "UPDATE items SET status = ? WHERE id = ?", (stage.value, item_id)
                ).rowcount
                > 0
            )

    def purge_fetched_before(self, cutoff: datetime) -> int:
        text = format_utc(cutoff)  # raises ValidationError for a naive or non-datetime value
        with self._db.transaction() as conn:
            return conn.execute("DELETE FROM items WHERE fetched_at < ?", (text,)).rowcount

    def purge_classified(self) -> int:
        with self._db.transaction() as conn:
            return conn.execute(
                "DELETE FROM items WHERE status = ?", (ItemStatus.CLASSIFIED.value,)
            ).rowcount

    # Defined last: the name shadows the builtin inside the class body.
    def list(self, *, source_id: str | None = None, limit: int = DEFAULT_LIST_LIMIT) -> list[Item]:
        validate_limit(limit)
        if source_id is None:
            rows = self._db.query(f"{_SELECT}{_NEWEST_FIRST}", (limit,))
        else:
            rows = self._db.query(
                f"{_SELECT} WHERE source_id = ?{_NEWEST_FIRST}", (source_id, limit)
            )
        return [_item_from_row(row) for row in rows]
