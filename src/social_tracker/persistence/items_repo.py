"""SQLite implementation of `ItemRepository`.

Validation lives in the domain (`Item`, `search_terms`, `validate_limit`) and is applied here on
every call, exactly like the in-memory reference. Every row read back is rebuilt through `Item`,
so a tampered row is a `StorageError`, never a half-valid object.

Dedup is decided inside the write transaction (`BEGIN IMMEDIATE`, so concurrent writers queue):
each item is inserted only if no row of the same source has its external id or its content hash
and no tombstone of the source has its external id, which makes a batch idempotent and a repeated
or racing ingest store nothing twice, and keeps a purged entry from coming back (see
`item_tombstones`, migration 0003). Quotas are checked in the same transaction, so two writers
cannot both squeeze in under the same cap.

User search text never reaches SQL as syntax: `search_terms` reduces it to plain words and each
word is bound inside a double-quoted FTS5 string, so `AND`, `NEAR`, `*`, `:` and quotes are data.
"""

import json
import sqlite3
from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Any

from social_tracker.items.domain import (
    DEFAULT_LIST_LIMIT,
    MAX_TOMBSTONES_PER_SOURCE,
    AddReport,
    Item,
    ItemStatus,
    SourceUsage,
    item_size_bytes,
    raw_json,
    search_terms,
    validate_limit,
    validate_quota,
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
    "AND (external_id = ? OR content_hash = ?)) "
    "AND NOT EXISTS (SELECT 1 FROM item_tombstones WHERE source_id = ? AND external_id = ?)"
)
_IS_KNOWN = (
    "SELECT 1 FROM items WHERE source_id = ? AND (external_id = ? OR content_hash = ?) "
    "UNION ALL SELECT 1 FROM item_tombstones WHERE source_id = ? AND external_id = ?"
)
_USAGE = (
    "SELECT count(*), COALESCE(SUM(length(CAST(title AS BLOB)) + length(CAST(text AS BLOB)) "
    "+ length(CAST(raw_payload AS BLOB))), 0) FROM items WHERE source_id = ?"
)
_UNDATED_FLAG = (
    "EXISTS (SELECT 1 FROM json_each(items.flags) WHERE json_each.value = 'published_at_missing')"
)
_TOMBSTONE_FROM = (
    "INSERT OR IGNORE INTO item_tombstones (source_id, external_id) "
    "SELECT source_id, external_id FROM items WHERE "
)
_FORGET_OLDEST = (
    "DELETE FROM item_tombstones WHERE source_id = ? AND seq <= "
    "(SELECT seq FROM item_tombstones WHERE source_id = ? ORDER BY seq DESC LIMIT 1 OFFSET ?)"
)
_NEWEST_FIRST = " ORDER BY published_at DESC, id DESC LIMIT ?"


def _stored_flags(text: str) -> tuple[str, ...]:
    flags = json.loads(text)
    if not isinstance(flags, list):
        raise TypeError("flags must be a JSON array")
    return tuple(flags)


def _stage(status: ItemStatus) -> ItemStatus:
    try:
        return ItemStatus(status)
    except ValueError:
        raise ValidationError("unknown item status") from None


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
        item.source_id,
        item.external_id,
    )


class SqliteItemRepository:
    def __init__(self, db: Database, max_tombstones_per_source: int = MAX_TOMBSTONES_PER_SOURCE):
        self._db = db
        self._max_tombstones = max_tombstones_per_source

    def add_new(
        self,
        items: Sequence[Item],
        *,
        max_items: int | None = None,
        max_bytes: int | None = None,
    ) -> AddReport:
        if not all(isinstance(item, Item) for item in items):
            raise ValidationError("items must be Item objects")
        item_cap = validate_quota(max_items, "max_items")
        byte_cap = validate_quota(max_bytes, "max_bytes")
        limited = item_cap is not None or byte_cap is not None
        new = duplicate = over = 0
        held: dict[str, list[int]] = {}
        with self._db.transaction() as conn:
            for item in items:
                cost = item_size_bytes(item) if limited else 0
                if limited:
                    count, used = held.setdefault(
                        item.source_id, list(conn.execute(_USAGE, (item.source_id,)).fetchone())
                    )
                    full = (item_cap is not None and count >= item_cap) or (
                        byte_cap is not None and used + cost > byte_cap
                    )
                    if full:
                        known = conn.execute(
                            _IS_KNOWN,
                            (
                                item.source_id,
                                item.external_id,
                                item.content_hash,
                                item.source_id,
                                item.external_id,
                            ),
                        ).fetchone()
                        duplicate, over = duplicate + bool(known), over + (known is None)
                        continue
                if conn.execute(_INSERT_IF_NEW, _insert_params(item)).rowcount:
                    new += 1
                    if limited:
                        held[item.source_id][0] += 1
                        held[item.source_id][1] += cost
                else:
                    duplicate += 1
        return AddReport(new=new, duplicate=duplicate, over_quota=over)

    def source_usage(self, source_id: str) -> SourceUsage:
        count, size = self._db.query(_USAGE, (source_id,))[0]
        return SourceUsage(items=int(count), bytes=int(size))

    def _remember(self, conn: sqlite3.Connection, condition: str, params: tuple[Any, ...]) -> None:
        """Leave a tombstone for each item `condition` selects, then forget the oldest ones of
        the touched sources beyond the cap. Runs before the delete, in its transaction."""
        sources = [
            row[0]
            for row in conn.execute(
                f"SELECT DISTINCT source_id FROM items WHERE {condition}", params
            ).fetchall()
        ]
        conn.execute(f"{_TOMBSTONE_FROM}{condition}", params)
        for source_id in sources:
            conn.execute(_FORGET_OLDEST, (source_id, source_id, self._max_tombstones))

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

    def list_by_status(
        self, status: ItemStatus, *, limit: int, source_ids: Collection[str] | None = None
    ) -> list[Item]:
        stage = _stage(status)
        validate_limit(limit)
        sql = f"{_SELECT} WHERE status = ?"
        params: list[Any] = [stage.value]
        if source_ids is not None:
            sql += f" AND source_id IN ({', '.join('?' * len(source_ids))})"
            params.extend(sorted(source_ids))
        rows = self._db.query(f"{sql} ORDER BY published_at, id LIMIT ?", (*params, limit))
        return [_item_from_row(row) for row in rows]

    def set_status(self, item_id: str, status: ItemStatus) -> bool:
        stage = _stage(status)
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
            self._remember(conn, f"fetched_at < ? AND {_UNDATED_FLAG}", (text,))
            return conn.execute("DELETE FROM items WHERE fetched_at < ?", (text,)).rowcount

    def purge_classified(self, *, out_of_scope_fetched_before: datetime | None = None) -> int:
        params: tuple[Any, ...] = (ItemStatus.CLASSIFIED.value, ItemStatus.OUT_OF_SCOPE.value)
        condition = "(status = ? OR status = ?)"
        if out_of_scope_fetched_before is not None:
            condition = "(status = ? OR (status = ? AND fetched_at < ?))"
            params = (*params, format_utc(out_of_scope_fetched_before))
        with self._db.transaction() as conn:
            self._remember(conn, condition, params)
            return conn.execute(f"DELETE FROM items WHERE {condition}", params).rowcount

    def count_by_status(self, status: ItemStatus) -> int:
        rows = self._db.query(
            "SELECT count(*) FROM items WHERE status = ?", (_stage(status).value,)
        )
        return int(rows[0][0])

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
