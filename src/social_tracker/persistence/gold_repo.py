"""SQLite implementation of the gold-set port (migration 0011).

Validation lives in the domain (`GoldExample`) and is applied here on every call; every row read
back is rebuilt through it, so a tampered row is a `StorageError`, never a half-valid object. All
SQL is parameterized: a text is data, whatever it says. Idempotence is decided by the database:
`add` is `INSERT ... ON CONFLICT (id) DO NOTHING` on the id the text makes, so two writers adding
the same text store one row between them and the loser is told so.
"""

import sqlite3
from typing import Any

from social_tracker.classification.domain import Label
from social_tracker.gold.domain import MAX_GOLD_EXAMPLES, GoldExample
from social_tracker.persistence.db import Database, format_utc, parse_utc
from social_tracker.shared.errors import DomainError, StorageError, ValidationError

_COLUMNS = "id, text, label, note, credibility, url, added_at, actor"
_INSERT = (
    f"INSERT INTO gold_examples ({_COLUMNS}) VALUES ({', '.join('?' * 8)}) "
    "ON CONFLICT (id) DO NOTHING"
)


def _from_row(row: sqlite3.Row) -> GoldExample:
    try:
        return GoldExample(
            id=row["id"],
            text=row["text"],
            label=row["label"],
            note=row["note"],
            credibility=row["credibility"],
            url=row["url"],
            added_at=parse_utc(row["added_at"]),
            actor=row["actor"],
        )
    except (DomainError, ValueError, TypeError):
        raise StorageError("stored gold example is invalid") from None


def _params(e: GoldExample) -> tuple[Any, ...]:
    return (
        e.id,
        e.text,
        e.label.value,
        e.note,
        e.credibility,
        e.url,
        format_utc(e.added_at),
        e.actor,
    )


class SqliteGoldRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, example: GoldExample) -> bool:
        if not isinstance(example, GoldExample):
            raise ValidationError("a gold example is required")
        with self._db.transaction() as conn:
            return conn.execute(_INSERT, _params(example)).rowcount > 0

    def get(self, example_id: str) -> GoldExample | None:
        rows = self._db.query(f"SELECT {_COLUMNS} FROM gold_examples WHERE id = ?", (example_id,))
        return _from_row(rows[0]) if rows else None

    def count(self) -> int:
        return int(self._db.query("SELECT count(*) FROM gold_examples")[0][0])

    def remove(self, example_id: str) -> bool:
        with self._db.transaction() as conn:
            return (
                conn.execute("DELETE FROM gold_examples WHERE id = ?", (example_id,)).rowcount > 0
            )

    # Defined last: the name shadows the builtin inside the class body.
    def list(
        self, *, label: Label | None = None, limit: int, offset: int = 0
    ) -> "list[GoldExample]":
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_GOLD_EXAMPLES
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
        ):
            raise ValidationError(f"limit must be from 1 to {MAX_GOLD_EXAMPLES}, offset from 0")
        where, params = ("", []) if label is None else (" WHERE label = ?", [Label(label).value])
        rows = self._db.query(
            f"SELECT {_COLUMNS} FROM gold_examples{where} ORDER BY seq LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return [_from_row(r) for r in rows]
