"""SQLite implementation of `CatalogRepository`.

Every value is bound as a parameter. Every row read back is rebuilt through the domain object, so
a tampered row is a `StorageError`, never a half-valid event. Writes run in the caller's
transaction; the claim is a single UPSERT statement, so two connections racing for it get exactly
one winner (`BEGIN IMMEDIATE` serialises the writers).
"""

import sqlite3
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from social_tracker.catalog.domain import CatalogEvent, CatalogSource, RefreshState
from social_tracker.catalog.ports import UpsertReport
from social_tracker.persistence.db import Database, format_utc, parse_utc
from social_tracker.shared.errors import DomainError, StorageError

_EVENT_COLUMNS = "source, external_id, origin_time, latitude, longitude, magnitude, place"
_STATE_COLUMNS = (
    "source, last_success_at, last_attempt_at, consecutive_failures, next_attempt_at, "
    "claimed_until, last_error, last_truncated"
)
_DELETE_BATCH = 500  # well under SQLite's bound-variable limit


def _event(row: sqlite3.Row) -> CatalogEvent:
    try:
        return CatalogEvent(
            row["source"],
            row["external_id"],
            parse_utc(row["origin_time"]),
            row["latitude"],
            row["longitude"],
            row["magnitude"],
            row["place"],
        )
    except DomainError:
        raise StorageError("a stored catalog event is invalid") from None


def _optional(value: Any) -> datetime | None:
    return None if value is None else parse_utc(value)


def _state(row: sqlite3.Row) -> RefreshState:
    try:
        return RefreshState(
            row["source"],
            _optional(row["last_success_at"]),
            _optional(row["last_attempt_at"]),
            row["consecutive_failures"],
            _optional(row["next_attempt_at"]),
            _optional(row["claimed_until"]),
            row["last_error"],
            bool(row["last_truncated"]),
        )
    except DomainError:
        raise StorageError("a stored catalog state is invalid") from None


class SqliteCatalogRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert_events(self, events: Sequence[CatalogEvent], fetched_at: datetime) -> UpsertReport:
        new = updated = unchanged = 0
        stamp = format_utc(fetched_at)
        with self._db.transaction() as conn:
            for event in events:
                key = (event.source.value, event.external_id)
                values = (
                    event.latitude,
                    event.longitude,
                    event.magnitude,
                    event.place,
                    format_utc(event.origin_time),
                )
                found = conn.execute(
                    "SELECT latitude, longitude, magnitude, place, origin_time "
                    "FROM catalog_events WHERE source = ? AND external_id = ?",
                    key,
                ).fetchone()
                if found is None:
                    conn.execute(
                        f"INSERT INTO catalog_events ({_EVENT_COLUMNS}, fetched_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (*key, values[4], *values[:4], stamp),
                    )
                    new += 1
                elif tuple(found) == values:
                    unchanged += 1
                else:
                    conn.execute(
                        "UPDATE catalog_events SET latitude = ?, longitude = ?, magnitude = ?, "
                        "place = ?, origin_time = ?, fetched_at = ? "
                        "WHERE source = ? AND external_id = ?",
                        (*values, stamp, *key),
                    )
                    updated += 1
        return UpsertReport(new, updated, unchanged)

    def events_between(self, start: datetime, end: datetime) -> list[CatalogEvent]:
        rows = self._db.query(
            f"SELECT {_EVENT_COLUMNS} FROM catalog_events "
            "WHERE origin_time >= ? AND origin_time <= ? "
            "ORDER BY origin_time, source, external_id",
            (format_utc(start), format_utc(end)),
        )
        return [_event(row) for row in rows]

    def count_by_source(self) -> dict[CatalogSource, int]:
        rows = self._db.query("SELECT source, count(*) FROM catalog_events GROUP BY source")
        return {CatalogSource(row[0]): row[1] for row in rows}

    def delete_events(self, source: CatalogSource, external_ids: Sequence[str]) -> int:
        name = CatalogSource(source).value
        removed = 0
        unique = list(dict.fromkeys(external_ids))
        with self._db.transaction() as conn:
            for start in range(0, len(unique), _DELETE_BATCH):
                batch = unique[start : start + _DELETE_BATCH]
                marks = ", ".join("?" * len(batch))
                removed += conn.execute(
                    f"DELETE FROM catalog_events WHERE source = ? AND external_id IN ({marks})",
                    (name, *batch),
                ).rowcount
        return removed

    def purge_before(self, cutoff: datetime) -> int:
        with self._db.transaction() as conn:
            return conn.execute(
                "DELETE FROM catalog_events WHERE origin_time < ?", (format_utc(cutoff),)
            ).rowcount

    def get_state(self, source: CatalogSource) -> RefreshState | None:
        rows = self._db.query(
            f"SELECT {_STATE_COLUMNS} FROM catalog_refresh_state WHERE source = ?",
            (CatalogSource(source).value,),
        )
        return _state(rows[0]) if rows else None

    def all_states(self) -> list[RefreshState]:
        rows = self._db.query(f"SELECT {_STATE_COLUMNS} FROM catalog_refresh_state ORDER BY source")
        return [_state(row) for row in rows]

    def try_claim(self, source: CatalogSource, now: datetime, until: datetime) -> bool:
        with self._db.transaction() as conn:
            changed = conn.execute(
                "INSERT INTO catalog_refresh_state (source, claimed_until) VALUES (?, ?) "
                "ON CONFLICT (source) DO UPDATE SET claimed_until = excluded.claimed_until "
                "WHERE catalog_refresh_state.claimed_until IS NULL "
                "OR catalog_refresh_state.claimed_until <= ?",
                (CatalogSource(source).value, format_utc(until), format_utc(now)),
            ).rowcount
        return changed > 0

    def release_claim(self, source: CatalogSource) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE catalog_refresh_state SET claimed_until = NULL WHERE source = ?",
                (CatalogSource(source).value,),
            )

    def record_success(
        self, source: CatalogSource, at: datetime, *, truncated: bool = False
    ) -> None:
        stamp = format_utc(at)
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO catalog_refresh_state "
                "(source, last_success_at, last_attempt_at, last_truncated) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (source) DO UPDATE SET "
                "last_success_at = excluded.last_success_at, "
                "last_attempt_at = excluded.last_attempt_at, consecutive_failures = 0, "
                "next_attempt_at = NULL, claimed_until = NULL, last_error = NULL, "
                "last_truncated = excluded.last_truncated",
                (CatalogSource(source).value, stamp, stamp, 1 if truncated else 0),
            )

    def record_failure(
        self, source: CatalogSource, at: datetime, next_attempt_at: datetime, error: str
    ) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO catalog_refresh_state (source, last_attempt_at, "
                "consecutive_failures, next_attempt_at, last_error) VALUES (?, ?, 1, ?, ?) "
                "ON CONFLICT (source) DO UPDATE SET last_attempt_at = excluded.last_attempt_at, "
                "consecutive_failures = catalog_refresh_state.consecutive_failures + 1, "
                "next_attempt_at = excluded.next_attempt_at, claimed_until = NULL, "
                "last_error = excluded.last_error",
                (
                    CatalogSource(source).value,
                    format_utc(at),
                    format_utc(next_attempt_at),
                    error[:200],
                ),
            )
