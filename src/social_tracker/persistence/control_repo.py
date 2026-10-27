"""SQLite implementation of `ControlRepository`.

Validation lives here, not in the database: `validate_setting` / `validate_setting_key` run on
every call, exactly like the reference fake. The schema CHECKs are the second line of defence,
and every row read back is re-validated so a tampered row is a `StorageError`, never a
half-valid object. The event log is append-only: this class has no way to change it, and the
schema triggers refuse raw UPDATE/DELETE too.
"""

import sqlite3
from typing import Any

from social_tracker.control.domain import (
    DEFAULT_SETTINGS,
    ControlEvent,
    SourceState,
    validate_setting,
    validate_setting_key,
    validate_setting_prefix,
)
from social_tracker.persistence.db import Database, format_utc, parse_utc
from social_tracker.shared.errors import DomainError, StorageError, ValidationError


def _state_from_row(row: sqlite3.Row) -> SourceState:
    try:
        next_run_at = row["next_run_at"]
        return SourceState(
            row["source_id"],
            row["state"],
            row["consecutive_failures"],
            parse_utc(next_run_at) if next_run_at is not None else None,
        )
    except DomainError:
        raise StorageError("stored source state is invalid") from None


def _event_from_row(row: sqlite3.Row) -> ControlEvent:
    try:
        return ControlEvent(
            at=parse_utc(row["at"]),
            kind=row["kind"],
            source_id=row["source_id"],
            detail=row["detail"],
        )
    except DomainError:
        raise StorageError("stored control event is invalid") from None


class SqliteControlRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get_setting(self, key: str) -> str | None:
        validate_setting_key(key)
        rows = self._db.query("SELECT value FROM settings WHERE key = ?", (key,))
        if not rows:
            return DEFAULT_SETTINGS.get(key)
        value = rows[0]["value"]
        try:
            validate_setting(key, value)
        except ValidationError:
            raise StorageError("stored setting is invalid") from None
        return value

    def set_setting(self, key: str, value: str) -> None:
        validate_setting(key, value)
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def delete_setting(self, key: str) -> None:
        validate_setting_key(key)
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM settings WHERE key = ?", (key,))

    def list_settings(self, prefix: str) -> dict[str, str]:
        validate_setting_prefix(prefix)
        # substr, not LIKE: the prefix is literal (LIKE would treat % and _ as wildcards).
        rows = self._db.query(
            "SELECT key, value FROM settings WHERE substr(key, 1, ?) = ? ORDER BY key",
            (len(prefix), prefix),
        )
        found = {k: v for k, v in DEFAULT_SETTINGS.items() if k.startswith(prefix)}
        for row in rows:
            try:
                validate_setting(row["key"], row["value"])
            except ValidationError:
                raise StorageError("stored setting is invalid") from None
            found[row["key"]] = row["value"]
        return dict(sorted(found.items()))

    def get_source_state(self, source_id: str) -> SourceState | None:
        rows = self._db.query(
            "SELECT source_id, state, consecutive_failures, next_run_at "
            "FROM source_state WHERE source_id = ?",
            (source_id,),
        )
        return _state_from_row(rows[0]) if rows else None

    def list_source_states(self) -> list[SourceState]:
        rows = self._db.query(
            "SELECT source_id, state, consecutive_failures, next_run_at "
            "FROM source_state ORDER BY source_id"
        )
        return [_state_from_row(row) for row in rows]

    def save_source_state(self, state: SourceState) -> None:
        if not isinstance(state, SourceState):
            raise ValidationError("state must be a SourceState")
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO source_state (source_id, state, consecutive_failures, next_run_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT (source_id) DO UPDATE SET "
                "state = excluded.state, consecutive_failures = excluded.consecutive_failures, "
                "next_run_at = excluded.next_run_at",
                (
                    state.source_id,
                    state.state.value,
                    state.consecutive_failures,
                    format_utc(state.next_run_at) if state.next_run_at is not None else None,
                ),
            )

    def delete_source_state(self, source_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM source_state WHERE source_id = ?", (source_id,))

    def append_control_event(self, event: ControlEvent) -> None:
        if not isinstance(event, ControlEvent):
            raise ValidationError("event must be a ControlEvent")
        params: tuple[Any, ...] = (format_utc(event.at), event.kind, event.source_id, event.detail)
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO control_events (at, kind, source_id, detail) VALUES (?, ?, ?, ?)",
                params,
            )

    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        sql = "SELECT at, kind, source_id, detail FROM control_events"
        if source_id is None:
            rows = self._db.query(sql + " ORDER BY id")
        else:
            rows = self._db.query(sql + " WHERE source_id = ? ORDER BY id", (source_id,))
        return [_event_from_row(row) for row in rows]
