"""Deliberately wrong SQLite repositories and schemas, each breaking exactly one rule.

`test_contract_catches_wrong_sql` asserts that the shared repository contracts fail for every
one of them, so a green contract run against the real SQLite adapters means something: it
could have been red.
"""

from collections.abc import Callable
from contextlib import nullcontext, suppress
from pathlib import Path

from social_tracker.control.domain import ControlEvent
from social_tracker.persistence.control_repo import SqliteControlRepository, _event_from_row
from social_tracker.persistence.db import Database, Migration
from social_tracker.persistence.sources_repo import _COLUMNS, SqliteSourceRepository, _from_row
from social_tracker.shared.errors import ConflictError, NotFoundError, StorageError
from social_tracker.sources.domain import Source
from tests.persistence.schema_helpers import migration_sql

SCHEMA = migration_sql()


def schema_without(fragment: str, replacement: str = "") -> list[Migration]:
    """The real migration with `fragment` removed; fails loudly if the fragment drifted."""
    assert fragment in SCHEMA, f"schema no longer contains: {fragment!r}"
    return [Migration(1, "init", SCHEMA.replace(fragment, replacement, 1))]


class NoLockDatabase(Database):
    """Threads share the single connection with nothing serialising them."""

    def __init__(self, path: Path, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self._lock = nullcontext()  # type: ignore[assignment]


# ---- sources ----


class ListWithoutOrderBy(SqliteSourceRepository):
    def list(self) -> list[Source]:
        rows = self._db.query(f"SELECT {_COLUMNS} FROM sources")
        return [_from_row(r) for r in rows]


class DeleteIgnoresMissing(SqliteSourceRepository):
    def delete(self, source_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))


class UpdateIgnoresMissing(SqliteSourceRepository):
    def update(self, source: Source) -> None:
        with suppress(NotFoundError):
            super().update(source)


class CaseSensitiveNameLookup(SqliteSourceRepository):
    def find_by_name_ci(self, name: str) -> Source | None:
        rows = self._db.query(f"SELECT {_COLUMNS} FROM sources WHERE name = ?", (name,))
        return _from_row(rows[0]) if rows else None


class DropsKeywords(SqliteSourceRepository):
    def add(self, source: Source) -> None:
        if source.type == "news_rss":
            source = source.model_copy(update={"keywords": None})
        super().add(source)


class ConflictBecomesStorageError(SqliteSourceRepository):
    def add(self, source: Source) -> None:
        try:
            super().add(source)
        except ConflictError:
            raise StorageError("duplicate") from None


class UpdateAsDeleteThenAdd(SqliteSourceRepository):
    """Not atomic: a conflicting update has already deleted the old row."""

    def update(self, source: Source) -> None:
        self.delete(source.id)
        self.add(source)


class UnsortedListAfterUpdate(SqliteSourceRepository):
    """Orders by insertion (rowid), which differs from id order once rows are unsorted."""

    def list(self) -> list[Source]:
        rows = self._db.query(f"SELECT {_COLUMNS} FROM sources ORDER BY rowid")
        return [_from_row(r) for r in rows]


# ---- control ----


def _control_list(repo: SqliteControlRepository, order: str, where: str, args: tuple) -> list:
    sql = f"SELECT at, kind, source_id, detail FROM control_events {where} ORDER BY {order}"
    return [_event_from_row(r) for r in repo._db.query(sql, args)]


class EventsOrderedByTimestamp(SqliteControlRepository):
    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        where, args = ("", ()) if source_id is None else ("WHERE source_id = ?", (source_id,))
        return _control_list(self, "at", where, args)


class EventsOrderedByKind(SqliteControlRepository):
    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        where, args = ("", ()) if source_id is None else ("WHERE source_id = ?", (source_id,))
        return _control_list(self, "kind", where, args)


class EmptyIdMeansNoFilter(SqliteControlRepository):
    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        return super().list_control_events(source_id or None)


class EventLogCappedAt100(SqliteControlRepository):
    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        return super().list_control_events(source_id)[-100:]


class SettingsWithoutValidation(SqliteControlRepository):
    def set_setting(self, key: str, value: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )


class StateInsertOnly(SqliteControlRepository):
    """No upsert: saving a second state for the same source blows up."""

    def save_source_state(self, state) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO source_state (source_id, state, consecutive_failures, next_run_at) "
                "VALUES (?, ?, ?, NULL)",
                (state.source_id, state.state.value, state.consecutive_failures),
            )


class StateRequiresRegisteredSource(SqliteControlRepository):
    def save_source_state(self, state) -> None:
        if not self._db.query("SELECT 1 FROM sources WHERE id = ?", (state.source_id,)):
            raise NotFoundError("source not found")
        super().save_source_state(state)


class DefaultsToTrue(SqliteControlRepository):
    def get_setting(self, key: str) -> str | None:
        value = super().get_setting(key)
        return "true" if value == "false" else value


class ExposesPurge(SqliteControlRepository):
    def purge_events(self) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM control_events")


class ForgetsNextRunAt(SqliteControlRepository):
    """Never persists the backoff schedule."""

    def save_source_state(self, state) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO source_state "
                "(source_id, state, consecutive_failures, next_run_at) VALUES (?, ?, ?, NULL)",
                (state.source_id, state.state.value, state.consecutive_failures),
            )


class EventsWithoutSourceId(SqliteControlRepository):
    def append_control_event(self, event: ControlEvent) -> None:
        super().append_control_event(
            ControlEvent(at=event.at, kind=event.kind, source_id=None, detail=event.detail)
        )


# ---- builders: name -> factory(db_path) -> repo ----

Builder = Callable[[Path], object]


def _plain(path: Path, repo_cls, *, db_cls=Database, migrations=None):
    kwargs = {"migrations": migrations} if migrations is not None else {}
    return repo_cls(db_cls(path, **kwargs))


SOURCE_WRONG: dict[str, Callable[[Path], object]] = {
    "no_unique_name_index": lambda p: _plain(
        p,
        SqliteSourceRepository,
        migrations=schema_without(
            "CREATE UNIQUE INDEX ux_sources_normalized_name ON sources (normalized_name);"
        ),
    ),
    "id_is_not_unique": lambda p: _plain(
        p,
        SqliteSourceRepository,
        migrations=schema_without(
            "id              TEXT    NOT NULL PRIMARY KEY",
            "id              TEXT    NOT NULL",
        ),
    ),
    "no_lock": lambda p: _plain(p, SqliteSourceRepository, db_cls=NoLockDatabase),
    "list_without_order_by": lambda p: _plain(p, ListWithoutOrderBy),
    "list_ordered_by_rowid": lambda p: _plain(p, UnsortedListAfterUpdate),
    "delete_ignores_missing": lambda p: _plain(p, DeleteIgnoresMissing),
    "update_ignores_missing": lambda p: _plain(p, UpdateIgnoresMissing),
    "case_sensitive_name_lookup": lambda p: _plain(p, CaseSensitiveNameLookup),
    "drops_keywords": lambda p: _plain(p, DropsKeywords),
    "conflict_becomes_storage_error": lambda p: _plain(p, ConflictBecomesStorageError),
    "update_is_delete_then_add": lambda p: _plain(p, UpdateAsDeleteThenAdd),
}

CONTROL_WRONG: dict[str, Callable[[Path], object]] = {
    "events_ordered_by_timestamp": lambda p: _plain(p, EventsOrderedByTimestamp),
    "events_ordered_by_kind": lambda p: _plain(p, EventsOrderedByKind),
    "empty_id_means_no_filter": lambda p: _plain(p, EmptyIdMeansNoFilter),
    "event_log_capped_at_100": lambda p: _plain(p, EventLogCappedAt100),
    "settings_without_validation": lambda p: _plain(p, SettingsWithoutValidation),
    "state_insert_only": lambda p: _plain(p, StateInsertOnly),
    "state_requires_registered_source": lambda p: _plain(p, StateRequiresRegisteredSource),
    "analysis_defaults_to_true": lambda p: _plain(p, DefaultsToTrue),
    "exposes_purge": lambda p: _plain(p, ExposesPurge),
    "forgets_next_run_at": lambda p: _plain(p, ForgetsNextRunAt),
    "events_lose_their_source_id": lambda p: _plain(p, EventsWithoutSourceId),
    "no_lock": lambda p: _plain(p, SqliteControlRepository, db_cls=NoLockDatabase),
    "settings_not_isolated_between_repositories": lambda p: SqliteControlRepository(
        Database(p.parent / "one-shared-file.db")
    ),
}
