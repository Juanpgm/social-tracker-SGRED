"""SQLite implementation of the alert port (migration 0010).

Validation lives in the domain (`Alert`, `AlertMember`, `AlertEvent`, `AlertFilter`) and is applied
here on every call; every row read back is rebuilt through it, so a tampered row is a
`StorageError`, never a half-valid object. All SQL is parameterized: a filter value is bound,
never spliced (the only text spliced into a statement is a fixed clause chosen by this module).

Idempotence is decided by the database: the alert is keyed by its id and the member by its item,
so storing the same thing twice leaves one row and two racing workers cannot give an item two
alerts. Times are ISO-8601 text in UTC, so comparing them as text is comparing instants.
"""

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

from social_tracker.alerts.domain import (
    WITHDRAWN,
    Alert,
    AlertCounts,
    AlertEvent,
    AlertFilter,
    AlertMember,
    Evidence,
    validate_page,
)
from social_tracker.classification.domain import RISK_LABELS
from social_tracker.items.domain import validate_limit
from social_tracker.persistence.db import Database, format_utc, map_sqlite_error, parse_utc
from social_tracker.shared.errors import DomainError, StorageError, ValidationError

_ALERT_COLUMNS = (
    "id, label, severity, confidence, rationale, source_id, source_name, item_id, excerpt, tags, "
    "evidence, linked_sources, status, occurrences, first_seen_at, last_seen_at, created_at, "
    "updated_at, status_changed_at, withdrawn_at, withdrawn_reason, downgraded_by_model, overflow, "
    "events_dropped"
)
_UPSERT_ALERT = (
    f"INSERT INTO alerts ({_ALERT_COLUMNS}) VALUES ({', '.join('?' * 24)}) "
    "ON CONFLICT (id) DO UPDATE SET "
    + ", ".join(
        f"{name} = excluded.{name}"
        for name in _ALERT_COLUMNS.split(", ")
        if name not in ("id", "created_at")
    )
)
_MEMBER_COLUMNS = (
    "item_id, alert_id, source_id, source_name, content_hash, excerpt, label, confidence, stage, "
    "rationale, severity, tags, evidence, active, occurred_at, linked_at, event_age_s, settled_at"
)
_UPSERT_MEMBER = (
    f"INSERT INTO alert_items ({_MEMBER_COLUMNS}) VALUES ({', '.join('?' * 18)}) "
    "ON CONFLICT (item_id) DO UPDATE SET "
    + ", ".join(
        f"{name} = excluded.{name}" for name in _MEMBER_COLUMNS.split(", ") if name != "item_id"
    )
)
_RISK = ", ".join(f"'{label.value}'" for label in RISK_LABELS)
# What the alert step has to look at: a qualifying verdict of an existing item that has no alert and
# was not forgotten by the purge, and a member whose verdict is no longer the one it was judged on.
_PENDING = (
    "SELECT c.item_id AS item_id FROM classifications c "
    f"WHERE c.label IN ({_RISK}) AND c.confidence >= ? "
    "AND NOT EXISTS (SELECT 1 FROM alert_items m WHERE m.item_id = c.item_id) "
    "AND NOT EXISTS (SELECT 1 FROM alert_forgotten f WHERE f.item_id = c.item_id) "
    "AND EXISTS (SELECT 1 FROM items i WHERE i.id = c.item_id)"
)
_CHANGED = (
    "SELECT m.item_id AS item_id FROM alert_items m "
    "JOIN classifications c ON c.item_id = m.item_id "
    "WHERE m.label <> c.label OR m.confidence <> c.confidence OR m.stage <> c.stage "
    "OR COALESCE(m.settled_at, '') <> COALESCE(c.settled_at, '')"
)
_EVERY_MEMBER = "SELECT m.item_id AS item_id FROM alert_items m"


def _alert_from_row(row: sqlite3.Row) -> Alert:
    try:
        return Alert(
            id=row["id"],
            label=row["label"],
            severity=row["severity"],
            confidence=row["confidence"],
            rationale=row["rationale"],
            source_id=row["source_id"],
            source_name=row["source_name"],
            item_id=row["item_id"],
            excerpt=row["excerpt"],
            tags=tuple(json.loads(row["tags"])),
            evidence=Evidence.from_json(row["evidence"]),
            status=row["status"],
            occurrences=row["occurrences"],
            first_seen_at=parse_utc(row["first_seen_at"]),
            last_seen_at=parse_utc(row["last_seen_at"]),
            created_at=parse_utc(row["created_at"]),
            updated_at=parse_utc(row["updated_at"]),
            status_changed_at=_optional_time(row["status_changed_at"]),
            withdrawn_at=_optional_time(row["withdrawn_at"]),
            withdrawn_reason=row["withdrawn_reason"],
            linked_sources=tuple((str(i), str(n)) for i, n in json.loads(row["linked_sources"])),
            downgraded_by_model=bool(row["downgraded_by_model"]),
            overflow=row["overflow"],
            events_dropped=row["events_dropped"],
        )
    except (DomainError, ValueError, TypeError):
        raise StorageError("stored alert is invalid") from None


def _member_from_row(row: sqlite3.Row) -> AlertMember:
    try:
        return AlertMember(
            alert_id=row["alert_id"],
            item_id=row["item_id"],
            source_id=row["source_id"],
            source_name=row["source_name"],
            content_hash=row["content_hash"],
            excerpt=row["excerpt"],
            label=row["label"],
            confidence=row["confidence"],
            stage=row["stage"],
            rationale=row["rationale"],
            severity=row["severity"],
            tags=tuple(json.loads(row["tags"])),
            evidence=Evidence.from_json(row["evidence"]),
            active=bool(row["active"]),
            occurred_at=parse_utc(row["occurred_at"]),
            linked_at=parse_utc(row["linked_at"]),
            event_age_s=row["event_age_s"],
            settled_at=_optional_time(row["settled_at"]),
        )
    except (DomainError, ValueError, TypeError):
        raise StorageError("stored alert member is invalid") from None


def _optional_time(value: Any) -> datetime | None:
    return parse_utc(value) if value is not None else None


def _alert_params(a: Alert) -> tuple[Any, ...]:
    return (
        a.id,
        a.label.value,
        a.severity.value,
        a.confidence,
        a.rationale,
        a.source_id,
        a.source_name,
        a.item_id,
        a.excerpt,
        json.dumps(list(a.tags)),
        a.evidence.to_json(),
        json.dumps([list(pair) for pair in a.linked_sources]),
        a.status.value,
        a.occurrences,
        format_utc(a.first_seen_at),
        format_utc(a.last_seen_at),
        format_utc(a.created_at),
        format_utc(a.updated_at),
        _text_or_none(a.status_changed_at),
        _text_or_none(a.withdrawn_at),
        a.withdrawn_reason,
        int(a.downgraded_by_model),
        a.overflow,
        a.events_dropped,
    )


def _member_params(m: AlertMember) -> tuple[Any, ...]:
    return (
        m.item_id,
        m.alert_id,
        m.source_id,
        m.source_name,
        m.content_hash,
        m.excerpt,
        m.label.value,
        m.confidence,
        m.stage,
        m.rationale,
        m.severity.value,
        json.dumps(list(m.tags)),
        m.evidence.to_json(),
        int(m.active),
        format_utc(m.occurred_at),
        format_utc(m.linked_at),
        m.event_age_s,
        _text_or_none(m.settled_at),
    )


def _text_or_none(value: datetime | None) -> str | None:
    return format_utc(value) if value is not None else None


def _where(flt: AlertFilter) -> tuple[str, list[Any]]:
    """The WHERE clause of a filter and its parameters (fixed clauses, bound values)."""
    clauses: list[str] = []
    params: list[Any] = []
    if flt.status == WITHDRAWN:
        clauses.append("a.withdrawn_at IS NOT NULL")
    else:
        clauses.append("a.withdrawn_at IS NULL")
        if flt.status is not None:
            clauses.append("a.status = ?")
            params.append(flt.status.value)  # type: ignore[union-attr]
    if flt.severity is not None:
        clauses.append("a.severity = ?")
        params.append(flt.severity.value)
    if flt.label is not None:
        clauses.append("a.label = ?")
        params.append(flt.label.value)
    if flt.source_id is not None:
        clauses.append(
            "(a.source_id = ? OR EXISTS "
            "(SELECT 1 FROM alert_items m WHERE m.alert_id = a.id AND m.source_id = ?))"
        )
        params += [flt.source_id, flt.source_id]
    if flt.tag is not None:
        clauses.append("EXISTS (SELECT 1 FROM json_each(a.tags) WHERE value = ?)")
        params.append(flt.tag)
    if flt.since is not None:
        clauses.append("a.last_seen_at >= ?")
        params.append(format_utc(flt.since))
    if flt.until is not None:
        clauses.append("a.last_seen_at <= ?")
        params.append(format_utc(flt.until))
    return " AND ".join(clauses), params


class SqliteAlertRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ---- alerts ----

    def get(self, alert_id: str) -> Alert | None:
        rows = self._db.query(f"SELECT {_ALERT_COLUMNS} FROM alerts WHERE id = ?", (alert_id,))
        return _alert_from_row(rows[0]) if rows else None

    def save(self, alert: Alert) -> None:
        if not isinstance(alert, Alert):
            raise ValidationError("an alert is required")
        with self._db.transaction() as conn:
            conn.execute(_UPSERT_ALERT, _alert_params(alert))

    # ---- members ----

    def member(self, item_id: str) -> AlertMember | None:
        rows = self._db.query(
            f"SELECT {_MEMBER_COLUMNS} FROM alert_items WHERE item_id = ?", (item_id,)
        )
        return _member_from_row(rows[0]) if rows else None

    def members(self, alert_id: str) -> list[AlertMember]:
        rows = self._db.query(
            f"SELECT {_MEMBER_COLUMNS} FROM alert_items WHERE alert_id = ? "
            "ORDER BY occurred_at, item_id",
            (alert_id,),
        )
        return [_member_from_row(r) for r in rows]

    def save_member(self, member: AlertMember) -> None:
        if not isinstance(member, AlertMember):
            raise ValidationError("an alert member is required")
        with self._db.transaction() as conn:
            conn.execute(_UPSERT_MEMBER, _member_params(member))

    # ---- the trail ----

    def add_event(self, event: AlertEvent) -> None:
        if not isinstance(event, AlertEvent):
            raise ValidationError("an alert event is required")
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO alert_events "
                "(alert_id, at, kind, actor, reason, from_value, to_value, from_absorbed) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.alert_id,
                    format_utc(event.at),
                    event.kind.value,
                    event.actor,
                    event.reason,
                    event.before,
                    event.after,
                    int(event.from_absorbed),
                ),
            )

    def events(self, alert_id: str) -> list[AlertEvent]:
        rows = self._db.query(
            "SELECT alert_id, at, kind, actor, reason, from_value, to_value, from_absorbed "
            "FROM alert_events "
            "WHERE alert_id = ? ORDER BY seq",
            (alert_id,),
        )
        try:
            return [
                AlertEvent(
                    r["alert_id"],
                    parse_utc(r["at"]),
                    r["kind"],
                    r["actor"],
                    r["reason"],
                    r["from_value"],
                    r["to_value"],
                    bool(r["from_absorbed"]),
                )
                for r in rows
            ]
        except (DomainError, ValueError, TypeError):
            raise StorageError("stored alert event is invalid") from None

    # ---- finding the alert of a copy ----

    def find_merge_targets(
        self,
        *,
        content_hash: str,
        related_items: Sequence[str],
        around: datetime,
        window: timedelta,
    ) -> list[str]:
        related = list(related_items)
        same = "m.content_hash = ?"
        if related:
            same = f"({same} OR m.item_id IN ({', '.join('?' * len(related))}))"
        rows = self._db.query(
            "SELECT DISTINCT a.id, a.created_at FROM alert_items m JOIN alerts a "
            f"ON a.id = m.alert_id WHERE {same} AND m.occurred_at >= ? AND m.occurred_at <= ? "
            "ORDER BY a.created_at, a.id",
            [content_hash, *related, format_utc(around - window), format_utc(around + window)],
        )
        return [str(r[0]) for r in rows]

    def delete(self, alert_id: str) -> bool:
        with self._db.transaction() as conn:
            return conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,)).rowcount > 0

    # ---- reading ----

    def count(self, flt: AlertFilter) -> int:
        where, params = _where(flt)
        return int(self._db.query(f"SELECT count(*) FROM alerts a WHERE {where}", params)[0][0])

    def counts(self) -> AlertCounts:
        try:
            with self._db.snapshot() as conn:
                total = conn.execute(
                    "SELECT count(*), COALESCE(SUM(status = 'new'), 0) FROM alerts "
                    "WHERE withdrawn_at IS NULL"
                ).fetchone()
                withdrawn = conn.execute(
                    "SELECT count(*) FROM alerts WHERE withdrawn_at IS NOT NULL"
                ).fetchone()
                by_status = conn.execute(
                    "SELECT status, count(*) FROM alerts WHERE withdrawn_at IS NULL "
                    "GROUP BY status ORDER BY status"
                ).fetchall()
                by_severity = conn.execute(
                    "SELECT severity, count(*) FROM alerts WHERE withdrawn_at IS NULL "
                    "GROUP BY severity ORDER BY severity"
                ).fetchall()
                new_by_severity = conn.execute(
                    "SELECT severity, count(*) FROM alerts WHERE withdrawn_at IS NULL "
                    "AND status = 'new' GROUP BY severity ORDER BY severity"
                ).fetchall()
        except sqlite3.Error as exc:
            raise map_sqlite_error(exc) from None
        return AlertCounts(
            active=int(total[0]),
            new=int(total[1]),
            withdrawn=int(withdrawn[0]),
            by_status=tuple((str(k), int(n)) for k, n in by_status),
            by_severity=tuple((str(k), int(n)) for k, n in by_severity),
            new_by_severity=tuple((str(k), int(n)) for k, n in new_by_severity),
        )

    # ---- the sync ----

    def sync_ids(
        self, *, min_confidence: float, full: bool, after: str | None, limit: int
    ) -> list[str]:
        validate_limit(limit)
        parts = [_PENDING, _EVERY_MEMBER if full else _CHANGED]
        rows = self._db.query(
            f"SELECT item_id FROM ({' UNION '.join(parts)}) WHERE item_id > ? "
            "ORDER BY item_id LIMIT ?",
            [float(min_confidence), after or "", limit],
        )
        return [str(r[0]) for r in rows]

    def is_forgotten(self, item_id: str) -> bool:
        return bool(self._db.query("SELECT 1 FROM alert_forgotten WHERE item_id = ?", (item_id,)))

    # ---- retention ----

    def purge(self, *, before: datetime | None, handled: bool, now: datetime) -> int:
        conditions: list[str] = []
        params: list[Any] = []
        if before is not None:
            conditions.append("last_seen_at < ?")
            params.append(format_utc(before))
        if handled:
            conditions.append("(status <> 'new' OR withdrawn_at IS NOT NULL)")
        removed = 0
        with self._db.transaction() as conn:
            if conditions:
                where = " OR ".join(conditions)
                conn.execute(
                    "INSERT INTO alert_forgotten (item_id, forgotten_at) "
                    "SELECT item_id, ? FROM alert_items "
                    f"WHERE alert_id IN (SELECT id FROM alerts WHERE {where}) "
                    "ON CONFLICT (item_id) DO NOTHING",
                    [format_utc(now), *params],
                )
                removed = conn.execute(f"DELETE FROM alerts WHERE {where}", params).rowcount
            conn.execute(
                "DELETE FROM alert_forgotten "
                "WHERE item_id NOT IN (SELECT item_id FROM classifications)"
            )
        return removed

    # Defined last: the name shadows the builtin inside the class body.
    def list(self, flt: AlertFilter, *, limit: int, offset: int) -> "list[Alert]":
        validate_page(limit, offset)
        where, params = _where(flt)
        rows = self._db.query(
            f"SELECT {_ALERT_COLUMNS} FROM alerts a WHERE {where} "
            "ORDER BY a.last_seen_at DESC, a.id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return [_alert_from_row(r) for r in rows]
