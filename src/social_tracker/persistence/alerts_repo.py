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
    ForgottenSignature,
    SyncRef,
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
    "events_dropped, published_at"
)
_UPSERT_ALERT = (
    f"INSERT INTO alerts ({_ALERT_COLUMNS}) VALUES ({', '.join('?' * 25)}) "
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
# A forgotten item whose verdict is still the one it was forgotten on stays forgotten; a row from
# before the signatures (label NULL) always suppresses.
_SAME_VERDICT = (
    "(f.label IS NULL OR (f.label = c.label AND f.confidence = c.confidence "
    "AND COALESCE(f.settled_at, '') = COALESCE(c.settled_at, '')))"
)
# What the alert step has to look at, each row with the instant it is ordered by (a chronological
# listing lets a copy find its original whichever reached the store first):
# - a qualifying verdict of an existing item that has no alert, was not counted as overflow of one
#   and was not forgotten on this very verdict;
# - a member whose verdict is no longer the one it was judged on;
# - every member (a full sync).
_PENDING = (
    "SELECT i.fetched_at AS ts, c.item_id AS item_id FROM classifications c "
    "JOIN items i ON i.id = c.item_id "
    f"WHERE c.label IN ({_RISK}) AND c.confidence >= ? "
    "AND NOT EXISTS (SELECT 1 FROM alert_items m WHERE m.item_id = c.item_id) "
    "AND NOT EXISTS (SELECT 1 FROM alert_overflow o WHERE o.item_id = c.item_id) "
    "AND NOT EXISTS (SELECT 1 FROM alert_forgotten f WHERE f.item_id = c.item_id "
    f"AND {_SAME_VERDICT})"
)
_CHANGED = (
    "SELECT m.occurred_at AS ts, m.item_id AS item_id FROM alert_items m "
    "JOIN classifications c ON c.item_id = m.item_id "
    "WHERE m.label <> c.label OR m.confidence <> c.confidence OR m.stage <> c.stage "
    "OR COALESCE(m.settled_at, '') <> COALESCE(c.settled_at, '')"
)
_EVERY_MEMBER = "SELECT m.occurred_at AS ts, m.item_id AS item_id FROM alert_items m"
# An item the step failed on waits (backoff) or is parked until a rebuild.
_NOT_BACKING_OFF = (
    "AND NOT EXISTS (SELECT 1 FROM alert_failures x WHERE x.item_id = {column} "
    "AND (x.parked = 1 OR x.next_try_at > ?))"
)
FORGOTTEN_KEEP = timedelta(days=365)  # how long the memory of a forgotten alert is kept
MAX_FORGOTTEN = 200_000  # and how many of them, the oldest going first
_SEVERITY_RANK = (
    "CASE severity WHEN 'low' THEN 0 WHEN 'medium' THEN 1 WHEN 'high' THEN 2 ELSE 3 END"
)
_CURSOR_SEPARATOR = "|"


def _cursor(ts: str, item_id: str) -> str:
    return f"{ts}{_CURSOR_SEPARATOR}{item_id}"


def _split_cursor(after: str | None) -> tuple[str, str]:
    if after is None:
        return "", ""
    if not isinstance(after, str):
        raise ValidationError("a sync cursor is text")
    ts, _, item_id = after.rpartition(_CURSOR_SEPARATOR)
    return ts, item_id


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
            published_at=_optional_time(row["published_at"]),
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
        _text_or_none(a.published_at),
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


_WHEN_COLUMN = {"first_seen": "a.first_seen_at", "last_seen": "a.last_seen_at"}


def _where(flt: AlertFilter) -> tuple[str, list[Any]]:
    """The WHERE clause of a filter and its parameters (fixed clauses, bound values)."""
    clauses: list[str] = []
    params: list[Any] = []
    if flt.status == WITHDRAWN:
        clauses.append("a.withdrawn_at IS NOT NULL")
    else:
        if not flt.include_withdrawn:
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
    column = _WHEN_COLUMN[flt.when]  # a fixed name chosen by this module, never the user's text
    if flt.since is not None:
        clauses.append(f"{column} >= ?")
        params.append(format_utc(flt.since))
    if flt.until is not None:
        clauses.append(f"{column} <= ?")
        params.append(format_utc(flt.until))
    if flt.hide_provisional:
        clauses.append("COALESCE(json_extract(a.evidence, '$.provisional'), 0) = 0")
    return " AND ".join(clauses) or "1", params


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

    def events(self, alert_id: str, limit: int | None = None) -> list[AlertEvent]:
        """The trail of an alert in the order it happened; with `limit`, only the latest ones."""
        columns = "alert_id, at, kind, actor, reason, from_value, to_value, from_absorbed"
        if limit is None:
            rows = self._db.query(
                f"SELECT {columns} FROM alert_events WHERE alert_id = ? ORDER BY seq", (alert_id,)
            )
        else:
            rows = self._db.query(
                f"SELECT {columns} FROM (SELECT seq, {columns} FROM alert_events "
                "WHERE alert_id = ? ORDER BY seq DESC LIMIT ?) ORDER BY seq",
                (alert_id, max(0, limit)),
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
        item_id: str | None = None,
    ) -> list[str]:
        """The alerts (oldest first, each once) that hold the same content, one of the originals
        this item points at, or (with `item_id`) a copy that points at THIS item, within the
        window of `around`."""
        low, high = format_utc(around - window), format_utc(around + window)
        span = "AND m.occurred_at >= ? AND m.occurred_at <= ?"
        parts = [f"SELECT m.alert_id FROM alert_items m WHERE m.content_hash = ? {span}"]
        params: list[Any] = [content_hash, low, high]
        # an item given up by a full alert is still found by its content and by its id
        parts.append(
            "SELECT o.alert_id FROM alert_overflow o WHERE o.content_hash = ? "
            "AND o.occurred_at >= ? AND o.occurred_at <= ?"
        )
        params += [content_hash, low, high]
        related = list(related_items)
        if related:
            marks = ", ".join("?" * len(related))
            parts.append(
                f"SELECT m.alert_id FROM alert_items m WHERE m.item_id IN ({marks}) {span}"
            )
            params += [*related, low, high]
            parts.append(
                f"SELECT o.alert_id FROM alert_overflow o WHERE o.item_id IN ({marks}) "
                "AND o.occurred_at >= ? AND o.occurred_at <= ?"
            )
            params += [*related, low, high]
        if item_id is not None:
            for key in ("near_of", "duplicate_of"):
                parts.append(
                    "SELECT m.alert_id FROM alert_items m "
                    f"WHERE json_extract(m.evidence, '$.{key}') = ? {span}"
                )
                params += [item_id, low, high]
        rows = self._db.query(
            "SELECT a.id FROM alerts a WHERE a.id IN (" + " UNION ".join(parts) + ") "
            "ORDER BY a.created_at, a.id",
            params,
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
                downgraded = conn.execute(
                    "SELECT count(*) FROM alerts WHERE withdrawn_at IS NULL "
                    "AND downgraded_by_model = 1"
                ).fetchone()
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
            downgraded=int(downgraded[0]),
        )

    def linked_sources(self) -> list[tuple[str, str]]:
        """Every source an alert is linked to, with the latest name an alert snapshot of it."""
        rows = self._db.query(
            "SELECT source_id, source_name FROM alert_items GROUP BY source_id "
            "HAVING linked_at = MAX(linked_at) ORDER BY source_id"
        )
        return [(str(r[0]), str(r[1])) for r in rows]

    # ---- the sync ----

    def sync_ids(
        self,
        *,
        min_confidence: float,
        full: bool,
        after: str | None,
        limit: int,
        now: datetime | None = None,
    ) -> list[SyncRef]:
        validate_limit(limit)
        parts = [_PENDING, _EVERY_MEMBER if full else _CHANGED]
        after_ts, after_id = _split_cursor(after)
        params: list[Any] = [float(min_confidence), after_ts, after_id]
        waiting = ""
        if now is not None:
            waiting = _NOT_BACKING_OFF.format(column="u.item_id")
            params.append(format_utc(now))
        rows = self._db.query(
            f"SELECT u.item_id, u.ts FROM ({' UNION '.join(parts)}) u "
            f"WHERE (u.ts, u.item_id) > (?, ?) {waiting} ORDER BY u.ts, u.item_id LIMIT ?",
            [*params, limit],
        )
        return [SyncRef(str(r[0]), _cursor(str(r[1]), str(r[0]))) for r in rows]

    def dirty_ids(
        self, *, after: str | None, limit: int, now: datetime | None = None
    ) -> list[SyncRef]:
        validate_limit(limit)
        after_ts, after_id = _split_cursor(after)
        params: list[Any] = [after_ts, after_id]
        waiting = ""
        if now is not None:
            waiting = _NOT_BACKING_OFF.format(column="d.item_id")
            params.append(format_utc(now))
        rows = self._db.query(
            "SELECT d.item_id, COALESCE(i.fetched_at, '') AS ts FROM alert_dirty d "
            "LEFT JOIN items i ON i.id = d.item_id "
            f"WHERE (COALESCE(i.fetched_at, ''), d.item_id) > (?, ?) {waiting} "
            "ORDER BY ts, d.item_id LIMIT ?",
            [*params, limit],
        )
        return [SyncRef(str(r[0]), _cursor(str(r[1]), str(r[0]))) for r in rows]

    def clear_dirty(self, item_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM alert_dirty WHERE item_id = ?", (item_id,))

    def is_forgotten(self, item_id: str) -> bool:
        return bool(self._db.query("SELECT 1 FROM alert_forgotten WHERE item_id = ?", (item_id,)))

    def forgotten_signature(self, item_id: str) -> ForgottenSignature | None:
        rows = self._db.query(
            "SELECT label, confidence, settled_at FROM alert_forgotten WHERE item_id = ?",
            (item_id,),
        )
        if not rows:
            return None
        row = rows[0]
        try:
            return ForgottenSignature(
                row["label"], row["confidence"], _optional_time(row["settled_at"])
            )
        except (ValueError, TypeError):
            raise StorageError("stored forgotten signature is invalid") from None

    def clear_forgotten(self, item_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM alert_forgotten WHERE item_id = ?", (item_id,))

    # ---- items the step could not evaluate ----

    def record_failure(
        self,
        item_id: str,
        now: datetime,
        *,
        park_after: int,
        backoff_base: timedelta,
        backoff_cap: timedelta,
    ) -> int:
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT failures FROM alert_failures WHERE item_id = ?", (item_id,)
            ).fetchone()
            failures = (int(row[0]) if row else 0) + 1
            delay = min(backoff_base * (2 ** min(failures - 1, 30)), backoff_cap)
            conn.execute(
                "INSERT INTO alert_failures "
                "(item_id, failures, last_failed_at, next_try_at, parked) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (item_id) DO UPDATE SET "
                "failures = excluded.failures, last_failed_at = excluded.last_failed_at, "
                "next_try_at = excluded.next_try_at, parked = excluded.parked",
                (
                    item_id,
                    failures,
                    format_utc(now),
                    format_utc(now + delay),
                    int(failures >= park_after),
                ),
            )
        return failures

    def clear_failure(self, item_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM alert_failures WHERE item_id = ?", (item_id,))

    def failure_counts(self, now: datetime) -> tuple[int, int]:
        """(items waiting to be tried again, items parked)."""
        rows = self._db.query(
            "SELECT COALESCE(SUM(parked = 0 AND next_try_at > ?), 0), "
            "COALESCE(SUM(parked = 1), 0) FROM alert_failures",
            (format_utc(now),),
        )
        return int(rows[0][0]), int(rows[0][1])

    def release_parked(self) -> int:
        with self._db.transaction() as conn:
            return conn.execute(
                "UPDATE alert_failures SET parked = 0, next_try_at = last_failed_at "
                "WHERE parked = 1"
            ).rowcount

    # ---- copies counted but not stored ----

    def overflow_of(self, item_id: str) -> str | None:
        rows = self._db.query("SELECT alert_id FROM alert_overflow WHERE item_id = ?", (item_id,))
        return str(rows[0][0]) if rows else None

    def add_overflow(self, alert_id: str, member: AlertMember) -> None:
        """Count a member without storing it: the item, its verdict signature, its content and
        its time stay, so that it is not counted twice and its copies still find the alert."""
        if not isinstance(member, AlertMember):
            raise ValidationError("an alert member is required")
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO alert_overflow (item_id, alert_id, label, confidence, settled_at, "
                "content_hash, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (item_id) DO UPDATE SET alert_id = excluded.alert_id, "
                "label = excluded.label, confidence = excluded.confidence, "
                "settled_at = excluded.settled_at, content_hash = excluded.content_hash, "
                "occurred_at = excluded.occurred_at",
                (
                    member.item_id,
                    alert_id,
                    member.label.value,
                    member.confidence,
                    _text_or_none(member.settled_at),
                    member.content_hash,
                    format_utc(member.occurred_at),
                ),
            )

    def move_overflow(self, from_id: str, to_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE alert_overflow SET alert_id = ? WHERE alert_id = ?", (to_id, from_id)
            )

    # ---- the cap on members and on the trail ----

    def count_members(self, alert_id: str) -> int:
        rows = self._db.query("SELECT count(*) FROM alert_items WHERE alert_id = ?", (alert_id,))
        return int(rows[0][0])

    def delete_member(self, item_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM alert_items WHERE item_id = ?", (item_id,))

    def weakest_member(self, alert_id: str, *, protect: str | None) -> AlertMember | None:
        """The member to give up first when the alert is full: an inactive one, then the least
        severe, least confident, newest. Never the earliest member nor `protect` (the lead)."""
        rows = self._db.query(
            f"SELECT {_MEMBER_COLUMNS} FROM alert_items WHERE alert_id = ? AND item_id <> ? "
            "AND item_id <> (SELECT item_id FROM alert_items WHERE alert_id = ? "
            "ORDER BY occurred_at, item_id LIMIT 1) "
            f"ORDER BY active, {_SEVERITY_RANK}, confidence, occurred_at DESC, item_id LIMIT 1",
            (alert_id, protect or "", alert_id),
        )
        return _member_from_row(rows[0]) if rows else None

    def count_events(self, alert_id: str) -> int:
        rows = self._db.query("SELECT count(*) FROM alert_events WHERE alert_id = ?", (alert_id,))
        return int(rows[0][0])

    def copy_events(self, from_id: str, to_id: str, *, limit: int) -> int:
        """Copy the trail of `from_id` under `to_id`, marked as absorbed and with its original
        times, at most `limit` entries (the entries that matter, status changes and withdrawals,
        before the routine ones). Returns how many were left out."""
        with self._db.transaction() as conn:
            total = int(
                conn.execute(
                    "SELECT count(*) FROM alert_events WHERE alert_id = ?", (from_id,)
                ).fetchone()[0]
            )
            take = max(0, min(total, limit))
            if take:
                conn.execute(
                    "INSERT INTO alert_events "
                    "(alert_id, at, kind, actor, reason, from_value, to_value, from_absorbed) "
                    "SELECT ?, at, kind, actor, reason, from_value, to_value, 1 FROM alert_events "
                    "WHERE seq IN (SELECT seq FROM alert_events WHERE alert_id = ? "
                    "ORDER BY (kind IN ('merged', 'updated')), seq LIMIT ?) ORDER BY seq",
                    (to_id, from_id, take),
                )
        return total - take

    def drop_events(self, alert_id: str, count: int) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE alerts SET events_dropped = events_dropped + ? WHERE id = ?",
                (count, alert_id),
            )

    # ---- retention ----

    def purge(self, *, before: datetime | None, handled: bool, now: datetime) -> int:
        """Remove the alerts whose latest occurrence is strictly before `before` and, with
        `handled`, those looked at (seen, dismissed) or withdrawn, EXCEPT while an active member
        still waits for the language model (its verdict may yet change what the alert means).
        Every item behind a removed alert is remembered with the verdict it had, so that it is
        raised again only when its verdict changes. The memory is dropped a year later."""
        conditions: list[str] = []
        params: list[Any] = []
        if before is not None:
            conditions.append("alerts.last_seen_at < ?")
            params.append(format_utc(before))
        if handled:
            conditions.append(
                "((alerts.status <> 'new' OR alerts.withdrawn_at IS NOT NULL) AND NOT EXISTS "
                "(SELECT 1 FROM alert_items w WHERE w.alert_id = alerts.id AND w.active = 1 "
                "AND json_extract(w.evidence, '$.provisional') = 1))"
            )
        removed = 0
        stamp = format_utc(now)
        with self._db.transaction() as conn:
            if conditions:
                where = " OR ".join(conditions)
                doomed = f"SELECT alerts.id FROM alerts WHERE {where}"
                conn.execute(
                    "INSERT INTO alert_forgotten (item_id, forgotten_at, label, confidence, "
                    "settled_at) SELECT item_id, ?, label, confidence, settled_at FROM ("
                    f"SELECT item_id, label, confidence, settled_at FROM alert_items "
                    f"WHERE alert_id IN ({doomed}) UNION ALL "
                    f"SELECT item_id, label, confidence, settled_at FROM alert_overflow "
                    f"WHERE alert_id IN ({doomed})) WHERE true "
                    "ON CONFLICT (item_id) DO UPDATE SET forgotten_at = excluded.forgotten_at, "
                    "label = excluded.label, confidence = excluded.confidence, "
                    "settled_at = excluded.settled_at",
                    [stamp, *params, *params],
                )
                removed = conn.execute(f"DELETE FROM alerts WHERE {where}", params).rowcount
            conn.execute(
                "DELETE FROM alert_forgotten WHERE forgotten_at < ?",
                (format_utc(now - FORGOTTEN_KEEP),),
            )
            conn.execute(
                "DELETE FROM alert_forgotten WHERE item_id IN (SELECT item_id FROM alert_forgotten "
                "ORDER BY forgotten_at DESC, item_id DESC LIMIT -1 OFFSET ?)",
                (MAX_FORGOTTEN,),
            )
            for table in ("alert_failures", "alert_dirty"):
                conn.execute(
                    f"DELETE FROM {table} "
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
