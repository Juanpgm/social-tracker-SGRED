"""SQLite implementation of the escalation port (migration 0008).

Validation lives in the domain (`Judgment`, `EscalationState`) and is applied on every call; a
stored row that no longer builds one is a `StorageError`. The verdicts themselves are changed by
`SqliteClassificationRepository.settle`; this repository keeps what surrounds a judgment.

Idempotence is decided by the database: a judgment is `INSERT ... ON CONFLICT DO NOTHING` on the
unique item id, the budget is one `INSERT ... ON CONFLICT DO UPDATE ... WHERE calls < cap` that
is atomic across processes, and the lease is one row that is taken only when free, expired or the
caller's own.
"""

import json
import sqlite3
from collections.abc import Collection, Sequence
from datetime import datetime

from social_tracker.items.domain import validate_limit
from social_tracker.llm.domain import (
    EscalationCounts,
    EscalationState,
    Judgment,
    QueueEntry,
    hour_start,
)
from social_tracker.persistence.db import Database, format_utc, map_sqlite_error, parse_utc
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import DomainError, StorageError, ValidationError

_JUDGMENT_COLUMNS = (
    "item_id, outcome, label, confidence, rationale, evidence, llm_model, prompt_version, "
    "raw_output, prior_label, prior_confidence, prior_stage, reused_from, judged_at"
)
_STATE_COLUMNS = "item_id, state, attempts, next_attempt_at, last_error, updated_at"
_QUEUE = (
    "SELECT c.item_id, c.priority, c.classified_at FROM classifications c "
    "JOIN items i ON i.id = c.item_id "
    "LEFT JOIN llm_escalations e ON e.item_id = c.item_id "
    "WHERE c.needs_llm = 1 AND i.status = 'classified' AND i.source_id IN ({marks}) "
    "AND (e.item_id IS NULL OR (e.state = 'escalation_failed' "
    "AND (e.next_attempt_at IS NULL OR e.next_attempt_at <= ?))) "
    "ORDER BY c.priority DESC, c.classified_at, c.item_id LIMIT ?"
)
_WAITING = (
    "FROM classifications c JOIN items i ON i.id = c.item_id "
    "WHERE c.needs_llm = 1 AND i.status = 'classified'"
)


def _judgment_from_row(row: sqlite3.Row) -> Judgment:
    try:
        return Judgment(
            item_id=row["item_id"],
            outcome=row["outcome"],
            label=row["label"],
            confidence=row["confidence"],
            rationale=row["rationale"],
            evidence=tuple(json.loads(row["evidence"])),
            llm_model=row["llm_model"],
            prompt_version=row["prompt_version"],
            raw_output=row["raw_output"],
            prior_label=row["prior_label"],
            prior_confidence=row["prior_confidence"],
            prior_stage=row["prior_stage"],
            reused_from=row["reused_from"],
            judged_at=parse_utc(row["judged_at"]),
        )
    except (DomainError, ValueError, TypeError):
        raise StorageError("stored judgment is invalid") from None


def _state_from_row(row: sqlite3.Row) -> EscalationState:
    try:
        wait = row["next_attempt_at"]
        return EscalationState(
            item_id=row["item_id"],
            state=row["state"],
            attempts=row["attempts"],
            next_attempt_at=parse_utc(wait) if wait is not None else None,
            last_error=row["last_error"],
            updated_at=parse_utc(row["updated_at"]),
        )
    except (DomainError, ValueError, TypeError):
        raise StorageError("stored escalation state is invalid") from None


def _budget_hour(hour: datetime) -> str:
    start = hour_start(hour)
    if start != ensure_utc(hour):
        raise ValidationError("a budget hour starts on the hour")
    return format_utc(start)


class SqliteEscalationRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ---- the queue ----

    def queue(self, *, source_ids: Collection[str], now: datetime, limit: int) -> list[QueueEntry]:
        validate_limit(limit)
        moment = format_utc(now)
        sources = sorted(source_ids)
        if not sources:
            return []
        sql = _QUEUE.format(marks=", ".join("?" * len(sources)))
        rows = self._db.query(sql, (*sources, moment, limit))
        return [
            QueueEntry(str(r["item_id"]), float(r["priority"]), parse_utc(r["classified_at"]))
            for r in rows
        ]

    def queued_count(self) -> int:
        return int(self._db.query(f"SELECT count(*) {_WAITING}")[0][0])

    def oldest_queued(self, limit: int) -> list[str]:
        validate_limit(limit)
        rows = self._db.query(
            f"SELECT c.item_id {_WAITING} ORDER BY c.classified_at, c.item_id LIMIT ?", (limit,)
        )
        return [str(r[0]) for r in rows]

    # ---- the budget ----

    def reserve_call(self, hour: datetime, cap: int) -> bool:
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValidationError("the cap must be a whole number from 1")
        start = _budget_hour(hour)
        with self._db.transaction() as conn:
            return (
                conn.execute(
                    "INSERT INTO llm_budget (hour_start, calls) VALUES (?, 1) "
                    "ON CONFLICT (hour_start) DO UPDATE SET calls = calls + 1 WHERE calls < ?",
                    (start, cap),
                ).rowcount
                > 0
            )

    def calls_in_hour(self, hour: datetime) -> int:
        rows = self._db.query(
            "SELECT calls FROM llm_budget WHERE hour_start = ?", (_budget_hour(hour),)
        )
        return int(rows[0][0]) if rows else 0

    # ---- bookkeeping and judgments ----

    def get_state(self, item_id: str) -> EscalationState | None:
        rows = self._db.query(
            f"SELECT {_STATE_COLUMNS} FROM llm_escalations WHERE item_id = ?", (item_id,)
        )
        return _state_from_row(rows[0]) if rows else None

    def put_state(self, state: EscalationState) -> None:
        if not isinstance(state, EscalationState):
            raise ValidationError("an escalation state is required")
        wait = format_utc(state.next_attempt_at) if state.next_attempt_at is not None else None
        with self._db.transaction() as conn:
            conn.execute(
                f"INSERT INTO llm_escalations ({_STATE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (item_id) DO UPDATE SET state = excluded.state, "
                "attempts = excluded.attempts, next_attempt_at = excluded.next_attempt_at, "
                "last_error = excluded.last_error, updated_at = excluded.updated_at",
                (
                    state.item_id,
                    state.state,
                    state.attempts,
                    wait,
                    state.last_error,
                    format_utc(state.updated_at),
                ),
            )

    def clear_state(self, item_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM llm_escalations WHERE item_id = ?", (item_id,))

    def add_judgment(self, judgment: Judgment) -> bool:
        if not isinstance(judgment, Judgment):
            raise ValidationError("a judgment is required")
        j = judgment
        with self._db.transaction() as conn:
            return (
                conn.execute(
                    f"INSERT INTO llm_judgments ({_JUDGMENT_COLUMNS}) "
                    f"VALUES ({', '.join('?' * 14)}) ON CONFLICT (item_id) DO NOTHING",
                    (
                        j.item_id,
                        j.outcome,
                        j.label,
                        j.confidence,
                        j.rationale,
                        json.dumps(list(j.evidence), ensure_ascii=False),
                        j.llm_model,
                        j.prompt_version,
                        j.raw_output,
                        j.prior_label,
                        j.prior_confidence,
                        j.prior_stage,
                        j.reused_from,
                        format_utc(j.judged_at),
                    ),
                ).rowcount
                > 0
            )

    def get_judgment(self, item_id: str) -> Judgment | None:
        rows = self._db.query(
            f"SELECT {_JUDGMENT_COLUMNS} FROM llm_judgments WHERE item_id = ?", (item_id,)
        )
        return _judgment_from_row(rows[0]) if rows else None

    def counts(self) -> EscalationCounts:
        try:
            with self._db.snapshot() as conn:
                outcomes = dict(
                    conn.execute("SELECT outcome, count(*) FROM llm_judgments GROUP BY outcome")
                )
                states = dict(
                    conn.execute("SELECT state, count(*) FROM llm_escalations GROUP BY state")
                )
                queued = int(conn.execute(f"SELECT count(*) {_WAITING}").fetchone()[0])
        except sqlite3.Error as exc:
            raise map_sqlite_error(exc) from None
        return EscalationCounts(
            judged=int(outcomes.get("judged", 0)),
            invalid=int(outcomes.get("invalid", 0)),
            reused=int(outcomes.get("reused", 0)),
            failed_waiting=int(states.get("escalation_failed", 0)),
            parked=int(states.get("parked", 0)),
            dropped=int(states.get("dropped", 0)),
            queued=queued,
        )

    # ---- the lease ----

    def try_lease(self, holder: str, now: datetime, until: datetime) -> bool:
        moment, end = ensure_utc(now), ensure_utc(until)
        if end <= moment or not isinstance(holder, str) or not 1 <= len(holder) <= 100:
            raise ValidationError("a lease needs a holder and an end after its start")
        with self._db.transaction() as conn:
            return (
                conn.execute(
                    "INSERT INTO llm_lease (id, holder, until) VALUES (1, ?, ?) "
                    "ON CONFLICT (id) DO UPDATE SET holder = excluded.holder, "
                    "until = excluded.until WHERE llm_lease.holder = excluded.holder "
                    "OR llm_lease.until <= ?",
                    (holder, format_utc(end), format_utc(moment)),
                ).rowcount
                > 0
            )

    def release_lease(self, holder: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM llm_lease WHERE id = 1 AND holder = ?", (holder,))

    def forget(self, item_ids: Sequence[str]) -> int:
        ids = list(item_ids)
        if not ids:
            return 0
        marks = ", ".join("?" * len(ids))
        with self._db.transaction() as conn:
            gone = {
                str(r[0])
                for r in conn.execute(
                    f"SELECT item_id FROM llm_judgments WHERE item_id IN ({marks}) "
                    f"UNION SELECT item_id FROM llm_escalations WHERE item_id IN ({marks})",
                    ids + ids,
                )
            }
            conn.execute(f"DELETE FROM llm_judgments WHERE item_id IN ({marks})", ids)
            conn.execute(f"DELETE FROM llm_escalations WHERE item_id IN ({marks})", ids)
        return len(gone)
