"""In-memory `EscalationRepository`: the reference for the contract and for service tests.

The queue needs the verdicts and the items, which live in their own in-memory repositories. They
are held through `_Peers`, which a `deepcopy` returns as it is: the in-memory unit of work copies
the state of its participants to roll it back, and a rollback of THIS repository must not swap the
other repositories for copies.
"""

import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from social_tracker.items.domain import ItemStatus, validate_limit
from social_tracker.llm.domain import (
    EscalationCounts,
    EscalationState,
    Judgment,
    QueueEntry,
    hour_start,
)
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError


@dataclass
class _Peers:
    classifications: Any
    items: Any

    def __deepcopy__(self, memo: dict) -> "_Peers":
        return self


class InMemoryEscalationRepository:
    def __init__(self, classifications: Any, items: Any) -> None:
        self._lock = threading.Lock()
        self._peers = _Peers(classifications, items)
        self._judgments: dict[str, Judgment] = {}
        self._states: dict[str, EscalationState] = {}
        self._budget: dict[datetime, int] = {}
        self._lease: tuple[str, datetime] | None = None

    # ---- the queue ----

    def _waiting(self) -> list[Any]:
        rows = list(self._peers.classifications._rows.values())
        live = []
        for row in rows:
            if not row.needs_llm:
                continue
            item = self._peers.items.get(row.item_id)
            if item is not None and item.status is ItemStatus.CLASSIFIED:
                live.append((row, item))
        return live

    def queue(self, *, source_ids: Collection[str], now: datetime, limit: int) -> list[QueueEntry]:
        validate_limit(limit)
        moment = ensure_utc(now)
        eligible = set(source_ids)
        with self._lock:
            states = dict(self._states)
        found = []
        for row, item in self._waiting():
            if item.source_id not in eligible:
                continue
            state = states.get(row.item_id)
            if state is not None and (
                state.state != "escalation_failed"
                or (state.next_attempt_at is not None and state.next_attempt_at > moment)
            ):
                continue
            found.append(QueueEntry(row.item_id, row.priority, row.classified_at))
        found.sort(key=lambda e: (-e.priority, e.classified_at, e.item_id))
        return found[:limit]

    def queued_count(self) -> int:
        return len(self._waiting())

    def oldest_queued(self, limit: int) -> list[str]:
        validate_limit(limit)
        ordered = sorted((row.classified_at, row.item_id) for row, _ in self._waiting())
        return [item_id for _, item_id in ordered[:limit]]

    # ---- the budget ----

    def reserve_call(self, hour: datetime, cap: int) -> bool:
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValidationError("the cap must be a whole number from 1")
        start = self._whole_hour(hour)
        with self._lock:
            used = self._budget.get(start, 0)
            if used >= cap:
                return False
            self._budget[start] = used + 1
            return True

    def calls_in_hour(self, hour: datetime) -> int:
        with self._lock:
            return self._budget.get(self._whole_hour(hour), 0)

    @staticmethod
    def _whole_hour(hour: datetime) -> datetime:
        start = hour_start(hour)
        if start != ensure_utc(hour):
            raise ValidationError("a budget hour starts on the hour")
        return start

    # ---- bookkeeping and judgments ----

    def get_state(self, item_id: str) -> EscalationState | None:
        with self._lock:
            return self._states.get(item_id)

    def put_state(self, state: EscalationState) -> None:
        if not isinstance(state, EscalationState):
            raise ValidationError("an escalation state is required")
        with self._lock:
            self._states[state.item_id] = state

    def clear_state(self, item_id: str) -> None:
        with self._lock:
            self._states.pop(item_id, None)

    def add_judgment(self, judgment: Judgment) -> bool:
        if not isinstance(judgment, Judgment):
            raise ValidationError("a judgment is required")
        with self._lock:
            if judgment.item_id in self._judgments:
                return False
            self._judgments[judgment.item_id] = replace(judgment)
            return True

    def get_judgment(self, item_id: str) -> Judgment | None:
        with self._lock:
            return self._judgments.get(item_id)

    def counts(self) -> EscalationCounts:
        with self._lock:
            outcomes = [j.outcome for j in self._judgments.values()]
            states = [s.state for s in self._states.values()]
        return EscalationCounts(
            judged=outcomes.count("judged"),
            invalid=outcomes.count("invalid"),
            reused=outcomes.count("reused"),
            failed_waiting=states.count("escalation_failed"),
            parked=states.count("parked"),
            dropped=states.count("dropped"),
            queued=self.queued_count(),
        )

    # ---- the lease ----

    def try_lease(self, holder: str, now: datetime, until: datetime) -> bool:
        moment, end = ensure_utc(now), ensure_utc(until)
        if end <= moment or not isinstance(holder, str) or not 1 <= len(holder) <= 100:
            raise ValidationError("a lease needs a holder and an end after its start")
        with self._lock:
            if self._lease is not None:
                owner, expires = self._lease
                if owner != holder and expires > moment:
                    return False
            self._lease = (holder, end)
            return True

    def release_lease(self, holder: str) -> None:
        with self._lock:
            if self._lease is not None and self._lease[0] == holder:
                self._lease = None

    def forget(self, item_ids: Sequence[str]) -> int:
        forgotten = 0
        with self._lock:
            for item_id in item_ids:
                gone = self._judgments.pop(item_id, None) is not None
                gone = self._states.pop(item_id, None) is not None or gone
                forgotten += gone
        return forgotten
