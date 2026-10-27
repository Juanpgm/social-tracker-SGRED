"""Ports of the LLM escalation."""

from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from social_tracker.llm.domain import (
    EscalationCounts,
    EscalationState,
    JudgeRequest,
    JudgeResponse,
    Judgment,
    QueueEntry,
)


@runtime_checkable
class LlmJudge(Protocol):
    """The judge: one question, one validated answer.

    `judge` raises `JudgeError` for everything that is not a valid verdict: nothing listening,
    a timeout, a server error, a body that is not an answer, no model, or a model output that
    breaks the schema (`invalid_output`). It never returns a half-valid response and never
    retries on its own beyond the fallback model. `list_models` names the models the server has."""

    def judge(self, request: JudgeRequest) -> JudgeResponse: ...

    def list_models(self) -> tuple[str, ...]: ...


@runtime_checkable
class EscalationRepository(Protocol):
    """Persistence port of the escalation bookkeeping. The verdicts themselves are changed
    through `ClassificationRepository.settle`; this port keeps the judgments, the failure and
    budget accounting and the cross-process lease."""

    def queue(self, *, source_ids: Collection[str], now: datetime, limit: int) -> list[QueueEntry]:
        """Items waiting for the judge, best first: highest priority, then oldest classified
        (then by id). Only items that still exist as `classified` and belong to `source_ids`,
        and not those that wait out a backoff or are parked/dropped."""
        ...

    def queued_count(self) -> int:
        """Items waiting for the judge, wherever they come from."""
        ...

    def oldest_queued(self, limit: int) -> list[str]:
        """Up to `limit` waiting items, oldest classified first (then by id)."""
        ...

    def reserve_call(self, hour: datetime, cap: int) -> bool:
        """Count one judgment started in the budget hour unless `cap` have been: True if there
        was room. Atomic across processes."""
        ...

    def calls_in_hour(self, hour: datetime) -> int: ...

    def get_state(self, item_id: str) -> EscalationState | None: ...

    def put_state(self, state: EscalationState) -> None:
        """Insert or replace the bookkeeping of an item."""
        ...

    def clear_state(self, item_id: str) -> None: ...

    def add_judgment(self, judgment: Judgment) -> bool:
        """Store the judgment unless the item has one: True when stored (never overwritten)."""
        ...

    def get_judgment(self, item_id: str) -> Judgment | None: ...

    def counts(self) -> EscalationCounts: ...

    def try_lease(self, holder: str, now: datetime, until: datetime) -> bool:
        """Take the single in-flight slot if it is free, expired or already `holder`'s."""
        ...

    def release_lease(self, holder: str) -> None: ...

    def forget(self, item_ids: Sequence[str]) -> int:
        """Forget the judgments and bookkeeping of these items (their verdicts are gone)."""
        ...
