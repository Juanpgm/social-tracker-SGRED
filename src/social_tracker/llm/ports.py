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

    def identify(self) -> None:
        """Check that what answers at the address is the model server it should be, before any
        item text is sent. Raises `JudgeError` (`not_ollama` for something else, `unreachable`
        for nothing)."""
        ...


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

    def queued_count(self, *, source_ids: Collection[str], now: datetime) -> int:
        """Items the queue could serve now: waiting, classified, from one of `source_ids`, and not
        waiting out a backoff or parked/dropped. This is what the queue bound counts."""
        ...

    def queued_by_source(self, *, source_ids: Collection[str], now: datetime) -> dict[str, int]:
        """The same items, counted per source (sources with none are absent)."""
        ...

    def drop_candidates(
        self,
        *,
        source_ids: Collection[str],
        now: datetime,
        limit: int,
        source_id: str | None = None,
    ) -> list[str]:
        """Up to `limit` (1..1000) of the same items in the order they are dropped from a full
        queue: the LOWEST priority first, then the oldest classified, then by id, so an older
        high-priority item never goes before a newer low-priority one. `source_id` limits it to
        one source."""
        ...

    def reserve_call(self, hour: datetime, cap: int) -> bool:
        """Count one judgment started in the budget hour unless `cap` have been: True if there
        was room. Atomic across processes."""
        ...

    def release_call(self, hour: datetime) -> None:
        """Give one reservation of the hour back (the call failed before the model gave
        anything). Never goes below zero."""
        ...

    def judged_since(self, since: datetime) -> int:
        """How many valid judgments (outcome `judged`) were made at or after `since`."""
        ...

    def parked(self, *, before: datetime | None, limit: int) -> list[EscalationState]:
        """Up to `limit` (1..1000) parked items, the longest parked first. With `before`, only
        those parked at or before that moment."""
        ...

    def orphaned_copies(self, *, source_ids: Collection[str], limit: int) -> list[str]:
        """Up to `limit` (1..1000) ids of exact copies (of a source in `source_ids`, classified)
        that wait for an original which will never be judged: its verdict or item is gone, it is
        not classified any more, it no longer waits, or its source is not in `source_ids`. An
        original that merely waits out a backoff is not one. By id."""
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
