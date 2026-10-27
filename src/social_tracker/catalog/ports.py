"""Ports of the catalog module."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from social_tracker.catalog.domain import CatalogEvent, CatalogSource, RefreshState


@dataclass(frozen=True)
class UpsertReport:
    new: int = 0
    updated: int = 0
    unchanged: int = 0


@runtime_checkable
class CatalogRepository(Protocol):
    """Persistence port of the catalog cache. Writes belong inside the caller's transaction (the
    refresher groups the events and the state of one catalog in one); reads need none."""

    def upsert_events(self, events: Sequence[CatalogEvent], fetched_at: datetime) -> UpsertReport:
        """Store the events; an event whose (source, id) is known replaces the stored one in
        place when anything changed, and is left alone when nothing did."""
        ...

    def events_between(self, start: datetime, end: datetime) -> list[CatalogEvent]:
        """Events with `start <= origin_time <= end`, by origin time then source and id."""
        ...

    def count_by_source(self) -> dict[CatalogSource, int]: ...

    def purge_before(self, cutoff: datetime) -> int:
        """Forget events whose origin time is strictly before `cutoff`; returns how many."""
        ...

    def get_state(self, source: CatalogSource) -> RefreshState | None: ...

    def all_states(self) -> list[RefreshState]:
        """Every known state, sorted by source name."""
        ...

    def try_claim(self, source: CatalogSource, now: datetime, until: datetime) -> bool:
        """Atomically take the right to refresh `source` until `until`: True unless an earlier
        claim is still running (`claimed_until > now`)."""
        ...

    def release_claim(self, source: CatalogSource) -> None: ...

    def record_success(self, source: CatalogSource, at: datetime) -> None:
        """A good refresh: remember it, forget failures, backoff and the claim."""
        ...

    def record_failure(
        self, source: CatalogSource, at: datetime, next_attempt_at: datetime, error: str
    ) -> None:
        """A failed refresh: count it, keep the last good time, set when to try again and drop
        the claim."""
        ...
