"""Ports of the classification module."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from social_tracker.classification.domain import (
    Classification,
    ClassificationStats,
    ItemSignature,
)
from social_tracker.classification.topics import StoredTopicConfig


@runtime_checkable
class ClassificationRepository(Protocol):
    """Persistence port for verdicts. One per item: `add` never overwrites.

    A verdict is validated by `Classification` before it reaches a store, and a stored row that no
    longer builds one is a `StorageError`. `stats` reads one consistent snapshot."""

    def add(self, classification: Classification) -> bool:
        """Store the verdict unless the item already has one. True when stored, False when the
        item was already classified (the stored verdict is left as it is)."""
        ...

    def get(self, item_id: str) -> Classification | None: ...

    def stats(self) -> ClassificationStats: ...

    def delete(self, item_id: str) -> bool:
        """Forget the verdict of an item so that it can be classified again; False if it had
        none."""
        ...

    def older_than(self, cutoff: datetime, limit: int, offset: int = 0) -> list[str]:
        """Item ids of at most `limit` (1..1000) verdicts classified strictly before `cutoff`,
        oldest first, after skipping `offset`: the candidates of the history pruning."""
        ...

    def delete_many(self, item_ids: Sequence[str]) -> int:
        """Forget these verdicts (and their failure counters); returns how many verdicts."""
        ...

    def note_failure(self, item_id: str, at: datetime) -> int:
        """Count one more consecutive per-item error for the item and return the new count."""
        ...

    def clear_failure(self, item_id: str) -> None:
        """Forget the error counter of an item (it was classified, or retried by hand)."""
        ...

    def settle(self, classification: Classification) -> bool:
        """Replace a verdict that still waits for the LLM (it is queued, `needs_llm`, or it is an
        exact copy whose original is, `awaiting_original`). Only the label, confidence, stage,
        rationale, low_confidence, needs_llm, awaiting_original and priority are written; the
        rest of the stored verdict stays. False, changing nothing, when the item has no verdict
        or nothing waits: a settled verdict is never settled twice."""
        ...

    def awaiting(self, original_id: str) -> list[Classification]:
        """The exact copies waiting for this original to be judged, by item id."""
        ...


@runtime_checkable
class SignatureRepository(Protocol):
    """Persistence port for the near-duplicate index. Neither lookup scans.

    `find_exact` returns the signatures with the same content hash OR the same folded-text hash as
    the item (optionally only those whose `ref_at` is within `[ref_from, ref_to]`), never the
    item's own, oldest `ref_at` first (then by item id), at most `limit` (1..1000). It is the
    lookup of identical content and has nothing to do with the candidate limit of the bands.

    `find_candidates` returns the signatures that share at least one of the item's band keys (see
    `similarity.band_keys`) or have its content hash, never the item's own, best first: the most
    shared bands, then the oldest `ref_at` (then by item id), at most `limit` (1..1000).
    Verifying a candidate against the full signature is the caller's job.
    """

    def add(self, signature: ItemSignature) -> bool:
        """Store a signature and its bands; False if the item already has one."""
        ...

    def get(self, item_id: str) -> ItemSignature | None: ...

    def find_exact(
        self,
        *,
        item_id: str,
        content_hash: str,
        text_hash: str | None,
        ref_from: datetime | None = None,
        ref_to: datetime | None = None,
        limit: int = 100,
    ) -> list[ItemSignature]: ...

    def find_candidates(
        self, *, item_id: str, content_hash: str, band_keys: Sequence[int], limit: int
    ) -> list[ItemSignature]: ...

    def prune_before(self, cutoff: datetime) -> int:
        """Forget signatures whose `ref_at` is strictly before `cutoff`; returns how many."""
        ...


@runtime_checkable
class TopicConfigRepository(Protocol):
    """Persistence port for the history of the topic configuration. Versions are 1, 2, 3...
    and are never changed or removed; the latest one is in force."""

    def latest(self) -> StoredTopicConfig | None: ...

    def append(self, stored: StoredTopicConfig) -> int:
        """Store a new version (the `version` of `stored` is ignored) and return its number."""
        ...

    def versions(self, limit: int) -> list[StoredTopicConfig]:
        """The newest `limit` versions (1..1000), newest first."""
        ...


@dataclass(frozen=True)
class CatalogResult:
    """Outcome of the cross-check against an official earthquake catalog (RC-6). Only
    `unavailable` exists yet: the unit that implements the check adds the others."""

    status: str


@runtime_checkable
class CatalogCheck(Protocol):
    """Port of the official-catalog cross-check (RC-1 stage 4). It must never raise for a catalog
    that is down or answers badly: that is `unavailable`, which never blocks a classification."""

    def check(self, text: str, published_at: datetime) -> CatalogResult: ...


class UnavailableCatalog:
    """The check until a real one is wired: the catalog is never reachable."""

    def check(self, text: str, published_at: datetime) -> CatalogResult:
        return CatalogResult("unavailable")
