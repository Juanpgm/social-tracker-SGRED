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


@runtime_checkable
class SignatureRepository(Protocol):
    """Persistence port for the near-duplicate index.

    `find_candidates` is the only lookup and it never scans: it returns the signatures that have
    the same content hash as the item OR share at least one of its band keys (see
    `similarity.band_keys`), never the item's own, oldest `ref_at` first (then by item id), at
    most `limit` (1..1000). Verifying a candidate against the full signature is the caller's job.
    """

    def add(self, signature: ItemSignature) -> bool:
        """Store a signature and its bands; False if the item already has one."""
        ...

    def get(self, item_id: str) -> ItemSignature | None: ...

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
