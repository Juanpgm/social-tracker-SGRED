"""In-memory classification ports: the reference for the contracts and for service tests."""

import threading
from collections import Counter
from dataclasses import replace
from datetime import datetime

from social_tracker.classification.domain import (
    Classification,
    ClassificationStats,
    ItemSignature,
)
from social_tracker.classification.similarity import BANDS, band_keys
from social_tracker.classification.topics import StoredTopicConfig
from social_tracker.items.domain import validate_limit
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError


class InMemoryClassificationRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, Classification] = {}

    def add(self, classification: Classification) -> bool:
        if not isinstance(classification, Classification):
            raise ValidationError("a classification is required")
        with self._lock:
            if classification.item_id in self._rows:
                return False
            self._rows[classification.item_id] = classification
            return True

    def get(self, item_id: str) -> Classification | None:
        with self._lock:
            return self._rows.get(item_id)

    def stats(self) -> ClassificationStats:
        with self._lock:
            rows = list(self._rows.values())
        labels = Counter(r.label.value for r in rows)
        reasons = Counter(r.scope_reason.value for r in rows if r.scope_reason is not None)
        return ClassificationStats(
            total=len(rows),
            by_label=tuple(sorted(labels.items())),
            out_of_scope_by_reason=tuple(sorted(reasons.items())),
            needs_llm=sum(r.needs_llm for r in rows),
            low_confidence=sum(r.low_confidence for r in rows),
            duplicates=sum(r.duplicate_of is not None for r in rows),
            recycled=sum(r.recycled_of is not None for r in rows),
            truncated=sum(r.truncated for r in rows),
        )


class InMemorySignatureRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, ItemSignature] = {}

    def add(self, signature: ItemSignature) -> bool:
        if not isinstance(signature, ItemSignature):
            raise ValidationError("a signature is required")
        with self._lock:
            if signature.item_id in self._rows:
                return False
            self._rows[signature.item_id] = signature
            return True

    def get(self, item_id: str) -> ItemSignature | None:
        with self._lock:
            return self._rows.get(item_id)

    def find_candidates(self, *, item_id, content_hash, band_keys, limit):
        validate_limit(limit)
        keys = tuple(band_keys) if band_keys is not None else ()
        if len(keys) != BANDS:
            raise ValidationError("one key per band is required")
        with self._lock:
            rows = list(self._rows.values())
        found = [
            row
            for row in rows
            if row.item_id != item_id
            and (
                row.content_hash == content_hash
                or any(a == b for a, b in zip(band_keys_of(row), keys, strict=True))
            )
        ]
        return sorted(found, key=lambda r: (r.ref_at, r.item_id))[:limit]

    def prune_before(self, cutoff: datetime) -> int:
        limit = ensure_utc(cutoff)
        with self._lock:
            doomed = [k for k, r in self._rows.items() if r.ref_at < limit]
            for key in doomed:
                del self._rows[key]
        return len(doomed)


def band_keys_of(signature: ItemSignature) -> tuple[int, ...]:
    return band_keys(signature.minhash)


class InMemoryTopicConfigRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._versions: list[StoredTopicConfig] = []

    def latest(self) -> StoredTopicConfig | None:
        with self._lock:
            return self._versions[-1] if self._versions else None

    def append(self, stored: StoredTopicConfig) -> int:
        if not isinstance(stored, StoredTopicConfig):
            raise ValidationError("a stored configuration is required")
        stored.config.to_json()  # the same size cap as the database
        with self._lock:
            number = len(self._versions) + 1
            self._versions.append(replace(stored, version=number))
            return number

    def versions(self, limit: int) -> list[StoredTopicConfig]:
        validate_limit(limit)
        with self._lock:
            return list(reversed(self._versions))[:limit]
