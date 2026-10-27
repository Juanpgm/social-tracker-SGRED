"""In-memory classification ports: the reference for the contracts and for service tests."""

import threading
from collections import Counter
from collections.abc import Sequence
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
        self._failures: dict[str, int] = {}

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
            near_duplicates=sum(r.near_of is not None for r in rows),
            awaiting_original=sum(
                r.awaiting_original and self._original_waits(r.duplicate_of, rows) for r in rows
            ),
            orphaned_awaiting=sum(
                r.awaiting_original and not self._original_waits(r.duplicate_of, rows) for r in rows
            ),
        )

    @staticmethod
    def _original_waits(original_id: str | None, rows: list[Classification]) -> bool:
        return any(o.item_id == original_id and o.needs_llm for o in rows)

    def _release_copies(self, gone: set[str]) -> None:
        """Copies that waited for an original that is forgotten become normal queued items."""
        for row in list(self._rows.values()):
            if not (row.awaiting_original and row.duplicate_of in gone):
                continue
            original = self._rows.get(row.duplicate_of)
            priority = max(0.1, original.priority if original is not None else 0.1)
            self._rows[row.item_id] = replace(
                row,
                awaiting_original=False,
                duplicate_of=None,
                needs_llm=True,
                priority=priority,
            )

    def delete(self, item_id: str) -> bool:
        with self._lock:
            self._release_copies({item_id})
            self._failures.pop(item_id, None)
            return self._rows.pop(item_id, None) is not None

    def older_than(self, cutoff: datetime, limit: int, offset: int = 0) -> list[str]:
        validate_limit(limit)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValidationError("offset must be a whole number from 0")
        before = ensure_utc(cutoff)
        with self._lock:
            old = [r for r in self._rows.values() if r.classified_at < before]
        ordered = sorted(old, key=lambda r: (r.classified_at, r.item_id))
        return [r.item_id for r in ordered][offset : offset + limit]

    def delete_many(self, item_ids: Sequence[str]) -> int:
        with self._lock:
            removed = 0
            self._release_copies(set(item_ids))
            for item_id in item_ids:
                self._failures.pop(item_id, None)
                removed += self._rows.pop(item_id, None) is not None
            return removed

    def note_failure(self, item_id: str, at: datetime) -> int:
        ensure_utc(at)
        with self._lock:
            self._failures[item_id] = min(self._failures.get(item_id, 0) + 1, 1000)
            return self._failures[item_id]

    def clear_failure(self, item_id: str) -> None:
        with self._lock:
            self._failures.pop(item_id, None)

    def settle(self, classification: Classification) -> bool:
        if not isinstance(classification, Classification):
            raise ValidationError("a classification is required")
        with self._lock:
            stored = self._rows.get(classification.item_id)
            if stored is None or not (stored.needs_llm or stored.awaiting_original):
                return False
            self._rows[stored.item_id] = replace(
                stored,
                label=classification.label,
                confidence=classification.confidence,
                stage=classification.stage,
                rationale=classification.rationale,
                low_confidence=classification.low_confidence,
                needs_llm=classification.needs_llm,
                awaiting_original=classification.awaiting_original,
                priority=classification.priority,
                duplicate_of=classification.duplicate_of,
                near_of=classification.near_of,
                settled_at=classification.settled_at,
                settle_reason=classification.settle_reason,
            )
            return True

    def reopen(self, item_id: str, priority: float) -> bool:
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int | float)
            or not 0.0 <= priority <= 1.0
        ):
            raise ValidationError("a priority is a number from 0 to 1")
        with self._lock:
            stored = self._rows.get(item_id)
            if stored is None or stored.needs_llm or stored.settle_reason != "parked":
                return False
            self._rows[item_id] = replace(
                stored, needs_llm=True, priority=priority, settled_at=None, settle_reason=None
            )
            return True

    def awaiting(self, original_id: str) -> list[Classification]:
        with self._lock:
            waiting = [
                r
                for r in self._rows.values()
                if r.awaiting_original and r.duplicate_of == original_id
            ]
        return sorted(waiting, key=lambda r: r.item_id)


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

    def find_exact(
        self, *, item_id, content_hash, text_hash, ref_from=None, ref_to=None, limit=100
    ):
        validate_limit(limit)
        with self._lock:
            rows = list(self._rows.values())
        low = None if ref_from is None else ensure_utc(ref_from)
        high = None if ref_to is None else ensure_utc(ref_to)
        found = [
            row
            for row in rows
            if row.item_id != item_id
            and (row.content_hash == content_hash or (text_hash and row.text_hash == text_hash))
            and (low is None or row.ref_at >= low)
            and (high is None or row.ref_at <= high)
        ]
        return sorted(found, key=lambda r: (r.ref_at, r.item_id))[:limit]

    def find_candidates(self, *, item_id, content_hash, band_keys, limit):
        validate_limit(limit)
        keys = tuple(band_keys) if band_keys is not None else ()
        if len(keys) != BANDS:
            raise ValidationError("one key per band is required")
        with self._lock:
            rows = list(self._rows.values())
        ranked = []
        for row in rows:
            if row.item_id == item_id:
                continue
            shared = sum(a == b for a, b in zip(band_keys_of(row), keys, strict=True))
            if row.content_hash == content_hash:
                shared = BANDS
            if shared:
                ranked.append((shared, row))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].ref_at, pair[1].item_id))
        return [row for _, row in ranked[:limit]]

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
