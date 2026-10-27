"""In-memory `ItemRepository`: the reference for the contract and for service tests."""

import threading
from collections.abc import Collection, Sequence
from dataclasses import replace
from datetime import datetime

from social_tracker.items.domain import (
    DEFAULT_LIST_LIMIT,
    MAX_TOMBSTONES_PER_SOURCE,
    AddReport,
    Item,
    ItemStatus,
    SourceUsage,
    item_size_bytes,
    plain_words,
    search_terms,
    validate_limit,
    validate_quota,
)
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError


def _stage(status: ItemStatus) -> ItemStatus:
    try:
        return ItemStatus(status)
    except ValueError:
        raise ValidationError("unknown item status") from None


def _newest_first(items: list[Item]) -> list[Item]:
    return sorted(items, key=lambda i: (i.published_at, i.id), reverse=True)


_UNDATED = "published_at_missing"


class InMemoryItemRepository:
    def __init__(self, max_tombstones_per_source: int = MAX_TOMBSTONES_PER_SOURCE) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Item] = {}
        self._tombstones: dict[tuple[str, str], None] = {}  # insertion order = age
        self._max_tombstones = max_tombstones_per_source

    def add_new(
        self,
        items: Sequence[Item],
        *,
        max_items: int | None = None,
        max_bytes: int | None = None,
    ) -> AddReport:
        if not all(isinstance(item, Item) for item in items):
            raise ValidationError("items must be Item objects")
        item_cap = validate_quota(max_items, "max_items")
        byte_cap = validate_quota(max_bytes, "max_bytes")
        with self._lock:
            external = {(i.source_id, i.external_id) for i in self._items.values()}
            hashes = {(i.source_id, i.content_hash) for i in self._items.values()}
            new = duplicate = over = 0
            for item in items:
                keys = ((item.source_id, item.external_id), (item.source_id, item.content_hash))
                if keys[0] in external or keys[1] in hashes or keys[0] in self._tombstones:
                    duplicate += 1
                    continue
                held = self._usage(item.source_id)
                cost = item_size_bytes(item)
                if (item_cap is not None and held.items >= item_cap) or (
                    byte_cap is not None and held.bytes + cost > byte_cap
                ):
                    over += 1
                    continue
                external.add(keys[0])
                hashes.add(keys[1])
                self._items[item.id] = item
                new += 1
        return AddReport(new, duplicate, over)

    def _usage(self, source_id: str) -> SourceUsage:
        mine = [i for i in self._items.values() if i.source_id == source_id]
        return SourceUsage(len(mine), sum(item_size_bytes(i) for i in mine))

    def source_usage(self, source_id: str) -> SourceUsage:
        with self._lock:
            return self._usage(source_id)

    def _remember(self, removed: list[Item]) -> None:
        touched = set()
        for item in removed:
            self._tombstones[(item.source_id, item.external_id)] = None
            touched.add(item.source_id)
        for source_id in touched:
            keys = [k for k in self._tombstones if k[0] == source_id]
            for key in keys[: max(0, len(keys) - self._max_tombstones)]:
                del self._tombstones[key]

    def get(self, item_id: str) -> Item | None:
        with self._lock:
            return self._items.get(item_id)

    def count(self, source_id: str | None = None) -> int:
        with self._lock:
            return sum(1 for i in self._items.values() if source_id in (None, i.source_id))

    def search(self, query: str, *, limit: int = DEFAULT_LIST_LIMIT) -> list[Item]:
        validate_limit(limit)
        terms = set(search_terms(query))
        if not terms:
            return []
        with self._lock:
            found = [
                i for i in self._items.values() if terms <= set(plain_words(f"{i.title} {i.text}"))
            ]
        return _newest_first(found)[:limit]

    def set_status(self, item_id: str, status: ItemStatus) -> bool:
        stage = _stage(status)
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False
            self._items[item_id] = replace(item, status=stage)
            return True

    def list_by_status(
        self, status: ItemStatus, *, limit: int, source_ids: Collection[str] | None = None
    ) -> list[Item]:
        stage = _stage(status)
        validate_limit(limit)
        with self._lock:
            found = [
                i
                for i in self._items.values()
                if i.status is stage and (source_ids is None or i.source_id in source_ids)
            ]
        return sorted(found, key=lambda i: (i.published_at, i.id))[:limit]

    def purge_fetched_before(self, cutoff: datetime) -> int:
        limit = ensure_utc(cutoff)
        with self._lock:
            doomed = [i for i in self._items.values() if i.fetched_at < limit]
            self._remember([i for i in doomed if _UNDATED in i.flags])
            for item in doomed:
                del self._items[item.id]
        return len(doomed)

    def purge_classified(self, *, out_of_scope_fetched_before: datetime | None = None) -> int:
        grace = (
            None if out_of_scope_fetched_before is None else ensure_utc(out_of_scope_fetched_before)
        )
        with self._lock:
            doomed = [
                i
                for i in self._items.values()
                if i.status is ItemStatus.CLASSIFIED
                or (i.status is ItemStatus.OUT_OF_SCOPE and (grace is None or i.fetched_at < grace))
            ]
            self._remember(doomed)
            for item in doomed:
                del self._items[item.id]
        return len(doomed)

    def count_by_status(self, status: ItemStatus) -> int:
        stage = _stage(status)
        with self._lock:
            return sum(1 for i in self._items.values() if i.status is stage)

    # Defined last: the name shadows the builtin inside the class body.
    def list(self, *, source_id: str | None = None, limit: int = DEFAULT_LIST_LIMIT) -> list[Item]:
        validate_limit(limit)
        with self._lock:
            found = [i for i in self._items.values() if source_id in (None, i.source_id)]
        return _newest_first(found)[:limit]
