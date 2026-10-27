"""In-memory `ItemRepository`: the reference for the contract and for service tests."""

import threading
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from social_tracker.items.domain import (
    DEFAULT_LIST_LIMIT,
    AddReport,
    Item,
    ItemStatus,
    plain_words,
    search_terms,
    validate_limit,
)
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError


def _newest_first(items: list[Item]) -> list[Item]:
    return sorted(items, key=lambda i: (i.published_at, i.id), reverse=True)


class InMemoryItemRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, Item] = {}

    def add_new(self, items: Sequence[Item]) -> AddReport:
        if not all(isinstance(item, Item) for item in items):
            raise ValidationError("items must be Item objects")
        with self._lock:
            external = {(i.source_id, i.external_id) for i in self._items.values()}
            hashes = {(i.source_id, i.content_hash) for i in self._items.values()}
            new = duplicate = 0
            for item in items:
                keys = ((item.source_id, item.external_id), (item.source_id, item.content_hash))
                if keys[0] in external or keys[1] in hashes:
                    duplicate += 1
                    continue
                external.add(keys[0])
                hashes.add(keys[1])
                self._items[item.id] = item
                new += 1
        return AddReport(new, duplicate)

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
        try:
            stage = ItemStatus(status)
        except ValueError:
            raise ValidationError("unknown item status") from None
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return False
            self._items[item_id] = replace(item, status=stage)
            return True

    def purge_fetched_before(self, cutoff: datetime) -> int:
        limit = ensure_utc(cutoff)
        with self._lock:
            doomed = [i.id for i in self._items.values() if i.fetched_at < limit]
            for item_id in doomed:
                del self._items[item_id]
        return len(doomed)

    def purge_classified(self) -> int:
        with self._lock:
            doomed = [i.id for i in self._items.values() if i.status is ItemStatus.CLASSIFIED]
            for item_id in doomed:
                del self._items[item_id]
        return len(doomed)

    # Defined last: the name shadows the builtin inside the class body.
    def list(self, *, source_id: str | None = None, limit: int = DEFAULT_LIST_LIMIT) -> list[Item]:
        validate_limit(limit)
        with self._lock:
            found = [i for i in self._items.values() if source_id in (None, i.source_id)]
        return _newest_first(found)[:limit]
