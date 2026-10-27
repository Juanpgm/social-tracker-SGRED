"""Ports of the items module."""

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from social_tracker.items.domain import DEFAULT_LIST_LIMIT, AddReport, Item, ItemStatus


@runtime_checkable
class ItemRepository(Protocol):
    """Persistence port for collected items.

    There is deliberately no link to the source store: an item carries `source_id` and a
    `source_name` snapshot, so deleting a source never removes its items (SR-9).

    `add_new` stores a batch all or nothing and skips what is already known, by
    `(source_id, external_id)` or by `(source_id, content_hash)`; the first of two duplicates inside
    one batch wins. Listing and search return newest first (`published_at`, then id, both
    descending). `search` takes arbitrary user text: it is reduced to plain words (see
    `search_terms`) that must ALL be present as whole words in the title or text, ignoring case and
    accents; it never raises for odd text. `limit` is 1..1000 (`ValidationError` otherwise).

    Retention purges delete by `fetched_at`, strictly older than the cutoff, and by status
    (`purge_classified`), and return how many items they removed.
    """

    def add_new(self, items: Sequence[Item]) -> AddReport: ...

    def get(self, item_id: str) -> Item | None: ...

    def count(self, source_id: str | None = None) -> int: ...

    def search(self, query: str, *, limit: int = DEFAULT_LIST_LIMIT) -> "list[Item]": ...

    def set_status(self, item_id: str, status: ItemStatus) -> bool:
        """Move an item to another pipeline stage; False if there is no such item."""
        ...

    def purge_fetched_before(self, cutoff: datetime) -> int: ...

    def purge_classified(self) -> int: ...

    # Defined last: the name shadows the builtin inside the class body.
    def list(
        self, *, source_id: str | None = None, limit: int = DEFAULT_LIST_LIMIT
    ) -> "list[Item]": ...
