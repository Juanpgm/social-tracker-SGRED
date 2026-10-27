"""Ports of the items module."""

from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from social_tracker.items.domain import (
    DEFAULT_LIST_LIMIT,
    AddReport,
    Item,
    ItemStatus,
    SourceUsage,
)


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
    (`purge_classified`), and return how many items they removed. A purge leaves a tombstone (the
    dedup key `(source_id, external_id)` alone) for every removed item whose publication date was
    unknown (flag `published_at_missing`) and, in `purge_classified`, for every one: `add_new`
    counts a tombstoned entry as a duplicate, so a feed that keeps listing it cannot bring it
    back. Dated entries need none: the collector skips what is older than the retention horizon.
    Each source remembers at most a fixed number of tombstones; beyond it the oldest are forgotten.
    """

    def add_new(
        self,
        items: Sequence[Item],
        *,
        max_items: int | None = None,
        max_bytes: int | None = None,
    ) -> AddReport:
        """Store the new items of a batch. The optional quotas bound what the item's SOURCE holds
        after the call (stored items already there count): a new item that would exceed
        `max_items` items or `max_bytes` bytes (`item_size_bytes`) is not stored and counts in
        `over_quota`, but a smaller one later in the batch may still fit. Duplicates are never
        charged. A quota that is not None must be a positive whole number."""
        ...

    def source_usage(self, source_id: str) -> SourceUsage: ...

    def get(self, item_id: str) -> Item | None: ...

    def count(self, source_id: str | None = None) -> int: ...

    def search(self, query: str, *, limit: int = DEFAULT_LIST_LIMIT) -> "list[Item]": ...

    def set_status(self, item_id: str, status: ItemStatus) -> bool:
        """Move an item to another pipeline stage; False if there is no such item."""
        ...

    def list_by_status(
        self, status: ItemStatus, *, limit: int, source_ids: Collection[str] | None = None
    ) -> "list[Item]":
        """Items at one pipeline stage, OLDEST published first (then by id): the processing
        order of a queue. `source_ids` restricts them to those sources (an empty collection
        matches nothing, None means every source). `limit` is 1..1000 and the status must be
        a known one (`ValidationError` otherwise)."""
        ...

    def purge_fetched_before(self, cutoff: datetime) -> int: ...

    def purge_classified(self, *, out_of_scope_fetched_before: datetime | None = None) -> int:
        """Remove the items that have been processed: classified ones always, and out-of-scope
        ones too, unless `out_of_scope_fetched_before` is given, in which case only those fetched
        strictly before it (a grace period: the topic vocabulary may change and an off-topic
        item is worth judging again). Items parked as failed are never removed here."""
        ...

    def count_by_status(self, status: ItemStatus) -> int:
        """How many items are at one pipeline stage."""
        ...

    # Defined last: the name shadows the builtin inside the class body.
    def list(
        self, *, source_id: str | None = None, limit: int = DEFAULT_LIST_LIMIT
    ) -> "list[Item]": ...
