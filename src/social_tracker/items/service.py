"""Item service: reading collected items and the retention purge.

Retention (CI-10): raw items are kept for `retention_days` days from the moment they were
fetched. One more instant and they go (see `items.domain`): an item exactly 30 days old is still
kept. A retention of 0 means "purge once classified": no time is involved, every item that has
been processed by the classifier is removed: `classified` ones at once, `out_of_scope` ones only
once they were fetched more than `OUT_OF_SCOPE_GRACE` (24 hours) ago, so that a vocabulary change
in the first day can still bring an off-topic item back (`classify reclassify`). Items parked as
`classification_failed` stay until a retry or the time-based retention.
An item whose verdict still waits for the language model is not purged merely for being old:
it goes only when it was fetched more than TWICE the retention ago (twice the grace, with a
retention of 0), so a judge that is down for a while does not make the queue lose its items.
What the model said about a purged item (the audit record with its raw output) goes with it.
Alerts are a different store and are never touched here.

Nothing here publishes or sends anything (G-1).
"""

from dataclasses import dataclass
from datetime import timedelta

from social_tracker.items.domain import (
    DEFAULT_LIST_LIMIT,
    DEFAULT_RETENTION_DAYS,
    Item,
    retention_cutoff,
    validate_retention_days,
)
from social_tracker.items.ports import ItemRepository
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import NotFoundError
from social_tracker.shared.ids import is_valid_slug
from social_tracker.shared.uow import UnitOfWork

OUT_OF_SCOPE_GRACE = timedelta(hours=24)


@dataclass(frozen=True)
class PurgeReport:
    deleted: int
    retention_days: int


class ItemService:
    def __init__(
        self,
        items: ItemRepository,
        uow: UnitOfWork,
        clock: Clock,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self._items = items
        self._uow = uow
        self._clock = clock
        self.retention_days = validate_retention_days(retention_days)

    def list_items(
        self, *, source_id: str | None = None, limit: int = DEFAULT_LIST_LIMIT
    ) -> list[Item]:
        return self._items.list(source_id=source_id, limit=limit)

    def get(self, item_id: str) -> Item:
        """One stored item, or `NotFoundError`."""
        found = self._items.get(item_id) if is_valid_slug(item_id) else None
        if found is None:
            raise NotFoundError("item not found")
        return found

    def search(self, query: str, *, limit: int = DEFAULT_LIST_LIMIT) -> list[Item]:
        return self._items.search(query, limit=limit)

    def purge_expired(self) -> PurgeReport:
        now = self._clock.now()
        with self._uow.transaction():
            if self.retention_days == 0:
                deleted = self._items.purge_classified(
                    out_of_scope_fetched_before=now - OUT_OF_SCOPE_GRACE,
                    purge_queued_fetched_before=now - 2 * OUT_OF_SCOPE_GRACE,
                )
            else:
                cutoff = retention_cutoff(now, self.retention_days)
                deleted = self._items.purge_fetched_before(
                    cutoff,
                    purge_queued_fetched_before=now - 2 * timedelta(days=self.retention_days),
                )
        return PurgeReport(deleted, self.retention_days)
