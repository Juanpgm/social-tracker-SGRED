"""Ports of the alerts module."""

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from social_tracker.alerts.domain import (
    Alert,
    AlertCounts,
    AlertEvent,
    AlertFilter,
    AlertMember,
)


@runtime_checkable
class AlertRepository(Protocol):
    """Persistence port of the alert store.

    An alert is a projection of its members (the items behind it): the service builds both and
    stores them together inside one transaction. `save` and `save_member` are upserts (by alert id
    and by item id), so storing the same thing twice leaves one row. An item belongs to at most one
    alert (the item id is the key of a member), which is what makes the whole step idempotent.
    Every value read back is rebuilt through the domain, so a tampered row is a `StorageError`.

    Listing is always newest first: the time of the latest occurrence, then the id, descending.
    A filter on `status` accepts `withdrawn` too; with no status filter withdrawn alerts are not
    listed. `source_id` matches any source linked to the alert, `tag` any of its tags, `since` and
    `until` the latest occurrence (both inclusive).
    """

    def get(self, alert_id: str) -> Alert | None: ...

    def save(self, alert: Alert) -> None:
        """Insert the alert or replace the one with its id."""
        ...

    def member(self, item_id: str) -> AlertMember | None:
        """The member an item is, if it raised (or joined) an alert."""
        ...

    def members(self, alert_id: str) -> list[AlertMember]:
        """The members of an alert, oldest occurrence first (then by item id)."""
        ...

    def save_member(self, member: AlertMember) -> None:
        """Insert the member or replace the one of its item. The alert must exist."""
        ...

    def add_event(self, event: AlertEvent) -> None:
        """Append to the audit trail of an alert. The alert must exist."""
        ...

    def events(self, alert_id: str) -> list[AlertEvent]:
        """The trail of an alert in the order it happened."""
        ...

    def find_merge_target(
        self,
        *,
        content_hash: str,
        related_items: Sequence[str],
        around: datetime,
        window: timedelta,
    ) -> str | None:
        """The id of the alert (the oldest one) that has a member with this content hash, or one
        of `related_items` (the originals a copy points to), whose occurrence is within `window`
        of `around`, boundaries included; None if there is none."""
        ...

    def count(self, flt: AlertFilter) -> int: ...

    def counts(self) -> AlertCounts:
        """The figures of the whole store, read from one consistent snapshot."""
        ...

    def sync_ids(
        self, *, min_confidence: float, full: bool, after: str | None, limit: int
    ) -> list[str]:
        """Up to `limit` (1..1000) item ids, ordered, that are greater than `after`, whose stored
        verdict the alert step has to look at: a risk verdict at or above `min_confidence` of an
        item that exists and has no alert yet (and whose alert was not forgotten by the retention
        purge), and a member whose stored verdict differs from the one it was evaluated on. With
        `full`, every member too (the threshold may have moved)."""
        ...

    def is_forgotten(self, item_id: str) -> bool:
        """True when the retention purge removed the alert of this item."""
        ...

    def purge(self, *, before: datetime | None, handled: bool, now: datetime) -> int:
        """Remove the alerts whose latest occurrence is strictly before `before`, and, with
        `handled`, those that are not new any more or are withdrawn; with their members and trail.
        Each member is remembered as forgotten so that it does not raise the alert again. Returns
        how many alerts were removed."""
        ...

    # Defined last: the name shadows the builtin inside the class body.
    def list(self, flt: AlertFilter, *, limit: int, offset: int) -> list[Alert]: ...
