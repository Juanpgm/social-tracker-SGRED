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
    ForgottenSignature,
    SyncRef,
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

    def count_members(self, alert_id: str) -> int: ...

    def delete_member(self, item_id: str) -> None:
        """Remove one member; nothing to do if there is none."""
        ...

    def weakest_member(self, alert_id: str, *, protect: str | None) -> AlertMember | None:
        """The member to give up first when the alert is full (an inactive one, then the least
        severe, least confident, newest), never the earliest member nor `protect`."""
        ...

    def add_event(self, event: AlertEvent) -> None:
        """Append to the audit trail of an alert. The alert must exist."""
        ...

    def events(self, alert_id: str, limit: int | None = None) -> list[AlertEvent]:
        """The trail of an alert in the order it happened; with `limit`, the latest ones."""
        ...

    def count_events(self, alert_id: str) -> int: ...

    def copy_events(self, from_id: str, to_id: str, *, limit: int) -> int:
        """Copy the trail of an alert that is being absorbed under the alert that keeps going,
        with the original times and marked as absorbed, at most `limit` entries (status changes
        and withdrawals before routine entries). Returns how many were left out."""
        ...

    def drop_events(self, alert_id: str, count: int) -> None:
        """Count trail entries that were not stored (the trail is capped)."""
        ...

    def find_merge_targets(
        self,
        *,
        content_hash: str,
        related_items: Sequence[str],
        around: datetime,
        window: timedelta,
        item_id: str | None = None,
    ) -> list[str]:
        """The ids of the alerts, oldest first and each once, that have a member with this content
        hash, or one of `related_items` (the originals a copy points to), whose occurrence is
        within `window` of `around`, boundaries included (members given up by a full alert count
        too). With `item_id` also those that hold a
        COPY pointing at that item (so an original that is seen after its copies still finds
        them). More than one means the item bridges alerts that must become one."""
        ...

    def delete(self, alert_id: str) -> bool:
        """Remove an alert with its members and trail; False if there is none."""
        ...

    def linked_sources(self) -> list[tuple[str, str]]:
        """Every source some alert is linked to, with the name its latest snapshot has."""
        ...

    def count(self, flt: AlertFilter) -> int: ...

    def counts(self) -> AlertCounts:
        """The figures of the whole store, read from one consistent snapshot."""
        ...

    def sync_ids(
        self,
        *,
        min_confidence: float,
        full: bool,
        after: str | None,
        limit: int,
        now: datetime | None = None,
    ) -> list[SyncRef]:
        """Up to `limit` (1..1000) items, chronological (fetch time, then id), after the cursor
        `after`, whose stored verdict the alert step has to look at: a risk verdict at or above
        `min_confidence` of an item that exists and has no alert yet (and whose alert was not
        forgotten on this very verdict, nor counted as overflow), and a member whose stored
        verdict differs from the one it was evaluated on. With `full`, every member too (the
        threshold may have moved). With `now`, items in backoff or parked are left out."""
        ...

    def dirty_ids(
        self, *, after: str | None, limit: int, now: datetime | None = None
    ) -> list[SyncRef]:
        """Up to `limit` items whose verdict was written since the step last looked at them,
        chronologically, after the cursor `after`. With `now`, items in backoff or parked are
        left out."""
        ...

    def clear_dirty(self, item_id: str) -> None: ...

    def is_forgotten(self, item_id: str) -> bool:
        """True when the retention purge removed the alert of this item."""
        ...

    def forgotten_signature(self, item_id: str) -> ForgottenSignature | None:
        """The verdict the item had when its alert was forgotten, None if it was not."""
        ...

    def clear_forgotten(self, item_id: str) -> None: ...

    def record_failure(
        self,
        item_id: str,
        now: datetime,
        *,
        park_after: int,
        backoff_base: timedelta,
        backoff_cap: timedelta,
    ) -> int:
        """The step failed on this item: count it, back off (doubling, capped) and park it once
        it has failed `park_after` times. Returns the number of failures."""
        ...

    def clear_failure(self, item_id: str) -> None: ...

    def failure_counts(self, now: datetime) -> tuple[int, int]:
        """(items waiting to be tried again, items parked)."""
        ...

    def release_parked(self) -> int:
        """Give the parked items another chance at once. Returns how many."""
        ...

    def overflow_of(self, item_id: str) -> str | None:
        """The alert that counted this item without storing it (the alert was full)."""
        ...

    def add_overflow(self, alert_id: str, member: AlertMember) -> None:
        """Count a member without storing it (its item, verdict signature, content and time stay,
        so that it is not counted twice and its copies still find the alert)."""
        ...

    def move_overflow(self, from_id: str, to_id: str) -> None: ...

    def purge(self, *, before: datetime | None, handled: bool, now: datetime) -> int:
        """Remove the alerts whose latest occurrence is strictly before `before`, and, with
        `handled`, those that are not new any more or are withdrawn (but not while an active
        member still waits for the language model); with their members and trail. Each item is
        remembered with its verdict as forgotten so that it does not raise the alert again unless
        its verdict changes. Returns how many alerts were removed."""
        ...

    # Defined last: the name shadows the builtin inside the class body.
    def list(self, flt: AlertFilter, *, limit: int, offset: int) -> list[Alert]: ...
