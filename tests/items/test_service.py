"""ItemService: retention purge boundaries, listing and search."""

from datetime import timedelta

import pytest

from social_tracker.items.domain import DEFAULT_RETENTION_DAYS, ItemStatus
from social_tracker.items.service import ItemService, PurgeReport
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import ValidationError
from tests.fakes.item_repository import InMemoryItemRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.items.factories import NOW, make_item


def build(retention_days: int = DEFAULT_RETENTION_DAYS):
    repo = InMemoryItemRepository()
    service = ItemService(repo, InMemoryUnitOfWork([repo]), FixedClock(NOW), retention_days)
    return service, repo


def aged(name: str, age: timedelta):
    return make_item(name, title=name, text=f"text of {name}", fetched_at=NOW - age)


def names(repo) -> set[str]:
    return {i.external_id for i in repo.list()}


def test_the_default_retention_is_thirty_days():
    service, repo = build()
    repo.add_new([aged("fresh", timedelta(days=1))])
    assert service.retention_days == 30


def test_items_inside_the_retention_are_kept_and_older_ones_are_purged():
    service, repo = build()
    repo.add_new(
        [
            aged("day29", timedelta(days=29)),
            aged("day29_23h", timedelta(days=29, hours=23, minutes=59)),
            aged("day31", timedelta(days=31)),
            aged("day90", timedelta(days=90)),
        ]
    )
    assert service.purge_expired() == PurgeReport(deleted=2, retention_days=30)
    assert names(repo) == {"day29", "day29_23h"}


def test_the_boundary_an_item_exactly_thirty_days_old_is_kept_one_microsecond_older_is_purged():
    service, repo = build()
    repo.add_new(
        [
            aged("exactly30", timedelta(days=30)),
            aged("just_over", timedelta(days=30, microseconds=1)),
        ]
    )
    assert service.purge_expired().deleted == 1
    assert names(repo) == {"exactly30"}


def test_a_shorter_retention_purges_sooner():
    service, repo = build(retention_days=7)
    repo.add_new([aged("day6", timedelta(days=6)), aged("day8", timedelta(days=8))])
    assert service.purge_expired() == PurgeReport(deleted=1, retention_days=7)
    assert names(repo) == {"day6"}


def test_the_clock_decides_what_is_expired():
    clock = FixedClock(NOW)
    repo = InMemoryItemRepository()
    service = ItemService(repo, InMemoryUnitOfWork([repo]), clock, 30)
    repo.add_new([aged("fresh", timedelta(days=1))])
    assert service.purge_expired().deleted == 0
    clock.advance(timedelta(days=40))
    assert service.purge_expired().deleted == 1


def test_a_zero_retention_purges_only_classified_items_whatever_their_age():
    service, repo = build(retention_days=0)
    done, pending = aged("done", timedelta(days=1)), aged("pending", timedelta(days=400))
    repo.add_new([done, pending])
    repo.set_status(done.id, ItemStatus.CLASSIFIED)
    assert service.purge_expired() == PurgeReport(deleted=1, retention_days=0)
    assert names(repo) == {"pending"}


def test_purging_an_empty_store_deletes_nothing():
    service, _ = build()
    assert service.purge_expired().deleted == 0


@pytest.mark.parametrize("days", [-1, -30, 3651, 1.5, "30", None, True])
def test_an_invalid_retention_is_refused(days):
    with pytest.raises(ValidationError):
        build(days)


def test_listing_and_search_delegate_with_limits():
    service, repo = build()
    repo.add_new([make_item("a"), make_item("b", title="Other", text="other words")])
    assert len(service.list_items()) == 2
    assert len(service.list_items(limit=1)) == 1
    assert len(service.list_items(source_id="src_none")) == 0
    assert [i.external_id for i in service.search("quake")] == ["a"]
    assert service.search("*** AND") == []
    with pytest.raises(ValidationError):
        service.list_items(limit=0)
