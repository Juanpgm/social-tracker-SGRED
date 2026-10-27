"""`ItemService.get`: one stored item by its id (the labeling tool reads items with it)."""

import pytest

from social_tracker.items.service import ItemService
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import NotFoundError
from tests.fakes.item_repository import InMemoryItemRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.items.factories import NOW, make_item


def build():
    repo = InMemoryItemRepository()
    return ItemService(repo, InMemoryUnitOfWork([repo]), FixedClock(NOW)), repo


def test_get_returns_a_stored_item():
    service, repo = build()
    item = make_item("a")
    repo.add_new([item])
    assert service.get(item.id) == item


@pytest.mark.parametrize("bad", ["itm_" + "0" * 32, "x", "' OR 1=1 --", "", "A" * 80, "itm_a b"])
def test_get_refuses_an_unknown_or_malformed_id(bad):
    service, _ = build()
    with pytest.raises(NotFoundError):
        service.get(bad)


def test_get_does_not_find_an_item_after_it_is_gone():
    service, repo = build()
    item = make_item("a")
    repo.add_new([item])
    repo.purge_fetched_before(item.fetched_at.replace(year=2100))
    with pytest.raises(NotFoundError):
        service.get(item.id)
