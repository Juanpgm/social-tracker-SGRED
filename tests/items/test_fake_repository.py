"""The in-memory fake satisfies the item repository contract (it is the reference)."""

from social_tracker.items.ports import ItemRepository
from tests.fakes.item_repository import InMemoryItemRepository
from tests.items.repository_contract import ItemRepositoryContract


class TestInMemoryItemRepository(ItemRepositoryContract):
    def make_repo(self, **options: int) -> ItemRepository:
        return InMemoryItemRepository(**options)
