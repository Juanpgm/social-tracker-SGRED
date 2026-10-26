"""The in-memory fake must honour the shared repository contract."""

from social_tracker.sources.ports import SourceRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.sources.repository_contract import SourceRepositoryContract


class TestInMemorySourceRepository(SourceRepositoryContract):
    def make_repo(self) -> SourceRepository:
        return InMemorySourceRepository()


def test_fake_satisfies_the_port_protocol():
    assert isinstance(InMemorySourceRepository(), SourceRepository)
