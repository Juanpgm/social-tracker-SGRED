"""The in-memory fake must honour the shared repository contract."""

from typing import get_type_hints

from social_tracker.sources.domain import Source
from social_tracker.sources.ports import SourceRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.sources.repository_contract import SourceRepositoryContract


class TestInMemorySourceRepository(SourceRepositoryContract):
    def make_repo(self) -> SourceRepository:
        return InMemorySourceRepository()


def test_fake_satisfies_the_port_protocol():
    assert isinstance(InMemorySourceRepository(), SourceRepository)


def test_port_signatures_use_the_source_union_not_the_base_class():
    assert get_type_hints(SourceRepository.add)["source"] == Source
    assert get_type_hints(SourceRepository.update)["source"] == Source
    assert get_type_hints(SourceRepository.get)["return"] == Source
