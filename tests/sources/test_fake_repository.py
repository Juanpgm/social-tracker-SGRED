"""The in-memory fake must honour the shared repository contract."""

from typing import get_type_hints

import pytest

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


class _AliasingRepository(InMemorySourceRepository):
    """Wrong on purpose: hands out and stores the very same objects it was given."""

    def add(self, source):
        super().add(source)
        self._by_id[source.id] = source

    def get(self, source_id):
        return self._by_id[source_id]


def test_contract_catches_a_repository_that_aliases_its_data():
    class Contract(SourceRepositoryContract):
        def make_repo(self):
            return _AliasingRepository()

    contract = Contract()
    with pytest.raises(AssertionError):
        contract.test_tampering_with_a_returned_source_cannot_change_the_stored_copy(
            contract.make_repo()
        )
    with pytest.raises(AssertionError):
        contract.test_tampering_with_the_added_source_cannot_change_the_stored_copy(
            contract.make_repo()
        )
