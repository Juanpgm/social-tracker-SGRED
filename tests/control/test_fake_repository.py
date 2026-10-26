"""The in-memory fake must honour the shared control repository contract."""

from social_tracker.control.ports import ControlRepository
from tests.control.repository_contract import ControlRepositoryContract
from tests.fakes.control_repository import InMemoryControlRepository


class TestInMemoryControlRepository(ControlRepositoryContract):
    def make_repo(self) -> ControlRepository:
        return InMemoryControlRepository()


def test_fake_satisfies_the_port_protocol():
    assert isinstance(InMemoryControlRepository(), ControlRepository)
