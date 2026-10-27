"""`SourceRegistry` contract on the in-memory fakes and on SQLite."""

import pytest

from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import FixedClock
from social_tracker.sources.service import SourceRegistry
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.sources.registry_contract import START, Env, RegistryContract


class TestInMemoryRegistry(RegistryContract):
    def make_env(self, registry_cls: type[SourceRegistry] = SourceRegistry) -> Env:
        sources, control = InMemorySourceRepository(), InMemoryControlRepository()
        uow, clock = InMemoryUnitOfWork([sources, control]), FixedClock(START)
        return Env(
            registry_cls(sources, control, uow, clock), sources, control, uow, clock, registry_cls
        )


class TestSqliteRegistry(RegistryContract):
    @pytest.fixture(autouse=True)
    def _databases(self, tmp_path):
        self._root, self._opened = tmp_path, []
        yield
        for db in self._opened:
            db.close()

    def make_env(self, registry_cls: type[SourceRegistry] = SourceRegistry) -> Env:
        db = Database(self._root / f"registry-{len(self._opened)}.db")
        self._opened.append(db)
        sources, control = SqliteSourceRepository(db), SqliteControlRepository(db)
        uow, clock = SqliteUnitOfWork(db), FixedClock(START)
        return Env(
            registry_cls(sources, control, uow, clock), sources, control, uow, clock, registry_cls
        )
