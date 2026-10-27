"""The `UnitOfWork` contract runs against the in-memory fake and the SQLite implementation."""

import pytest

from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.shared.uow_contract import Env, UnitOfWorkContract


class TestInMemoryUnitOfWork(UnitOfWorkContract):
    def make_env(self) -> Env:
        sources, control = InMemorySourceRepository(), InMemoryControlRepository()
        return Env(InMemoryUnitOfWork([sources, control]), sources, control)


class TestSqliteUnitOfWork(UnitOfWorkContract):
    @pytest.fixture
    def env(self, tmp_path):
        db = Database(tmp_path / "uow.db")
        yield Env(SqliteUnitOfWork(db), SqliteSourceRepository(db), SqliteControlRepository(db))
        db.close()
