"""SQLite `UnitOfWork`: a thin wrapper over `Database.transaction()`.

`Database.transaction()` already provides everything the port promises: the outermost block is
one `BEGIN IMMEDIATE` transaction, nested blocks on the same thread join it through SAVEPOINTs
and other threads wait for the lock. This class only hides the connection from the services.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from social_tracker.persistence.db import Database


class SqliteUnitOfWork:
    def __init__(self, db: Database) -> None:
        self._db = db

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._db.transaction():
            yield

    @contextmanager
    def snapshot(self) -> Iterator[None]:
        with self._db.snapshot():
            yield
