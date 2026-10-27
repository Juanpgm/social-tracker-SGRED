"""Composition root: the one place that builds the storage adapters and wires them to services.

Entrypoints (the CLI today, an API later) call `build_container` and never import the
persistence package themselves; an import-linter contract enforces it.

Environment:
- `SOCIAL_TRACKER_DB_PATH`: SQLite file (default `data/social_tracker.db`, relative to the
  working directory). Missing parent directories are created. A blank value means the default.
- `SOCIAL_TRACKER_BUSY_TIMEOUT_MS`: how long a write waits for another process holding the
  database before failing (default 5000, accepted range 1..600000). A blank value means the
  default.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from social_tracker.control.service import ControlService
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import DEFAULT_BUSY_TIMEOUT_MS, Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import SystemClock
from social_tracker.shared.errors import ValidationError
from social_tracker.sources.service import SourceRegistry

DB_PATH_ENV = "SOCIAL_TRACKER_DB_PATH"
BUSY_TIMEOUT_ENV = "SOCIAL_TRACKER_BUSY_TIMEOUT_MS"
DEFAULT_DB_PATH = "data/social_tracker.db"
MAX_BUSY_TIMEOUT_MS = 600_000
_DIGITS = re.compile(r"[0-9]{1,7}")


@dataclass(frozen=True)
class Container:
    sources: SourceRegistry
    control: ControlService
    _db: Database

    def close(self) -> None:
        """Release the database. Safe to call more than once."""
        self._db.close()


def _busy_timeout_ms(raw: str) -> int:
    value = int(raw) if _DIGITS.fullmatch(raw) else 0
    if not 1 <= value <= MAX_BUSY_TIMEOUT_MS:
        message = f"{BUSY_TIMEOUT_ENV} must be a whole number of milliseconds (1..600000)"
        raise ValidationError(message)
    return value


def build_container(environ: Mapping[str, str] | None = None) -> Container:
    """Open the database (running pending migrations) and wire the services around it."""
    env = os.environ if environ is None else environ
    path = env.get(DB_PATH_ENV, "").strip() or DEFAULT_DB_PATH
    raw_timeout = env.get(BUSY_TIMEOUT_ENV, "").strip()
    timeout = _busy_timeout_ms(raw_timeout) if raw_timeout else DEFAULT_BUSY_TIMEOUT_MS
    db = Database(path, busy_timeout_ms=timeout)
    sources, control = SqliteSourceRepository(db), SqliteControlRepository(db)
    uow, clock = SqliteUnitOfWork(db), SystemClock()
    return Container(
        SourceRegistry(sources, control, uow, clock),
        ControlService(control, sources, uow, clock),
        db,
    )
