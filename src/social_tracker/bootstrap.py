"""Composition root: the one place that builds the storage adapters and wires them to services.

Entrypoints (the CLI today, an API later) call `build_container` and never import the
persistence package themselves; an import-linter contract enforces it.

Environment:
- `SOCIAL_TRACKER_DB_PATH`: SQLite file (default `data/social_tracker.db`, relative to the
  working directory). A leading `~` is expanded and the result is made absolute once, here; the
  value is operator-owned, not untrusted input. Missing parent directories are created (mode
  0700, see `persistence.db`). A blank value means the default.
- `SOCIAL_TRACKER_BUSY_TIMEOUT_MS`: how long a write waits for another process holding the
  database before failing (default 5000, accepted range 1..600000). A blank value means the
  default.
- `SOCIAL_TRACKER_RETENTION_DAYS`: how long raw items are kept, counted from when they were
  fetched (default 30, accepted range 0..3650; 0 means "purge once classified"). A blank value
  means the default. An invalid value stops the start before the database is opened.
- `SOCIAL_TRACKER_ALLOW_PRIVATE_HOSTS`: comma-separated host names (or IP addresses) the collector
  may contact although they resolve to private addresses. Empty by default, which blocks every
  loopback, private, link-local and metadata address (SSRF protection); every use is logged.

`build_container` is for the CLI and the web UI and never creates a network client. `build_worker`
adds the collector and the HTTP fetcher (the application's only outbound path) for
`social-tracker run`.
"""

import ipaddress
import os
import random
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from social_tracker.collector.http_fetcher import HttpFetcher
from social_tracker.collector.ports import Fetcher
from social_tracker.collector.service import CollectorService
from social_tracker.collector.worker import DEFAULT_TICK_S, Worker
from social_tracker.control.service import ControlService
from social_tracker.items.domain import DEFAULT_RETENTION_DAYS, validate_retention_days
from social_tracker.items.service import ItemService
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import DEFAULT_BUSY_TIMEOUT_MS, Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import SystemClock
from social_tracker.shared.errors import ValidationError
from social_tracker.sources.service import SourceRegistry

DB_PATH_ENV = "SOCIAL_TRACKER_DB_PATH"
BUSY_TIMEOUT_ENV = "SOCIAL_TRACKER_BUSY_TIMEOUT_MS"
RETENTION_DAYS_ENV = "SOCIAL_TRACKER_RETENTION_DAYS"
ALLOW_PRIVATE_HOSTS_ENV = "SOCIAL_TRACKER_ALLOW_PRIVATE_HOSTS"
DEFAULT_DB_PATH = "data/social_tracker.db"
MAX_BUSY_TIMEOUT_MS = 600_000
_DIGITS = re.compile(r"[0-9]{1,7}")
_RETENTION_DIGITS = re.compile(r"[0-9]{1,4}")
_HOST_NAME = re.compile(r"[a-z0-9._-]{1,253}")
_BACKOFF_JITTER_FRACTION = 0.1


@dataclass(frozen=True)
class Container:
    sources: SourceRegistry
    control: ControlService
    items: ItemService
    _db: Database

    @property
    def db_path(self) -> Path:
        """The absolute path of the open database file."""
        return self._db.path

    def close(self) -> None:
        """Release the database. Safe to call more than once."""
        self._db.close()


@dataclass(frozen=True)
class WorkerContainer:
    """Everything `social-tracker run` needs: the services plus the worker around the collector."""

    container: Container
    worker: Worker
    fetcher: Fetcher

    @property
    def sources(self) -> SourceRegistry:
        return self.container.sources

    @property
    def control(self) -> ControlService:
        return self.container.control

    @property
    def items(self) -> ItemService:
        return self.container.items

    def close(self) -> None:
        self.container.close()


def resolve_db_path(raw: str) -> Path:
    """The database file as an absolute path, worked out once: a leading `~` is the home
    directory and a relative path (the default included) is taken from the current working
    directory, so a later `chdir` cannot move the database. Symbolic links are not resolved."""
    chosen = raw.strip() or DEFAULT_DB_PATH
    return Path(os.path.abspath(os.path.expanduser(chosen)))


def _busy_timeout_ms(raw: str) -> int:
    value = int(raw) if _DIGITS.fullmatch(raw) else 0
    if not 1 <= value <= MAX_BUSY_TIMEOUT_MS:
        message = f"{BUSY_TIMEOUT_ENV} must be a whole number of milliseconds (1..600000)"
        raise ValidationError(message)
    return value


def parse_retention_days(raw: str) -> int:
    """Days to keep raw items from `SOCIAL_TRACKER_RETENTION_DAYS` (blank: the default, 30)."""
    text = raw.strip()
    if not text:
        return DEFAULT_RETENTION_DAYS
    message = f"{RETENTION_DAYS_ENV} must be a whole number of days (0..3650)"
    if not _RETENTION_DIGITS.fullmatch(text):
        raise ValidationError(message)
    try:
        return validate_retention_days(int(text))
    except ValidationError:
        raise ValidationError(message) from None


def parse_private_hosts(raw: str) -> frozenset[str]:
    """Host names allowed to resolve to private addresses, from a comma-separated list."""
    hosts: set[str] = set()
    for entry in raw.split(","):
        host = entry.strip().lower()
        if not host:
            continue
        if not _HOST_NAME.fullmatch(host):
            try:
                ipaddress.ip_address(host)
            except ValueError:
                message = f"{ALLOW_PRIVATE_HOSTS_ENV} must list host names or IP addresses"
                raise ValidationError(message) from None
        hosts.add(host)
    return frozenset(hosts)


def _random_jitter(base_s: float) -> float:
    return random.uniform(0.0, _BACKOFF_JITTER_FRACTION * base_s)


@dataclass(frozen=True)
class _Wiring:
    db: Database
    sources: SqliteSourceRepository
    control: SqliteControlRepository
    items: SqliteItemRepository
    uow: SqliteUnitOfWork
    clock: SystemClock
    container: Container


def _open(env: Mapping[str, str]) -> _Wiring:
    retention_days = parse_retention_days(env.get(RETENTION_DAYS_ENV, ""))
    path = resolve_db_path(env.get(DB_PATH_ENV, ""))
    raw_timeout = env.get(BUSY_TIMEOUT_ENV, "").strip()
    timeout = _busy_timeout_ms(raw_timeout) if raw_timeout else DEFAULT_BUSY_TIMEOUT_MS
    db = Database(path, busy_timeout_ms=timeout)
    sources, control, items = (
        SqliteSourceRepository(db),
        SqliteControlRepository(db),
        SqliteItemRepository(db),
    )
    uow, clock = SqliteUnitOfWork(db), SystemClock()
    container = Container(
        SourceRegistry(sources, control, uow, clock),
        ControlService(control, sources, uow, clock),
        ItemService(items, uow, clock, retention_days),
        db,
    )
    return _Wiring(db, sources, control, items, uow, clock, container)


def build_container(environ: Mapping[str, str] | None = None) -> Container:
    """Open the database (running pending migrations) and wire the services around it."""
    return _open(os.environ if environ is None else environ).container


def build_worker(
    environ: Mapping[str, str] | None = None,
    *,
    fetcher: Fetcher | None = None,
    stop: threading.Event | None = None,
    sleeper: Callable[[float], object] | None = None,
    tick_s: float = DEFAULT_TICK_S,
) -> WorkerContainer:
    """Wire the collector and its worker. `fetcher` replaces the HTTP client (tests, tooling)."""
    env = os.environ if environ is None else environ
    private_hosts = parse_private_hosts(env.get(ALLOW_PRIVATE_HOSTS_ENV, ""))
    wiring = _open(env)
    try:
        chosen = fetcher or HttpFetcher(wiring.clock, private_hosts=private_hosts)
        container = wiring.container
        collector = CollectorService(
            wiring.sources,
            container.control,
            wiring.control,
            wiring.items,
            chosen,
            wiring.uow,
            wiring.clock,
            jitter=_random_jitter,
        )
        worker = Worker(
            collector,
            container.control,
            wiring.clock,
            stop or threading.Event(),
            sleeper=sleeper,
            tick_s=tick_s,
            purge=container.items.purge_expired,
        )
    except BaseException:
        wiring.db.close()
        raise
    return WorkerContainer(container, worker, chosen)
