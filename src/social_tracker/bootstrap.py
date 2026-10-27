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
- `SOCIAL_TRACKER_MAX_ITEMS_PER_SOURCE`, `SOCIAL_TRACKER_MAX_BYTES_PER_SOURCE`,
  `SOCIAL_TRACKER_MAX_BYTES_PER_FETCH`: ceilings on what one feed may cost (items a source holds,
  bytes a source holds, bytes of the items of one fetch; whole numbers from 1). Defaults are
  generous (100000 items, 512 MiB, 16 MiB); entries beyond a ceiling are dropped and counted.
- `SOCIAL_TRACKER_RECYCLE_DAYS`: content first SEEN here STRICTLY more than this many days apart
  is "recycled" when it comes back (default 7, accepted range 1..3650). A blank value means the
  default.
- `SOCIAL_TRACKER_NEAR_DUPLICATE_CANDIDATES`: how many near-duplicate candidates are verified per
  item, the best-ranked first (default 500, accepted range 1..1000). A blank value means the
  default.
- `SOCIAL_TRACKER_CLASSIFICATION_RETENTION_DAYS`: how long the verdict of an item that no longer
  exists is kept (default 180, accepted range 1..3650); the hourly retention purge forgets the
  older ones. A blank value means the default.
- `SOCIAL_TRACKER_NEAR_DUPLICATE_THRESHOLD`: the estimated similarity (a decimal above 0 up to 1,
  at most four decimals) from which two texts are near duplicates, inclusive (default 0.8). A
  blank value means the default.
- `SOCIAL_TRACKER_ALLOWED_PORTS`: comma-separated destination ports the collector may contact
  (feeds and every redirect hop). Default `80,443,8080,8443`; a value replaces the list.
- Official-catalog cross-check (USGS and EMSC only): `SOCIAL_TRACKER_CATALOG_CENTER`
  (`lat,lon`, default `3.4516,-76.5320`, Cali), `SOCIAL_TRACKER_CATALOG_RADIUS_KM` (default
  300), `SOCIAL_TRACKER_CATALOG_MIN_MAGNITUDE` (regional floor, default 2.5),
  `SOCIAL_TRACKER_CATALOG_GLOBAL_MIN_MAGNITUDE` (floor of the felt-elsewhere window, default
  5.5), `SOCIAL_TRACKER_CATALOG_INTERVAL_S` (worker refresh interval, 60..86400, default 600)
  `SOCIAL_TRACKER_CATALOG_MAX_AGE_S` (a cache whose last good refresh is older is stale
  and the check says `unavailable`; from the interval up, default 7200) and
  `SOCIAL_TRACKER_CATALOG_INGEST_GRACE_S` (how long the catalogs may take to publish a quake; a
  claim newer than the last refresh minus this is never called a mismatch; 0..86400, default
  1800). A blank value means the default. The catalog fetcher is separate from the feed fetcher:
  only
  `earthquake.usgs.gov` and `www.seismicportal.eu`, https on port 443, no private hosts.

`build_container` is for the CLI and the web UI and never creates a network client. `build_worker`
adds the collector and the HTTP fetcher (the application's only outbound path) for
`social-tracker run`, and takes the single-worker lock next to the database (see
`persistence.worker_lock`): a second worker on the same database is refused.
"""

import ipaddress
import os
import random
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from social_tracker.catalog.domain import CATALOG_HOSTS, CatalogSettings
from social_tracker.catalog.service import CachedCatalogCheck, CatalogCache, CatalogRefresher
from social_tracker.classification.domain import ClassifierSettings
from social_tracker.classification.service import ClassifierService, TopicConfigService
from social_tracker.classification.yaml_io import default_topic_config
from social_tracker.collector.domain import IngestLimits
from social_tracker.collector.http_fetcher import DEFAULT_ALLOWED_PORTS, FetchLimits, HttpFetcher
from social_tracker.collector.ports import Fetcher
from social_tracker.collector.service import CollectorService
from social_tracker.collector.worker import DEFAULT_TICK_S, Worker
from social_tracker.control.service import ControlService
from social_tracker.items.domain import DEFAULT_RETENTION_DAYS, validate_retention_days
from social_tracker.items.service import ItemService
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.classification_repo import (
    SqliteClassificationRepository,
    SqliteSignatureRepository,
    SqliteTopicConfigRepository,
)
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import DEFAULT_BUSY_TIMEOUT_MS, Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.persistence.worker_lock import WorkerLock
from social_tracker.shared.clock import SystemClock
from social_tracker.shared.errors import ValidationError
from social_tracker.sources.service import SourceRegistry

DB_PATH_ENV = "SOCIAL_TRACKER_DB_PATH"
BUSY_TIMEOUT_ENV = "SOCIAL_TRACKER_BUSY_TIMEOUT_MS"
RETENTION_DAYS_ENV = "SOCIAL_TRACKER_RETENTION_DAYS"
ALLOW_PRIVATE_HOSTS_ENV = "SOCIAL_TRACKER_ALLOW_PRIVATE_HOSTS"
ALLOWED_PORTS_ENV = "SOCIAL_TRACKER_ALLOWED_PORTS"
RECYCLE_DAYS_ENV = "SOCIAL_TRACKER_RECYCLE_DAYS"
NEAR_DUPLICATE_THRESHOLD_ENV = "SOCIAL_TRACKER_NEAR_DUPLICATE_THRESHOLD"
NEAR_DUPLICATE_CANDIDATES_ENV = "SOCIAL_TRACKER_NEAR_DUPLICATE_CANDIDATES"
CLASSIFICATION_RETENTION_DAYS_ENV = "SOCIAL_TRACKER_CLASSIFICATION_RETENTION_DAYS"
MAX_ITEMS_PER_SOURCE_ENV = "SOCIAL_TRACKER_MAX_ITEMS_PER_SOURCE"
MAX_BYTES_PER_SOURCE_ENV = "SOCIAL_TRACKER_MAX_BYTES_PER_SOURCE"
MAX_BYTES_PER_FETCH_ENV = "SOCIAL_TRACKER_MAX_BYTES_PER_FETCH"
CATALOG_CENTER_ENV = "SOCIAL_TRACKER_CATALOG_CENTER"
CATALOG_RADIUS_ENV = "SOCIAL_TRACKER_CATALOG_RADIUS_KM"
CATALOG_MIN_MAGNITUDE_ENV = "SOCIAL_TRACKER_CATALOG_MIN_MAGNITUDE"
CATALOG_GLOBAL_MIN_MAGNITUDE_ENV = "SOCIAL_TRACKER_CATALOG_GLOBAL_MIN_MAGNITUDE"
CATALOG_INTERVAL_ENV = "SOCIAL_TRACKER_CATALOG_INTERVAL_S"
CATALOG_MAX_AGE_ENV = "SOCIAL_TRACKER_CATALOG_MAX_AGE_S"
CATALOG_INGEST_GRACE_ENV = "SOCIAL_TRACKER_CATALOG_INGEST_GRACE_S"
CATALOG_FETCH_LIMITS = FetchLimits(
    timeout_s=10.0, deadline_s=20.0, max_bytes=4 * 1024 * 1024, max_redirects=2
)
DEFAULT_DB_PATH = "data/social_tracker.db"
MAX_BUSY_TIMEOUT_MS = 600_000
_DIGITS = re.compile(r"[0-9]{1,7}")
_RETENTION_DIGITS = re.compile(r"[0-9]{1,4}")
_HOST_NAME = re.compile(r"[a-z0-9._-]{1,253}")
_PORT_DIGITS = re.compile(r"[0-9]{1,5}")
_LIMIT_DIGITS = re.compile(r"[0-9]{1,15}")
_DAY_DIGITS = re.compile(r"[0-9]{1,4}")
_THRESHOLD = re.compile(r"[01](?:\.[0-9]{1,4})?")
_DECIMAL = re.compile(r"-?[0-9]{1,5}(?:\.[0-9]{1,6})?")
_SECONDS = re.compile(r"[0-9]{1,9}")
_BACKOFF_JITTER_FRACTION = 0.1
CLAIM_MARGIN_S = 60.0  # what a claim may outlast the fetch deadline (parse and store) unchallenged


@dataclass(frozen=True)
class Container:
    sources: SourceRegistry
    control: ControlService
    items: ItemService
    classifier: ClassifierService
    topics: TopicConfigService
    classifier_settings: ClassifierSettings
    catalog: CatalogCache
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
    lock: WorkerLock | None = None

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
        """Release the database, then the single-worker lock. Safe to call more than once."""
        self.container.close()
        if self.lock is not None:
            self.lock.release()


@dataclass(frozen=True)
class CatalogContainer:
    """Everything `social-tracker catalog refresh` needs: the services plus the refresher and
    the catalog fetcher. No worker lock: the refresher's own claim keeps two refreshes apart."""

    container: Container
    refresher: CatalogRefresher
    fetcher: Fetcher

    @property
    def control(self) -> ControlService:
        return self.container.control

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


def parse_allowed_ports(raw: str) -> frozenset[int]:
    """Destination ports from a comma-separated list (blank: the default web ports)."""
    ports: set[int] = set()
    for entry in raw.split(","):
        text = entry.strip()
        if not text:
            continue
        if not _PORT_DIGITS.fullmatch(text) or not 1 <= int(text) <= 65535:
            raise ValidationError(f"{ALLOWED_PORTS_ENV} must list port numbers (1..65535)")
        ports.add(int(text))
    return frozenset(ports) or DEFAULT_ALLOWED_PORTS


def _limit(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    if not _LIMIT_DIGITS.fullmatch(raw) or int(raw) < 1:
        raise ValidationError(f"{name} must be a whole number from 1 (at most 15 digits)")
    return int(raw)


def parse_ingest_limits(env: Mapping[str, str]) -> IngestLimits:
    """The ceilings on what one feed may cost, from the environment (blank: the defaults)."""
    defaults = IngestLimits()
    return IngestLimits(
        _limit(env, MAX_ITEMS_PER_SOURCE_ENV, defaults.max_items_per_source),
        _limit(env, MAX_BYTES_PER_SOURCE_ENV, defaults.max_bytes_per_source),
        _limit(env, MAX_BYTES_PER_FETCH_ENV, defaults.max_bytes_per_fetch),
    )


def parse_classifier_settings(env: Mapping[str, str]) -> ClassifierSettings:
    """The tunable thresholds of the classification cascade from the environment (blank or absent:
    the defaults)."""
    defaults = ClassifierSettings()
    threshold_text = env.get(NEAR_DUPLICATE_THRESHOLD_ENV, "").strip()
    threshold = defaults.near_duplicate_threshold
    days = _whole_setting(env, RECYCLE_DAYS_ENV, defaults.recycle_days, 3650, "of days")
    kept = _whole_setting(
        env, CLASSIFICATION_RETENTION_DAYS_ENV, defaults.classification_retention_days, 3650,
        "of days",
    )  # fmt: skip
    candidates = _whole_setting(
        env, NEAR_DUPLICATE_CANDIDATES_ENV, defaults.candidate_limit, 1000, "of candidates"
    )
    if threshold_text:
        if not _THRESHOLD.fullmatch(threshold_text) or not 0 < float(threshold_text) <= 1:
            message = f"{NEAR_DUPLICATE_THRESHOLD_ENV} must be a decimal above 0 up to 1"
            raise ValidationError(message)
        threshold = float(threshold_text)
    return ClassifierSettings(
        recycle_days=days,
        near_duplicate_threshold=threshold,
        candidate_limit=candidates,
        classification_retention_days=kept,
    )


def _whole_setting(env: Mapping[str, str], name: str, default: int, high: int, what: str) -> int:
    """A whole number from 1 to `high` in the environment (blank or absent: `default`)."""
    text = env.get(name, "").strip()
    if not text:
        return default
    if not _DAY_DIGITS.fullmatch(text) or not 1 <= int(text) <= high:
        raise ValidationError(f"{name} must be a whole number {what} (1..{high})")
    return int(text)


def _decimal_setting(
    env: Mapping[str, str], name: str, default: float, low: float, high: float, *, open_low: bool
) -> float:
    text = env.get(name, "").strip()
    if not text:
        return default
    if not _DECIMAL.fullmatch(text):
        raise ValidationError(f"{name} must be a plain decimal number")
    value = float(text)
    if value > high or value < low or (open_low and value == low):
        bound = "above" if open_low else "from"
        raise ValidationError(f"{name} must be a number {bound} {low:g} up to {high:g}")
    return value


def _seconds_setting(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    text = env.get(name, "").strip()
    if not text:
        return default
    if not _SECONDS.fullmatch(text) or not low <= int(text) <= high:
        raise ValidationError(f"{name} must be a whole number of seconds ({low}..{high})")
    return int(text)


def parse_catalog_settings(env: Mapping[str, str]) -> CatalogSettings:
    """The official-catalog settings from the environment (blank or absent: the defaults)."""
    defaults = CatalogSettings()
    latitude, longitude = defaults.center_latitude, defaults.center_longitude
    center = env.get(CATALOG_CENTER_ENV, "").strip()
    if center:
        parts = [part.strip() for part in center.split(",")]
        if len(parts) != 2 or not all(_DECIMAL.fullmatch(part) for part in parts):  # noqa: PLR2004
            raise ValidationError(f"{CATALOG_CENTER_ENV} must be 'latitude,longitude' in degrees")
        latitude, longitude = float(parts[0]), float(parts[1])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:  # noqa: PLR2004
            raise ValidationError(f"{CATALOG_CENTER_ENV} is outside the valid coordinates")
    interval = _seconds_setting(env, CATALOG_INTERVAL_ENV, defaults.interval_s, 60, 86_400)
    max_age = _seconds_setting(env, CATALOG_MAX_AGE_ENV, defaults.max_age_s, interval, 7 * 86_400)
    return CatalogSettings(
        center_latitude=latitude,
        center_longitude=longitude,
        radius_km=_decimal_setting(
            env, CATALOG_RADIUS_ENV, defaults.radius_km, 0, 20_000, open_low=True
        ),
        min_magnitude=_decimal_setting(
            env, CATALOG_MIN_MAGNITUDE_ENV, defaults.min_magnitude, -2, 10, open_low=False
        ),
        global_min_magnitude=_decimal_setting(
            env,
            CATALOG_GLOBAL_MIN_MAGNITUDE_ENV,
            defaults.global_min_magnitude,
            -2,
            10,
            open_low=False,
        ),
        interval_s=interval,
        max_age_s=max_age,
        ingest_grace_s=_seconds_setting(
            env, CATALOG_INGEST_GRACE_ENV, defaults.ingest_grace_s, 0, 86_400
        ),
    )


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
    catalog_repo: SqliteCatalogRepository
    catalog_settings: CatalogSettings


def _open(env: Mapping[str, str]) -> _Wiring:
    retention_days = parse_retention_days(env.get(RETENTION_DAYS_ENV, ""))
    classifier_settings = parse_classifier_settings(env)
    catalog_settings = parse_catalog_settings(env)
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
    control_service = ControlService(control, sources, uow, clock)
    topics = TopicConfigService(
        SqliteTopicConfigRepository(db), control, uow, clock, default_topic_config()
    )
    catalog_repo = SqliteCatalogRepository(db)
    catalog_cache = CatalogCache(catalog_repo, clock, catalog_settings)
    classifier = ClassifierService(
        items,
        sources,
        control_service,
        SqliteClassificationRepository(db),
        SqliteSignatureRepository(db),
        topics,
        uow,
        clock,
        catalog=CachedCatalogCheck(catalog_repo, clock, catalog_settings),
        settings=classifier_settings,
    )
    container = Container(
        SourceRegistry(sources, control, uow, clock),
        control_service,
        ItemService(items, uow, clock, retention_days),
        classifier,
        topics,
        classifier_settings,
        catalog_cache,
        db,
    )
    return _Wiring(
        db, sources, control, items, uow, clock, container, catalog_repo, catalog_settings
    )


def _purge_everything(container: Container) -> Callable[[], object]:
    """The worker's hourly housekeeping: the raw items past their retention, then the verdicts
    (and error counters) of items that no longer exist and are older than their own retention.
    Then the cached catalog events past theirs. Each step runs even when an earlier one
    fails, and a failure is raised after."""

    def purge() -> object:
        try:
            return container.items.purge_expired()
        finally:
            try:
                container.classifier.prune_history()
            finally:
                container.catalog.purge_expired()

    return purge


def build_container(environ: Mapping[str, str] | None = None) -> Container:
    """Open the database (running pending migrations) and wire the services around it."""
    return _open(os.environ if environ is None else environ).container


def _catalog_fetcher(clock: SystemClock) -> HttpFetcher:
    """The fetcher of the catalogs: its own, confined to the two catalog hosts over https on
    port 443, with no private-host allowance and small limits."""
    return HttpFetcher(
        clock,
        limits=CATALOG_FETCH_LIMITS,
        allowed_hosts=CATALOG_HOSTS,
        https_only=True,
        allowed_ports=frozenset({443}),
    )


def _refresher(wiring: _Wiring, fetcher: Fetcher) -> CatalogRefresher:
    return CatalogRefresher(
        fetcher,
        wiring.catalog_repo,
        wiring.uow,
        wiring.clock,
        wiring.catalog_settings,
        jitter=_random_jitter,
    )


def build_catalog(
    environ: Mapping[str, str] | None = None, *, fetcher: Fetcher | None = None
) -> CatalogContainer:
    """Wire the catalog refresher for `catalog refresh`. `fetcher` replaces the catalog fetcher
    (tests, tooling)."""
    wiring = _open(os.environ if environ is None else environ)
    try:
        chosen = fetcher or _catalog_fetcher(wiring.clock)
        return CatalogContainer(wiring.container, _refresher(wiring, chosen), chosen)
    except BaseException:
        wiring.db.close()
        raise


def build_worker(
    environ: Mapping[str, str] | None = None,
    *,
    fetcher: Fetcher | None = None,
    stop: threading.Event | None = None,
    sleeper: Callable[[float], object] | None = None,
    tick_s: float = DEFAULT_TICK_S,
) -> WorkerContainer:
    """Wire the collector and its worker. `fetcher` replaces the HTTP client (tests, tooling);
    a replaced fetcher also leaves the catalogs unfetched, so no test or tool reaches them."""
    env = os.environ if environ is None else environ
    private_hosts = parse_private_hosts(env.get(ALLOW_PRIVATE_HOSTS_ENV, ""))
    allowed_ports = parse_allowed_ports(env.get(ALLOWED_PORTS_ENV, ""))
    ingest_limits = parse_ingest_limits(env)
    wiring = _open(env)
    lock = WorkerLock(wiring.db.path)
    try:
        lock.acquire()  # before anything else touches the collection state
        limits = FetchLimits()
        chosen = fetcher or HttpFetcher(
            wiring.clock, limits=limits, private_hosts=private_hosts, allowed_ports=allowed_ports
        )
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
            claim_timeout_s=limits.deadline_s + CLAIM_MARGIN_S,
            retention_days=container.items.retention_days,
            limits=ingest_limits,
        )
        worker = Worker(
            collector,
            container.control,
            wiring.clock,
            stop or threading.Event(),
            sleeper=sleeper,
            tick_s=tick_s,
            purge=_purge_everything(container),
            classify=container.classifier.classify_pending,
            catalog=_refresher(wiring, _catalog_fetcher(wiring.clock)).refresh
            if fetcher is None
            else None,
        )
    except BaseException:
        wiring.db.close()
        lock.release()
        raise
    return WorkerContainer(container, worker, chosen, lock)
