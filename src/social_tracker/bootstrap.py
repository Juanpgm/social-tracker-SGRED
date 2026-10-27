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
  5.5), `SOCIAL_TRACKER_CATALOG_INTERVAL_S` (worker refresh interval, 60..86400, default 600),
  `SOCIAL_TRACKER_CATALOG_MAX_AGE_S` (a cache whose last good refresh is older is stale
  and the check says `unavailable`; from the interval up, default 7200) and
  `SOCIAL_TRACKER_CATALOG_INGEST_GRACE_S` (how long the catalogs may take to publish a quake; a
  claim newer than the last refresh minus this is never called a mismatch; 0..86400, default
  1800). A blank value means the default. The catalog fetcher is separate from the feed fetcher:
  only `earthquake.usgs.gov` and `www.seismicportal.eu`, https on port 443, no private hosts.

- LLM escalation (a model server on THIS machine only; see the README):
  `SOCIAL_TRACKER_LLM_BASE_URL` (default `http://127.0.0.1:11434`; only `http://127.0.0.1` or
  `http://localhost` with an optional port is accepted, anything else stops the start),
  `SOCIAL_TRACKER_LLM_MODEL` (default
  `st-gemma4-e4b`), `SOCIAL_TRACKER_LLM_FALLBACK_MODEL` (default `st-qwen35-4b`, or `none`),
  `SOCIAL_TRACKER_LLM_KEEP_ALIVE_S` (how long the server keeps the model loaded, 1..86400, default
  300), `SOCIAL_TRACKER_LLM_TIMEOUT_S` (deadline of one call, 1..600, default 60),
  `SOCIAL_TRACKER_LLM_HOURLY_CAP` (judgments per UTC hour, 1..100000, default 200),
  `SOCIAL_TRACKER_LLM_QUEUE_MAX` (items waiting; the oldest beyond it are dropped, default 2000),
  `SOCIAL_TRACKER_LLM_BATCH` (judged per worker pass, 1..1000, default 20),
  `SOCIAL_TRACKER_LLM_MAX_FAILURES` (failures that may be the item's own before it is parked,
  default 3; a failure of the server never counts),
  `SOCIAL_TRACKER_LLM_LOAD_TIMEOUT_S` (deadline of a call that has to load the model first,
  1..1800, default 180), `SOCIAL_TRACKER_LLM_NUM_CTX` (context window sent to the server,
  2560..262144, default 8192; the prompt is cut to fit it),
  `SOCIAL_TRACKER_LLM_PASS_BUDGET_S` (wall-clock budget of one escalation pass, 1..86400, default
  120), `SOCIAL_TRACKER_LLM_PARKED_TTL_S` (a parked item re-enters the queue after this long,
  60..2592000, default 86400) and `SOCIAL_TRACKER_LLM_SOURCE_SHARE` (the share of the queue bound
  one source may hold when the queue overflows, above 0 up to 1, at most four decimals, default
  0.25),
  `SOCIAL_TRACKER_LLM_RETRY_BASE_S` and `SOCIAL_TRACKER_LLM_RETRY_CAP_S` (backoff after a failed
  call: 60 doubling up to 3600) and `SOCIAL_TRACKER_LLM_PROMPT_CHARS` (item text longer than this
  is read at both ends, 100..100000, default 6000). A blank value means the default.

- Alerts (stored and shown on this machine only, never sent anywhere):
  `SOCIAL_TRACKER_ALERT_RETENTION_DAYS` (an alert is kept this long after its latest occurrence,
  0..3650, default 180; 0 keeps an alert only until it has been seen or dismissed),
  `SOCIAL_TRACKER_ALERT_DEDUP_HOURS` (copies seen within this many hours of each other are one
  alert, 1..720, default 24), `SOCIAL_TRACKER_ALERT_CRITICAL_WINDOW_HOURS` (how fresh an event is
  for a catalog contradiction to be critical, 1..720, default 24) and the confidences from which
  a verdict is `critical` (`SOCIAL_TRACKER_ALERT_CRITICAL_DELIBERATE`, 0.85, and
  `SOCIAL_TRACKER_ALERT_CRITICAL_CATALOG`, 0.8), `high` (`SOCIAL_TRACKER_ALERT_HIGH`, 0.7) and
  `medium` (`SOCIAL_TRACKER_ALERT_MEDIUM`, 0.5): decimals from 0 to 1 with at most four
  decimals, the critical ones above 0, medium not above high. The minimum confidence that raises
  an alert at all is not an environment variable but the audited setting `alerts threshold set`.
  A blank value means the default.

`build_container` is for the CLI and the web UI and never creates a network client. `build_worker`
adds the collector and the HTTP fetcher (the application's only outbound path) for
`social-tracker run` (its escalation reaches the model server on this machine only), and takes
the single-worker lock next to the database (see `persistence.worker_lock`): a second worker on
the same database is refused.
"""

import ipaddress
import os
import random
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from social_tracker.alerts.domain import MAX_RETENTION_DAYS as ALERT_MAX_RETENTION_DAYS
from social_tracker.alerts.domain import AlertSettings
from social_tracker.alerts.service import AlertService
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
from social_tracker.gold.service import GoldService
from social_tracker.items.domain import DEFAULT_RETENTION_DAYS, validate_retention_days
from social_tracker.items.service import ItemService
from social_tracker.llm.domain import (
    DEFAULT_FALLBACK_MODEL,
    JudgeError,
    JudgeErrorKind,
    LlmSettings,
    parse_loopback_url,
)
from social_tracker.llm.ollama import OllamaJudge
from social_tracker.llm.ports import LlmJudge
from social_tracker.llm.service import EscalationService
from social_tracker.persistence.alerts_repo import SqliteAlertRepository
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.classification_repo import (
    SqliteClassificationRepository,
    SqliteSignatureRepository,
    SqliteTopicConfigRepository,
)
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import DEFAULT_BUSY_TIMEOUT_MS, Database
from social_tracker.persistence.gold_repo import SqliteGoldRepository
from social_tracker.persistence.items_repo import SqliteItemRepository
from social_tracker.persistence.llm_repo import SqliteEscalationRepository
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
LLM_BASE_URL_ENV = "SOCIAL_TRACKER_LLM_BASE_URL"
LLM_MODEL_ENV = "SOCIAL_TRACKER_LLM_MODEL"
LLM_FALLBACK_MODEL_ENV = "SOCIAL_TRACKER_LLM_FALLBACK_MODEL"
LLM_KEEP_ALIVE_ENV = "SOCIAL_TRACKER_LLM_KEEP_ALIVE_S"
LLM_TIMEOUT_ENV = "SOCIAL_TRACKER_LLM_TIMEOUT_S"
LLM_HOURLY_CAP_ENV = "SOCIAL_TRACKER_LLM_HOURLY_CAP"
LLM_QUEUE_MAX_ENV = "SOCIAL_TRACKER_LLM_QUEUE_MAX"
LLM_BATCH_ENV = "SOCIAL_TRACKER_LLM_BATCH"
LLM_MAX_FAILURES_ENV = "SOCIAL_TRACKER_LLM_MAX_FAILURES"
LLM_RETRY_BASE_ENV = "SOCIAL_TRACKER_LLM_RETRY_BASE_S"
LLM_RETRY_CAP_ENV = "SOCIAL_TRACKER_LLM_RETRY_CAP_S"
LLM_PROMPT_CHARS_ENV = "SOCIAL_TRACKER_LLM_PROMPT_CHARS"
LLM_LOAD_TIMEOUT_ENV = "SOCIAL_TRACKER_LLM_LOAD_TIMEOUT_S"
LLM_NUM_CTX_ENV = "SOCIAL_TRACKER_LLM_NUM_CTX"
LLM_PASS_BUDGET_ENV = "SOCIAL_TRACKER_LLM_PASS_BUDGET_S"
LLM_PARKED_TTL_ENV = "SOCIAL_TRACKER_LLM_PARKED_TTL_S"
LLM_SOURCE_SHARE_ENV = "SOCIAL_TRACKER_LLM_SOURCE_SHARE"
ALERT_RETENTION_DAYS_ENV = "SOCIAL_TRACKER_ALERT_RETENTION_DAYS"
ALERT_DEDUP_WINDOW_ENV = "SOCIAL_TRACKER_ALERT_DEDUP_HOURS"
ALERT_CRITICAL_WINDOW_ENV = "SOCIAL_TRACKER_ALERT_CRITICAL_WINDOW_HOURS"
ALERT_CRITICAL_DELIBERATE_ENV = "SOCIAL_TRACKER_ALERT_CRITICAL_DELIBERATE"
ALERT_CRITICAL_CATALOG_ENV = "SOCIAL_TRACKER_ALERT_CRITICAL_CATALOG"
ALERT_HIGH_ENV = "SOCIAL_TRACKER_ALERT_HIGH"
ALERT_MEDIUM_ENV = "SOCIAL_TRACKER_ALERT_MEDIUM"
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
    escalation: EscalationService
    llm_settings: LlmSettings
    alerts: AlertService
    alert_settings: AlertSettings
    gold: GoldService
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


@dataclass(frozen=True)
class LlmContainer:
    """Everything `classify escalate` and `llm status|ping` need: the services plus the
    escalation wired to the real judge (the model server on this machine). No worker lock: the
    single in-flight slot of the database keeps a one-shot run and the worker apart."""

    container: Container
    service: EscalationService
    judge: LlmJudge
    settings: LlmSettings

    def close(self) -> None:
        close = getattr(self.judge, "close", None)
        try:
            if callable(close):
                close()
        finally:
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


def _bounded(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    """A whole number from `low` to `high` in the environment (blank or absent: `default`)."""
    text = env.get(name, "").strip()
    if not text:
        return default
    if not _LIMIT_DIGITS.fullmatch(text) or not low <= int(text) <= high:
        raise ValidationError(f"{name} must be a whole number from {low} to {high}")
    return int(text)


def _unit_setting(env: Mapping[str, str], name: str, default: float) -> float:
    """A decimal from 0 to 1 with at most four decimals (blank or absent: `default`)."""
    text = env.get(name, "").strip()
    if not text:
        return default
    if not _THRESHOLD.fullmatch(text) or not 0 <= float(text) <= 1:
        raise ValidationError(f"{name} must be a decimal from 0 to 1 (at most four decimals)")
    return float(text)


def parse_alert_settings(env: Mapping[str, str]) -> AlertSettings:
    """The alert policy's numbers from the environment (blank or absent: the defaults)."""
    defaults = AlertSettings()
    retention_text = env.get(ALERT_RETENTION_DAYS_ENV, "").strip()
    retention = defaults.retention_days
    if retention_text:
        in_range = _RETENTION_DIGITS.fullmatch(retention_text) and (
            0 <= int(retention_text) <= ALERT_MAX_RETENTION_DAYS
        )
        if not in_range:
            message = f"{ALERT_RETENTION_DAYS_ENV} must be a whole number of days (0..3650)"
            raise ValidationError(message)
        retention = int(retention_text)
    try:
        return AlertSettings(
            critical_disinformation=_unit_setting(
                env, ALERT_CRITICAL_DELIBERATE_ENV, defaults.critical_disinformation
            ),
            critical_catalog=_unit_setting(
                env, ALERT_CRITICAL_CATALOG_ENV, defaults.critical_catalog
            ),
            high=_unit_setting(env, ALERT_HIGH_ENV, defaults.high),
            medium=_unit_setting(env, ALERT_MEDIUM_ENV, defaults.medium),
            critical_window=timedelta(hours=_bounded(env, ALERT_CRITICAL_WINDOW_ENV, 24, 1, 720)),
            dedup_window=timedelta(hours=_bounded(env, ALERT_DEDUP_WINDOW_ENV, 24, 1, 720)),
            retention_days=retention,
        )
    except ValidationError as exc:
        raise ValidationError(f"alert settings: {exc.message}") from None


def _model_setting(env: Mapping[str, str], name: str, default: str | None) -> str | None:
    text = env.get(name, "").strip()
    if not text:
        return default
    if name == LLM_FALLBACK_MODEL_ENV and text.lower() == "none":
        return None
    try:
        LlmSettings(model=text)
    except ValidationError:
        raise ValidationError(
            f"{name} must be a model name (1 to 100 letters, digits or . _ : / -)"
        ) from None
    return text


def _share_setting(env: Mapping[str, str], name: str, default: float) -> float:
    """A share above 0 and up to 1, at most four decimals (blank or absent: `default`)."""
    text = env.get(name, "").strip()
    if not text:
        return default
    if not _THRESHOLD.fullmatch(text) or not 0 < float(text) <= 1:
        raise ValidationError(f"{name} must be a number above 0 up to 1 (at most four decimals)")
    return float(text)


def parse_llm_settings(env: Mapping[str, str]) -> LlmSettings:
    """The LLM escalation settings from the environment (blank or absent: the defaults). The
    address must be the loopback one: anything else stops the start."""
    defaults = LlmSettings()
    base_url = env.get(LLM_BASE_URL_ENV, "").strip() or defaults.base_url
    try:
        parse_loopback_url(base_url)
    except ValidationError:
        message = (
            f"{LLM_BASE_URL_ENV} must be http://127.0.0.1 or http://localhost, "
            "optionally with a port"
        )
        raise ValidationError(message) from None
    retry_base = _bounded(env, LLM_RETRY_BASE_ENV, defaults.retry_base_s, 1, 86_400)
    retry_cap = _bounded(env, LLM_RETRY_CAP_ENV, defaults.retry_cap_s, 1, 604_800)
    if retry_cap < retry_base:
        raise ValidationError(f"{LLM_RETRY_CAP_ENV} must not be under {LLM_RETRY_BASE_ENV}")
    return LlmSettings(
        base_url=base_url,
        model=_model_setting(env, LLM_MODEL_ENV, defaults.model) or defaults.model,
        fallback_model=_model_setting(env, LLM_FALLBACK_MODEL_ENV, DEFAULT_FALLBACK_MODEL),
        keep_alive_s=_bounded(env, LLM_KEEP_ALIVE_ENV, defaults.keep_alive_s, 1, 86_400),
        call_deadline_s=float(
            _bounded(env, LLM_TIMEOUT_ENV, int(defaults.call_deadline_s), 1, 600)
        ),
        hourly_cap=_bounded(env, LLM_HOURLY_CAP_ENV, defaults.hourly_cap, 1, 100_000),
        queue_max=_bounded(env, LLM_QUEUE_MAX_ENV, defaults.queue_max, 1, 1_000_000),
        batch_limit=_bounded(env, LLM_BATCH_ENV, defaults.batch_limit, 1, 1000),
        max_prompt_chars=_bounded(
            env, LLM_PROMPT_CHARS_ENV, defaults.max_prompt_chars, 100, 100_000
        ),
        retry_base_s=retry_base,
        retry_cap_s=retry_cap,
        max_failures=_bounded(env, LLM_MAX_FAILURES_ENV, defaults.max_failures, 1, 100),
        load_deadline_s=float(
            _bounded(env, LLM_LOAD_TIMEOUT_ENV, int(defaults.load_deadline_s), 1, 1800)
        ),
        num_ctx=_bounded(env, LLM_NUM_CTX_ENV, defaults.num_ctx, 2560, 262_144),
        pass_budget_s=float(
            _bounded(env, LLM_PASS_BUDGET_ENV, int(defaults.pass_budget_s), 1, 86_400)
        ),
        parked_ttl_s=_bounded(env, LLM_PARKED_TTL_ENV, defaults.parked_ttl_s, 60, 2_592_000),
        source_share=_share_setting(env, LLM_SOURCE_SHARE_ENV, defaults.source_share),
    )


class _UnreachableJudge:
    """The judge of a container that never talks to a model server: the escalation of the CLI and
    the web UI can read what was kept, and any attempt to ask is answered "unreachable"."""

    def judge(self, request: object) -> object:
        raise JudgeError(JudgeErrorKind.UNREACHABLE, "this process has no model server client")

    def list_models(self) -> tuple[str, ...]:
        raise JudgeError(JudgeErrorKind.UNREACHABLE, "this process has no model server client")

    def identify(self) -> None:
        raise JudgeError(JudgeErrorKind.UNREACHABLE, "this process has no model server client")


def _escalation(
    items: SqliteItemRepository,
    control: ControlService,
    classifications: SqliteClassificationRepository,
    escalations: SqliteEscalationRepository,
    uow: SqliteUnitOfWork,
    clock: SystemClock,
    judge: LlmJudge,
    settings: LlmSettings,
    stop: threading.Event | None,
) -> EscalationService:
    return EscalationService(
        items,
        control,
        classifications,
        escalations,
        judge,
        uow,
        clock,
        settings=settings,
        should_stop=stop.is_set if stop is not None else (lambda: False),
        holder=_holder(),
    )


def _holder() -> str:
    """Who holds the single in-flight slot: this process, told apart from another by a suffix."""
    return f"pid{os.getpid()}-{random.randrange(16**6):06x}"


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
    escalation_repo: SqliteEscalationRepository


def _open(env: Mapping[str, str]) -> _Wiring:
    retention_days = parse_retention_days(env.get(RETENTION_DAYS_ENV, ""))
    classifier_settings = parse_classifier_settings(env)
    catalog_settings = parse_catalog_settings(env)
    llm_settings = parse_llm_settings(env)
    alert_settings = parse_alert_settings(env)
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
    classification_repo = SqliteClassificationRepository(db)
    classifier = ClassifierService(
        items,
        sources,
        control_service,
        classification_repo,
        SqliteSignatureRepository(db),
        topics,
        uow,
        clock,
        catalog=CachedCatalogCheck(catalog_repo, clock, catalog_settings),
        settings=classifier_settings,
    )
    escalation_repo = SqliteEscalationRepository(db)
    escalation = _escalation(
        items, control_service, classification_repo, escalation_repo, uow, clock,
        _UnreachableJudge(), llm_settings, None,
    )  # fmt: skip
    alerts = AlertService(
        SqliteAlertRepository(db),
        classification_repo,
        items,
        sources,
        escalation_repo,
        control_service,
        uow,
        clock,
        settings=alert_settings,
    )
    container = Container(
        SourceRegistry(sources, control, uow, clock),
        control_service,
        ItemService(items, uow, clock, retention_days),
        classifier,
        topics,
        classifier_settings,
        catalog_cache,
        escalation,
        llm_settings,
        alerts,
        alert_settings,
        GoldService(SqliteGoldRepository(db), topics, uow, clock, settings=classifier_settings),
        db,
    )
    return _Wiring(
        db, sources, control, items, uow, clock, container, catalog_repo, catalog_settings,
        escalation_repo,
    )  # fmt: skip


def _purge_everything(container: Container) -> Callable[[], object]:
    """The worker's hourly housekeeping: the raw items past their retention, then the verdicts
    (and error counters) of items that no longer exist and are older than their own retention.
    Then the cached catalog events past theirs and, last, the alerts past theirs (they clean up
    after the verdicts the earlier steps forgot). Each step runs even when an earlier one
    fails, and a failure is raised after."""

    def purge() -> object:
        try:
            return container.items.purge_expired()
        finally:
            try:
                container.classifier.prune_history()
            finally:
                try:
                    container.catalog.purge_expired()
                finally:
                    container.alerts.purge_expired()

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


def _refresher(
    wiring: _Wiring, fetcher: Fetcher, should_stop: Callable[[], bool] | None = None
) -> CatalogRefresher:
    return CatalogRefresher(
        fetcher,
        wiring.catalog_repo,
        wiring.uow,
        wiring.clock,
        wiring.catalog_settings,
        jitter=_random_jitter,
        should_stop=should_stop,
    )


def _worker_escalation(
    wiring: _Wiring,
    container: Container,
    stop: threading.Event,
    judge: LlmJudge | None,
    *,
    replaced: bool,
) -> Callable[[], object] | None:
    """The escalation hook of the worker: the real model server by default, a given judge for
    tests, nothing when the HTTP client was replaced and no judge was given."""
    if judge is None and replaced:
        return None
    settings = container.llm_settings
    service = _escalation(
        wiring.items, container.control, SqliteClassificationRepository(wiring.db),
        wiring.escalation_repo, wiring.uow, wiring.clock,
        judge if judge is not None else OllamaJudge(settings), settings, stop,
    )  # fmt: skip
    return service.escalate


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


def build_llm(
    environ: Mapping[str, str] | None = None,
    *,
    judge: LlmJudge | None = None,
    stop: threading.Event | None = None,
) -> LlmContainer:
    """Wire the escalation for the one-shot commands. `judge` replaces the Ollama adapter (tests,
    tooling). `stop` is the flag the run checks between items."""
    wiring = _open(os.environ if environ is None else environ)
    try:
        settings = wiring.container.llm_settings
        chosen = judge if judge is not None else OllamaJudge(settings)
        service = _escalation(
            wiring.items, wiring.container.control, SqliteClassificationRepository(wiring.db),
            wiring.escalation_repo, wiring.uow, wiring.clock, chosen, settings, stop,
        )  # fmt: skip
        return LlmContainer(wiring.container, service, chosen, settings)
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
    judge: LlmJudge | None = None,
) -> WorkerContainer:
    """Wire the collector and its worker. `fetcher` replaces the HTTP client (tests, tooling);
    a replaced fetcher also leaves the catalogs unfetched and the escalation off (unless a `judge`
    is given), so no test or tool reaches them or a model server."""
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
        stop_event = stop or threading.Event()
        escalate = _worker_escalation(
            wiring, container, stop_event, judge, replaced=fetcher is not None
        )
        worker = Worker(
            collector,
            container.control,
            wiring.clock,
            stop_event,
            sleeper=sleeper,
            tick_s=tick_s,
            purge=_purge_everything(container),
            classify=container.classifier.classify_pending,
            catalog=_refresher(wiring, _catalog_fetcher(wiring.clock), stop_event.is_set).refresh
            if fetcher is None
            else None,
            escalate=escalate,
            alerts=container.alerts.refresh,
        )
    except BaseException:
        wiring.db.close()
        lock.release()
        raise
    return WorkerContainer(container, worker, chosen, lock)
