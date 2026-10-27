"""The catalog use cases (application layer): refresh the cache, read its status, purge it, and
answer the classifier's cross-check from it.

REFRESH (`CatalogRefresher`), run by the worker and by `catalog refresh`
- One catalog at a time (USGS, then EMSC). A catalog is asked twice (regional and global, see
  `queries`) and stored only if BOTH answers are good, in one transaction together with its state:
  a failed, malformed or half-answered catalog writes no event at all.
- All network I/O happens before any write transaction opens, so a slow catalog never holds the
  database: another writer (the classifier, the collector) can commit meanwhile.
- A claim row keeps two refreshers (the worker and a manual run) from fetching the same catalog at
  once: the loser reports `skipped_busy` and fetches nothing. A claim expires, so a refresher that
  died cannot block the catalog for ever.
- A failure (timeout, DNS, 4xx/5xx, `Retry-After`, an answer that is not what was asked for, an
  unexpected error) is recorded with a fixed message, never raised, and backs the catalog off
  exponentially (60 s doubling, capped at one hour; a longer `Retry-After` wins, capped at six
  hours). Without `force`, a catalog still in backoff or refreshed less than `interval_s` ago is
  skipped.
- Logs and stored errors carry fixed messages: never a URL, a header or a response text.

STATUS and PURGE (`CatalogCache`) need no network.

CHECK (`CachedCatalogCheck`) is the `CatalogCheck` port of the classifier. It reads ONLY the cache:
no network, no write. See its docstring for when the answer is `unavailable`.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from social_tracker.catalog.domain import CatalogSettings, CatalogSource, RefreshState
from social_tracker.catalog.parsers import CatalogFormatError, parse_catalog
from social_tracker.catalog.ports import CatalogRepository
from social_tracker.catalog.queries import catalog_requests
from social_tracker.classification.catalog_match import cross_check
from social_tracker.classification.claims import extract_claims
from social_tracker.classification.ports import CatalogResult
from social_tracker.collector.domain import FetchError, FetchRequest
from social_tracker.collector.ports import Fetcher
from social_tracker.control.domain import backoff_delay_s
from social_tracker.shared.clock import Clock, ensure_utc
from social_tracker.shared.errors import DomainError
from social_tracker.shared.uow import UnitOfWork

CLAIM_TTL_S = 120.0
MAX_RETRY_AFTER_S = 6 * 3600.0
UNREADABLE = "unreadable answer"
UNEXPECTED = "unexpected error"
_LOG = logging.getLogger("social_tracker.catalog")


class Outcome(StrEnum):
    REFRESHED = "refreshed"
    DRY_RUN = "dry_run"
    FAILED = "failed"
    NOT_DUE = "not_due"
    SKIPPED_BACKOFF = "skipped_backoff"
    SKIPPED_BUSY = "skipped_busy"


@dataclass(frozen=True)
class SourceResult:
    """What one refresh did to one catalog. `error` is a fixed message, never response text."""

    source: CatalogSource
    outcome: Outcome
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    discarded: int = 0
    error: str | None = None


@dataclass(frozen=True)
class RefreshReport:
    results: tuple[SourceResult, ...]

    @property
    def failed(self) -> int:
        return sum(r.outcome is Outcome.FAILED for r in self.results)


class CatalogRefresher:
    def __init__(
        self,
        fetcher: Fetcher,
        repo: CatalogRepository,
        uow: UnitOfWork,
        clock: Clock,
        settings: CatalogSettings | None = None,
        *,
        jitter: Callable[[float], float] | None = None,
        claim_ttl_s: float = CLAIM_TTL_S,
    ) -> None:
        self._fetcher = fetcher
        self._repo = repo
        self._uow = uow
        self._clock = clock
        self._settings = settings or CatalogSettings()
        self._jitter = jitter or (lambda base: 0.0)
        self._claim_ttl = timedelta(seconds=claim_ttl_s)

    def refresh(self, *, force: bool = False, dry_run: bool = False) -> RefreshReport:
        """Refresh every catalog that is due (all of them with `force`). With `dry_run` the
        catalogs are fetched and parsed and the counts reported, and nothing is written."""
        results = []
        for source in CatalogSource:
            try:
                results.append(self._refresh_one(source, force, dry_run))
            except DomainError as exc:  # the state could not be read or written: next time
                _LOG.warning("catalog %s not refreshed: %s", source.value, exc.code)
                results.append(SourceResult(source, Outcome.FAILED, error=exc.code))
        return RefreshReport(tuple(results))

    # ---- one catalog ----

    def _refresh_one(self, source: CatalogSource, force: bool, dry_run: bool) -> SourceResult:
        now = self._clock.now()
        state = self._repo.get_state(source)
        if not force and state is not None:
            skipped = self._skip(source, state, now)
            if skipped is not None:
                return skipped
        if not dry_run and not self._repo.try_claim(source, now, now + self._claim_ttl):
            return SourceResult(source, Outcome.SKIPPED_BUSY)
        try:
            events, discarded = self._fetch(source, now)
        except (FetchError, CatalogFormatError) as exc:
            return self._failed(source, state, exc, dry_run)
        except Exception:  # noqa: BLE001 - a catalog must never stop the worker
            return self._failed(source, state, None, dry_run)
        if dry_run:
            known = self._known(source, events)
            return SourceResult(
                source,
                Outcome.DRY_RUN,
                new=len(events) - known,
                unchanged=known,
                discarded=discarded,
            )
        try:
            with self._uow.transaction():
                report = self._repo.upsert_events(events, self._clock.now())
                self._repo.record_success(source, self._clock.now())
        except BaseException:
            self._release(source)
            raise
        _LOG.info(
            "catalog %s refreshed: %d new, %d updated, %d unchanged, %d discarded",
            source.value,
            report.new,
            report.updated,
            report.unchanged,
            discarded,
        )
        return SourceResult(
            source,
            Outcome.REFRESHED,
            report.new,
            report.updated,
            report.unchanged,
            discarded,
        )

    def _skip(
        self, source: CatalogSource, state: RefreshState, now: datetime
    ) -> SourceResult | None:
        if state.next_attempt_at is not None and now < state.next_attempt_at:
            return SourceResult(source, Outcome.SKIPPED_BACKOFF)
        interval = timedelta(seconds=self._settings.interval_s)
        last = state.last_success_at
        if last is not None and state.next_attempt_at is None and last <= now < last + interval:
            return SourceResult(source, Outcome.NOT_DUE)
        return None

    def _fetch(self, source: CatalogSource, now: datetime):
        """Both answers of a catalog, parsed, as one list of events (the last one of an id wins)."""
        kept = {}
        discarded = 0
        for url in catalog_requests(source, self._settings, now):
            response = self._fetcher.fetch(FetchRequest(url))
            if response.not_modified:  # nothing was asked conditionally: not a valid answer
                raise CatalogFormatError
            parsed = parse_catalog(source, response.body, self._clock.now())
            discarded += parsed.discarded
            for event in parsed.events:
                kept[event.external_id] = event
        return list(kept.values()), discarded

    def _known(self, source: CatalogSource, events) -> int:
        if not events:
            return 0
        stored = self._repo.events_between(
            min(e.origin_time for e in events), max(e.origin_time for e in events)
        )
        keys = {e.external_id for e in stored if e.source is source}
        return sum(e.external_id in keys for e in events)

    def _failed(
        self,
        source: CatalogSource,
        state: RefreshState | None,
        exc: Exception | None,
        dry_run: bool,
    ) -> SourceResult:
        if isinstance(exc, FetchError):
            message = exc.message
        elif isinstance(exc, CatalogFormatError):
            message = UNREADABLE
        else:
            message = UNEXPECTED
        _LOG.warning("catalog %s not refreshed: %s", source.value, message)
        if not dry_run:
            at = self._clock.now()
            failures = state.consecutive_failures if state is not None else 0
            delay = backoff_delay_s(failures, self._jitter(60.0 * 2 ** min(failures, 20)))
            if isinstance(exc, FetchError) and exc.retry_after_s:
                delay = max(delay, min(exc.retry_after_s, MAX_RETRY_AFTER_S))
            with self._uow.transaction():
                self._repo.record_failure(source, at, at + timedelta(seconds=delay), message)
        return SourceResult(source, Outcome.FAILED, error=message)

    def _release(self, source: CatalogSource) -> None:
        try:
            self._repo.release_claim(source)
        except DomainError:
            _LOG.warning("catalog %s claim could not be released", source.value)


# ---- status and purge ----


@dataclass(frozen=True)
class SourceStatus:
    source: CatalogSource
    events: int
    last_success_at: datetime | None
    last_attempt_at: datetime | None
    consecutive_failures: int
    next_attempt_at: datetime | None
    last_error: str | None
    stale: bool


@dataclass(frozen=True)
class CatalogStatus:
    """What `catalog status` shows. `available` says the check can answer (every catalog but at
    least one fresh); `complete` that every catalog is fresh."""

    sources: tuple[SourceStatus, ...]
    available: bool
    complete: bool
    newest_event_at: datetime | None


class CatalogCache:
    def __init__(self, repo: CatalogRepository, clock: Clock, settings: CatalogSettings) -> None:
        self._repo = repo
        self._clock = clock
        self._settings = settings

    def status(self) -> CatalogStatus:
        now = self._clock.now()
        states = {s.source: s for s in self._repo.all_states()}
        counts = self._repo.count_by_source()
        max_age = timedelta(seconds=self._settings.max_age_s)
        rows = []
        for source in CatalogSource:
            state = states.get(source) or RefreshState(source)
            fresh = state.last_success_at is not None and now - state.last_success_at <= max_age
            rows.append(
                SourceStatus(
                    source,
                    counts.get(source, 0),
                    state.last_success_at,
                    state.last_attempt_at,
                    state.consecutive_failures,
                    state.next_attempt_at,
                    state.last_error,
                    stale=not fresh,
                )
            )
        newest = self._newest()
        fresh_count = sum(not r.stale for r in rows)
        return CatalogStatus(
            tuple(rows),
            available=fresh_count > 0 and sum(counts.values()) > 0,
            complete=fresh_count == len(rows),
            newest_event_at=newest,
        )

    def _newest(self) -> datetime | None:
        now = self._clock.now()
        window = timedelta(days=self._settings.retention_days)
        events = self._repo.events_between(now - window, now + timedelta(days=1))
        return events[-1].origin_time if events else None

    def purge_expired(self) -> int:
        """Forget events older than the retention. Returns how many."""
        cutoff = self._clock.now() - timedelta(days=self._settings.retention_days)
        return self._repo.purge_before(cutoff)


# ---- the check ----


class CachedCatalogCheck:
    """The classifier's `CatalogCheck`, answering from the cache alone (no network, no write).

    `unavailable` (never blocks a classification, never raises risk): no catalog has been refreshed
    successfully within `max_age_s` (stale), the cache holds no event at all (nothing was ever
    fetched, or everything was purged), or no event lies in the window while one of the catalogs is
    stale (the absence proves nothing). `no_claim`: the text claims no magnitude, or a date or
    instant the cache cannot answer. Otherwise `consistent` or `mismatch`, see
    `classification.catalog_match`. A storage error propagates; the classifier reads it as
    `unavailable`."""

    def __init__(self, repo: CatalogRepository, clock: Clock, settings: CatalogSettings) -> None:
        self._repo = repo
        self._clock = clock
        self._settings = settings
        self._match = settings.match_settings()

    def check(self, text: str, published_at: datetime) -> CatalogResult:
        settings = self._settings
        now = self._clock.now()
        max_age = timedelta(seconds=settings.max_age_s)
        fresh = [
            s.last_success_at
            for s in self._repo.all_states()
            if s.last_success_at is not None and now - s.last_success_at <= max_age
        ]
        if not fresh or not self._repo.count_by_source():
            return CatalogResult("unavailable")
        claims = extract_claims(text, published_at, utc_offset_hours=settings.utc_offset_hours)
        if not claims.magnitudes:
            return CatalogResult("no_claim")
        try:
            reference = claims.when.at if claims.when.at is not None else ensure_utc(published_at)
            reach = self._match.time_window + timedelta(seconds=self._match.duplicate_seconds)
            events = self._repo.events_between(reference - reach, reference + reach)
        except OverflowError:  # a publication time at the edge of the calendar: nothing to compare
            return CatalogResult("no_claim")
        finding = cross_check(
            claims,
            published_at,
            [e.to_point() for e in events],
            now=now,
            settings=self._match,
            complete=len(fresh) == len(CatalogSource),
            covered_from=max(fresh) - timedelta(hours=settings.lookback_hours),
            covered_until=min(fresh),
        )
        return CatalogResult(finding.status)
