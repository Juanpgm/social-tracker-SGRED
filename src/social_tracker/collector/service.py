"""Collector service: one collection of one source, with the control state machine around it.

`collect_once(source_id)` is the only way content enters the system. Its steps and guarantees:

1. CLAIM (one transaction). Nothing is fetched unless the source exists, is an RSS source, is
   enabled AND global analysis is on (CI-11, PC-2); the per-source analysis flag gates analysis,
   not collection. A source in backoff is retried only once its retry time has come; one that is
   collecting (another run owns it), failed (the operator must reset it) or disabled is skipped.
   The claim moves the source to `collecting`, so of two concurrent calls exactly one proceeds.
2. FETCH (no transaction held): the fetcher port, then parsing and normalization. Nothing is
   written, so a failure here leaves no partial result (CI-5).
3. COMMIT (one transaction). If the source was disabled, deleted or re-enabled meanwhile, or is
   no longer `collecting`, the unit is DISCARDED: nothing is stored and no transition is made
   (the registry already moved the state). Otherwise the items of the whole fetch, the
   `collection_succeeded` transition and the fetch status commit together or not at all.
   Global analysis turned off meanwhile does not waste the unit already fetched (PC-6).
4. FAILURE (one transaction) for any error in steps 2-3: `collection_failed` with exponential
   backoff, the sanitized reason stored as the source's last error, and `Retry-After` honoured
   as a minimum wait: the exponential
   delay stops at one hour, but an explicit `Retry-After` may ask for more, up to
   `MAX_RETRY_AFTER_S` (7 days; a larger value is clamped and the clamp is logged). The fifth
   consecutive failure ends in `failed`. A refusal (401/403/429) is the same transition with its
   reason (CI-6).

The reasons stored and audited come from a fixed vocabulary (`FetchError.message`, `FeedError`
text) and go through `sanitize_error`: never a URL, a header or a response body.
Everything is attributed to the actor `system:worker`. Nothing here publishes or sends anything
(G-1): the fetcher only reads configured source URLs.
"""

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from social_tracker.collector.domain import (
    MAX_RETRY_AFTER_S,
    FetchError,
    FetchRequest,
    FetchResponse,
    is_due,
)
from social_tracker.collector.feed import FeedError, parse_feed
from social_tracker.collector.normalize import NormalizeReport, normalize_entries
from social_tracker.collector.ports import Fetcher
from social_tracker.control.domain import (
    CollectionState,
    FetchStatus,
    backoff_delay_s,
    sanitize_error,
)
from social_tracker.control.ports import ControlRepository
from social_tracker.control.service import ControlService
from social_tracker.items.ports import ItemRepository
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import (
    InvalidTransition,
    NotFoundError,
    StorageError,
    ValidationError,
)
from social_tracker.shared.ids import is_valid_slug
from social_tracker.shared.uow import UnitOfWork
from social_tracker.sources.domain import NewsRssSource
from social_tracker.sources.ports import SourceRepository

ACTOR = "system:worker"
INTERRUPTED = "interrupted collection"
_LOG = logging.getLogger("social_tracker.collector")


class CollectOutcome(StrEnum):
    COLLECTED = "collected"
    NOT_MODIFIED = "not_modified"
    FAILED = "failed"
    DISCARDED = "discarded"  # the source changed while fetching: nothing was stored
    SKIPPED_DISABLED = "skipped_disabled"
    SKIPPED_ANALYSIS_OFF = "skipped_analysis_off"
    SKIPPED_NOT_ELIGIBLE = "skipped_not_eligible"  # collecting, in backoff, failed...
    SKIPPED_UNSUPPORTED = "skipped_unsupported"


@dataclass(frozen=True)
class CollectResult:
    source_id: str
    outcome: CollectOutcome
    new: int = 0
    duplicate: int = 0
    discarded: Mapping[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class _Claim:
    source: NewsRssSource
    status: FetchStatus | None


@dataclass(frozen=True)
class _Fetched:
    response: FetchResponse
    report: NormalizeReport | None  # None for "not modified"
    at: datetime


def _no_jitter(base_s: float) -> float:
    return 0.0


def _unexpected(exc: Exception) -> str:
    return f"unexpected error ({type(exc).__name__})"


class CollectorService:
    def __init__(
        self,
        sources: SourceRepository,
        control: ControlService,
        control_repo: ControlRepository,
        items: ItemRepository,
        fetcher: Fetcher,
        uow: UnitOfWork,
        clock: Clock,
        *,
        jitter: Callable[[float], float] = _no_jitter,
    ) -> None:
        self._sources = sources
        self._control = control
        self._repo = control_repo
        self._items = items
        self._fetcher = fetcher
        self._uow = uow
        self._clock = clock
        self._jitter = jitter

    # ---- one collection ----

    def collect_once(self, source_id: str) -> CollectResult:
        if not is_valid_slug(source_id):
            raise ValidationError("source id is not a valid slug", ("source_id",))
        claim = self._claim(source_id)
        if isinstance(claim, CollectResult):
            return claim
        try:
            fetched = self._fetch(source_id, claim)
        except FetchError as exc:
            return self._fail(source_id, exc.message, exc.retry_after_s)
        except FeedError as exc:
            return self._fail(source_id, f"invalid feed: {exc.message}", None)
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop the others
            _LOG.error("unexpected error fetching %s: %s", source_id, type(exc).__name__)
            return self._fail(source_id, _unexpected(exc), None)
        try:
            return self._commit(source_id, fetched)
        except Exception as exc:  # noqa: BLE001 - the unit is rolled back; record the failure
            message = "storage error" if isinstance(exc, StorageError) else _unexpected(exc)
            _LOG.error("could not store the collection of %s: %s", source_id, type(exc).__name__)
            return self._fail(source_id, message, None)

    def _skipped(self, source_id: str, outcome: CollectOutcome) -> CollectResult:
        return CollectResult(source_id, outcome)

    def _claim(self, source_id: str) -> _Claim | CollectResult:
        with self._uow.transaction():
            source = self._sources.get(source_id)
            if not isinstance(source, NewsRssSource):
                return self._skipped(source_id, CollectOutcome.SKIPPED_UNSUPPORTED)
            if not source.enabled:
                return self._skipped(source_id, CollectOutcome.SKIPPED_DISABLED)
            if not self._control.global_analysis_enabled():
                return self._skipped(source_id, CollectOutcome.SKIPPED_ANALYSIS_OFF)
            current = self._control.get_state(source_id)
            if current.state is CollectionState.BACKOFF:
                if current.next_run_at is None or self._clock.now() < current.next_run_at:
                    return self._skipped(source_id, CollectOutcome.SKIPPED_NOT_ELIGIBLE)
                self._control.retry_collection(source_id, actor=ACTOR, reason="retry time reached")
            try:  # a collecting, failed or disabled source refuses to start
                self._control.start_collection(source_id, actor=ACTOR, reason="collection started")
            except InvalidTransition:
                return self._skipped(source_id, CollectOutcome.SKIPPED_NOT_ELIGIBLE)
            return _Claim(source, self._repo.get_fetch_status(source_id))

    def _fetch(self, source_id: str, claim: _Claim) -> _Fetched:
        status = claim.status
        request = FetchRequest(
            claim.source.feed_url,
            etag=status.etag if status else None,
            last_modified=status.last_modified if status else None,
        )
        response = self._fetcher.fetch(request)
        at = self._clock.now()
        if response.not_modified:
            return _Fetched(response, None, at)
        parsed = parse_feed(response.body)
        report = normalize_entries(
            parsed.entries,
            source_id=source_id,
            source_name=claim.source.name,
            fetched_at=at,
            over_limit=parsed.over_limit,
        )
        return _Fetched(response, report, at)

    def _still_ours(self, source_id: str) -> bool:
        """True while the unit we claimed is still the source's current collection."""
        try:
            source = self._sources.get(source_id)
            state = self._control.get_state(source_id)
        except NotFoundError:
            return False
        return source.enabled and state.state is CollectionState.COLLECTING

    def _commit(self, source_id: str, fetched: _Fetched) -> CollectResult:
        with self._uow.transaction():
            if not self._still_ours(source_id):
                return self._skipped(source_id, CollectOutcome.DISCARDED)
            previous = self._repo.get_fetch_status(source_id)
            if fetched.report is None:
                new = duplicate = 0
                discarded: Mapping[str, int] = {}
            else:
                added = self._items.add_new(fetched.report.items)
                new, duplicate, discarded = added.new, added.duplicate, fetched.report.discarded
            etag, modified = fetched.response.etag, fetched.response.last_modified
            if fetched.report is None and previous is not None:
                # a "not modified" answer keeps the validators it did not resend
                etag, modified = etag or previous.etag, modified or previous.last_modified
            self._control.finish_collection(
                source_id, actor=ACTOR, reason=f"collected {new} new, {duplicate} duplicate"
            )
            self._repo.save_fetch_status(FetchStatus(source_id, fetched.at, None, etag, modified))
        outcome = (
            CollectOutcome.NOT_MODIFIED if fetched.report is None else CollectOutcome.COLLECTED
        )
        return CollectResult(source_id, outcome, new, duplicate, discarded)

    def _safe_jitter(self, base_s: float) -> float:
        jitter = self._jitter(base_s)
        return jitter if math.isfinite(jitter) and jitter > 0 else 0.0

    def _fail(self, source_id: str, message: str, retry_after_s: float | None) -> CollectResult:
        text = sanitize_error(message)
        with self._uow.transaction():
            if not self._still_ours(source_id):
                return self._skipped(source_id, CollectOutcome.DISCARDED)
            current = self._control.get_state(source_id)
            base = backoff_delay_s(current.consecutive_failures)
            self._control.fail_collection(
                source_id,
                actor=ACTOR,
                reason=text,
                jitter_s=self._safe_jitter(base),
                retry_after_s=retry_after_s,
            )
            if retry_after_s is not None and retry_after_s >= MAX_RETRY_AFTER_S:
                _LOG.info(
                    "source=%s: Retry-After clamped to the maximum of %d s",
                    source_id,
                    MAX_RETRY_AFTER_S,
                )
            previous = self._repo.get_fetch_status(source_id)
            self._repo.save_fetch_status(
                FetchStatus(
                    source_id,
                    previous.last_fetch_at if previous else None,
                    text,
                    previous.etag if previous else None,
                    previous.last_modified if previous else None,
                )
            )
        return CollectResult(source_id, CollectOutcome.FAILED, error=text)

    # ---- scheduling support ----

    def due_sources(self) -> list[str]:
        """Ids of the sources that should be collected now, sorted (see `is_due`)."""
        with self._uow.snapshot():
            sources = self._sources.list()
            states = {s.source_id: s for s in self._repo.list_source_states()}
            fetches = {f.source_id: f for f in self._repo.list_fetch_statuses()}
        now = self._clock.now()
        return [s.id for s in sources if is_due(s, states.get(s.id), fetches.get(s.id), now)]

    def recover_interrupted(self) -> list[str]:
        """Fail every source left in `collecting` by a run that died (a crash, a kill).

        A worker runs one collection at a time, so at the start of a pass any `collecting` source
        is a leftover: it is moved on like any failed attempt (counted, with backoff), instead of
        staying stuck forever. Run ONE worker per database: a second worker would see the first
        one's in-flight collection as interrupted.
        """
        recovered: list[str] = []
        for state in self._repo.list_source_states():
            if state.state is not CollectionState.COLLECTING:
                continue
            with self._uow.transaction():
                try:
                    self._control.fail_collection(
                        state.source_id, actor=ACTOR, reason="interrupted collection recovered"
                    )
                except (InvalidTransition, NotFoundError):
                    continue  # disabled or deleted meanwhile: nothing left to recover
                previous = self._repo.get_fetch_status(state.source_id)
                self._repo.save_fetch_status(
                    FetchStatus(
                        state.source_id,
                        previous.last_fetch_at if previous else None,
                        INTERRUPTED,
                        previous.etag if previous else None,
                        previous.last_modified if previous else None,
                    )
                )
                recovered.append(state.source_id)
        return recovered
