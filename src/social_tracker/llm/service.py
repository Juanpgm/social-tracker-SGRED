"""The LLM escalation (RC-7, RC-9, RC-12): the last stage of the cascade (application layer).

`EscalationService.escalate` takes the items the CPU stages could not settle (the ones the
classifier marked `needs_llm`), best first (priority, then oldest), and asks the judge about each
one: the item text as untrusted data plus the structured evidence found so far. It runs after the
cascade in the worker pass and only while global analysis is on.

Rules of the road
- Only the work of the CPU stages is ever replaced, and only by a VALID verdict: one of four
  labels with a confidence from 0 to 1 and a short rationale (see `schema`). An invalid answer is
  asked again ONCE, with a corrective message that names the class of the problem and another
  seed; if that is invalid too the verdict of the earlier stage stays (RC-9), but only when the
  judge has been answering OTHER items: a judge that answers nothing valid for anyone is sick,
  not the item. The credibility of the source stays a prior: it is recorded, never applied to a
  judged confidence.
- A valid answer that is suspicious is not taken at face value (`llm.domain.judged`): the model's
  confidence is at most 0.95, and a `no_risk` that clears a text the earlier stages found
  obfuscated, a risk signal of at least 0.6, or a claim the catalog contradicts is capped at 0.7,
  flagged `low_confidence` and noted for review. Each such case is counted (`contradicted`, logged
  as `llm_contradicts_prior`).
- The judge is a different failure domain from the CPU. A failure is either the JUDGE's (nothing
  listening, a refused or rate-limited request, a 5xx, a model that is slow to load, a server that
  is not Ollama) or possibly the ITEM's (a warm call that never finishes, a body that is not an
  answer). A judge-level failure never counts against the item: the reservation of the hour is
  given back when the model gave nothing, the item goes behind the others with a backoff, the
  whole judge backs off (doubling, capped) and the batch ends. An item-level failure counts towards
  parking the item (after `max_failures`) ONLY while other items are being judged successfully;
  otherwise it is treated as the judge's. A parked item is not lost: it re-enters the queue after
  `parked_ttl_s`, or at once with `requeue_parked`.
- One judgment at a time: a lock in the process and a lease in the database keep two workers (or
  a worker and a one-shot command) from loading the GPU together. The lease outlives the longest
  possible judgment (a cold load, then the retry) and expires, so a crashed holder frees it.
- No network call inside a write transaction. Everything is read first, the judge is called with
  no transaction open, and the answer is written in one short transaction that first checks that
  the item still exists, is still classified, its source is still eligible (PC-2) and its verdict
  still waits: a source switched off, an item purged or a verdict settled meanwhile discard the
  answer instead of writing it. A repeated or racing run never judges an item twice.
- Budget: at most `hourly_cap` judgments are started per UTC hour (an invalid answer's one retry
  belongs to the same judgment). A server that is down does not burn it. A pass also stops after
  `pass_budget_s` of wall-clock time, and checks the stop flag between items and between the two
  calls of a judgment.
- The queue is bounded. When more than `queue_max` items could be served, one source's excess over
  its fair share is dropped first, then the LOWEST priority first (the oldest among equals); they
  keep their CPU verdict, are counted and settle as `parked`.
- Exact copies that wait for their original (`awaiting_original`) are settled when the original
  leaves the queue: a copy whose text is IDENTICAL to the original's takes its judged verdict (its
  own credibility prior stays on its record); a copy that differs by a character (punctuation, a
  look-alike letter, another address) is judged for itself; if there is no verdict to copy, the
  copy keeps its provisional one with the same low-confidence flag. A copy whose original is gone,
  settled or ineligible never waits for ever: it is released into the queue as a normal item.
- Item text, rationales and model output are never logged: logs carry counts, ids and codes.
"""

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from social_tracker.classification.domain import Classification, Label, Stage
from social_tracker.classification.ports import ClassificationRepository
from social_tracker.control.service import ControlService
from social_tracker.items.domain import Item, ItemStatus, text_identity
from social_tracker.items.ports import ItemRepository
from social_tracker.llm.domain import (
    PROMPT_VERSION,
    EscalationCounts,
    EscalationState,
    JudgeError,
    JudgeErrorKind,
    JudgeEvidence,
    JudgeRequest,
    JudgeResponse,
    Judgment,
    LlmSettings,
    QueueEntry,
    backoff_delay,
    contradicts_prior,
    evidence_of,
    hour_start,
    judged,
    left_unjudged,
    prompt_text,
    released_duplicate,
    requeued_copy,
    reused_by_duplicate,
)
from social_tracker.llm.ports import EscalationRepository, LlmJudge
from social_tracker.llm.prompt import fit_request
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import DomainError, NotFoundError, StorageError, ValidationError
from social_tracker.shared.uow import UnitOfWork

_LOG = logging.getLogger("social_tracker.llm")

MAX_BATCH = 1000
BREAKER = 2  # failed items in a row that end a batch
LEASE_MARGIN_S = 30
GATE_POLL_S = 0.2
DROP_PAGE = 1000
MAX_DROP_PAGES = 100
SWEEP_PAGE = 100  # parked items re-queued, and orphaned copies released, per pass
REQUEUE_PRIORITY = 0.1  # a parked item that comes back queues behind the healthy ones
IDENTITY_TTL_S = 300  # how long "this address is Ollama" is trusted
SUCCESS_WINDOW = timedelta(hours=1)  # "the judge has been answering" looks this far back
UNEXPECTED = "llm_unexpected"
BAD_REQUEST = "llm_bad_request"
BUSY = "busy"
PING_ITEM_ID = "ping_check"
PING_TEXT = "Prueba de conexion: este mensaje no trata ningun tema."

_CONTINUE = False
_STOP = True
# Failures that say nothing about which item was asked: no server, no model, not Ollama.
_NOT_THERE = frozenset(
    {JudgeErrorKind.UNREACHABLE, JudgeErrorKind.MODEL_NOT_FOUND, JudgeErrorKind.NOT_OLLAMA}
)


def _never() -> bool:
    return False


class _Interrupted(Exception):  # noqa: N818 - control flow inside this module, never leaves it
    """A stop request or the pass budget came between the two calls of a judgment."""

    def __init__(self, *, stopped: bool) -> None:
        super().__init__()
        self.stopped = stopped


@dataclass(frozen=True)
class EscalationReport:
    """What one `escalate` run did. `judged` got a valid verdict; `invalid` got two invalid
    answers (RC-9); `reused` are copies settled from an original (or a recycled copy of a judged
    item) without a call; `skipped` stopped being eligible between listing and writing; `failed`
    will be retried; `parked` gave up; `dropped` left a full queue. `requeued` are parked items
    that came back, `released` copies that stopped waiting for an original that is gone,
    `contradicted` valid `no_risk` answers that clear a suspicious text (capped, flagged).
    `candidates` and `pending_drops` are what a dry run would have sent and dropped. `error` is
    the code of the judge-level failure that ended the run, `backing_off` says the judge is
    resting after one, `out_of_time` that the pass budget ran out."""

    judged: int = 0
    invalid: int = 0
    reused: int = 0
    skipped: int = 0
    failed: int = 0
    parked: int = 0
    dropped: int = 0
    candidates: int = 0
    cap_reached: bool = False
    busy: bool = False
    stopped: bool = False
    analysis_off: bool = False
    unreachable: bool = False
    storage_errors: int = 0
    eligible_sources: int = 0
    requeued: int = 0
    released: int = 0
    contradicted: int = 0
    pending_drops: int = 0
    backing_off: bool = False
    out_of_time: bool = False
    error: str | None = None

    @property
    def settled(self) -> int:
        """Items that left the queue with a verdict of this stage."""
        return self.judged + self.invalid + self.reused

    @property
    def nothing_could_be_judged(self) -> bool:
        """The run judged nothing because the judge was unreachable or resting, busy with another
        process, or the run was stopped: worth a different exit status than a quiet queue."""
        return self.settled == 0 and (
            self.unreachable or self.backing_off or self.busy or self.stopped
        )


@dataclass
class _Tally:
    judged: int = 0
    invalid: int = 0
    reused: int = 0
    skipped: int = 0
    failed: int = 0
    parked: int = 0
    dropped: int = 0
    cap_reached: bool = False
    busy: bool = False
    stopped: bool = False
    unreachable: bool = False
    storage_errors: int = 0
    eligible_sources: int = 0
    requeued: int = 0
    released: int = 0
    contradicted: int = 0
    backing_off: bool = False
    out_of_time: bool = False
    error: str | None = None
    streak: int = field(default=0)
    deadline: datetime | None = field(default=None)

    def report(self) -> EscalationReport:
        data = {k: v for k, v in vars(self).items() if k not in {"streak", "deadline"}}
        return EscalationReport(**data)


@dataclass(frozen=True)
class LlmStatus:
    """What `llm status` shows. `installed` are the models the server has, `missing` the
    configured ones it lacks (empty when the server cannot be asked). `error` is the code of what
    went wrong asking: nothing listening, or something that is not Ollama (then nothing more is
    asked of it)."""

    reachable: bool
    error: str | None
    base_url: str
    configured: tuple[str, ...]
    installed: tuple[str, ...]
    missing: tuple[str, ...]
    queued: int
    used_this_hour: int
    hourly_cap: int
    counts: EscalationCounts
    analysis_on: bool


@dataclass(frozen=True)
class LocalLlmStatus:
    """What can be said about the model from this machine's own records, without asking the
    server or the model anything: the web page shows it. The server's reachability and its
    installed models are not here; `llm status` asks for them."""

    base_url: str
    configured: tuple[str, ...]
    queued: int
    used_this_hour: int
    hourly_cap: int
    counts: EscalationCounts
    analysis_on: bool


@dataclass(frozen=True)
class PingResult:
    ok: bool
    model: str | None = None
    label: Label | None = None
    elapsed_s: float = 0.0
    error: str | None = None


def _has_model(installed: tuple[str, ...], name: str) -> bool:
    wanted = {name, f"{name}:latest"} if ":" not in name else {name}
    return any(entry in wanted for entry in installed)


class EscalationService:
    def __init__(
        self,
        items: ItemRepository,
        control: ControlService,
        classifications: ClassificationRepository,
        escalations: EscalationRepository,
        judge: LlmJudge,
        uow: UnitOfWork,
        clock: Clock,
        *,
        settings: LlmSettings | None = None,
        should_stop: Callable[[], bool] = _never,
        holder: str = "escalation",
    ) -> None:
        self._items = items
        self._control = control
        self._classifications = classifications
        self._escalations = escalations
        self._judge = judge
        self._uow = uow
        self._clock = clock
        self._settings = settings if settings is not None else LlmSettings()
        self._should_stop = should_stop
        self._holder = holder
        self._gate = threading.Lock()
        self._judge_failures = 0  # judge-level failures in a row (sets how long it rests)
        self._rests_until: datetime | None = None
        self._identified_until: datetime | None = None

    # ---- reading ----

    def judgment(self, item_id: str) -> Judgment | None:
        """What was kept of the judgment of an item, if it went through the judge."""
        return self._escalations.get_judgment(item_id)

    def counts(self) -> EscalationCounts:
        return self._escalations.counts()

    def state(self, item_id: str) -> EscalationState | None:
        """The bookkeeping of an item that has no judgment (yet): a failure waiting for its retry,
        a parked or a dropped item."""
        return self._escalations.get_state(item_id)

    def local_status(self) -> LocalLlmStatus:
        """The queue, the hourly budget, the counts and the settings: stored figures only. Nothing
        is sent to the model server and nothing is loaded (the web page uses this)."""
        settings = self._settings
        counts = self._escalations.counts()
        return LocalLlmStatus(
            base_url=settings.endpoint.url,
            configured=settings.models,
            queued=counts.queued,
            used_this_hour=self._escalations.calls_in_hour(hour_start(self._clock.now())),
            hourly_cap=settings.hourly_cap,
            counts=counts,
            analysis_on=self._control.global_analysis_enabled(),
        )

    def status(self) -> LlmStatus:
        """Ask the server who it is and which models it has (two `GET`s, no model is loaded) and
        put it next to the local figures. A server that is down, or is not Ollama, is a status,
        not an error."""
        settings = self._settings
        used = self._escalations.calls_in_hour(hour_start(self._clock.now()))
        counts = self._escalations.counts()
        installed: tuple[str, ...] = ()
        error: str | None = None
        try:
            self._judge.identify()
            installed = self._judge.list_models()
        except JudgeError as exc:
            error = exc.code
        missing = () if error else tuple(m for m in settings.models if not _has_model(installed, m))
        return LlmStatus(
            reachable=error is None,
            error=error,
            base_url=settings.endpoint.url,
            configured=settings.models,
            installed=installed,
            missing=missing,
            queued=counts.queued,
            used_this_hour=used,
            hourly_cap=settings.hourly_cap,
            counts=counts,
            analysis_on=self._control.global_analysis_enabled(),
        )

    def judge_text(self, item_id: str, text: str, evidence: JudgeEvidence) -> JudgeResponse | None:
        """One judgment of a text that is NOT an item (the calibration of the gold set asks it).
        It leaves no verdict, no judgment record, no failure bookkeeping and uses none of the
        hourly budget: the answer is all there is. The server is identified first, so the text
        goes nowhere but to a model server on this machine; the single in-flight slot is the same
        as any judgment's, and `None` says another process holds it. An answer that breaks the
        schema is asked for once more (RC-9); every other failure of the judge is raised as the
        `JudgeError` it is. `ValidationError` for an id or evidence the judge cannot be asked."""
        settings = self._settings
        body, truncated = prompt_text("", text, settings.max_prompt_chars)
        request = fit_request(
            JudgeRequest(item_id, body, truncated, evidence),
            num_ctx=settings.num_ctx,
            output_tokens=settings.max_output_tokens,
        )
        with self._exclusive() as got:
            if not got:
                return None
            self._judge.identify()
            try:
                return self._judge.judge(request)
            except JudgeError as exc:
                if exc.kind is not JudgeErrorKind.INVALID_OUTPUT:
                    raise
                problem = exc.problem
            return self._judge.judge(replace(request, attempt=1, previous_problem=problem))

    def ping(self) -> PingResult:
        """One tiny harmless question to the judge, to see that it answers. It touches no verdict
        and uses no budget; it waits for the same single slot as a judgment."""
        request = JudgeRequest(
            PING_ITEM_ID,
            PING_TEXT,
            False,
            JudgeEvidence(
                prior_label=Label.UNCLASSIFIED,
                prior_confidence=0.0,
                prior_stage=Stage.TOPIC,
                prior_rationale="connectivity check",
                topics=(),
                signals=(),
                catalog_status="not_checked",
                source_credibility=3,
            ),
        )
        with self._exclusive() as got:
            if not got:
                return PingResult(False, error=BUSY)
            started = self._clock.now()
            try:
                self._judge.identify()
                answer = self._judge.judge(request)
            except JudgeError as exc:
                return PingResult(False, error=exc.code)
            except Exception as exc:  # noqa: BLE001 - a broken adapter is a failed ping
                _LOG.warning("ping failed: %s", type(exc).__name__)
                return PingResult(False, error=UNEXPECTED)
            elapsed = (self._clock.now() - started).total_seconds()
            return PingResult(True, answer.model, answer.label, elapsed)

    # ---- running ----

    def escalate(self, limit: int | None = None, *, dry_run: bool = False) -> EscalationReport:
        """Judge up to `limit` (default: the configured batch) waiting items, best first."""
        count = self._batch(limit)
        overview = self._control.overview()
        if not overview.global_analysis:
            return EscalationReport(analysis_off=True)
        eligible = sorted(o.source.id for o in overview.sources if o.effective)
        now = self._clock.now()
        if dry_run:
            return self._dry_run(eligible, now, count)
        tally = _Tally(eligible_sources=len(eligible))
        tally.deadline = now + timedelta(seconds=self._settings.pass_budget_s)
        try:
            self._housekeeping(eligible, tally, now)
            entries = self._servable(eligible, now, count, tally)
            if entries and self._identify(tally, now) is _CONTINUE:
                self._serve(entries, overview.alert_min_confidence, tally)
        except StorageError as exc:
            _LOG.warning("escalation stopped by a storage error: %s", exc.code)
            tally.storage_errors += 1
        if tally.contradicted:
            _LOG.warning("llm_contradicts_prior=%d", tally.contradicted)
        return tally.report()

    def requeue_parked(self, limit: int | None = None) -> int:
        """Send up to `limit` parked items (the longest parked first) back to the queue now,
        instead of waiting for their time-to-live. Returns how many came back."""
        count = self._batch(limit)
        tally = _Tally()
        for state in self._escalations.parked(before=None, limit=count):
            self._requeue_one(state.item_id, tally)
        return tally.requeued

    def _batch(self, limit: int | None) -> int:
        if limit is None:
            return self._settings.batch_limit
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_BATCH:
            raise ValidationError(f"limit must be a whole number from 1 to {MAX_BATCH}")
        return limit

    def _servable(
        self, eligible: list[str], now: datetime, count: int, tally: _Tally
    ) -> list[QueueEntry]:
        if not eligible:
            return []
        if self._rests_until is not None and now < self._rests_until:
            tally.backing_off = True
            tally.unreachable = True
            return []
        return self._escalations.queue(source_ids=eligible, now=now, limit=count)

    def _serve(self, entries: list[QueueEntry], alert_min: float, tally: _Tally) -> None:
        for entry in entries:
            if self._should_stop():
                tally.stopped = True
                return
            if tally.deadline is not None and self._clock.now() >= tally.deadline:
                tally.out_of_time = True
                return
            if self._cycle(entry, alert_min, tally) is _STOP:
                return

    def _dry_run(self, eligible: list[str], now: datetime, count: int) -> EscalationReport:
        """What a run would do, doing none of it: the items it would send (not those that settle
        without a call), the cap, and the items a full queue would drop."""
        entries = (
            self._escalations.queue(source_ids=eligible, now=now, limit=count) if eligible else []
        )
        sendable = [e for e in entries if self._needs_a_call(e.item_id)]
        used = self._escalations.calls_in_hour(hour_start(now))
        room = max(0, self._settings.hourly_cap - used)
        waiting = self._escalations.queued_count(source_ids=eligible, now=now)
        return EscalationReport(
            candidates=min(len(sendable), room),
            cap_reached=len(sendable) > room,
            eligible_sources=len(eligible),
            pending_drops=max(0, waiting - self._settings.queue_max),
        )

    def _needs_a_call(self, item_id: str) -> bool:
        current = self._classifications.get(item_id)
        item = self._items.get(item_id)
        if current is None or not current.needs_llm or not self._eligible(item):
            return False
        return not (current.recycled_of is not None and self._already_judged(current.recycled_of))

    # ---- housekeeping before the queue is served ----

    def _housekeeping(self, eligible: list[str], tally: _Tally, now: datetime) -> None:
        self._requeue_expired(tally, now)
        if eligible:
            self._release_orphans(eligible, tally)
            self._drop_overflow(eligible, tally, now)

    def _requeue_expired(self, tally: _Tally, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._settings.parked_ttl_s)
        for state in self._escalations.parked(before=cutoff, limit=SWEEP_PAGE):
            self._requeue_one(state.item_id, tally)

    def _requeue_one(self, item_id: str, tally: _Tally) -> None:
        """A parked item goes back in the queue, behind the healthy ones, with a clean slate."""
        with self._uow.transaction():
            if not self._classifications.reopen(item_id, REQUEUE_PRIORITY):
                return
            self._escalations.clear_state(item_id)
        tally.requeued += 1

    def _release_orphans(self, eligible: list[str], tally: _Tally) -> None:
        """Copies that wait for an original that will never be judged (purged, pruned, settled
        without them, or from a source that is off) stop waiting: they are judged for themselves."""
        for copy_id in self._escalations.orphaned_copies(source_ids=eligible, limit=SWEEP_PAGE):
            with self._uow.transaction():
                fresh = self._classifications.get(copy_id)
                if fresh is None or not fresh.awaiting_original:
                    continue
                if not self._eligible(self._items.get(copy_id)):
                    continue
                original = (
                    self._classifications.get(fresh.duplicate_of) if fresh.duplicate_of else None
                )
                priority = original.priority if original is not None else 0.0
                near = original.item_id if original is not None else None
                if self._classifications.settle(
                    requeued_copy(fresh, near_of=near, priority=priority)
                ):
                    tally.released += 1

    # ---- the judge: who is there, and when it rests ----

    def _identify(self, tally: _Tally, now: datetime) -> bool:
        """Before any item text is sent, check that the address answers like Ollama. A loopback
        port can be held by any local process."""
        if self._identified_until is not None and now < self._identified_until:
            return _CONTINUE
        try:
            self._judge.identify()
        except JudgeError as exc:
            self._judge_down(exc.code, tally, now)
            return _STOP
        except Exception as exc:  # noqa: BLE001 - a broken adapter is a judge that is not there
            _LOG.warning("the judge could not be identified: %s", type(exc).__name__)
            self._judge_down(UNEXPECTED, tally, now)
            return _STOP
        self._identified_until = now + timedelta(seconds=IDENTITY_TTL_S)
        return _CONTINUE

    def _judge_down(self, code: str, tally: _Tally, now: datetime) -> None:
        """A judge-level failure: the whole judge rests, doubling up to the cap."""
        settings = self._settings
        self._judge_failures += 1
        delay = backoff_delay(self._judge_failures, settings.retry_base_s, settings.retry_cap_s)
        self._rests_until = now + delay
        self._identified_until = None
        tally.unreachable = True
        tally.error = code
        _LOG.warning("the judge failed (%s); it rests for %d s", code, int(delay.total_seconds()))

    def _judge_up(self) -> None:
        self._judge_failures = 0
        self._rests_until = None

    # ---- one item at a time ----

    @contextmanager
    def _exclusive(self) -> Iterator[bool]:
        """The single in-flight slot: the lock of this process, then the lease of the database.
        Yields False when the slot cannot be had (a stop was requested while waiting, or another
        process holds it)."""
        while not self._gate.acquire(timeout=GATE_POLL_S):
            if self._should_stop():
                yield False
                return
        try:
            now = self._clock.now()
            span = self._settings.longest_judgment_s + LEASE_MARGIN_S
            taken = self._escalations.try_lease(self._holder, now, now + timedelta(seconds=span))
            if not taken:
                yield False
                return
            try:
                yield True
            finally:
                self._escalations.release_lease(self._holder)
        finally:
            self._gate.release()

    def _cycle(self, entry: QueueEntry, alert_min: float, tally: _Tally) -> bool:
        with self._exclusive() as got:
            if not got:
                if self._should_stop():
                    tally.stopped = True
                else:
                    tally.busy = True
                return _STOP
            return self._judge_one(entry.item_id, alert_min, tally)

    def _eligible(self, item: Item | None) -> bool:
        """The item exists, is classified and its source may be analysed now (PC-2)."""
        if item is None or item.status is not ItemStatus.CLASSIFIED:
            return False
        try:
            return self._control.effective_analysis(item.source_id)
        except NotFoundError:
            return False

    def _request(self, current: Classification, item: Item) -> JudgeRequest:
        """The question for one item: its text cleaned, bounded and fitted to the context window,
        and the evidence found so far. Raises `ValidationError` for an item it cannot describe."""
        settings = self._settings
        text, truncated = prompt_text(item.title, item.text, settings.max_prompt_chars)
        request = JudgeRequest(current.item_id, text, truncated, evidence_of(current))
        return fit_request(
            request, num_ctx=settings.num_ctx, output_tokens=settings.max_output_tokens
        )

    def _judge_one(self, item_id: str, alert_min: float, tally: _Tally) -> bool:
        current = self._classifications.get(item_id)
        item = self._items.get(item_id)
        if current is None or not current.needs_llm or not self._eligible(item):
            tally.skipped += 1
            return _CONTINUE
        assert item is not None  # _eligible said so
        if current.recycled_of is not None and self._already_judged(current.recycled_of):
            return self._settle_without_call(current, item, alert_min, tally)
        try:
            request = self._request(current, item)
        except DomainError as exc:
            # The item cannot be described to the judge: no call, no budget, and it is the
            # item's own fault.
            _LOG.warning("%s cannot be put to the judge: %s", item_id, exc.code)
            return self._fail_item(current, BAD_REQUEST, tally, called=False)
        hour = hour_start(self._clock.now())
        if not self._escalations.reserve_call(hour, self._settings.hourly_cap):
            tally.cap_reached = True
            return _STOP
        try:
            answer = self._ask(request, tally)
        except _Interrupted as stop:
            tally.stopped = stop.stopped
            tally.out_of_time = not stop.stopped
            return _STOP
        except JudgeError as exc:
            return self._on_error(current, item, exc, hour, alert_min, tally)
        except Exception as exc:  # noqa: BLE001 - one broken call must not stall the queue
            _LOG.warning("the judge raised an unexpected error: %s", type(exc).__name__)
            return self._on_trouble(current, UNEXPECTED, tally)
        self._judge_up()
        return self._settle_answer(current, item, answer, alert_min, tally)

    def _ask(self, request: JudgeRequest, tally: _Tally) -> JudgeResponse:
        """One question, asked again ONCE if the answer breaks the schema (RC-9): with a
        corrective message that names the class of the problem and another seed. A stop or the
        pass budget is honoured between the two calls."""
        try:
            return self._judge.judge(request)
        except JudgeError as exc:
            if exc.kind is not JudgeErrorKind.INVALID_OUTPUT:
                raise
            first = exc
        if self._should_stop():
            raise _Interrupted(stopped=True)
        if tally.deadline is not None and self._clock.now() >= tally.deadline:
            raise _Interrupted(stopped=False)
        return self._judge.judge(replace(request, attempt=1, previous_problem=first.problem))

    def _already_judged(self, original_id: str) -> bool:
        original = self._classifications.get(original_id)
        return original is not None and original.stage is Stage.LLM

    def _judge_has_answered(self, now: datetime) -> bool:
        """The judge gave a valid verdict for some item in the last hour (this pass included: a
        judgment is stored the moment it is made)."""
        return self._escalations.judged_since(now - SUCCESS_WINDOW) > 0

    # ---- writing the outcome (one short transaction each) ----

    def _still_waiting(self, item_id: str) -> Classification | None:
        """Inside a transaction: the verdict, if the item may still be settled."""
        fresh = self._classifications.get(item_id)
        if fresh is None or not fresh.needs_llm or not self._eligible(self._items.get(item_id)):
            return None
        return fresh

    def _settle_answer(
        self,
        current: Classification,
        item: Item,
        answer: JudgeResponse,
        alert_min: float,
        tally: _Tally,
    ) -> bool:
        with self._uow.transaction():
            fresh = self._still_waiting(current.item_id)
            now = self._clock.now()
            new = judged(fresh, answer, alert_min, now) if fresh is not None else None
            if fresh is None or new is None or not self._classifications.settle(new):
                tally.skipped += 1
                return _CONTINUE
            self._escalations.add_judgment(
                Judgment(
                    item_id=current.item_id,
                    outcome="judged",
                    label=answer.label.value,
                    confidence=answer.confidence,
                    rationale=answer.rationale,
                    evidence=answer.evidence,
                    llm_model=answer.model,
                    prompt_version=PROMPT_VERSION,
                    raw_output=answer.raw_output,
                    prior_label=fresh.label.value,
                    prior_confidence=fresh.confidence,
                    prior_stage=fresh.stage.value,
                    reused_from=None,
                    judged_at=now,
                )
            )
            self._escalations.clear_state(current.item_id)
            copies = self._settle_copies(new, item, fresh.priority, answer, alert_min, now)
        if contradicts_prior(fresh, answer):
            tally.contradicted += 1
        tally.judged += 1
        tally.reused += copies
        tally.streak = 0
        return _CONTINUE

    def _settle_invalid(
        self, current: Classification, item: Item, exc: JudgeError, alert_min: float, tally: _Tally
    ) -> bool:
        with self._uow.transaction():
            fresh = self._still_waiting(current.item_id)
            now = self._clock.now()
            if fresh is None or not self._classifications.settle(
                left_unjudged(fresh, now, "invalid")
            ):
                tally.skipped += 1
                return _CONTINUE
            self._escalations.add_judgment(
                Judgment(
                    item_id=current.item_id,
                    outcome="invalid",
                    label=None,
                    confidence=None,
                    rationale=None,
                    evidence=(),
                    llm_model=None,
                    prompt_version=PROMPT_VERSION,
                    raw_output=exc.raw_output or None,
                    prior_label=fresh.label.value,
                    prior_confidence=fresh.confidence,
                    prior_stage=fresh.stage.value,
                    reused_from=None,
                    judged_at=now,
                )
            )
            self._escalations.clear_state(current.item_id)
            copies = self._settle_copies(
                left_unjudged(fresh, now, "invalid"),
                item,
                fresh.priority,
                None,
                alert_min,
                now,
                reason="invalid",
            )
        _LOG.warning("the judge gave two invalid answers for %s", current.item_id)
        tally.invalid += 1
        tally.reused += copies
        tally.streak = 0
        self._judge_up()
        return _CONTINUE

    def _settle_without_call(
        self, current: Classification, item: Item, alert_min: float, tally: _Tally
    ) -> bool:
        """A recycled copy of an item that was already judged: it keeps the verdict of the cascade
        and leaves the queue without a call (as the cascade does for a copy it classifies later)."""
        with self._uow.transaction():
            fresh = self._still_waiting(current.item_id)
            now = self._clock.now()
            if fresh is None:
                tally.skipped += 1
                return _CONTINUE
            settled = replace(
                fresh, needs_llm=False, priority=0.0, settled_at=now, settle_reason="duplicate"
            )
            if not self._classifications.settle(settled):
                tally.skipped += 1
                return _CONTINUE
            self._escalations.clear_state(current.item_id)
            self._settle_copies(
                settled,
                item,
                fresh.priority,
                None,
                alert_min,
                now,
                reason="duplicate",
                unjudged=False,
            )
        tally.reused += 1
        return _CONTINUE

    def _settle_copies(
        self,
        original: Classification,
        original_item: Item,
        original_priority: float,
        answer: JudgeResponse | None,
        alert_min: float,
        now: datetime,
        *,
        reason: str = "duplicate",
        unjudged: bool = True,
    ) -> int:
        """Settle the exact copies that wait for `original`, inside the caller's transaction. A
        copy whose item is gone, unclassified or from a source that is off now (PC-2) is left
        alone. With a judged verdict, a copy whose text is IDENTICAL to the original's takes it;
        one that differs is queued to be judged for itself. Without a verdict (invalid, parked,
        dropped) the copy keeps its provisional one and stops waiting. Returns how many copies took
        a verdict from the original."""
        reused = 0
        for copy in self._classifications.awaiting(original.item_id):
            copy_item = self._items.get(copy.item_id)
            if not self._eligible(copy_item):
                continue
            assert copy_item is not None  # _eligible said so
            shares = answer is not None and _same_text(original_item, copy_item)
            if answer is not None and not shares:
                update = requeued_copy(copy, near_of=original.item_id, priority=original_priority)
            elif answer is not None:
                update = reused_by_duplicate(copy, original, alert_min, now)
            else:
                update = released_duplicate(copy, now, reason)
                if not unjudged:
                    update = replace(update, low_confidence=copy.low_confidence)
            if not self._classifications.settle(update):
                continue
            if answer is not None and shares:
                reused += 1
            if (answer is None and reason != "invalid") or (answer is not None and not shares):
                continue  # nothing was judged for this copy: no record to keep
            self._escalations.add_judgment(
                Judgment(
                    item_id=copy.item_id,
                    outcome="reused",
                    label=answer.label.value if answer else None,
                    confidence=original.confidence if answer else None,
                    rationale=update.rationale if answer else None,
                    evidence=answer.evidence if answer else (),
                    llm_model=answer.model if answer else None,
                    prompt_version=PROMPT_VERSION,
                    raw_output=None,
                    prior_label=copy.label.value,
                    prior_confidence=copy.confidence,
                    prior_stage=copy.stage.value,
                    reused_from=original.item_id,
                    judged_at=now,
                )
            )
        return reused

    # ---- failures ----

    def _on_error(
        self,
        current: Classification,
        item: Item,
        exc: JudgeError,
        hour: datetime,
        alert_min: float,
        tally: _Tally,
    ) -> bool:
        """The call failed. A judge-level failure is the server's; an invalid answer settles the
        item only while the judge answers other items; anything else may be the item's."""
        now = self._clock.now()
        if exc.kind is JudgeErrorKind.INVALID_OUTPUT:
            if self._judge_has_answered(now):
                return self._settle_invalid(current, item, exc, alert_min, tally)
            # Nothing valid has come out of the judge for anyone: a sick judge, not a bad item.
            return self._judge_failed(current, exc.code, tally, refund=None)
        if exc.judge_level:
            nowhere = exc.kind in _NOT_THERE
            return self._judge_failed(current, exc.code, tally, refund=hour, behind=not nowhere)
        return self._on_trouble(current, exc.code, tally)

    def _on_trouble(self, current: Classification, code: str, tally: _Tally) -> bool:
        """A failure that may be the item's: it counts against the item only while the judge is
        answering other items; otherwise it is the judge's."""
        if self._judge_has_answered(self._clock.now()):
            return self._fail_item(current, code, tally)
        return self._judge_failed(current, code, tally, refund=None)

    def _judge_failed(
        self,
        current: Classification,
        code: str,
        tally: _Tally,
        *,
        refund: datetime | None,
        behind: bool = True,
    ) -> bool:
        """A failure of the judge: give the reservation back when the model gave nothing, put the
        item behind the others (it is not counted against it), rest the whole judge and end the
        batch. The item waits one step longer than the judge rests, so that when the judge is
        back the others are served first, not the same item again; when nothing is there at all
        (`behind` off) the item says nothing and waits as long as the judge does."""
        now = self._clock.now()
        if refund is not None:
            self._escalations.release_call(refund)
        before = self._escalations.get_state(current.item_id)
        attempts = (before.attempts if before else 0) + 1
        failures = before.item_failures if before else 0
        step = attempts + 1 if behind else attempts
        wait = now + backoff_delay(step, self._settings.retry_base_s, self._settings.retry_cap_s)
        self._escalations.put_state(
            EscalationState(
                current.item_id, "escalation_failed", attempts, wait, code, now, failures
            )
        )
        tally.failed += 1
        self._judge_down(code, tally, now)
        return _STOP

    def _fail_item(
        self, current: Classification, code: str, tally: _Tally, *, called: bool = True
    ) -> bool:
        """The call failed for a reason that may be the item's own and the judge is answering
        other items. Remember it with a backoff; park the item after `max_failures`."""
        settings = self._settings
        now = self._clock.now()
        before = self._escalations.get_state(current.item_id)
        attempts = (before.attempts if before else 0) + 1
        failures = (before.item_failures if before else 0) + 1
        _LOG.warning("escalation of %s failed: %s", current.item_id, code)
        if failures >= settings.max_failures:
            with self._uow.transaction():
                fresh = self._classifications.get(current.item_id)
                if fresh is not None and fresh.needs_llm:
                    parked = left_unjudged(fresh, now, "parked")
                    self._classifications.settle(parked)
                    item = self._items.get(current.item_id)
                    if item is not None:
                        self._settle_copies(
                            parked, item, fresh.priority, None, 0.0, now, reason="parked"
                        )
                self._escalations.put_state(
                    EscalationState(current.item_id, "parked", attempts, None, code, now, failures)
                )
            tally.parked += 1
        else:
            wait = now + backoff_delay(attempts, settings.retry_base_s, settings.retry_cap_s)
            self._escalations.put_state(
                EscalationState(
                    current.item_id, "escalation_failed", attempts, wait, code, now, failures
                )
            )
            tally.failed += 1
        if not called:
            return _CONTINUE
        tally.streak += 1
        return _STOP if tally.streak >= BREAKER else _CONTINUE

    # ---- the queue bound ----

    def _drop_overflow(self, eligible: list[str], tally: _Tally, now: datetime) -> None:
        """More than `queue_max` items could be served: drop the excess. A source over its fair
        share loses its own excess first (the lowest priority, then the oldest of it), so one
        source cannot crowd out the others; if the queue is still over, the lowest priority goes
        first, the oldest among equals. Their CPU verdicts stay, they are counted, and their
        waiting copies are released."""
        excess = (
            self._escalations.queued_count(source_ids=eligible, now=now) - self._settings.queue_max
        )
        if excess <= 0:
            return
        cap = self._settings.source_cap
        per_source = self._escalations.queued_by_source(source_ids=eligible, now=now)
        for source_id in sorted(per_source):
            over = per_source[source_id] - cap
            if over > 0 and excess > 0:
                excess -= self._drop(eligible, now, tally, min(over, excess), source_id)
        if excess > 0:
            self._drop(eligible, now, tally, excess, None)

    def _drop(
        self, eligible: list[str], now: datetime, tally: _Tally, wanted: int, source_id: str | None
    ) -> int:
        """Drop up to `wanted` items, a page at a time. Returns how many were dropped."""
        dropped = 0
        for _ in range(MAX_DROP_PAGES):
            if dropped >= wanted:
                return dropped
            ids = self._escalations.drop_candidates(
                source_ids=eligible,
                now=now,
                limit=min(wanted - dropped, DROP_PAGE),
                source_id=source_id,
            )
            if not ids:
                return dropped
            done = sum(self._drop_one(item_id, now) for item_id in ids)
            tally.dropped += done
            dropped += done
            if done == 0:
                return dropped
        _LOG.warning("the queue is still over its bound after %d pages of drops", MAX_DROP_PAGES)
        return dropped

    def _drop_one(self, item_id: str, now: datetime) -> bool:
        with self._uow.transaction():
            fresh = self._classifications.get(item_id)
            if fresh is None or not fresh.needs_llm:
                return False
            dropped = left_unjudged(fresh, now, "parked")
            if not self._classifications.settle(dropped):
                return False
            self._escalations.put_state(
                EscalationState(item_id, "dropped", 0, None, "queue_full", now, 0)
            )
            item = self._items.get(item_id)
            if item is not None:
                self._settle_copies(dropped, item, fresh.priority, None, 0.0, now, reason="parked")
        return True


def _same_text(original: Item, copy: Item) -> bool:
    """The copy may take a model's verdict only if its text is the original's to the byte."""
    return text_identity(original.title, original.text) == text_identity(copy.title, copy.text)
