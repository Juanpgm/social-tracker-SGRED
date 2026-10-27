"""The LLM escalation (RC-7, RC-9, RC-12): the last stage of the cascade (application layer).

`EscalationService.escalate` takes the items the CPU stages could not settle (the ones the
classifier marked `needs_llm`), best first (priority, then oldest), and asks the judge about each
one: the item text as untrusted data plus the structured evidence found so far. It runs after the
cascade in the worker pass and only while global analysis is on.

Rules of the road
- Only the work of the CPU stages is ever replaced, and only by a VALID verdict: one of four
  labels with a confidence from 0 to 1 and a short rationale (see `schema`). Anything else is
  retried once and then counts as `unclassified`: the verdict of the earlier stage stays (RC-9).
  The credibility of the source stays a prior: it is recorded, never applied to a judged
  confidence. A judged label under 0.5 is applied but flagged `low_confidence`.
- The judge is a different failure domain from the CPU. A judge that is down, slow or sick costs
  the escalation and nothing else: the item keeps its CPU verdict, is marked `escalation_failed`
  with an exponential backoff and is tried again later. A failure that may be the item's own (a
  timeout, a 5xx, a body that is not an answer) counts towards parking it for good after
  `max_failures`; "nothing is listening" and "no model" do not. Two failed items in a row, or one
  that says the judge is unreachable, end the batch.
- One judgment at a time: a lock in the process and a lease in the database keep two workers (or
  a worker and a one-shot command) from loading the GPU together. The lease expires, so a crashed
  holder frees it.
- No network call inside a write transaction. Everything is read first, the judge is called with
  no transaction open, and the answer is written in one short transaction that first checks that
  the item still exists, is still classified, its source is still eligible (PC-2) and its verdict
  still waits: a source switched off, an item purged or a verdict settled meanwhile discard the
  answer instead of writing it. A repeated or racing run never judges an item twice.
- Budget: at most `hourly_cap` judgments are started per UTC hour (an invalid answer's one retry
  belongs to the same judgment). Items beyond it wait. If more than `queue_max` wait, the OLDEST
  are dropped, counted, and keep their CPU verdict.
- Exact copies that wait for their original (`awaiting_original`) are settled when the original
  leaves the queue: with the judged verdict (their own credibility prior stays on their record)
  or, if there is none, with the provisional one they already have.
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
from social_tracker.items.domain import Item, ItemStatus
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
    evidence_of,
    hour_start,
    judged,
    left_unjudged,
    prompt_text,
    released_duplicate,
    reused_by_duplicate,
)
from social_tracker.llm.ports import EscalationRepository, LlmJudge
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import NotFoundError, StorageError, ValidationError
from social_tracker.shared.uow import UnitOfWork

_LOG = logging.getLogger("social_tracker.llm")

MAX_BATCH = 1000
BREAKER = 2  # failed items in a row that end a batch
LEASE_MARGIN_S = 30
GATE_POLL_S = 0.2
DROP_PAGE = 1000
MAX_DROP_PAGES = 100
UNEXPECTED = "llm_unexpected"
BUSY = "busy"
PING_ITEM_ID = "ping_check"
PING_TEXT = "Prueba de conexion: este mensaje no trata ningun tema."

_CONTINUE = False
_STOP = True


def _never() -> bool:
    return False


@dataclass(frozen=True)
class EscalationReport:
    """What one `escalate` run did. `judged` got a valid verdict; `invalid` got two invalid
    answers (RC-9); `reused` are copies settled from an original (or a recycled copy of a judged
    item) without a call; `skipped` stopped being eligible between listing and writing; `failed`
    will be retried; `parked` gave up for good; `dropped` left a full queue. `candidates` is what
    a dry run would have sent."""

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
    streak: int = field(default=0)

    def report(self) -> EscalationReport:
        data = {k: v for k, v in vars(self).items() if k != "streak"}
        return EscalationReport(**data)


@dataclass(frozen=True)
class LlmStatus:
    """What `llm status` shows. `installed` are the models the server has, `missing` the
    configured ones it lacks (empty when the server cannot be asked)."""

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

    # ---- reading ----

    def judgment(self, item_id: str) -> Judgment | None:
        """What was kept of the judgment of an item, if it went through the judge."""
        return self._escalations.get_judgment(item_id)

    def counts(self) -> EscalationCounts:
        return self._escalations.counts()

    def status(self) -> LlmStatus:
        """Ask the server which models it has (a `GET`, no model is loaded) and put it next to the
        local figures. A server that is down is a status, not an error."""
        settings = self._settings
        used = self._escalations.calls_in_hour(hour_start(self._clock.now()))
        counts = self._escalations.counts()
        installed: tuple[str, ...] = ()
        error: str | None = None
        try:
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
        tally = _Tally()
        try:
            self._drop_overflow(tally, now)
            entries = (
                self._escalations.queue(source_ids=eligible, now=now, limit=count)
                if eligible
                else []
            )
            for entry in entries:
                if self._should_stop():
                    tally.stopped = True
                    break
                if self._cycle(entry, overview.alert_min_confidence, tally) is _STOP:
                    break
        except StorageError as exc:
            _LOG.warning("escalation stopped by a storage error: %s", exc.code)
            tally.storage_errors += 1
        report = tally.report()
        return report

    def _batch(self, limit: int | None) -> int:
        if limit is None:
            return self._settings.batch_limit
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_BATCH:
            raise ValidationError(f"limit must be a whole number from 1 to {MAX_BATCH}")
        return limit

    def _dry_run(self, eligible: list[str], now: datetime, count: int) -> EscalationReport:
        entries = (
            self._escalations.queue(source_ids=eligible, now=now, limit=count) if eligible else []
        )
        used = self._escalations.calls_in_hour(hour_start(now))
        room = max(0, self._settings.hourly_cap - used)
        return EscalationReport(candidates=min(len(entries), room), cap_reached=len(entries) > room)

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
            span = 2 * self._settings.call_deadline_s + LEASE_MARGIN_S
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

    def _judge_one(self, item_id: str, alert_min: float, tally: _Tally) -> bool:
        current = self._classifications.get(item_id)
        item = self._items.get(item_id)
        if current is None or not current.needs_llm or not self._eligible(item):
            tally.skipped += 1
            return _CONTINUE
        assert item is not None  # _eligible said so
        if current.recycled_of is not None and self._already_judged(current.recycled_of):
            return self._settle_without_call(current, alert_min, tally)
        if not self._escalations.reserve_call(
            hour_start(self._clock.now()), self._settings.hourly_cap
        ):
            tally.cap_reached = True
            return _STOP
        text, truncated = prompt_text(item.title, item.text, self._settings.max_prompt_chars)
        request = JudgeRequest(item_id, text, truncated, evidence_of(current))
        try:
            answer = self._ask(request)
        except JudgeError as exc:
            if exc.kind is JudgeErrorKind.INVALID_OUTPUT:
                return self._settle_invalid(item_id, exc, alert_min, tally)
            return self._fail(current, exc.code, exc.item_specific, tally)
        except Exception as exc:  # noqa: BLE001 - one broken call must not stall the queue
            _LOG.warning("the judge raised an unexpected error: %s", type(exc).__name__)
            return self._fail(current, UNEXPECTED, True, tally)
        return self._settle_answer(item_id, answer, alert_min, tally)

    def _ask(self, request: JudgeRequest) -> JudgeResponse:
        """One question, asked again ONCE if the answer breaks the schema (RC-9)."""
        try:
            return self._judge.judge(request)
        except JudgeError as exc:
            if exc.kind is not JudgeErrorKind.INVALID_OUTPUT:
                raise
        return self._judge.judge(request)

    def _already_judged(self, original_id: str) -> bool:
        original = self._classifications.get(original_id)
        return original is not None and original.stage is Stage.LLM

    # ---- writing the outcome (one short transaction each) ----

    def _still_waiting(self, item_id: str) -> Classification | None:
        """Inside a transaction: the verdict, if the item may still be settled."""
        fresh = self._classifications.get(item_id)
        if fresh is None or not fresh.needs_llm or not self._eligible(self._items.get(item_id)):
            return None
        return fresh

    def _settle_answer(
        self, item_id: str, answer: JudgeResponse, alert_min: float, tally: _Tally
    ) -> bool:
        with self._uow.transaction():
            fresh = self._still_waiting(item_id)
            new = judged(fresh, answer, alert_min) if fresh is not None else None
            if fresh is None or new is None or not self._classifications.settle(new):
                tally.skipped += 1
                return _CONTINUE
            now = self._clock.now()
            self._escalations.add_judgment(
                Judgment(
                    item_id=item_id,
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
            self._escalations.clear_state(item_id)
            copies = self._settle_copies(new, answer, alert_min, now)
        tally.judged += 1
        tally.reused += copies
        tally.streak = 0
        return _CONTINUE

    def _settle_invalid(
        self, item_id: str, exc: JudgeError, alert_min: float, tally: _Tally
    ) -> bool:
        with self._uow.transaction():
            fresh = self._still_waiting(item_id)
            if fresh is None or not self._classifications.settle(left_unjudged(fresh)):
                tally.skipped += 1
                return _CONTINUE
            now = self._clock.now()
            self._escalations.add_judgment(
                Judgment(
                    item_id=item_id,
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
            self._escalations.clear_state(item_id)
            copies = self._settle_copies(left_unjudged(fresh), None, alert_min, now, invalid=True)
        _LOG.warning("the judge gave two invalid answers for %s", item_id)
        tally.invalid += 1
        tally.reused += copies
        tally.streak = 0
        return _CONTINUE

    def _settle_without_call(
        self, current: Classification, alert_min: float, tally: _Tally
    ) -> bool:
        """A recycled copy of an item that was already judged: it keeps the verdict of the cascade
        and leaves the queue without a call (as the cascade does for a copy it classifies later)."""
        with self._uow.transaction():
            fresh = self._still_waiting(current.item_id)
            if fresh is None or not self._classifications.settle(_out_of_queue(fresh)):
                tally.skipped += 1
                return _CONTINUE
            self._escalations.clear_state(current.item_id)
            now = self._clock.now()
            self._settle_copies(_out_of_queue(fresh), None, alert_min, now)
        tally.reused += 1
        return _CONTINUE

    def _settle_copies(
        self,
        original: Classification,
        answer: JudgeResponse | None,
        alert_min: float,
        now: datetime,
        *,
        invalid: bool = False,
    ) -> int:
        """Settle the exact copies that wait for `original`: with its judged verdict if there is
        one, else release them with the provisional verdict they have. Inside the caller's
        transaction. Returns how many were settled."""
        settled = 0
        for copy in self._classifications.awaiting(original.item_id):
            if answer is not None:
                update = reused_by_duplicate(copy, original, alert_min)
            else:
                update = released_duplicate(copy)
            if not self._classifications.settle(update):
                continue
            settled += 1
            if answer is None and not invalid:
                continue  # nothing was judged: no record to keep
            self._escalations.add_judgment(
                Judgment(
                    item_id=copy.item_id,
                    outcome="reused",
                    label=answer.label.value if answer else None,
                    confidence=answer.confidence if answer else None,
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
        return settled

    # ---- failures ----

    def _fail(self, current: Classification, code: str, item_specific: bool, tally: _Tally) -> bool:
        """The call failed. Remember it with a backoff; park the item if it keeps failing for
        reasons that may be its own; end the batch if the judge itself is the problem."""
        settings = self._settings
        now = self._clock.now()
        before = self._escalations.get_state(current.item_id)
        attempts = (before.attempts if before else 0) + 1
        failures = (before.item_failures if before else 0) + int(item_specific)
        _LOG.warning("escalation of %s failed: %s", current.item_id, code)
        if failures >= settings.max_failures:
            with self._uow.transaction():
                fresh = self._classifications.get(current.item_id)
                if fresh is not None and fresh.needs_llm:
                    self._classifications.settle(left_unjudged(fresh))
                    self._settle_copies(left_unjudged(fresh), None, 0.0, now)
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
        tally.streak += 1
        if not item_specific:
            tally.unreachable = True
            return _STOP
        return _STOP if tally.streak >= BREAKER else _CONTINUE

    # ---- the queue bound ----

    def _drop_overflow(self, tally: _Tally, now: datetime) -> None:
        """More than `queue_max` waiting: drop the OLDEST beyond it, keep their CPU verdict, count
        them. Their waiting copies are released."""
        for _ in range(MAX_DROP_PAGES):
            excess = self._escalations.queued_count() - self._settings.queue_max
            if excess <= 0:
                return
            for item_id in self._escalations.oldest_queued(min(excess, DROP_PAGE)):
                with self._uow.transaction():
                    fresh = self._classifications.get(item_id)
                    if fresh is None or not fresh.needs_llm:
                        continue
                    self._classifications.settle(left_unjudged(fresh))
                    self._escalations.put_state(
                        EscalationState(item_id, "dropped", 0, None, "queue_full", now, 0)
                    )
                    self._settle_copies(left_unjudged(fresh), None, 0.0, now)
                tally.dropped += 1
        _LOG.warning("the queue is still over its bound after %d pages of drops", MAX_DROP_PAGES)


def _out_of_queue(verdict: Classification) -> Classification:
    """The verdict as it is, out of the queue, with nothing else changed."""
    return replace(verdict, needs_llm=False, priority=0.0)
