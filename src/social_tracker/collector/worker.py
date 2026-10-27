"""The collection worker: startup reset, a scheduling loop and a graceful stop.

What it does, in order:

- START: global analysis is reset to OFF with an audited event (actor `system:worker`, reason
  `worker startup`; PC-2, PC-10). Nothing is collected until the operator runs
  `social-tracker analysis on` afterwards (the loop reads the switch on every pass). CLI commands
  never reset it; only starting a worker does. Collections a dead worker left running are then
  recovered as failed attempts (see `CollectorService.recover_interrupted`).
- EACH PASS: housekeeping (the retention purge, at most once per `purge_every_s`), then, if
  global analysis is on, every due source is collected, one at a time and in id order
  (concurrency 1), through `CollectorService.collect_once`. A source that fails, or a storage
  error, costs that source or that pass and nothing more.
- STOP (`request_stop`, wired to SIGINT/SIGTERM by the entrypoint): no new collection starts; the
  one in flight is completed and committed, never abandoned half-written (PC-6). The wait for
  it is bounded by the fetcher's deadline, well inside the 30 s the product allows.

The clock and the sleeper are injected so the loop is deterministic under test. Logs name source
ids, counts and fixed reasons; never URLs, item text or exception messages.
"""

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

from social_tracker.collector.service import ACTOR, CollectorService, CollectOutcome, CollectResult
from social_tracker.control.service import ControlService
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import DomainError

DEFAULT_TICK_S = 5.0
DEFAULT_PURGE_EVERY_S = 3600.0
STARTUP_REASON = "worker startup"
_LOG = logging.getLogger("social_tracker.worker")


@dataclass(frozen=True)
class PassSummary:
    global_off: bool
    results: tuple[CollectResult, ...]
    errors: int = 0


def _positive(value: float) -> bool:
    return isinstance(value, int | float) and math.isfinite(value) and value > 0


class Worker:
    def __init__(
        self,
        collector: CollectorService,
        control: ControlService,
        clock: Clock,
        stop: threading.Event,
        *,
        sleeper: Callable[[float], object] | None = None,
        tick_s: float = DEFAULT_TICK_S,
        purge: Callable[[], object] | None = None,
        purge_every_s: float = DEFAULT_PURGE_EVERY_S,
    ) -> None:
        if not _positive(tick_s) or not _positive(purge_every_s):
            raise ValueError("tick_s and purge_every_s must be positive, finite numbers")
        self._collector = collector
        self._control = control
        self._clock = clock
        self._stop = stop
        self._sleeper = sleeper if sleeper is not None else stop.wait
        self._tick_s = tick_s
        self._purge = purge
        self._purge_every_s = purge_every_s
        self._next_purge: float | None = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    # ---- lifecycle ----

    def start(self) -> None:
        """Reset global analysis to OFF (audited) and recover interrupted collections."""
        change = self._control.set_global_analysis(False, actor=ACTOR, reason=STARTUP_REASON)
        _LOG.info("worker started; global analysis is off%s", " (was on)" if change.changed else "")
        for source_id in self._collector.recover_interrupted():
            _LOG.warning("recovered an interrupted collection of %s", source_id)

    def run(self, *, once: bool = False) -> PassSummary | None:
        """Start, then pass after pass until stopped (or after one pass with `once`)."""
        self.start()
        summary = None
        while not self._stop.is_set():
            summary = self.run_pass()
            if once or self._stop.is_set():
                break
            self._sleeper(self._tick_s)
        _LOG.info("worker stopped")
        return summary

    # ---- one pass ----

    def run_pass(self) -> PassSummary:
        self._housekeeping()
        try:
            if not self._control.global_analysis_enabled():
                return PassSummary(global_off=True, results=())
            self._collector.recover_interrupted()
            due = self._collector.due_sources()
        except DomainError as exc:
            _LOG.warning("pass skipped: %s", exc.code)
            return PassSummary(global_off=False, results=(), errors=1)
        results: list[CollectResult] = []
        errors = 0
        for source_id in due:
            if self._stop.is_set():
                break
            result = self._collect(source_id)
            if result is None:
                errors += 1
            else:
                results.append(result)
        return PassSummary(global_off=False, results=tuple(results), errors=errors)

    def _collect(self, source_id: str) -> CollectResult | None:
        try:
            result = self._collector.collect_once(source_id)
        except DomainError as exc:
            _LOG.warning("collection of %s not done: %s", source_id, exc.code)
            return None
        except Exception as exc:  # noqa: BLE001 - one source must not stop the worker
            _LOG.error("unexpected error collecting %s: %s", source_id, type(exc).__name__)
            return None
        self._log_result(result)
        return result

    @staticmethod
    def _log_result(result: CollectResult) -> None:
        if result.outcome in (CollectOutcome.COLLECTED, CollectOutcome.NOT_MODIFIED):
            _LOG.info(
                "%s source=%s: %d new, %d duplicate, discarded %s",
                result.outcome.value,
                result.source_id,
                result.new,
                result.duplicate,
                dict(result.discarded),
            )
        elif result.outcome is CollectOutcome.FAILED:
            _LOG.warning("failed source=%s: %s", result.source_id, result.error)
        else:
            _LOG.info("%s source=%s", result.outcome.value, result.source_id)

    def _housekeeping(self) -> None:
        if self._purge is None:
            return
        now = self._clock.now().timestamp()
        if self._next_purge is not None and now < self._next_purge:
            return
        self._next_purge = now + self._purge_every_s
        try:
            self._purge()
        except Exception as exc:  # noqa: BLE001 - housekeeping must never stop collection
            _LOG.warning("retention purge failed: %s", type(exc).__name__)
