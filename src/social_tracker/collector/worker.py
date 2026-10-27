"""The collection worker: startup reset, a scheduling loop and a graceful stop.

What it does, in order:

- START: global analysis is reset to OFF with an audited event (actor `system:worker`, reason
  `worker startup`; PC-2, PC-10). Nothing is collected until the operator runs
  `social-tracker analysis on` afterwards (the loop reads the switch on every pass). CLI commands
  never reset it; only starting a worker does. A one-shot diagnostic run (`once`) may skip the
  reset (`reset_analysis=False`, the CLI's `--no-reset-analysis`): without it a single pass right
  after the reset could never collect anything. The long-running worker always resets. Collections
  a dead worker left running are then recovered as failed attempts (see
  `CollectorService.recover_interrupted`).
- EACH PASS: housekeeping (the retention purge, at most once per `purge_every_s`), then, if
  global analysis is on, every due source is collected, one at a time and in id order
  (concurrency 1), through `CollectorService.collect_once`. A source that fails, or a storage
  error, costs that source or that pass and nothing more.
- AFTER THE COLLECTION, if a classify hook was given (the composition root passes
  `ClassifierService.classify_pending`), the cascade runs over the pending items of the eligible
  sources: the CPU stages only, in batches. A failure there costs that step and nothing more.
- LAST, if a catalog hook was given (the composition root passes the refresher of the USGS and
  EMSC catalogs), the catalogs that are due are refreshed for the NEXT pass. They run after the
  classification on purpose: a refresh is network I/O of up to two catalogs times two requests,
  and it must never hold the classification back. The refresher itself checks the stop flag
  between catalogs and requests. It follows the global analysis switch like the sources: nothing
  is fetched while it is off. A failure there costs that step and nothing more.
- AFTER ALL OF THAT, if an escalate hook was given (the composition root passes the LLM
  escalation, `EscalationService.escalate`), the items the cascade could not settle are sent to
  the local language model, a bounded batch per pass. It comes last because it is the slowest
  and the only optional stage: a judge that is down or slow costs this step and nothing else
  (the CPU verdicts are already stored). The step is bounded by the item count AND by a
  wall-clock budget (`SOCIAL_TRACKER_LLM_PASS_BUDGET_S`, enforced by the escalation service), and
  it checks the stop flag between items and between the two calls of one judgment. A judge that
  failed rests (doubling, capped) across passes: the hook returns at once until it is over. It
  follows the global analysis switch too.
- LAST OF ALL, if an alerts hook was given (the composition root passes the alert step,
  `AlertService.refresh`), the verdicts that are new or changed since their alert was last looked
  at are turned into alerts: it reads only what is stored (nothing is sent anywhere), and it comes
  after the model so that a verdict the model just settled reaches its alert in the same pass. A
  failure there costs that step and nothing more.
- STOP (`request_stop`, wired to SIGINT/SIGTERM by the entrypoint): no new collection starts; the
  one in flight is completed and committed, never abandoned half-written (PC-6). The wait for
  it is bounded by the fetcher's deadline (a wall-clock budget enforced inside every connect,
  read, write and name lookup, see `collector.http_fetcher`), well inside the 30 s the product
  allows. A second signal does not wait: the entrypoint exits at once (`entrypoints.signals`) and
  the next start recovers the abandoned collection.

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
    """What one pass did. `classification` is whatever the classify hook returned (the report of
    the cascade run), None when there is no hook, nothing ran or it failed."""

    global_off: bool
    results: tuple[CollectResult, ...]
    errors: int = 0
    classification: object | None = None
    catalog: object | None = None
    escalation: object | None = None
    alerts: object | None = None


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
        classify: Callable[[], object] | None = None,
        catalog: Callable[[], object] | None = None,
        escalate: Callable[[], object] | None = None,
        alerts: Callable[[], object] | None = None,
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
        self._classify = classify
        self._catalog = catalog
        self._escalate = escalate
        self._alerts = alerts
        self._next_purge: float | None = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    # ---- lifecycle ----

    def start(self, reset_analysis: bool = True) -> None:
        """Reset global analysis to OFF (audited, unless `reset_analysis` is false) and recover
        interrupted collections. The caller holds the single-worker lock, so every collection
        still marked as running belongs to a dead worker and is recovered at once."""
        if reset_analysis:
            change = self._control.set_global_analysis(False, actor=ACTOR, reason=STARTUP_REASON)
            _LOG.info(
                "worker started; global analysis is off%s", " (was on)" if change.changed else ""
            )
        else:
            _LOG.info("worker started; global analysis left as it is")
        for source_id in self._collector.recover_interrupted(force=True):
            _LOG.warning("recovered an interrupted collection of %s", source_id)

    def run(self, *, once: bool = False, reset_analysis: bool = True) -> PassSummary | None:
        """Start, then pass after pass until stopped (or after one pass with `once`)."""
        self.start(reset_analysis)
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
        classification, failed = self._classify_pending()
        refreshed = self._refresh_catalog()
        escalation, escalation_failed = self._escalate_queued()
        alerted, alerts_failed = self._refresh_alerts()
        return PassSummary(
            global_off=False,
            results=tuple(results),
            errors=errors + failed + escalation_failed + alerts_failed,
            classification=classification,
            catalog=refreshed,
            escalation=escalation,
            alerts=alerted,
        )

    def _refresh_alerts(self) -> tuple[object | None, int]:
        """Turn the new and changed verdicts into alerts. Skipped once a stop is requested. A
        failure costs this step only; the next pass tries again."""
        if self._alerts is None or self._stop.is_set():
            return None, 0
        try:
            report = self._alerts()
        except DomainError as exc:
            _LOG.warning("alerts not refreshed: %s", exc.code)
            return None, 1
        except Exception as exc:  # noqa: BLE001 - the alert step must never stop the worker
            _LOG.error("unexpected error refreshing the alerts: %s", type(exc).__name__)
            return None, 1
        _LOG.info(
            "alerts: created %d, merged %d, updated %d, withdrawn %d, reactivated %d, errors %d",
            getattr(report, "created", 0),
            getattr(report, "merged", 0),
            getattr(report, "updated", 0),
            getattr(report, "withdrawn", 0),
            getattr(report, "reactivated", 0),
            getattr(report, "errors", 0),
        )
        return report, 0

    def _escalate_queued(self) -> tuple[object | None, int]:
        """Send the items the cascade could not settle to the language model. Skipped once a stop
        is requested. A failure costs this step only; the next pass tries again."""
        if self._escalate is None or self._stop.is_set():
            return None, 0
        try:
            report = self._escalate()
        except DomainError as exc:
            _LOG.warning("escalation not done: %s", exc.code)
            return None, 1
        except Exception as exc:  # noqa: BLE001 - the optional stage must never stop the worker
            _LOG.error("unexpected error escalating: %s", type(exc).__name__)
            return None, 1
        _LOG.info(
            "escalation: judged %d, invalid %d, reused %d, skipped %d, failed %d, parked %d, "
            "dropped %d",
            getattr(report, "judged", 0),
            getattr(report, "invalid", 0),
            getattr(report, "reused", 0),
            getattr(report, "skipped", 0),
            getattr(report, "failed", 0),
            getattr(report, "parked", 0),
            getattr(report, "dropped", 0),
        )
        return report, 0

    def _refresh_catalog(self) -> object | None:
        """Refresh the official catalogs that are due (each knows its own interval and backoff).
        Only reached while global analysis is on, like the sources. The refresher records its own
        failures and does not raise; whatever else goes wrong costs this step only."""
        if self._catalog is None or self._stop.is_set():
            return None
        try:
            return self._catalog()
        except Exception as exc:  # noqa: BLE001 - the cross-check must never stop collection
            _LOG.error("unexpected error refreshing the catalogs: %s", type(exc).__name__)
            return None

    def _classify_pending(self) -> tuple[object | None, int]:
        """Run the cascade over what was collected (CPU only). Skipped once a stop is requested.
        A failure costs this step only; the next pass tries again."""
        if self._classify is None or self._stop.is_set():
            return None, 0
        try:
            report = self._classify()
        except DomainError as exc:
            _LOG.warning("classification not done: %s", exc.code)
            return None, 1
        except Exception as exc:  # noqa: BLE001 - classification must never stop the worker
            _LOG.error("unexpected error classifying: %s", type(exc).__name__)
            return None, 1
        _LOG.info(
            "classification: classified %d, out of scope %d, skipped %d, errors %d, "
            "queued for the LLM %d",
            getattr(report, "classified", 0),
            getattr(report, "out_of_scope", 0),
            getattr(report, "skipped", 0),
            getattr(report, "errors", 0),
            getattr(report, "needs_llm", 0),
        )
        return report, 0

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
