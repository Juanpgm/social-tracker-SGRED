"""Control service: the only writer of pipeline control state.

It owns the global analysis switch, the per-source analysis flags, the collection state machine
of every source and the alert threshold. Rules enforced here (the repositories only store):

- Every REAL change appends exactly one control event (actor, reason, from, to) inside the same
  `UnitOfWork` transaction as the state or setting write, so the audit log and the persisted
  state cannot diverge. A request that changes nothing (redundant switch, repeated threshold,
  disabling a disabled source) writes neither. An illegal transition raises `InvalidTransition`
  and writes nothing.
- Replaying the event log in order reproduces the persisted state of every source that exists:
  each event carries its `from` and `to` (and, for collection events, `failures` and
  `next_run_at`). The registry's events that move a state carry `state_from`/`state_to`, and
  `source_deleted` (which removes the state row and the flag row without a collection event)
  carries the previous `state` and `analysis` value.
- A source's `enabled` flag (SourceRegistry) is the operator's intent and the only way to turn a
  source on or off. This service never enables or disables a source: it only runs the worker
  transitions (start, succeed, fail, retry, reset) and refuses them on a disabled source with
  `InvalidTransition`, writing nothing. The registry keeps the state row in step with the flag
  in its own transaction, so `source.enabled <=> collection state is not disabled`.
- Defaults are the safe ones: the system starts paused, analysis flags start off, a source with
  no state row is `disabled`.
- Nothing here ever publishes or sends anything outside the system (G-1).

Worker eligibility (the rule slice 3's collection worker must apply): a source may be picked up
only when `source.enabled` AND its state is `idle`, or `backoff` once `next_run_at` has elapsed
(`retry_collection` then moves it back to idle). Disabled, collecting and failed sources are
never eligible. This service enforces the first half by refusing worker transitions on a
disabled source; choosing WHICH eligible source to run is the worker's job.

Event kinds are stable identifiers (`EVENT_*` and `COLLECTION_EVENT_KINDS`). Details are compact
JSON objects with known keys only; `actor` and `reason` are validated free text and never echo
into error messages.

Per-source analysis flags are settings named by `source_analysis_key(source_id)`: "true" or
"false", absent meaning off. The alert threshold is the `alert_min_confidence` setting. There is
no environment override: the settings store is the single source of truth.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from social_tracker.control.domain import (
    DEFAULT_ALERT_MIN_CONFIDENCE,
    SETTING_ALERT_MIN_CONFIDENCE,
    SETTING_GLOBAL_ANALYSIS_ENABLED,
    SETTING_SOURCE_ANALYSIS_PREFIX,
    CollectionState,
    ControlEvent,
    Event,
    GlobalState,
    SourceState,
    analysis_effective,
    apply_event,
    apply_global,
    event_detail_json,
    format_confidence,
    parse_confidence,
    source_analysis_key,
    validate_actor,
    validate_confidence,
    validate_reason,
)
from social_tracker.control.ports import ControlRepository
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import (
    InvalidTransition,
    NotFoundError,
    StorageError,
    ValidationError,
)
from social_tracker.shared.ids import is_valid_slug
from social_tracker.shared.uow import UnitOfWork
from social_tracker.sources.domain import Source
from social_tracker.sources.ports import SourceRepository

EVENT_GLOBAL_ANALYSIS = "global_analysis_changed"
EVENT_SOURCE_ANALYSIS = "source_analysis_changed"
EVENT_ALERT_THRESHOLD = "alert_threshold_changed"
# Worker transitions only: enabling and disabling are registry events (`source_enabled`...).
COLLECTION_EVENT_KINDS: dict[Event, str] = {
    Event.START: "collection_started",
    Event.SUCCEED: "collection_succeeded",
    Event.FAIL: "collection_failed",
    Event.RETRY: "collection_retried",
    Event.RESET: "collection_reset",
}

_REGISTRY_ONLY_EVENTS = frozenset({Event.ENABLE, Event.DISABLE})
_ACTION_SET = "set"
_ON, _OFF = "on", "off"


@dataclass(frozen=True)
class FlagChange:
    """Outcome of a switch request; `changed` is False for an idempotent no-op."""

    enabled: bool
    changed: bool


@dataclass(frozen=True)
class CollectionChange:
    state: SourceState
    changed: bool


@dataclass(frozen=True)
class ThresholdChange:
    value: float
    changed: bool


@dataclass(frozen=True)
class SourceOverview:
    """One source with its collection state, analysis flag and whether analysis really runs."""

    source: Source
    state: CollectionState
    analysis: bool
    effective: bool
    consecutive_failures: int = 0
    next_run_at: datetime | None = None  # set only while in backoff
    last_fetch_at: datetime | None = None  # the last successful fetch
    last_error: str | None = None  # sanitized reason of the last failure, until the next success


@dataclass(frozen=True)
class SystemOverview:
    global_analysis: bool
    alert_min_confidence: float
    sources: tuple[SourceOverview, ...]  # sorted by source id


def _require_bool(on: Any) -> bool:
    if not isinstance(on, bool):
        raise ValidationError("on must be a bool", ("on",))
    return on


def _check_source_id(source_id: Any) -> str:
    if not is_valid_slug(source_id):
        raise ValidationError("source id is not a valid slug", ("source_id",))
    return source_id


def _stored_flag(value: str | None) -> bool:
    if value in (None, "false"):
        return False
    if value == "true":
        return True
    raise StorageError("stored flag is invalid")


def _coerce_event(event: Any) -> Event:
    try:
        return Event(event)
    except (ValueError, TypeError):
        raise InvalidTransition("unknown event") from None


class ControlService:
    def __init__(
        self,
        control: ControlRepository,
        sources: SourceRepository,
        uow: UnitOfWork,
        clock: Clock,
    ) -> None:
        self._control = control
        self._sources = sources
        self._uow = uow
        self._clock = clock

    # ---- helpers ----

    def _audit(
        self,
        at: datetime,
        kind: str,
        source_id: str | None,
        *,
        action: str,
        actor: str,
        reason: str,
        before: str,
        after: str,
        **extra: Any,
    ) -> None:
        detail = {
            "action": action,
            "actor": actor,
            "reason": reason,
            "from": before,
            "to": after,
            **extra,
        }
        text = event_detail_json(detail)
        self._control.append_control_event(ControlEvent(at, kind, source_id, text))

    def _known_source(self, source_id: Any) -> str:
        """Validate the id and make sure the source exists (`NotFoundError` otherwise)."""
        if not self._sources.exists(_check_source_id(source_id)):
            raise NotFoundError("source not found")
        return source_id

    def _global_state(self) -> GlobalState:
        active = _stored_flag(self._control.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED))
        return GlobalState.ACTIVE if active else GlobalState.PAUSED

    # ---- global switch ----

    def global_analysis_enabled(self) -> bool:
        return self._global_state() is GlobalState.ACTIVE

    def set_global_analysis(self, on: bool, *, actor: str, reason: str) -> FlagChange:
        target = GlobalState.ACTIVE if _require_bool(on) else GlobalState.PAUSED
        actor, reason = validate_actor(actor), validate_reason(reason)
        with self._uow.transaction():
            now = self._clock.now()
            current = self._global_state()
            if not apply_global(current, target).changed:
                return FlagChange(on, changed=False)
            self._control.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true" if on else "false")
            self._audit(
                now,
                EVENT_GLOBAL_ANALYSIS,
                None,
                action=_ACTION_SET,
                actor=actor,
                reason=reason,
                before=current.value,
                after=target.value,
            )
        return FlagChange(on, changed=True)

    # ---- per-source analysis ----

    def source_analysis_enabled(self, source_id: str) -> bool:
        self._known_source(source_id)
        return _stored_flag(self._control.get_setting(source_analysis_key(source_id)))

    def set_source_analysis(
        self, source_id: str, on: bool, *, actor: str, reason: str
    ) -> FlagChange:
        _require_bool(on)
        actor, reason = validate_actor(actor), validate_reason(reason)
        with self._uow.transaction():
            now = self._clock.now()
            self._known_source(source_id)
            key = source_analysis_key(source_id)
            current = _stored_flag(self._control.get_setting(key))
            if current is on:
                return FlagChange(on, changed=False)
            self._control.set_setting(key, "true" if on else "false")
            self._audit(
                now,
                EVENT_SOURCE_ANALYSIS,
                source_id,
                action=_ACTION_SET,
                actor=actor,
                reason=reason,
                before=_ON if current else _OFF,
                after=_ON if on else _OFF,
            )
        return FlagChange(on, changed=True)

    def effective_analysis(self, source_id: str) -> bool:
        """Analysis runs only when the system is active, the source is enabled AND the source has
        analysis enabled."""
        source = self._sources.get(_check_source_id(source_id))
        flag = _stored_flag(self._control.get_setting(source_analysis_key(source_id)))
        return analysis_effective(self._global_state(), flag, source_enabled=source.enabled)

    def overview(self) -> SystemOverview:
        """Everything an operator listing needs in a fixed number of queries (sources, states,
        flags, two settings), however many sources exist, all read inside one `uow.snapshot()`.

        The snapshot makes the result a single point in time: a source another process adds,
        enables or switches while this runs is either wholly in or wholly out, so an enabled
        source is never listed with a missing (hence DISABLED) state. It takes no write lock.
        """
        with self._uow.snapshot():
            global_state = self._global_state()
            states = {s.source_id: s for s in self._control.list_source_states()}
            fetches = {f.source_id: f for f in self._control.list_fetch_statuses()}
            flags = {
                key.removeprefix(SETTING_SOURCE_ANALYSIS_PREFIX): _stored_flag(value)
                for key, value in self._control.list_settings(
                    SETTING_SOURCE_ANALYSIS_PREFIX
                ).items()
            }
            sources = self._sources.list()
            threshold = self.alert_min_confidence()
        rows: list[SourceOverview] = []
        for source in sources:
            flag = flags.get(source.id, False)
            effective = analysis_effective(global_state, flag, source_enabled=source.enabled)
            stored = states.get(source.id)
            state = stored.state if stored else CollectionState.DISABLED
            failures = stored.consecutive_failures if stored else 0
            retry_at = stored.next_run_at if stored else None
            fetch = fetches.get(source.id)
            rows.append(
                SourceOverview(
                    source,
                    state,
                    flag,
                    effective,
                    failures,
                    retry_at,
                    fetch.last_fetch_at if fetch else None,
                    fetch.last_error if fetch else None,
                )
            )
        return SystemOverview(global_state is GlobalState.ACTIVE, threshold, tuple(rows))

    # ---- collection state machine ----

    def get_state(self, source_id: str) -> SourceState:
        self._known_source(source_id)
        return self._control.get_source_state(source_id) or SourceState(
            source_id, CollectionState.DISABLED
        )

    def apply_collection_event(
        self, source_id: str, event: Event, *, actor: str, reason: str, jitter_s: float = 0.0
    ) -> CollectionChange:
        """Apply one WORKER event to a source, auditing it when it changes something.

        `ENABLE` and `DISABLE` are refused: use `SourceRegistry.enable/disable`, which keep the
        source flag and the state row in step. A worker event on a disabled source is refused
        too, whatever the stored row says.
        """
        actor, reason = validate_actor(actor), validate_reason(reason)
        domain_event = _coerce_event(event)
        if domain_event in _REGISTRY_ONLY_EVENTS:
            raise InvalidTransition(
                "enabling and disabling a source is done in the source registry"
            )
        with self._uow.transaction():
            now = self._clock.now()
            if not self._sources.get(_check_source_id(source_id)).enabled:
                raise InvalidTransition("the source is disabled")
            stored = self._control.get_source_state(source_id)
            current = stored or SourceState(source_id, CollectionState.DISABLED)
            result = apply_event(current, domain_event, now, jitter_s=jitter_s)
            if not result.changed:
                return CollectionChange(current, changed=False)
            new = result.state
            self._control.save_source_state(new)
            self._audit(
                now,
                COLLECTION_EVENT_KINDS[domain_event],
                source_id,
                action=domain_event.value,
                actor=actor,
                reason=reason,
                before=current.state.value,
                after=new.state.value,
                failures=new.consecutive_failures,
                next_run_at=new.next_run_at.isoformat() if new.next_run_at else None,
            )
        return CollectionChange(new, changed=True)

    def start_collection(self, source_id: str, *, actor: str, reason: str) -> CollectionChange:
        return self.apply_collection_event(source_id, Event.START, actor=actor, reason=reason)

    def finish_collection(self, source_id: str, *, actor: str, reason: str) -> CollectionChange:
        return self.apply_collection_event(source_id, Event.SUCCEED, actor=actor, reason=reason)

    def fail_collection(
        self, source_id: str, *, actor: str, reason: str, jitter_s: float = 0.0
    ) -> CollectionChange:
        return self.apply_collection_event(
            source_id, Event.FAIL, actor=actor, reason=reason, jitter_s=jitter_s
        )

    def retry_collection(self, source_id: str, *, actor: str, reason: str) -> CollectionChange:
        return self.apply_collection_event(source_id, Event.RETRY, actor=actor, reason=reason)

    def reset_failed(self, source_id: str, *, actor: str, reason: str) -> CollectionChange:
        return self.apply_collection_event(source_id, Event.RESET, actor=actor, reason=reason)

    # ---- alert threshold ----

    def alert_min_confidence(self) -> float:
        """The minimum confidence for an alert: stored override, else 0.5."""
        stored = self._control.get_setting(SETTING_ALERT_MIN_CONFIDENCE)
        if stored is None:
            return DEFAULT_ALERT_MIN_CONFIDENCE
        try:
            return parse_confidence(stored)
        except ValidationError:
            raise StorageError("stored alert threshold is invalid") from None

    def set_alert_min_confidence(self, value: float, *, actor: str, reason: str) -> ThresholdChange:
        actor, reason = validate_actor(actor), validate_reason(reason)
        new = validate_confidence(value)
        with self._uow.transaction():
            now = self._clock.now()
            current = self.alert_min_confidence()
            if new == current:
                return ThresholdChange(new, changed=False)
            self._control.set_setting(SETTING_ALERT_MIN_CONFIDENCE, format_confidence(new))
            self._audit(
                now,
                EVENT_ALERT_THRESHOLD,
                None,
                action=_ACTION_SET,
                actor=actor,
                reason=reason,
                before=format_confidence(current),
                after=format_confidence(new),
            )
        return ThresholdChange(new, changed=True)
