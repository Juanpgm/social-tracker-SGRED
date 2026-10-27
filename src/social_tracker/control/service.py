"""Control service: the only writer of pipeline control state.

It owns the global analysis switch, the per-source analysis flags, the collection state machine
of every source and the alert threshold. Rules enforced here (the repositories only store):

- Every REAL change appends exactly one control event (actor, reason, from, to) inside the same
  `UnitOfWork` transaction as the state or setting write, so the audit log and the persisted
  state cannot diverge. A request that changes nothing (redundant switch, repeated threshold,
  disabling a disabled source) writes neither. An illegal transition raises `InvalidTransition`
  and writes nothing.
- Replaying the event log in order reproduces the persisted state: each event carries its
  `from` and `to` (and, for collection events, `failures` and `next_run_at`).
- Defaults are the safe ones: the system starts paused, analysis flags start off, a source with
  no state row is `disabled`.
- Nothing here ever publishes or sends anything outside the system (G-1).

Event kinds are stable identifiers (`EVENT_*` and `COLLECTION_EVENT_KINDS`). Details are compact
JSON objects with known keys only; `actor` and `reason` are validated free text and never echo
into error messages.

Per-source analysis flags are settings named by `source_analysis_key(source_id)`: "true" or
"false", absent meaning off. The alert threshold is the `alert_min_confidence` setting. There is
no environment override: the settings store is the single source of truth.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from social_tracker.control.domain import (
    DEFAULT_ALERT_MIN_CONFIDENCE,
    SETTING_ALERT_MIN_CONFIDENCE,
    SETTING_GLOBAL_ANALYSIS_ENABLED,
    CollectionState,
    ControlEvent,
    Event,
    GlobalState,
    SourceState,
    analysis_effective,
    apply_event,
    apply_global,
    format_confidence,
    parse_confidence,
    source_analysis_key,
    validate_actor,
    validate_confidence,
    validate_reason,
)
from social_tracker.control.ports import ControlRepository
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import InvalidTransition, StorageError, ValidationError
from social_tracker.shared.ids import is_valid_slug
from social_tracker.shared.uow import UnitOfWork
from social_tracker.sources.ports import SourceRepository

EVENT_GLOBAL_ANALYSIS = "global_analysis_changed"
EVENT_SOURCE_ANALYSIS = "source_analysis_changed"
EVENT_ALERT_THRESHOLD = "alert_threshold_changed"
COLLECTION_EVENT_KINDS: dict[Event, str] = {
    Event.ENABLE: "collection_enabled",
    Event.DISABLE: "collection_disabled",
    Event.START: "collection_started",
    Event.SUCCEED: "collection_succeeded",
    Event.FAIL: "collection_failed",
    Event.RETRY: "collection_retried",
    Event.RESET: "collection_reset",
}

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
        text = json.dumps(detail, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self._control.append_control_event(ControlEvent(at, kind, source_id, text))

    def _known_source(self, source_id: Any) -> str:
        """Validate the id and make sure the source exists (`NotFoundError` otherwise)."""
        self._sources.get(_check_source_id(source_id))
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
        now = self._clock.now()
        with self._uow.transaction():
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
        now = self._clock.now()
        with self._uow.transaction():
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
        """Analysis runs only when the system is active AND the source has analysis enabled."""
        flag = self.source_analysis_enabled(source_id)
        return analysis_effective(self._global_state(), flag)

    # ---- collection state machine ----

    def get_state(self, source_id: str) -> SourceState:
        self._known_source(source_id)
        return self._control.get_source_state(source_id) or SourceState(
            source_id, CollectionState.DISABLED
        )

    def apply_collection_event(
        self, source_id: str, event: Event, *, actor: str, reason: str, jitter_s: float = 0.0
    ) -> CollectionChange:
        """Apply one state machine event to a source, auditing it when it changes something."""
        actor, reason = validate_actor(actor), validate_reason(reason)
        domain_event = _coerce_event(event)
        now = self._clock.now()
        with self._uow.transaction():
            self._known_source(source_id)
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

    def enable_collection(self, source_id: str, *, actor: str, reason: str) -> CollectionChange:
        return self.apply_collection_event(source_id, Event.ENABLE, actor=actor, reason=reason)

    def disable_collection(self, source_id: str, *, actor: str, reason: str) -> CollectionChange:
        return self.apply_collection_event(source_id, Event.DISABLE, actor=actor, reason=reason)

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
        now = self._clock.now()
        with self._uow.transaction():
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
