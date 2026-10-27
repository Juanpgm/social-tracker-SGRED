"""Deliberately wrong control services. The contract must fail each one (see the meta-test)."""

import time
from typing import Any

from social_tracker.control.domain import (
    SETTING_ALERT_MIN_CONFIDENCE,
    SETTING_GLOBAL_ANALYSIS_ENABLED,
    CollectionState,
    Event,
    GlobalState,
    SourceState,
    apply_event,
    format_confidence,
    source_analysis_key,
    validate_actor,
    validate_confidence,
    validate_reason,
)
from social_tracker.control.service import (
    COLLECTION_EVENT_KINDS,
    EVENT_ALERT_THRESHOLD,
    EVENT_GLOBAL_ANALYSIS,
    CollectionChange,
    ControlService,
    FlagChange,
    ThresholdChange,
)
from social_tracker.shared.errors import InvalidTransition


class StateThenEventOutsideTransaction(ControlService):
    """Writes the state in a transaction and the audit event after it."""

    def apply_collection_event(
        self, source_id: str, event: Event, *, actor: str, reason: str, jitter_s: float = 0.0
    ) -> CollectionChange:
        domain_event = Event(event)
        now = self._clock.now()
        with self._uow.transaction():
            self._known_source(source_id)
            current = self._control.get_source_state(source_id) or SourceState(
                source_id, CollectionState.DISABLED
            )
            result = apply_event(current, domain_event, now, jitter_s=jitter_s)
            if not result.changed:
                return CollectionChange(current, changed=False)
            self._control.save_source_state(result.state)
        self._audit(
            now,
            COLLECTION_EVENT_KINDS[domain_event],
            source_id,
            action=domain_event.value,
            actor=actor,
            reason=reason,
            before=current.state.value,
            after=result.state.state.value,
            failures=result.state.consecutive_failures,
            next_run_at=None,
        )
        return CollectionChange(result.state, changed=True)


class GlobalWithoutTransaction(ControlService):
    def set_global_analysis(self, on: bool, *, actor: str, reason: str) -> FlagChange:
        now = self._clock.now()
        current = self._global_state()
        target = GlobalState.ACTIVE if on else GlobalState.PAUSED
        if current is target:
            return FlagChange(on, changed=False)
        self._control.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true" if on else "false")
        self._audit(
            now,
            EVENT_GLOBAL_ANALYSIS,
            None,
            action="set",
            actor=actor,
            reason=reason,
            before=current.value,
            after=target.value,
        )
        return FlagChange(on, changed=True)


class GlobalAlwaysAudits(ControlService):
    def set_global_analysis(self, on: bool, *, actor: str, reason: str) -> FlagChange:
        actor, reason = validate_actor(actor), validate_reason(reason)
        now = self._clock.now()
        target = GlobalState.ACTIVE if on else GlobalState.PAUSED
        with self._uow.transaction():
            current = self._global_state()
            self._control.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true" if on else "false")
            self._audit(
                now,
                EVENT_GLOBAL_ANALYSIS,
                None,
                action="set",
                actor=actor,
                reason=reason,
                before=current.value,
                after=target.value,
            )
        return FlagChange(on, changed=current is not target)


class GlobalReadsOutsideTheTransaction(ControlService):
    """Read-modify-write race: the decision is taken before the lock is held."""

    def set_global_analysis(self, on: bool, *, actor: str, reason: str) -> FlagChange:
        actor, reason = validate_actor(actor), validate_reason(reason)
        now = self._clock.now()
        current = self._global_state()
        target = GlobalState.ACTIVE if on else GlobalState.PAUSED
        if current is target:
            return FlagChange(on, changed=False)
        time.sleep(0.02)
        with self._uow.transaction():
            self._control.set_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, "true" if on else "false")
            self._audit(
                now,
                EVENT_GLOBAL_ANALYSIS,
                None,
                action="set",
                actor=actor,
                reason=reason,
                before=current.value,
                after=target.value,
            )
        return FlagChange(on, changed=True)


class CollectionReadsOutsideTheTransaction(ControlService):
    def apply_collection_event(
        self, source_id: str, event: Event, *, actor: str, reason: str, jitter_s: float = 0.0
    ) -> CollectionChange:
        domain_event = Event(event)
        now = self._clock.now()
        self._known_source(source_id)
        current = self._control.get_source_state(source_id) or SourceState(
            source_id, CollectionState.DISABLED
        )
        result = apply_event(current, domain_event, now, jitter_s=jitter_s)
        if not result.changed:
            return CollectionChange(current, changed=False)
        time.sleep(0.02)
        with self._uow.transaction():
            self._control.save_source_state(result.state)
            self._audit(
                now,
                COLLECTION_EVENT_KINDS[domain_event],
                source_id,
                action=domain_event.value,
                actor=actor,
                reason=reason,
                before=current.state.value,
                after=result.state.state.value,
                failures=result.state.consecutive_failures,
                next_run_at=(
                    result.state.next_run_at.isoformat() if result.state.next_run_at else None
                ),
            )
        return CollectionChange(result.state, changed=True)


class SkipsReasonValidationOnNoOps(ControlService):
    """Validates actor and reason only when something actually changes."""

    def set_global_analysis(self, on: bool, *, actor: str, reason: str) -> FlagChange:
        if self.global_analysis_enabled() is on:
            return FlagChange(on, changed=False)
        return super().set_global_analysis(on, actor=actor, reason=reason)


class EffectiveIsGlobalOrSource(ControlService):
    def effective_analysis(self, source_id: str) -> bool:
        return self.global_analysis_enabled() or self.source_analysis_enabled(source_id)


class EffectiveIgnoresTheGlobalSwitch(ControlService):
    def effective_analysis(self, source_id: str) -> bool:
        return self.source_analysis_enabled(source_id)


class NeverChecksThatTheSourceExists(ControlService):
    def _known_source(self, source_id: Any) -> str:
        return source_id


class WrongDefaultThreshold(ControlService):
    def alert_min_confidence(self) -> float:
        stored = self._control.get_setting(SETTING_ALERT_MIN_CONFIDENCE)
        return 0.6 if stored in (None, "0.5") else super().alert_min_confidence()


class ThresholdIsClamped(ControlService):
    def set_alert_min_confidence(self, value: float, *, actor: str, reason: str) -> ThresholdChange:
        try:
            validate_confidence(value)
        except Exception:  # noqa: BLE001
            value = min(max(float(value), 0.0), 1.0) if isinstance(value, int | float) else 0.5
        return super().set_alert_min_confidence(value, actor=actor, reason=reason)


class ThresholdAlwaysAudits(ControlService):
    def set_alert_min_confidence(self, value: float, *, actor: str, reason: str) -> ThresholdChange:
        actor, reason = validate_actor(actor), validate_reason(reason)
        new = validate_confidence(value)
        now = self._clock.now()
        with self._uow.transaction():
            current = self.alert_min_confidence()
            self._control.set_setting(SETTING_ALERT_MIN_CONFIDENCE, format_confidence(new))
            self._audit(
                now,
                EVENT_ALERT_THRESHOLD,
                None,
                action="set",
                actor=actor,
                reason=reason,
                before=format_confidence(current),
                after=format_confidence(new),
            )
        return ThresholdChange(new, changed=new != current)


class InvalidTransitionsAreSwallowed(ControlService):
    """Turns an illegal transition into a quiet no-op instead of raising."""

    def apply_collection_event(
        self, source_id: str, event: Event, *, actor: str, reason: str, jitter_s: float = 0.0
    ) -> CollectionChange:
        try:
            return super().apply_collection_event(
                source_id, event, actor=actor, reason=reason, jitter_s=jitter_s
            )
        except InvalidTransition:
            return CollectionChange(self.get_state(source_id), changed=False)


class ResetFailedFromAnywhere(ControlService):
    def reset_failed(self, source_id: str, *, actor: str, reason: str) -> CollectionChange:
        self._known_source(source_id)
        with self._uow.transaction():
            new = SourceState(source_id, CollectionState.IDLE)
            self._control.save_source_state(new)
            self._audit(
                self._clock.now(),
                COLLECTION_EVENT_KINDS[Event.RESET],
                source_id,
                action="reset",
                actor=actor,
                reason=reason,
                before="failed",
                after="idle",
                failures=0,
                next_run_at=None,
            )
        return CollectionChange(new, changed=True)


class AuditsTheNewStateAsFrom(ControlService):
    def _audit(self, at, kind, source_id, *, before, after, **kwargs):  # type: ignore[override]
        super()._audit(at, kind, source_id, before=after, after=after, **kwargs)


class InvertedGlobalFlag(ControlService):
    def global_analysis_enabled(self) -> bool:
        return self._control.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) != "true"


class SourceFlagSharedAcrossSources(ControlService):
    """Stores every per-source flag under one key."""

    def set_source_analysis(
        self, source_id: str, on: bool, *, actor: str, reason: str
    ) -> FlagChange:
        self._known_source(source_id)
        self._control.set_setting(source_analysis_key("src_shared"), "true" if on else "false")
        return FlagChange(on, changed=True)


WRONG_SERVICES = {
    "state_then_event_outside_tx": (
        StateThenEventOutsideTransaction,
        "test_a_failing_event_append_leaves_the_collection_state_unchanged",
    ),
    "global_without_tx": (
        GlobalWithoutTransaction,
        "test_a_failing_event_append_leaves_the_setting_unchanged",
    ),
    "global_always_audits": (GlobalAlwaysAudits, "test_a_redundant_request_is_a_noop"),
    "global_read_outside_tx": (
        GlobalReadsOutsideTheTransaction,
        "test_concurrent_identical_requests_produce_exactly_one_change",
    ),
    "collection_read_outside_tx": (
        CollectionReadsOutsideTheTransaction,
        "test_many_threads_starting_one_source_let_exactly_one_win",
    ),
    "skips_validation_on_noops": (
        SkipsReasonValidationOnNoOps,
        "test_a_redundant_request_with_an_invalid",
    ),
    "effective_or": (EffectiveIsGlobalOrSource, "test_effective_analysis_needs_the_global"),
    "effective_ignores_global": (
        EffectiveIgnoresTheGlobalSwitch,
        "test_effective_analysis_needs_the_global",
    ),
    "unknown_source_unchecked": (
        NeverChecksThatTheSourceExists,
        "test_unknown_sources_are_not_found_and_nothing_is_written",
    ),
    "default_threshold_06": (WrongDefaultThreshold, "test_the_default_threshold_is_one_half"),
    "threshold_clamped": (ThresholdIsClamped, "test_invalid_thresholds_are_rejected"),
    "threshold_always_audits": (
        ThresholdAlwaysAudits,
        "test_setting_the_default_value_explicitly",
    ),
    "invalid_transition_swallowed": (
        InvalidTransitionsAreSwallowed,
        "test_every_operation_in_every_state_follows_the_state_machine",
    ),
    "reset_from_anywhere": (
        ResetFailedFromAnywhere,
        "test_reset_failed_works_only_on_failed_sources",
    ),
    "audits_new_state_as_from": (
        AuditsTheNewStateAsFrom,
        "test_turning_the_global_switch_on_writes_the_setting",
    ),
    "inverted_global_flag": (InvertedGlobalFlag, "test_a_fresh_system_is_paused_with_analysis_off"),
    "shared_source_flag_key": (
        SourceFlagSharedAcrossSources,
        "test_the_source_flag_is_stored_per_source_with_an_exact_event",
    ),
}
