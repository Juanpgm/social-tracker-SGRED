"""Pure control state machine for source collection and the global switch. No I/O.

Every function takes the current state and an explicit `now`; nothing reads a clock or
mutates its input, so behaviour is fully deterministic (jitter is injected as a parameter).
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import InvalidTransition, ValidationError
from social_tracker.shared.ids import validate_slug

MAX_CONSECUTIVE_FAILURES = 5
BACKOFF_BASE_S = 60
BACKOFF_CAP_S = 3600
_MAX_EXPONENT = 12  # 60 * 2**12 is far above the cap; avoids huge integers for big inputs


class CollectionState(StrEnum):
    DISABLED = "disabled"
    IDLE = "idle"
    COLLECTING = "collecting"
    BACKOFF = "backoff"
    FAILED = "failed"


class Event(StrEnum):
    ENABLE = "enable"
    DISABLE = "disable"
    START = "start"  # idle -> collecting
    SUCCEED = "succeed"  # collecting -> idle
    FAIL = "fail"  # collecting -> backoff | failed
    RETRY = "retry"  # backoff -> idle, once next_run_at has been reached
    RESET = "reset"  # failed -> idle, operator action


class GlobalState(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"


def _coerce[E: StrEnum](enum_cls: type[E], value: Any, what: str) -> E:
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        raise InvalidTransition(f"unknown {what}: {repr(value)[:50]}") from None


@dataclass(frozen=True)
class SourceState:
    source_id: str
    state: CollectionState
    consecutive_failures: int = 0
    next_run_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_slug(self.source_id)
        object.__setattr__(self, "state", _coerce(CollectionState, self.state, "state"))
        failures = self.consecutive_failures
        if not isinstance(failures, int) or isinstance(failures, bool) or failures < 0:
            raise ValidationError("consecutive_failures must be a non-negative integer")
        if self.next_run_at is not None:
            object.__setattr__(self, "next_run_at", ensure_utc(self.next_run_at))
        if (self.state is CollectionState.BACKOFF) != (self.next_run_at is not None):
            raise ValidationError("next_run_at is required in backoff and forbidden otherwise")


@dataclass(frozen=True)
class TransitionResult:
    state: SourceState
    changed: bool  # False for an idempotent no-op


@dataclass(frozen=True)
class GlobalResult:
    state: GlobalState
    changed: bool


def _check_jitter(jitter_s: Any) -> float:
    if (
        isinstance(jitter_s, bool)
        or not isinstance(jitter_s, int | float)
        or not math.isfinite(jitter_s)
        or jitter_s < 0
    ):
        raise ValidationError("jitter_s must be a finite number >= 0")
    return float(jitter_s)


def backoff_delay_s(failures_before: int, jitter_s: float = 0.0) -> float:
    """Seconds to wait after a failure: 60 * 2**n capped at 3600, plus injected jitter."""
    if (
        not isinstance(failures_before, int)
        or isinstance(failures_before, bool)
        or failures_before < 0
    ):
        raise ValidationError("failure count must be a non-negative integer")
    jitter = _check_jitter(jitter_s)
    exponent = min(failures_before, _MAX_EXPONENT)
    return float(min(BACKOFF_BASE_S * 2**exponent, BACKOFF_CAP_S)) + jitter


def _noop(current: SourceState) -> TransitionResult:
    return TransitionResult(current, changed=False)


def _moved(current: SourceState, state: CollectionState, **fields: Any) -> TransitionResult:
    new = SourceState(
        current.source_id,
        state,
        fields.get("consecutive_failures", current.consecutive_failures),
        fields.get("next_run_at"),
    )
    return TransitionResult(new, changed=True)


def apply_event(
    current: SourceState, event: Event | str, now: datetime, *, jitter_s: float = 0.0
) -> TransitionResult:
    """Apply `event` to `current`. Redundant requests are no-ops; illegal ones raise."""
    now = ensure_utc(now)
    event = _coerce(Event, event, "event")
    state = current.state
    S = CollectionState

    if event is Event.ENABLE:
        if state is S.DISABLED:
            return _moved(current, S.IDLE, consecutive_failures=0)
        return _noop(current)
    if event is Event.DISABLE:
        if state is S.DISABLED:
            return _noop(current)
        return _moved(current, S.DISABLED, consecutive_failures=0)
    if event is Event.START and state is S.IDLE:
        return _moved(current, S.COLLECTING)
    if event is Event.SUCCEED and state is S.COLLECTING:
        return _moved(current, S.IDLE, consecutive_failures=0)
    if event is Event.FAIL and state is S.COLLECTING:
        failures = current.consecutive_failures + 1
        if failures >= MAX_CONSECUTIVE_FAILURES:
            return _moved(current, S.FAILED, consecutive_failures=failures)
        delay = backoff_delay_s(current.consecutive_failures, jitter_s)
        return _moved(
            current,
            S.BACKOFF,
            consecutive_failures=failures,
            next_run_at=now + timedelta(seconds=delay),
        )
    if event is Event.RETRY and state is S.BACKOFF:
        if current.next_run_at is not None and now >= current.next_run_at:
            return _moved(current, S.IDLE)
        raise InvalidTransition("cannot leave backoff before next_run_at")
    if event is Event.RESET and state is S.FAILED:
        return _moved(current, S.IDLE, consecutive_failures=0)
    raise InvalidTransition(f"cannot apply {event.value!r} while {state.value}")


def apply_global(current: GlobalState | str, target: GlobalState | str) -> GlobalResult:
    """Pause or resume the whole system. Requesting the current state is a no-op."""
    current_state = _coerce(GlobalState, current, "global state")
    target_state = _coerce(GlobalState, target, "global state")
    return GlobalResult(target_state, changed=target_state is not current_state)


def analysis_effective(global_state: GlobalState | str, source_analysis_enabled: bool) -> bool:
    """Analysis runs only when the system is active AND the source has analysis enabled."""
    if not isinstance(source_analysis_enabled, bool):
        raise ValidationError("source_analysis_enabled must be a bool")
    return _coerce(GlobalState, global_state, "global state") is GlobalState.ACTIVE and (
        source_analysis_enabled
    )


# ---- control events and settings -------------------------------------------------------

SETTING_GLOBAL_ANALYSIS_ENABLED = "global_analysis_enabled"
DEFAULT_SETTINGS: dict[str, str] = {SETTING_GLOBAL_ANALYSIS_ENABLED: "false"}
MAX_SETTING_KEY = 100
MAX_SETTING_VALUE = 1000
MAX_EVENT_KIND = 64
MAX_EVENT_DETAIL = 2000


def validate_setting(key: Any, value: Any) -> tuple[str, str]:
    validate_setting_key(key)
    if not isinstance(value, str) or len(value) > MAX_SETTING_VALUE:
        raise ValidationError(f"setting value must be a string up to {MAX_SETTING_VALUE} chars")
    return key, value


def validate_setting_key(key: Any) -> str:
    if not isinstance(key, str) or not key.strip() or len(key) > MAX_SETTING_KEY:
        raise ValidationError(
            f"setting key must be a non-blank string up to {MAX_SETTING_KEY} chars"
        )
    return key


@dataclass(frozen=True)
class ControlEvent:
    """Immutable audit record of a control action (append-only log entry)."""

    at: datetime
    kind: str
    source_id: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", ensure_utc(self.at))
        if (
            not isinstance(self.kind, str)
            or not self.kind.strip()
            or len(self.kind) > MAX_EVENT_KIND
        ):
            raise ValidationError(f"kind must be a non-blank string up to {MAX_EVENT_KIND} chars")
        if self.source_id is not None:
            validate_slug(self.source_id)
        if not isinstance(self.detail, str) or len(self.detail) > MAX_EVENT_DETAIL:
            raise ValidationError(f"detail must be a string up to {MAX_EVENT_DETAIL} chars")
