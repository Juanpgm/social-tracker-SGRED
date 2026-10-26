"""In-memory `ControlRepository` used by service tests and as the contract reference."""

import threading

from social_tracker.control.domain import (
    DEFAULT_SETTINGS,
    ControlEvent,
    SourceState,
    validate_setting,
    validate_setting_key,
)
from social_tracker.shared.errors import ValidationError


class InMemoryControlRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings: dict[str, str] = dict(DEFAULT_SETTINGS)
        self._states: dict[str, SourceState] = {}
        self._events: list[ControlEvent] = []

    def get_setting(self, key: str) -> str | None:
        validate_setting_key(key)
        with self._lock:
            return self._settings.get(key)

    def set_setting(self, key: str, value: str) -> None:
        validate_setting(key, value)
        with self._lock:
            self._settings[key] = value

    def get_source_state(self, source_id: str) -> SourceState | None:
        with self._lock:
            return self._states.get(source_id)

    def save_source_state(self, state: SourceState) -> None:
        with self._lock:
            self._states[state.source_id] = state

    def append_control_event(self, event: ControlEvent) -> None:
        if not isinstance(event, ControlEvent):
            raise ValidationError("event must be a ControlEvent")
        with self._lock:
            self._events.append(event)

    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        with self._lock:
            return [e for e in self._events if source_id is None or e.source_id == source_id]
