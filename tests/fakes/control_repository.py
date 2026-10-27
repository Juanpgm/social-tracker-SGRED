"""In-memory `ControlRepository` used by service tests and as the contract reference."""

import threading

from social_tracker.control.domain import (
    DEFAULT_SETTINGS,
    ControlEvent,
    FetchStatus,
    SourceState,
    validate_setting,
    validate_setting_key,
    validate_setting_prefix,
)
from social_tracker.shared.errors import ValidationError


class InMemoryControlRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._settings: dict[str, str] = dict(DEFAULT_SETTINGS)
        self._states: dict[str, SourceState] = {}
        self._events: list[ControlEvent] = []
        self._fetch: dict[str, FetchStatus] = {}

    def get_setting(self, key: str) -> str | None:
        validate_setting_key(key)
        with self._lock:
            return self._settings.get(key)

    def set_setting(self, key: str, value: str) -> None:
        validate_setting(key, value)
        with self._lock:
            self._settings[key] = value

    def delete_setting(self, key: str) -> None:
        validate_setting_key(key)
        with self._lock:
            if key in DEFAULT_SETTINGS:
                self._settings[key] = DEFAULT_SETTINGS[key]
            else:
                self._settings.pop(key, None)

    def list_settings(self, prefix: str) -> dict[str, str]:
        validate_setting_prefix(prefix)
        with self._lock:
            return {k: v for k, v in sorted(self._settings.items()) if k.startswith(prefix)}

    def get_source_state(self, source_id: str) -> SourceState | None:
        with self._lock:
            return self._states.get(source_id)

    def list_source_states(self) -> list[SourceState]:
        with self._lock:
            return [self._states[key] for key in sorted(self._states)]

    def save_source_state(self, state: SourceState) -> None:
        with self._lock:
            self._states[state.source_id] = state

    def get_fetch_status(self, source_id: str) -> FetchStatus | None:
        with self._lock:
            return self._fetch.get(source_id)

    def list_fetch_statuses(self) -> list[FetchStatus]:
        with self._lock:
            return [self._fetch[key] for key in sorted(self._fetch)]

    def save_fetch_status(self, status: FetchStatus) -> None:
        if not isinstance(status, FetchStatus):
            raise ValidationError("status must be a FetchStatus")
        with self._lock:
            self._fetch[status.source_id] = status

    def delete_source_state(self, source_id: str) -> None:
        with self._lock:
            self._states.pop(source_id, None)
            self._fetch.pop(source_id, None)

    def append_control_event(self, event: ControlEvent) -> None:
        if not isinstance(event, ControlEvent):
            raise ValidationError("event must be a ControlEvent")
        with self._lock:
            self._events.append(event)

    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        with self._lock:
            return [e for e in self._events if source_id is None or e.source_id == source_id]
