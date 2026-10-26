"""Ports of the pipeline control module."""

from typing import Protocol, runtime_checkable

from social_tracker.control.domain import ControlEvent, SourceState


@runtime_checkable
class ControlRepository(Protocol):
    """Persistence port for settings, per-source runtime state and the control event log.

    Settings are string key/values; `global_analysis_enabled` defaults to "false" and unknown
    keys read as None. The event log is append-only: there is deliberately no way to update
    or delete an event.
    """

    def get_setting(self, key: str) -> str | None: ...

    def set_setting(self, key: str, value: str) -> None: ...

    def get_source_state(self, source_id: str) -> SourceState | None: ...

    def save_source_state(self, state: SourceState) -> None:
        """Insert or replace the state of `state.source_id`."""
        ...

    def append_control_event(self, event: ControlEvent) -> None: ...

    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        """Events in append order; filtered by source when `source_id` is given."""
        ...
