"""Ports of the pipeline control module."""

from typing import Protocol, runtime_checkable

from social_tracker.control.domain import ControlEvent, SourceState


@runtime_checkable
class ControlRepository(Protocol):
    """Persistence port for settings, per-source runtime state and the control event log.

    Settings are string key/values; `global_analysis_enabled` defaults to "false" and unknown
    keys read as None.

    Validation is part of the contract and lives in the repository, not in the database:
    every implementation (SQLite included) MUST apply `validate_setting` on `set_setting` and
    `validate_setting_key` on `get_setting`, raising `ValidationError` and leaving the stored
    value untouched. `global_analysis_enabled` accepts only "true" or "false".

    `save_source_state` does NOT check that the source exists: sources and control state are
    separate stores with no cross-repository foreign key.

    The event log is append-only and listed in append order (never sorted by timestamp or
    kind). There is deliberately no way to update or delete an event, and implementations must
    not expose one under any name.
    """

    def get_setting(self, key: str) -> str | None: ...

    def set_setting(self, key: str, value: str) -> None: ...

    def get_source_state(self, source_id: str) -> SourceState | None: ...

    def save_source_state(self, state: SourceState) -> None:
        """Insert or replace the state of `state.source_id`."""
        ...

    def delete_source_state(self, source_id: str) -> None:
        """Forget the runtime state of `source_id`; a no-op if there is none (idempotent).

        Called when a source is deleted. Repositories stay independent (no cross-repository
        foreign key), so the application service removes the state in the same joined
        `Database.transaction()` as the source. Control events are NOT touched: they are the
        audit trail and outlive the source (SR-9).
        """
        ...

    def append_control_event(self, event: ControlEvent) -> None: ...

    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        """Events in append order. `None` means no filter; any other value, including the empty
        string, filters by exactly that source id."""
        ...
