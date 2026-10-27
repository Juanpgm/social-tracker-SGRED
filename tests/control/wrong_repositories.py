"""Deliberately wrong `ControlRepository` implementations.

Each one breaks exactly one rule of the port. `test_contract_catches_wrong_repositories`
asserts the shared contract fails for every one of them, so the contract itself is tested:
a contract that cannot fail protects nothing.
"""

from social_tracker.control.domain import ControlEvent
from social_tracker.shared.errors import NotFoundError
from tests.fakes.control_repository import InMemoryControlRepository


class EmptyIdMeansNoFilter(InMemoryControlRepository):
    """`list_control_events("")` wrongly behaves like `list_control_events()`."""

    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        return super().list_control_events(source_id or None)


class SortedByTimestamp(InMemoryControlRepository):
    """Returns events ordered by `at` instead of by append order."""

    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        return sorted(super().list_control_events(source_id), key=lambda e: e.at)


class SortedByKind(InMemoryControlRepository):
    """Returns events ordered by `kind` instead of by append order."""

    def list_control_events(self, source_id: str | None = None) -> list[ControlEvent]:
        return sorted(super().list_control_events(source_id), key=lambda e: e.kind)


class ExposesPurge(InMemoryControlRepository):
    """Adds a public way to destroy the log, which the append-only port forbids."""

    def purge_events(self) -> None:
        self._events.clear()


class ExposesTruncate(InMemoryControlRepository):
    def truncate_log(self, keep: int) -> None:
        del self._events[:-keep]


class KeepsOnlyTheLast30(InMemoryControlRepository):
    """A ring buffer: silently drops old events."""

    def append_control_event(self, event: ControlEvent) -> None:
        super().append_control_event(event)
        del self._events[:-30]


class RewritesOnAppend(InMemoryControlRepository):
    """Appending also rewrites an earlier event (here: its detail), which must never happen."""

    def append_control_event(self, event: ControlEvent) -> None:
        super().append_control_event(event)
        if len(self._events) > 1:
            old = self._events[0]
            self._events[0] = ControlEvent(old.at, old.kind, old.source_id, "rewritten")


class SharesSettingsAcrossInstances(InMemoryControlRepository):
    """Settings live in a class-level dict, leaking between repositories."""

    _shared: dict[str, str] = {"global_analysis_enabled": "false"}

    def __init__(self) -> None:
        super().__init__()
        self._settings = SharesSettingsAcrossInstances._shared


class SkipsSettingValidation(InMemoryControlRepository):
    """Stores whatever it is given, even values the port says are invalid."""

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self._settings[key] = value  # type: ignore[assignment]

    def get_setting(self, key: str) -> str | None:
        with self._lock:
            return self._settings.get(key)


class DeleteSettingNotIdempotent(InMemoryControlRepository):
    """Deleting a key that is not stored raises."""

    def delete_setting(self, key: str) -> None:
        with self._lock:
            if key not in self._settings:
                raise NotFoundError("no such setting")
            del self._settings[key]


class DeleteSettingDropsTheDefault(InMemoryControlRepository):
    """Deleting a key that has a default makes it vanish instead of reading as the default."""

    def delete_setting(self, key: str) -> None:
        with self._lock:
            self._settings.pop(key, None)


class ListSettingsBySubstring(InMemoryControlRepository):
    """Matches the prefix anywhere in the key."""

    def list_settings(self, prefix: str) -> dict[str, str]:
        with self._lock:
            return {k: v for k, v in sorted(self._settings.items()) if prefix in k}


WRONG_REPOSITORIES: dict[str, type[InMemoryControlRepository]] = {
    "empty_id_means_no_filter": EmptyIdMeansNoFilter,
    "sorted_by_timestamp": SortedByTimestamp,
    "sorted_by_kind": SortedByKind,
    "exposes_purge": ExposesPurge,
    "exposes_truncate": ExposesTruncate,
    "keeps_only_the_last_30": KeepsOnlyTheLast30,
    "rewrites_on_append": RewritesOnAppend,
    "shares_settings_across_instances": SharesSettingsAcrossInstances,
    "skips_setting_validation": SkipsSettingValidation,
    "delete_setting_not_idempotent": DeleteSettingNotIdempotent,
    "delete_setting_drops_the_default": DeleteSettingDropsTheDefault,
    "list_settings_by_substring": ListSettingsBySubstring,
}
