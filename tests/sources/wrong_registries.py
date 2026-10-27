"""Deliberately wrong registries. The contract must fail each one (see the meta-test)."""

import json
from collections.abc import Mapping
from typing import Any

import yaml

from social_tracker.control.domain import ControlEvent
from social_tracker.shared.errors import ConflictError, ValidationError
from social_tracker.sources.domain import Source, parse_source
from social_tracker.sources.service import (
    SOURCE_ADDED,
    SOURCE_ENABLED,
    ImportReport,
    SourceChange,
    SourceRegistry,
)


def _event(registry: SourceRegistry, kind: str, source_id: str | None, detail: dict[str, Any]):
    registry._control.append_control_event(  # noqa: SLF001
        ControlEvent(registry._clock.now(), kind, source_id, json.dumps(detail))  # noqa: SLF001
    )


class ForgetsTheAuditEventOnAdd(SourceRegistry):
    def add(self, data: Mapping[str, Any]) -> Source:
        now = self._clock.now()
        source = parse_source({**data, "created_at": now, "updated_at": now})
        with self._uow.transaction():
            self._ensure_free(source)
            self._sources.add(source)
        return source


class WritesTheEventOutsideTheTransaction(SourceRegistry):
    def add(self, data: Mapping[str, Any]) -> Source:
        now = self._clock.now()
        source = parse_source({**data, "created_at": now, "updated_at": now})
        with self._uow.transaction():
            self._ensure_free(source)
            self._sources.add(source)
        self._audit(now, SOURCE_ADDED, source.id, {"type": source.type})
        return source


class AddWithoutAnyTransaction(SourceRegistry):
    def add(self, data: Mapping[str, Any]) -> Source:
        now = self._clock.now()
        source = parse_source({**data, "created_at": now, "updated_at": now})
        self._ensure_free(source)
        self._sources.add(source)
        self._audit(now, SOURCE_ADDED, source.id, {"type": source.type})
        return source


class EchoesTheNameInConflicts(SourceRegistry):
    def _ensure_free(self, source: Source) -> None:
        try:
            super()._ensure_free(source)
        except ConflictError:
            raise ConflictError(f"source name already exists: {source.name}") from None


class EnableAlwaysWritesAnEvent(SourceRegistry):
    def _set_enabled(self, source_id: str, *, enabled: bool) -> SourceChange:
        now = self._clock.now()
        with self._uow.transaction():
            current = self._require(source_id)
            updated = current.with_changes({"enabled": enabled, "updated_at": now})
            self._sources.update(updated)
            self._audit(now, SOURCE_ENABLED, updated.id, {})
        return SourceChange(updated, changed=current.enabled is not enabled)


class EditAlwaysStampsUpdatedAt(SourceRegistry):
    def edit(self, source_id: str, changes: Mapping[str, Any]) -> SourceChange:
        now = self._clock.now()
        with self._uow.transaction():
            current = self._require(source_id)
            updated = current.with_changes({**changes, "updated_at": now})
            self._sources.update(updated)
        return SourceChange(updated, changed=updated != current)


class EditAllowsImmutableFields(SourceRegistry):
    def edit(self, source_id: str, changes: Mapping[str, Any]) -> SourceChange:
        now = self._clock.now()
        with self._uow.transaction():
            current = self._require(source_id)
            updated = current.with_changes({**changes, "updated_at": now})
            self._sources.update(updated)
            self._audit(now, "source_edited", updated.id, {"changed": []})
        return SourceChange(updated, changed=True)


class EditIgnoresNameOwnership(SourceRegistry):
    def _ensure_name_available(self, source: Source) -> None:
        return None


class DeleteLeavesTheStateRow(SourceRegistry):
    def delete(self, source_id: str) -> None:
        now = self._clock.now()
        with self._uow.transaction():
            current = self._require(source_id)
            self._sources.delete(current.id)
            self._audit(now, "source_deleted", current.id, {"type": current.type, "state": None})


class DeleteWithoutTransaction(SourceRegistry):
    def delete(self, source_id: str) -> None:
        now = self._clock.now()
        current = self._require(source_id)
        state = self._control.get_source_state(current.id)
        self._sources.delete(current.id)
        self._control.delete_source_state(current.id)
        detail = {"type": current.type, "state": state.state.value if state else None}
        self._audit(now, "source_deleted", current.id, detail)


class DeleteForgetsTheEvent(SourceRegistry):
    def delete(self, source_id: str) -> None:
        with self._uow.transaction():
            current = self._require(source_id)
            self._sources.delete(current.id)
            self._control.delete_source_state(current.id)


class DeleteKeepsTheAnalysisFlag(SourceRegistry):
    def delete(self, source_id: str) -> None:
        now = self._clock.now()
        with self._uow.transaction():
            current = self._require(source_id)
            state = self._control.get_source_state(current.id)
            self._sources.delete(current.id)
            self._control.delete_source_state(current.id)
            detail = {"type": current.type, "state": state.state.value if state else None}
            self._audit(now, "source_deleted", current.id, detail)


class ImportAppliesEntriesOneByOne(SourceRegistry):
    """No plan, no transaction: entries are written as they are read (partial application)."""

    def import_sources(self, text: str, *, dry_run: bool = False) -> ImportReport:
        from social_tracker.sources.yaml_io import load_sources  # noqa: PLC0415

        added = []
        for entry in load_sources(text, default_timestamp=self._clock.now()):
            self._sources.add(entry)
            self._audit(self._clock.now(), SOURCE_ADDED, entry.id, {"type": entry.type})
            added.append(entry.id)
        return ImportReport(tuple(added), (), (), dry_run)


class ImportSkipsInvalidEntries(SourceRegistry):
    """Parses leniently: invalid entries are silently dropped instead of rejecting the file."""

    def import_sources(self, text: str, *, dry_run: bool = False) -> ImportReport:
        document = yaml.safe_load(text) or {}
        added = []
        with self._uow.transaction():
            for raw in document.get("sources", []):
                try:
                    entry = parse_source(
                        {
                            "created_at": self._clock.now(),
                            "updated_at": self._clock.now(),
                            **raw,
                        }
                    )
                except (ValidationError, TypeError):
                    continue
                self._sources.add(entry)
                added.append(entry.id)
        return ImportReport(tuple(added), (), (), dry_run)


class ImportIgnoresTypeImmutability(SourceRegistry):
    def _plan_import(self, incoming: list[Source], now: Any):
        stored = {s.id: s for s in self._sources.list()}
        safe = [e for e in incoming if e.id not in stored or e.type == stored[e.id].type]
        return super()._plan_import(safe, now)


class ImportAlwaysRewrites(SourceRegistry):
    """Not idempotent: known entries are re-stamped and re-audited on every import."""

    def import_sources(self, text: str, *, dry_run: bool = False) -> ImportReport:
        from social_tracker.sources.yaml_io import load_sources  # noqa: PLC0415

        now = self._clock.now()
        incoming = load_sources(text, default_timestamp=now)
        with self._uow.transaction():
            for entry in incoming:
                try:
                    self._sources.get(entry.id)
                except Exception:  # noqa: BLE001
                    self._sources.add(entry)
                else:
                    self._sources.update(entry.with_changes({"updated_at": now}))
                self._audit(now, "source_edited", entry.id, {"changed": [], "via": "import"})
        return ImportReport((), tuple(e.id for e in incoming), (), dry_run)


class DryRunWrites(SourceRegistry):
    def import_sources(self, text: str, *, dry_run: bool = False) -> ImportReport:
        report = super().import_sources(text, dry_run=False)
        return ImportReport(report.added, report.updated, report.unchanged, dry_run)


class ImportWithoutSummaryEvent(SourceRegistry):
    def _audit(self, at, kind, source_id, detail):  # type: ignore[override]
        if kind == "sources_imported":
            return
        super()._audit(at, kind, source_id, detail)


class ImportEventsOutsideTheTransaction(SourceRegistry):
    """Writes the sources in the transaction but the audit events after it."""

    def import_sources(self, text: str, *, dry_run: bool = False) -> ImportReport:
        from social_tracker.sources.yaml_io import load_sources  # noqa: PLC0415

        now = self._clock.now()
        incoming = load_sources(text, default_timestamp=now)
        with self._uow.transaction():
            plan = self._plan_import(incoming, now)
            if not dry_run:
                for op in plan.operations:
                    (self._sources.add if op.changed is None else self._sources.update)(op.source)
        if not dry_run:
            for op in plan.operations:
                self._audit(now, SOURCE_ADDED, op.source.id, {"type": op.source.type})
        return ImportReport(
            tuple(s.id for s in plan.added),
            tuple(s.id for s in plan.updated),
            plan.unchanged,
            dry_run,
        )


WRONG_REGISTRIES = {
    "forgets_event_on_add": (ForgetsTheAuditEventOnAdd, "test_add_writes_exactly_one"),
    "event_outside_transaction": (
        WritesTheEventOutsideTheTransaction,
        "test_a_failing_audit_event_rolls_back_the_new_source",
    ),
    "add_without_transaction": (
        AddWithoutAnyTransaction,
        "test_a_failing_audit_event_rolls_back_the_new_source",
    ),
    "echoes_names": (EchoesTheNameInConflicts, "test_conflict_messages_do_not_echo_names"),
    "enable_always_audits": (
        EnableAlwaysWritesAnEvent,
        "test_redundant_enable_and_disable_are_noops",
    ),
    "edit_always_stamps": (
        EditAlwaysStampsUpdatedAt,
        "test_edit_with_identical_values_is_a_noop",
    ),
    "edit_allows_immutable": (EditAllowsImmutableFields, "test_edit_refuses_immutable"),
    "edit_ignores_name_owner": (
        EditIgnoresNameOwnership,
        "test_edit_rename_to_another_sources_name_conflicts",
    ),
    "delete_leaves_state": (
        DeleteLeavesTheStateRow,
        "test_delete_removes_the_source_and_its_state",
    ),
    "delete_without_tx": (DeleteWithoutTransaction, "test_a_failing_state_cleanup_rolls_back"),
    "delete_forgets_event": (DeleteForgetsTheEvent, "test_delete_removes_the_source_and_its_state"),
    "delete_keeps_flag": (DeleteKeepsTheAnalysisFlag, "test_delete_switches_off_the_per_source"),
    "import_one_by_one": (
        ImportAppliesEntriesOneByOne,
        "test_import_is_atomic_and_reports_every_conflict_together",
    ),
    "import_skips_invalid": (ImportSkipsInvalidEntries, "test_import_is_atomic_when_a_later_entry"),
    "import_ignores_type": (
        ImportIgnoresTypeImmutability,
        "test_import_is_atomic_and_reports_every_conflict_together",
    ),
    "import_not_idempotent": (ImportAlwaysRewrites, "test_import_is_idempotent"),
    "dry_run_writes": (DryRunWrites, "test_import_dry_run_classifies_without_writing"),
    "import_no_summary": (ImportWithoutSummaryEvent, "test_import_into_an_empty_registry_adds"),
    "import_events_outside_tx": (
        ImportEventsOutsideTheTransaction,
        "test_import_rolls_back_every_write_when_a_later_audit_event_fails",
    ),
}
