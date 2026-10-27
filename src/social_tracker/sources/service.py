"""Source registry: the application service that owns every change to the set of sources.

Rules enforced here (the repositories only store):
- Every real change writes ONE control event inside the same `UnitOfWork` transaction as the
  repository write, so a source change and its audit record commit or fail together. A request
  that changes nothing (redundant enable, edit with identical values, import of known data)
  writes neither.
- Ids and types are immutable; names are unique case-insensitively (`normalized_name`).
- Nothing is built in: a fresh registry is empty.
- Error messages never echo user-supplied names or keys; conflicts are described by field.

Event kinds are stable identifiers (see the `SOURCE_*` constants). Event details are compact
JSON objects with known keys only; they never contain URLs, secret references or free text.

Deleting a source removes its runtime state row and switches its per-source analysis flag off,
but the audit events stay (SR-9). Deleting a source that is currently `collecting` is allowed:
the running collection is not interrupted by this service, and the detail of the
`source_deleted` event records the state it had.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from social_tracker.control.domain import ControlEvent, source_analysis_key
from social_tracker.control.ports import ControlRepository
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import ConflictError, NotFoundError, ValidationError
from social_tracker.shared.ids import is_valid_slug
from social_tracker.shared.uow import UnitOfWork
from social_tracker.sources.domain import Source, normalize_name, parse_source
from social_tracker.sources.ports import SourceRepository
from social_tracker.sources.yaml_io import SourceFileProblem, dump_sources, load_sources

SOURCE_ADDED = "source_added"
SOURCE_EDITED = "source_edited"
SOURCE_ENABLED = "source_enabled"
SOURCE_DISABLED = "source_disabled"
SOURCE_DELETED = "source_deleted"
SOURCES_IMPORTED = "sources_imported"

_CODE_CONFLICT = "conflict"
_IMMUTABLE_FIELDS = ("id", "type", "enabled", "created_at", "updated_at")
_VIA_IMPORT = {"via": "import"}
_ID_CONFLICT = "a source with this id already exists"
_NAME_CONFLICT = "a source with this name already exists"
_UNKNOWN_FIELD = "unknown field (not part of this source type)"


@dataclass(frozen=True)
class SourceChange:
    """Outcome of an edit/enable/disable; `changed` is False for an idempotent no-op."""

    source: Source
    changed: bool


@dataclass(frozen=True)
class ImportReport:
    """Source ids by outcome, in file order. `dry_run` reports without having written."""

    added: tuple[str, ...]
    updated: tuple[str, ...]
    unchanged: tuple[str, ...]
    dry_run: bool


class ImportConflictError(ConflictError):
    """The file is valid but cannot be applied; `problems` lists every conflict found."""

    def __init__(self, problems: tuple[SourceFileProblem, ...]) -> None:
        super().__init__(f"import rejected: {len(problems)} conflict(s) with existing sources")
        self.problems = problems


@dataclass(frozen=True)
class _Operation:
    """One write of an import, in file order. `changed` is None for an add."""

    source: Source
    changed: tuple[str, ...] | None


@dataclass(frozen=True)
class _Plan:
    operations: tuple[_Operation, ...]
    unchanged: tuple[str, ...]

    @property
    def added(self) -> tuple[Source, ...]:
        return tuple(op.source for op in self.operations if op.changed is None)

    @property
    def updated(self) -> tuple[Source, ...]:
        return tuple(op.source for op in self.operations if op.changed is not None)


def _check_id(source_id: Any) -> str:
    if not is_valid_slug(source_id):
        raise ValidationError("source id is not a valid slug", ("id",))
    return source_id


def _changed_fields(before: Source, after: Source) -> tuple[str, ...]:
    names = type(before).model_fields
    return tuple(
        sorted(n for n in names if n != "updated_at" and getattr(before, n) != getattr(after, n))
    )


class SourceRegistry:
    def __init__(
        self, sources: SourceRepository, control: ControlRepository, uow: UnitOfWork, clock: Clock
    ) -> None:
        self._sources = sources
        self._control = control
        self._uow = uow
        self._clock = clock

    # ---- helpers ----

    def _audit(
        self, at: datetime, kind: str, source_id: str | None, detail: Mapping[str, Any]
    ) -> None:
        text = json.dumps(detail, sort_keys=True, separators=(",", ":"))
        self._control.append_control_event(ControlEvent(at, kind, source_id, text))

    def _require(self, source_id: str) -> Source:
        return self._sources.get(_check_id(source_id))

    def _ensure_free(self, source: Source) -> None:
        try:
            self._sources.get(source.id)
        except NotFoundError:
            pass
        else:
            raise ConflictError(_ID_CONFLICT)
        if self._sources.find_by_name_ci(source.name) is not None:
            raise ConflictError(_NAME_CONFLICT)

    def _ensure_name_available(self, source: Source) -> None:
        owner = self._sources.find_by_name_ci(source.name)
        if owner is not None and owner.id != source.id:
            raise ConflictError(_NAME_CONFLICT)

    # ---- reads ----

    def get(self, source_id: str) -> Source:
        return self._require(source_id)

    def export_sources(self) -> str:
        return dump_sources(self._sources.list())

    # ---- writes ----

    def add(self, data: Mapping[str, Any]) -> Source:
        """Create a source. Timestamps come from the clock; any supplied ones are ignored."""
        if not isinstance(data, Mapping):
            raise ValidationError("source data must be a mapping")
        now = self._clock.now()
        source = parse_source({**data, "created_at": now, "updated_at": now})
        with self._uow.transaction():
            self._ensure_free(source)
            try:
                self._sources.add(source)
            except ConflictError:  # lost a race the pre-check could not see
                raise ConflictError(_NAME_CONFLICT) from None
            self._audit(now, SOURCE_ADDED, source.id, {"type": source.type})
        return source

    def edit(self, source_id: str, changes: Mapping[str, Any]) -> SourceChange:
        """Apply field changes. `id`, `type`, `enabled` and the timestamps cannot be edited."""
        if not isinstance(changes, Mapping) or not all(isinstance(k, str) for k in changes):
            raise ValidationError("changes must be a mapping of field names to values")
        for field in _IMMUTABLE_FIELDS:
            if field in changes:
                raise ValidationError(f"{field} cannot be edited", (field,))
        now = self._clock.now()
        with self._uow.transaction():
            current = self._require(source_id)
            if any(key not in type(current).model_fields for key in changes):
                raise ValidationError(_UNKNOWN_FIELD)  # never echo a user-supplied key
            candidate = current.with_changes(changes)
            if candidate == current:
                return SourceChange(current, changed=False)
            self._ensure_name_available(candidate)
            updated = current.with_changes({**changes, "updated_at": now})
            self._sources.update(updated)
            self._audit(
                now, SOURCE_EDITED, updated.id, {"changed": _changed_fields(current, updated)}
            )
        return SourceChange(updated, changed=True)

    def enable(self, source_id: str) -> SourceChange:
        return self._set_enabled(source_id, enabled=True)

    def disable(self, source_id: str) -> SourceChange:
        return self._set_enabled(source_id, enabled=False)

    def _set_enabled(self, source_id: str, *, enabled: bool) -> SourceChange:
        now = self._clock.now()
        with self._uow.transaction():
            current = self._require(source_id)
            if current.enabled is enabled:
                return SourceChange(current, changed=False)
            updated = current.with_changes({"enabled": enabled, "updated_at": now})
            self._sources.update(updated)
            kind = SOURCE_ENABLED if enabled else SOURCE_DISABLED
            self._audit(now, kind, updated.id, {})
        return SourceChange(updated, changed=True)

    def delete(self, source_id: str) -> None:
        now = self._clock.now()
        with self._uow.transaction():
            current = self._require(source_id)
            state = self._control.get_source_state(current.id)
            self._sources.delete(current.id)
            self._control.delete_source_state(current.id)
            flag = source_analysis_key(current.id)
            if self._control.get_setting(flag) is not None:
                self._control.set_setting(flag, "false")  # a re-added id must not inherit it
            detail = {"type": current.type, "state": state.state.value if state else None}
            self._audit(now, SOURCE_DELETED, current.id, detail)

    # ---- import ----

    def import_sources(self, text: str, *, dry_run: bool = False) -> ImportReport:
        """Upsert every source in `text` (keyed by id) all-or-nothing, or change nothing.

        An entry REPLACES the stored source with the same id: fields it omits fall back to their
        defaults (`enabled` becomes False). Sources missing from the file are left alone. New
        sources keep the timestamps in the file (the clock fills in missing ones); updated ones
        keep their `created_at` and get `updated_at` from the clock.
        """
        now = self._clock.now()
        incoming = load_sources(text, default_timestamp=now)
        with self._uow.transaction():
            plan = self._plan_import(incoming, now)
            report = ImportReport(
                tuple(s.id for s in plan.added),
                tuple(s.id for s in plan.updated),
                plan.unchanged,
                dry_run,
            )
            if dry_run or not plan.operations:
                return report
            for op in plan.operations:
                if op.changed is None:
                    self._sources.add(op.source)
                    detail = {"type": op.source.type, **_VIA_IMPORT}
                    self._audit(now, SOURCE_ADDED, op.source.id, detail)
                else:
                    self._sources.update(op.source)
                    detail = {"changed": op.changed, **_VIA_IMPORT}
                    self._audit(now, SOURCE_EDITED, op.source.id, detail)
            summary = {
                "added": len(plan.added),
                "updated": len(plan.updated),
                "unchanged": len(plan.unchanged),
            }
            self._audit(now, SOURCES_IMPORTED, None, summary)
        return report

    def _plan_import(self, incoming: list[Source], now: datetime) -> _Plan:
        """Classify every entry against the stored sources, in file order, or raise with ALL the
        conflicts. Name ownership is simulated entry by entry, exactly as the writes will happen.
        """
        stored = {s.id: s for s in self._sources.list()}
        owner = {normalize_name(s.name): s.id for s in stored.values()}
        problems: list[SourceFileProblem] = []
        operations: list[_Operation] = []
        unchanged: list[str] = []

        for index, entry in enumerate(incoming):
            current = stored.get(entry.id)
            if current is not None and entry.type != current.type:
                problems.append(_conflict(index, "type", "type of an existing source is immutable"))
                continue
            new_key = normalize_name(entry.name)
            if owner.get(new_key, entry.id) != entry.id:
                problems.append(_conflict(index, "name", "name is used by another source"))
                continue
            if current is None:
                owner[new_key] = entry.id
                operations.append(_Operation(entry, None))
                continue
            wanted = entry.with_changes(
                {"created_at": current.created_at, "updated_at": current.updated_at}
            )
            if wanted == current:
                unchanged.append(entry.id)
                continue
            owner.pop(normalize_name(current.name), None)
            owner[new_key] = entry.id
            stamped = wanted.with_changes({"updated_at": now})
            operations.append(_Operation(stamped, _changed_fields(current, wanted)))

        if problems:
            raise ImportConflictError(tuple(problems))
        return _Plan(tuple(operations), tuple(unchanged))

    # Defined last: the name shadows the builtin inside the class body.
    def list(self) -> "list[Source]":
        return self._sources.list()


def _conflict(index: int, field: str, message: str) -> SourceFileProblem:
    return SourceFileProblem(index, _CODE_CONFLICT, (field,), message)
