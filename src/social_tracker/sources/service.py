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
JSON objects with known keys only; they never contain URLs or secret references. The only free
text in them is the validated `actor` and `reason` of the operator.

`Source.enabled` is the operator's intent and the ONLY way to enable or disable a source: the
collection state row is kept in step inside the same transaction (`_sync_state`), so
`source.enabled <=> collection state is not disabled` (a missing row means disabled).

Deleting a source removes its runtime state row and its per-source analysis setting row (the
previous values are recorded in the `source_deleted` event), but the audit events stay (SR-9).
Deleting a source that is currently `collecting` is allowed: the running collection is not
interrupted by this service, and the detail of the `source_deleted` event records the state it
had.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from social_tracker.control.domain import (
    CollectionState,
    ControlEvent,
    Event,
    SourceState,
    apply_event,
    event_detail_json,
    source_analysis_key,
    validate_actor,
    validate_reason,
)
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
_PLACEHOLDER_TIME = datetime(1970, 1, 1, tzinfo=UTC)
_IMMUTABLE_FIELDS = ("id", "type", "enabled", "created_at", "updated_at")
_VIA_IMPORT = {"via": "import"}
_ID_CONFLICT = "a source with this id already exists"
_NAME_CONFLICT = "a source with this name already exists"
STALE_SOURCE = "the source was changed after the version this edit was based on"


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


def _who(actor: Any, reason: Any) -> dict[str, str]:
    """Validated actor and reason (the control-domain rules), ready to merge into event details."""
    return {"actor": validate_actor(actor), "reason": validate_reason(reason)}


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
        self,
        at: datetime,
        kind: str,
        source_id: str | None,
        who: Mapping[str, str],
        detail: Mapping[str, Any],
    ) -> None:
        text = event_detail_json({**detail, **who})
        self._control.append_control_event(ControlEvent(at, kind, source_id, text))

    def _sync_state(self, source: Source, now: datetime) -> dict[str, str]:
        """Make the collection state agree with `source.enabled` (the operator's intent).

        Runs inside the caller's transaction, through the domain transition, so a flag change and
        its state change commit or fail together. Enabling moves disabled -> idle; disabling moves
        any state -> disabled with the failures reset; anything already in step is left alone (a
        running source keeps its backoff). Returns the event detail fragment for a real move.
        """
        stored = self._control.get_source_state(source.id)
        current = stored or SourceState(source.id, CollectionState.DISABLED)
        event = Event.ENABLE if source.enabled else Event.DISABLE
        result = apply_event(current, event, now)
        if not result.changed:
            return {}
        self._control.save_source_state(result.state)
        return {"state_from": current.state.value, "state_to": result.state.state.value}

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

    def add(self, data: Mapping[str, Any], *, actor: str, reason: str) -> Source:
        """Create a source. Timestamps come from the clock; any supplied ones are ignored."""
        who = _who(actor, reason)
        if not isinstance(data, Mapping):
            raise ValidationError("source data must be a mapping")
        with self._uow.transaction():
            now = self._clock.now()
            source = parse_source({**data, "created_at": now, "updated_at": now})
            self._ensure_free(source)
            try:
                self._sources.add(source)
            except ConflictError:  # lost a race the pre-check could not see
                raise ConflictError(_NAME_CONFLICT) from None
            moved = self._sync_state(source, now)
            self._audit(now, SOURCE_ADDED, source.id, who, {"type": source.type, **moved})
        return source

    def edit(
        self,
        source_id: str,
        changes: Mapping[str, Any],
        *,
        actor: str,
        reason: str,
        expected_updated_at: datetime | None = None,
    ) -> SourceChange:
        """Apply field changes. `id`, `type`, `enabled` and the timestamps cannot be edited.

        With `expected_updated_at` the edit applies only if the source still carries that
        `updated_at`, checked inside the same transaction as the write: an editor working from a
        stale copy gets a `ConflictError` (`STALE_SOURCE`) and nothing is written.
        """
        who = _who(actor, reason)
        if not isinstance(changes, Mapping) or not all(isinstance(k, str) for k in changes):
            raise ValidationError("changes must be a mapping of field names to values")
        for field in _IMMUTABLE_FIELDS:
            if field in changes:
                raise ValidationError(f"{field} cannot be edited", (field,))
        with self._uow.transaction():
            now = self._clock.now()
            current = self._require(source_id)
            if expected_updated_at is not None and current.updated_at != expected_updated_at:
                raise ConflictError(STALE_SOURCE)
            # Unknown keys (`self`, another type's field...) are rejected by the model itself
            # (extra="forbid") with a fixed message that never echoes the key.
            candidate = current.with_changes(changes)
            if candidate == current:
                return SourceChange(current, changed=False)
            self._ensure_name_available(candidate)
            updated = current.with_changes({**changes, "updated_at": now})
            self._sources.update(updated)
            self._audit(
                now, SOURCE_EDITED, updated.id, who, {"changed": _changed_fields(current, updated)}
            )
        return SourceChange(updated, changed=True)

    def enable(self, source_id: str, *, actor: str, reason: str) -> SourceChange:
        return self._set_enabled(source_id, enabled=True, who=_who(actor, reason))

    def disable(self, source_id: str, *, actor: str, reason: str) -> SourceChange:
        return self._set_enabled(source_id, enabled=False, who=_who(actor, reason))

    def _set_enabled(
        self, source_id: str, *, enabled: bool, who: Mapping[str, str]
    ) -> SourceChange:
        with self._uow.transaction():
            now = self._clock.now()
            current = self._require(source_id)
            if current.enabled is enabled:
                return SourceChange(current, changed=False)
            updated = current.with_changes({"enabled": enabled, "updated_at": now})
            self._sources.update(updated)
            moved = self._sync_state(updated, now)
            kind = SOURCE_ENABLED if enabled else SOURCE_DISABLED
            self._audit(now, kind, updated.id, who, moved)
        return SourceChange(updated, changed=True)

    def delete(self, source_id: str, *, actor: str, reason: str) -> None:
        who = _who(actor, reason)
        with self._uow.transaction():
            now = self._clock.now()
            current = self._require(source_id)
            state = self._control.get_source_state(current.id)
            flag = source_analysis_key(current.id)
            analysis = self._control.get_setting(flag)
            self._sources.delete(current.id)
            self._control.delete_source_state(current.id)
            self._control.delete_setting(flag)  # a re-added id must not inherit the old flag
            detail = {
                "type": current.type,
                "state": state.state.value if state else None,
                "analysis": analysis,
            }
            self._audit(now, SOURCE_DELETED, current.id, who, detail)

    # ---- import ----

    def import_sources(
        self,
        text: str,
        *,
        actor: str | None = None,
        reason: str | None = None,
        dry_run: bool = False,
    ) -> ImportReport:
        """Upsert every source in `text` (keyed by id) all-or-nothing, or change nothing.

        An entry REPLACES the stored source with the same id (this is deliberate and can
        surprise an operator): every field the entry omits falls back to its default, so a
        hand-written entry without `enabled` DISABLES a source that was enabled, and omitted
        optional fields (keywords, poll interval...) are reset. Sources missing from the file are
        left alone.

        Timestamps in the file are never trusted: new sources get `created_at` and `updated_at`
        from the clock, and updated ones keep the stored `created_at` and get `updated_at` from
        the clock. Importing a file therefore does not preserve history from another registry.
        """
        who = _who(actor, reason) if not dry_run else {}
        # Timestamps in the file are never used (see above), so parsing needs only a placeholder;
        # it happens before the write lock is taken and the real time is read inside it.
        incoming = load_sources(text, default_timestamp=_PLACEHOLDER_TIME)
        with self._uow.transaction():
            now = self._clock.now()
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
                    moved = self._sync_state(op.source, now)
                    detail = {"type": op.source.type, **_VIA_IMPORT, **moved}
                    self._audit(now, SOURCE_ADDED, op.source.id, who, detail)
                else:
                    self._sources.update(op.source)
                    moved = self._sync_state(op.source, now)
                    detail = {"changed": op.changed, **_VIA_IMPORT, **moved}
                    self._audit(now, SOURCE_EDITED, op.source.id, who, detail)
            summary = {
                "added": len(plan.added),
                "updated": len(plan.updated),
                "unchanged": len(plan.unchanged),
            }
            self._audit(now, SOURCES_IMPORTED, None, who, summary)
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
                stamped_new = entry.with_changes({"created_at": now, "updated_at": now})
                operations.append(_Operation(stamped_new, None))
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
