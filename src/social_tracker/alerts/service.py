"""The alert service (application layer): turns stored verdicts into alerts and looks after them.

The step (`refresh`, and `rebuild` for everything) is a pure policy applied to what is stored:

- AL-1/AL-2: `alerts.domain.assess` decides whether a verdict alerts and how severe. The service
  only reads the verdict of an item, the item itself (once, for its excerpt, its source and its
  times) and, when the model judged it, the judgment (model and prompt version).
- AL-6: an item whose content (or whose original, for a near copy) is already behind an alert seen
  within `dedup_window` joins that alert; one that bridges several makes them one (the oldest
  stays). The alert is a projection of its members, so its severity is the highest of the active
  ones: merging can raise it and never lowers it.
- AL-10: an item is one member whatever happens to it. A verdict that changes (the model settled
  it, a reclassification, a copy that took its original's verdict) updates the alert it is in;
  one that stops qualifying withdraws the alert when no other member holds it up, with a reason;
  one that qualifies again brings it back. Nothing is ever deleted by the step.
- Every item is one short transaction that re-reads what it needs, so two workers (or a rebuild
  next to the worker) cannot give an item two alerts; a failure belongs to the item, a storage
  failure ends the run and the next one resumes.

What the operator does goes through `set_status` (AL-5): the allowed moves, an audit entry with
actor and reason, nothing on a redundant request. Nothing here ever sends anything outside this
machine (G-1): no publication, webhook, email, draft or approval flow exists.

The text kept in an alert is the bounded, cleaned and redacted excerpt of `alerts.excerpt`; the
rest is names and numbers. Item text is untrusted and never interpreted.
"""

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum

from social_tracker.alerts.domain import (
    ACTOR_SYSTEM,
    DEFAULT_PAGE_SIZE,
    Alert,
    AlertCounts,
    AlertEvent,
    AlertEventKind,
    AlertFilter,
    AlertMember,
    AlertPage,
    AlertSettings,
    AlertStatus,
    Evidence,
    Severity,
    SyncReport,
    alert_id_for,
    apply_status,
    assess,
    event_age,
    project,
    severity_rank,
    tags_for,
    validate_page,
)
from social_tracker.alerts.excerpt import clean_name, clean_text, make_excerpt, make_rationale
from social_tracker.alerts.ports import AlertRepository
from social_tracker.classification.domain import RISK_LABELS, Classification, Label, Stage
from social_tracker.classification.ports import ClassificationRepository
from social_tracker.control.domain import validate_actor, validate_reason
from social_tracker.control.service import ControlService
from social_tracker.items.domain import Item, validate_limit
from social_tracker.items.ports import ItemRepository
from social_tracker.llm.ports import EscalationRepository
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import (
    DomainError,
    InvalidTransition,
    NotFoundError,
    StorageError,
    ValidationError,
)
from social_tracker.shared.ids import is_valid_slug
from social_tracker.shared.uow import UnitOfWork
from social_tracker.sources.ports import SourceRepository

DEFAULT_BATCH = 200
_LOG = logging.getLogger("social_tracker.alerts")
_MAX_REASON = 480


class Outcome(StrEnum):
    CREATED = "created"
    MERGED = "merged"
    UPDATED = "updated"
    WITHDRAWN = "withdrawn"
    REACTIVATED = "reactivated"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class StatusResult:
    """The alert as it is after a status request, the status it had before, and whether anything
    changed (False for a redundant request)."""

    alert: Alert
    changed: bool
    previous: AlertStatus


@dataclass(frozen=True)
class AlertDetail:
    """An alert with the items behind it, its trail, and the sources that no longer exist."""

    alert: Alert
    members: tuple[AlertMember, ...]
    events: tuple[AlertEvent, ...]
    removed_sources: frozenset[str]


def _withdrawal_reason(label: Label) -> str:
    if label in RISK_LABELS:
        return "the confidence fell under the alert threshold"
    return f"the verdict is now {label.value}"


class AlertService:
    def __init__(
        self,
        alerts: AlertRepository,
        classifications: ClassificationRepository,
        items: ItemRepository,
        sources: SourceRepository,
        escalations: EscalationRepository,
        control: ControlService,
        uow: UnitOfWork,
        clock: Clock,
        *,
        settings: AlertSettings | None = None,
    ) -> None:
        self._alerts = alerts
        self._classifications = classifications
        self._items = items
        self._sources = sources
        self._escalations = escalations
        self._control = control
        self._uow = uow
        self._clock = clock
        self._settings = settings if settings is not None else AlertSettings()

    @property
    def settings(self) -> AlertSettings:
        return self._settings

    # ---- the step ----

    def refresh(self, *, full: bool = False, batch: int = DEFAULT_BATCH) -> SyncReport:
        """Look at the verdicts that are new or changed since their alert was last evaluated and
        update the alerts. With `full`, at every member too (the threshold may have moved): that
        is `rebuild`. Idempotent: a second run finds nothing to do."""
        validate_limit(batch)
        threshold = self._control.alert_min_confidence()
        tally: Counter[str] = Counter()
        after: str | None = None
        while True:
            ids = self._alerts.sync_ids(
                min_confidence=threshold, full=full, after=after, limit=batch
            )
            for item_id in ids:
                try:
                    tally[self._sync(item_id, threshold).value] += 1
                except StorageError as exc:
                    _LOG.warning("the alert step was stopped by a storage error: %s", exc.code)
                    tally["errors"] += 1
                    return self._report(tally)
                except Exception as exc:  # noqa: BLE001 - one poison item must not stall the step
                    _LOG.warning("an item could not be evaluated for alerts: %s", _reason(exc))
                    tally["errors"] += 1
            if len(ids) < batch:
                return self._report(tally)
            after = ids[-1]

    def rebuild(self) -> SyncReport:
        """Evaluate every verdict again with the threshold in force. Idempotent."""
        return self.refresh(full=True)

    def sync_item(self, item_id: str, min_confidence: float | None = None) -> Outcome:
        """Evaluate the verdict of one item now (its own transaction)."""
        threshold = (
            self._control.alert_min_confidence() if min_confidence is None else min_confidence
        )
        return self._sync(item_id, threshold)

    @staticmethod
    def _report(tally: Counter[str]) -> SyncReport:
        return SyncReport(
            created=tally[Outcome.CREATED],
            merged=tally[Outcome.MERGED],
            updated=tally[Outcome.UPDATED],
            withdrawn=tally[Outcome.WITHDRAWN],
            reactivated=tally[Outcome.REACTIVATED],
            unchanged=tally[Outcome.UNCHANGED],
            skipped=tally[Outcome.SKIPPED],
            errors=tally["errors"],
        )

    def _sync(self, item_id: str, threshold: float) -> Outcome:
        with self._uow.transaction():
            verdict = self._classifications.get(item_id)
            if verdict is None:
                return Outcome.SKIPPED
            existing = self._alerts.member(item_id)
            if existing is None:
                return self._raise(verdict, threshold)
            return self._update(existing, verdict, threshold)

    # ---- a new member ----

    def _raise(self, verdict: Classification, threshold: float) -> Outcome:
        item = self._items.get(verdict.item_id)
        if item is None or self._alerts.is_forgotten(verdict.item_id):
            return Outcome.SKIPPED
        age = event_age(item.published_at, item.fetched_at)
        severity = assess(
            verdict.label,
            verdict.confidence,
            min_confidence=threshold,
            catalog_status=verdict.catalog_status,
            event_age=age,
            settings=self._settings,
        )
        if severity is None:
            return Outcome.SKIPPED
        now = self._clock.now()
        targets = self._alerts.find_merge_targets(
            content_hash=item.content_hash,
            related_items=[x for x in (verdict.duplicate_of, verdict.near_of) if x],
            around=item.fetched_at,
            window=self._settings.dedup_window,
        )
        alert_id = targets[0] if targets else alert_id_for(item.id)
        member = self._member(alert_id, verdict, item, severity, age, now)
        previous = self._alerts.get(alert_id)
        status, basis = (
            self._decisive_status(previous, targets[1:]) if previous is not None else (None, None)
        )
        for other in targets[1:]:
            self._absorb(alert_id, other, now)
        members = [*self._alerts.members(alert_id), member]
        alert = project(alert_id, members, previous=previous, now=now)
        if status is not None:
            alert = replace(alert, status=status)
        alert = self._with_withdrawal(previous, alert, members, member)
        if previous is not None and not any(m.active for m in members[:-1]):
            alert = replace(alert, downgraded_by_model=False)  # a live copy: nothing to flag
        rise = self._rise(alert, basis)
        if rise is not None:
            alert = replace(alert, status=AlertStatus.NEW, status_changed_at=now)
        self._alerts.save(alert)
        self._alerts.save_member(member)
        if previous is None:
            self._event(alert_id, now, AlertEventKind.CREATED, _says(verdict), "", "1")
            return Outcome.CREATED
        self._event(
            alert_id,
            now,
            AlertEventKind.MERGED,
            f"a copy joined: {_says(verdict)}",
            str(previous.occurrences),
            str(alert.occurrences),
        )
        self._write_rise(alert_id, rise, status, now)
        if previous.withdrawn and not alert.withdrawn:
            self._event(alert_id, now, AlertEventKind.REACTIVATED, "a new occurrence qualifies")
            return Outcome.REACTIVATED
        return Outcome.MERGED

    @staticmethod
    def _rise(alert: Alert, basis: Severity | None, why: str = "") -> str | None:
        """A5/A1: the reason to put an alert back to `new`, when a person had handled it
        (seen, dismissed) at the severity `basis` and it is now more severe. Equal or lower
        severity keeps what the person decided."""
        if basis is None or alert.withdrawn or alert.status is AlertStatus.NEW:
            return None
        if severity_rank(alert.severity) <= severity_rank(basis):
            return None
        reason = f"severity rose from {basis.value} to {alert.severity.value}"
        return f"{reason}: {why}" if why else reason

    def _write_rise(
        self, alert_id: str, reason: str | None, was: AlertStatus | None, now: datetime
    ) -> None:
        if reason is not None and was is not None:
            self._event(
                alert_id,
                now,
                AlertEventKind.STATUS_CHANGED,
                reason,
                was.value,
                AlertStatus.NEW.value,
            )

    def _absorb(self, keep_id: str, other_id: str, now: datetime) -> None:
        """Move the members of `other_id` under `keep_id` and forget the empty alert."""
        for member in self._alerts.members(other_id):
            self._alerts.save_member(replace(member, alert_id=keep_id))
        self._alerts.delete(other_id)
        self._event(keep_id, now, AlertEventKind.MERGED, f"absorbed alert {other_id}")

    def _decisive_status(self, kept: Alert, others: list[str]) -> tuple[AlertStatus, Severity]:
        """The status of alerts that become one: what needs attention most decides (new, then
        seen, then dismissed), so merging can never hide an alert nobody has looked at. With it,
        the severity that status was chosen at (the highest among the alerts that held it): the
        merged alert goes back to `new` only if it is MORE severe than that."""
        held = [kept, *(a for a in (self._alerts.get(o) for o in others) if a is not None)]
        statuses = {a.status for a in held}
        for status in (AlertStatus.NEW, AlertStatus.SEEN, AlertStatus.DISMISSED):
            if status in statuses:
                basis = max((a.severity for a in held if a.status is status), key=severity_rank)
                return status, basis
        return kept.status, kept.severity

    def _member(
        self,
        alert_id: str,
        verdict: Classification,
        item: Item,
        severity: Severity,
        age: timedelta,
        now: datetime,
    ) -> AlertMember:
        return AlertMember(
            alert_id=alert_id,
            item_id=item.id,
            source_id=item.source_id,
            source_name=clean_name(item.source_name, item.source_id),
            content_hash=item.content_hash,
            excerpt=make_excerpt(item.title, item.text),
            label=verdict.label,
            confidence=verdict.confidence,
            stage=verdict.stage.value,
            rationale=make_rationale(verdict.rationale),
            severity=severity,
            tags=tags_for(
                [t.value for t in verdict.topics],
                [s.name for s in verdict.signals],
                verdict.catalog_status,
            ),
            evidence=self._evidence(verdict),
            active=True,
            occurred_at=item.fetched_at,
            linked_at=now,
            event_age_s=int(age.total_seconds()),
            settled_at=verdict.settled_at,
        )

    def _evidence(self, verdict: Classification) -> Evidence:
        """The cross-check behind a verdict. Every string of it is cleaned like the excerpt (A10):
        a model name or a reason can come from a file or an answer that an outsider influenced."""
        judgment = self._escalations.get_judgment(verdict.item_id)
        judged = judgment is not None and judgment.llm_model is not None
        return Evidence(
            catalog_status=verdict.catalog_status,
            stage=verdict.stage.value,
            classifier_version=clean_text(verdict.classifier_version, 64, "unknown"),
            signals=tuple((s.name, s.label.value, s.weight) for s in verdict.signals),
            topics=tuple(t.value for t in verdict.topics),
            settle_reason=_tidy(verdict.settle_reason),
            provisional=verdict.needs_llm or verdict.awaiting_original,
            llm_model=_tidy(judgment.llm_model) if judged and judgment else None,
            prompt_version=_tidy(judgment.prompt_version) if judged and judgment else None,
            duplicate_of=verdict.duplicate_of,
            near_of=verdict.near_of,
            recycled_of=verdict.recycled_of,
            source_credibility=verdict.source_credibility_used,
            truncated=verdict.truncated,
        )

    # ---- a member whose verdict changed ----

    def _update(self, old: AlertMember, verdict: Classification, threshold: float) -> Outcome:
        age = timedelta(seconds=old.event_age_s)
        severity = assess(
            verdict.label,
            verdict.confidence,
            min_confidence=threshold,
            catalog_status=verdict.catalog_status,
            event_age=age,
            settings=self._settings,
        )
        fresh = replace(
            old,
            label=verdict.label,
            confidence=verdict.confidence,
            stage=verdict.stage.value,
            rationale=make_rationale(verdict.rationale),
            severity=severity if severity is not None else old.severity,
            tags=tags_for(
                [t.value for t in verdict.topics],
                [s.name for s in verdict.signals],
                verdict.catalog_status,
            ),
            evidence=self._evidence(verdict),
            active=severity is not None,
            settled_at=verdict.settled_at,
        )
        if fresh == old:
            return Outcome.UNCHANGED
        now = self._clock.now()
        previous = self._alerts.get(old.alert_id)
        if previous is None:  # a member without its alert cannot exist; be loud, not wrong
            raise StorageError("an alert member has no alert")
        members = [
            fresh if m.item_id == old.item_id else m for m in self._alerts.members(old.alert_id)
        ]
        projected = project(previous.id, members, previous=previous, now=now)
        # A11: the language model made this verdict (it settled a waiting one). A verdict it
        # lowers, or one that stops qualifying, is flagged and kept in view, never silently hidden
        model_made = verdict.stage is Stage.LLM and (
            old.stage != Stage.LLM.value or old.settled_at != verdict.settled_at
        )
        dark = not any(m.active for m in members)
        lowered = not previous.withdrawn and (
            severity_rank(projected.severity) < severity_rank(previous.severity)
        )
        downgrade = model_made and (dark or lowered)
        flag = True if downgrade else (False if model_made else previous.downgraded_by_model)
        alert = self._with_withdrawal(
            previous, projected, members, fresh, keep_visible=dark and flag
        )
        alert = replace(alert, downgraded_by_model=flag)
        if downgrade and dark:
            alert = replace(alert, rationale=fresh.rationale)  # what the model said
        reason = None
        if downgrade and alert.status is not AlertStatus.NEW:
            reason = f"the language model settled a lower verdict: {_says(verdict)}"
        elif not alert.withdrawn:
            reason = self._rise(alert, previous.severity, "verdict upgraded" if model_made else "")
        if reason is not None:
            alert = replace(alert, status=AlertStatus.NEW, status_changed_at=now)
        self._alerts.save(alert)
        self._alerts.save_member(fresh)
        if alert.withdrawn and not previous.withdrawn:
            self._event(
                alert.id, now, AlertEventKind.WITHDRAWN, alert.withdrawn_reason or "withdrawn"
            )
            return Outcome.WITHDRAWN
        self._write_rise(alert.id, reason, previous.status, now)
        if previous.withdrawn and not alert.withdrawn:
            self._event(alert.id, now, AlertEventKind.REACTIVATED, "the verdict qualifies again")
            return Outcome.REACTIVATED
        shown = (alert.severity, alert.label, alert.confidence)
        if shown != (previous.severity, previous.label, previous.confidence) or (
            alert.downgraded_by_model and not previous.downgraded_by_model
        ):
            what = (
                "the language model settled a lower verdict" if downgrade else "the verdict changed"
            )
            self._event(
                alert.id,
                now,
                AlertEventKind.UPDATED,
                f"{what}: {_says(verdict)}",
                previous.severity.value,
                alert.severity.value,
            )
        return Outcome.UPDATED

    @staticmethod
    def _with_withdrawal(
        previous: Alert | None,
        alert: Alert,
        members: list[AlertMember],
        changed: AlertMember,
        *,
        keep_visible: bool = False,
    ) -> Alert:
        """Set or clear the withdrawal of `alert`: it stands while any member is active; with none
        left it is withdrawn, once, with the reason the member that changed gives. `keep_visible`
        is the one exception (A11): the language model made the last member stop qualifying, and
        an alert is not hidden by what a model said."""
        if keep_visible or any(m.active for m in members):
            return replace(alert, withdrawn_at=None, withdrawn_reason=None)
        if previous is not None and previous.withdrawn:
            return replace(
                alert,
                withdrawn_at=previous.withdrawn_at,
                withdrawn_reason=previous.withdrawn_reason,
            )
        return replace(
            alert, withdrawn_at=alert.updated_at, withdrawn_reason=_withdrawal_reason(changed.label)
        )

    def _event(
        self,
        alert_id: str,
        at: datetime,
        kind: AlertEventKind,
        reason: str,
        before: str = "",
        after: str = "",
    ) -> None:
        event = AlertEvent(alert_id, at, kind, ACTOR_SYSTEM, reason[:_MAX_REASON], before, after)
        self._alerts.add_event(event)

    # ---- what the operator does ----

    def set_status(self, alert_id: str, target: object, *, actor: str, reason: str) -> StatusResult:
        """Move an alert to `target` (AL-5): the legal moves only, audited with actor and reason.
        The status it already has is a no-op that writes nothing. An unknown id is
        `NotFoundError`, an unknown status `ValidationError`, an illegal move
        `InvalidTransition`."""
        actor, reason = validate_actor(actor), validate_reason(reason)
        try:
            wanted = AlertStatus(target)
        except (ValueError, TypeError):
            raise ValidationError("unknown alert status") from None
        with self._uow.transaction():
            alert = self._known(alert_id)
            change = apply_status(alert.status, wanted)
            if not change.changed:
                return StatusResult(alert, changed=False, previous=alert.status)
            now = self._clock.now()
            updated = replace(
                alert, status=wanted, status_changed_at=now, downgraded_by_model=False
            )
            self._alerts.save(updated)
            self._alerts.add_event(
                AlertEvent(
                    alert.id,
                    now,
                    AlertEventKind.STATUS_CHANGED,
                    actor,
                    reason,
                    alert.status.value,
                    wanted.value,
                )
            )
        return StatusResult(updated, changed=True, previous=alert.status)

    def reopen(self, alert_id: str, *, actor: str, reason: str) -> StatusResult:
        """Bring a dismissed alert back as seen. A seen alert stays as it is (no entry); one nobody
        has looked at is not "reopened": `InvalidTransition`."""
        actor, reason = validate_actor(actor), validate_reason(reason)
        with self._uow.transaction():
            alert = self._known(alert_id)
            if alert.status is AlertStatus.NEW:
                raise InvalidTransition("only a dismissed alert can be reopened")
            return self.set_status(alert.id, AlertStatus.SEEN, actor=actor, reason=reason)

    def _known(self, alert_id: object) -> Alert:
        if not is_valid_slug(alert_id):
            raise NotFoundError("alert not found")
        alert = self._alerts.get(str(alert_id))
        if alert is None:
            raise NotFoundError("alert not found")
        return alert

    # ---- reading ----

    def list(
        self, flt: AlertFilter | None = None, *, limit: int = DEFAULT_PAGE_SIZE, offset: int = 0
    ) -> AlertPage:
        """A page of alerts, newest first (AL-4)."""
        validate_page(limit, offset)
        chosen = flt if flt is not None else AlertFilter()
        with self._uow.snapshot():
            found = self._alerts.list(chosen, limit=limit, offset=offset)
            total = self._alerts.count(chosen)
        return AlertPage(tuple(found), total, limit, offset)

    def detail(self, alert_id: str) -> AlertDetail:
        """An alert with its members and trail; `NotFoundError` for one that does not exist."""
        with self._uow.snapshot():
            alert = self._known(alert_id)
            members = tuple(self._alerts.members(alert.id))
            events = tuple(self._alerts.events(alert.id))
        ids = {m.source_id for m in members} | {alert.source_id}
        return AlertDetail(alert, members, events, self.removed_sources(ids))

    def removed_sources(self, source_ids: Iterable[str]) -> frozenset[str]:
        """Which of these sources no longer exist (their alerts keep the name they had). One read
        of the registry, however many sources are asked about."""
        wanted = set(source_ids)
        if not wanted:
            return frozenset()
        return frozenset(wanted - {source.id for source in self._sources.list()})

    def counts(self) -> AlertCounts:
        return self._alerts.counts()

    # ---- retention (AL-8) ----

    def purge_expired(self) -> int:
        """Forget the alerts past their retention: strictly older than `retention_days` counted
        from their latest occurrence; with 0 days, those already looked at (seen, dismissed) or
        withdrawn. Returns how many. The items they came from do not raise them again."""
        now = self._clock.now()
        days = self._settings.retention_days
        if days == 0:
            return self._alerts.purge(before=None, handled=True, now=now)
        return self._alerts.purge(before=now - timedelta(days=days), handled=False, now=now)


def _tidy(value: str | None) -> str | None:
    """A short optional string of the evidence, cleaned (A10); None when nothing is left."""
    return (clean_text(value, 100, "") or None) if value is not None else None


def _says(verdict: Classification) -> str:
    return f"{verdict.label.value} at {verdict.confidence:.2f} by {verdict.stage.value}"


def _reason(exc: Exception) -> str:
    return exc.code if isinstance(exc, DomainError) else type(exc).__name__
