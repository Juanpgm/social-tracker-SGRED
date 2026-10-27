"""Vocabulary and policy of the alerts. Pure, no I/O.

An alert says that a stored verdict deserves a person's attention. The policy is two questions:

- AL-1, does it alert at all: the label is one of the three risk labels (`unintentional_rumor`,
  `deliberate_disinformation`, `recycled_content`) and the confidence is at least the alert
  threshold (`alert_min_confidence`, default 0.5, inclusive). `no_risk`, `unclassified` and
  `out_of_scope` never alert, whatever their confidence.
- AL-2, how severe: `critical` for deliberate disinformation at 0.85 or more, or for a verdict
  that the official catalogs contradict (`catalog_status == "mismatch"`) about an event at most 24
  hours old (inclusive) at 0.8 or more; otherwise `high` from 0.7, `medium` from 0.5, `low` under
  that (only reachable when the operator lowers the threshold). Every number is configurable
  (`AlertSettings`).

The age of an event is the time between the item's publication date and the moment THIS
application first saw it (`event_age`). The moment of the first sighting is the application's own
clock; the publication date is the feed's, which can write anything. So the date can only be
trusted so far: a date in the future counts as no age, and a date that is missing (the collector
put the fetch time in its place and flagged the item) means the age is UNKNOWN, and an unknown age
is never fresh (`UNKNOWN_AGE`, older than any window). A backdated date can only make an event look
older: at most it lowers a critical contradiction of the catalog to high; it never raises a
severity. Both the policy and the age are pure functions, so the same verdict always gets the
same severity.

An alert is a projection of its members (the items that raised it, AL-6): `project` builds it
from them, so merging a duplicate can only raise the severity (it is the highest of the active
members) and a member whose verdict stops qualifying leaves the alert as `withdrawn` (AL-10)
instead of deleting it.

Item text is untrusted: the excerpt kept here (see `alerts.excerpt`) is bounded and stripped of
control and hidden characters before it reaches this module, and every string is HTML-escaped by
the template that renders it.
"""

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, NamedTuple

from social_tracker.classification.domain import RISK_LABELS, Label, Topic
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import InvalidTransition, ValidationError
from social_tracker.shared.ids import validate_slug
from social_tracker.shared.text import has_control_chars, has_unsafe_chars

MAX_EXCERPT_CHARS = 500  # AL-11
MAX_RATIONALE_CHARS = 300
MAX_SOURCE_NAME_CHARS = 100
MAX_EVIDENCE_BYTES = 4096
MAX_TAGS = 12
DEFAULT_RETENTION_DAYS = 180  # AL-8
MAX_RETENTION_DAYS = 3650
MAX_WINDOW = timedelta(days=30)
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 200
MAX_OFFSET = 1_000_000
DEFAULT_MAX_MEMBERS = 500  # the items stored behind one alert
MAX_MEMBERS_LIMIT = 5_000
DEFAULT_MAX_EVENTS = 200  # the entries of the trail stored and shown per alert
MAX_EVENTS_LIMIT = 2_000
CATALOG_MISMATCH = "mismatch"
WITHDRAWN = "withdrawn"  # a lifecycle state a list can be filtered by, next to the three statuses
ACTOR_SYSTEM = "system:alerts"

_TAG_RE = re.compile(r"[a-z_]{1,32}")
_STATUS_RE = re.compile(r"[a-z_]{1,32}")
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_VERSION_MAX = 64
_NAME_MAX = 100

TAG_SECRETARIAT_ATTRIBUTION = "secretariat_attribution"
TAG_CATALOG_CONTRADICTION = "catalog_contradiction"
TAG_OBFUSCATED = "obfuscated_text"
TAG_PREDICTION = "prediction"
# Every tag an alert can carry (the topics of the filter and the notable evidence), for screens
# that offer them as choices.
KNOWN_TAGS = tuple(
    sorted(
        {
            Topic.EARTHQUAKE.value,
            Topic.SECRETARIAT.value,
            Topic.OTHER_DISASTER.value,
            TAG_SECRETARIAT_ATTRIBUTION,
            TAG_CATALOG_CONTRADICTION,
            TAG_OBFUSCATED,
            TAG_PREDICTION,
        }
    )
)


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_ORDER = (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)


def severity_rank(severity: Severity) -> int:
    return SEVERITY_ORDER.index(severity)


class AlertStatus(StrEnum):
    NEW = "new"
    SEEN = "seen"
    DISMISSED = "dismissed"


# AL-5: the legal moves. Anything else (and `x -> new`) is refused; the same status again is a
# no-op.
_TRANSITIONS = frozenset(
    {
        (AlertStatus.NEW, AlertStatus.SEEN),
        (AlertStatus.NEW, AlertStatus.DISMISSED),
        (AlertStatus.SEEN, AlertStatus.DISMISSED),
        (AlertStatus.DISMISSED, AlertStatus.SEEN),
    }
)


class AlertEventKind(StrEnum):
    CREATED = "created"
    MERGED = "merged"
    UPDATED = "updated"
    WITHDRAWN = "withdrawn"
    REACTIVATED = "reactivated"
    STATUS_CHANGED = "status_changed"


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _probability(value: object, what: str) -> float:
    if not _is_number(value) or not math.isfinite(value) or not 0.0 <= value <= 1.0:  # type: ignore[operator,arg-type]
        raise ValidationError(f"{what} must be a number from 0 to 1")
    return float(value)  # type: ignore[arg-type]


def _whole(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _coerce[E: StrEnum](enum_cls: type[E], value: Any, what: str) -> E:
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        raise ValidationError(f"unknown {what}") from None


def _aware(value: Any, what: str) -> datetime:
    try:
        return ensure_utc(value)
    except ValidationError:
        raise ValidationError(f"{what} must be a timezone-aware time") from None


# ---- settings ----


@dataclass(frozen=True)
class AlertSettings:
    """The numbers of the policy; the defaults are the product's (AL-2, AL-6, AL-8).

    - `critical_disinformation`: confidence from which deliberate disinformation is critical.
    - `critical_catalog` and `critical_window`: confidence from which a catalog contradiction
      about an event at most this old is critical.
    - `high`, `medium`: the confidence from which a verdict is high / medium (`medium <= high`).
    - `dedup_window`: items with the same content or near copies seen within this time of each
      other are one alert.
    - `retention_days`: an alert is kept this many days after its latest occurrence (strictly
      older goes); 0 keeps an alert only until it has been looked at (seen or dismissed).
    - `max_members`: the items stored behind one alert. Past it a copy is only counted (and a
      stronger one takes the place of the weakest), so a flood of near copies cannot grow an alert
      or the cost of merging into it without bound.
    - `max_events`: the entries of the trail stored (and shown) per alert; past it routine
      entries (merged, updated) are only counted.
    """

    critical_disinformation: float = 0.85
    critical_catalog: float = 0.8
    high: float = 0.7
    medium: float = 0.5
    critical_window: timedelta = timedelta(hours=24)
    dedup_window: timedelta = timedelta(hours=24)
    retention_days: int = DEFAULT_RETENTION_DAYS
    max_members: int = DEFAULT_MAX_MEMBERS
    max_events: int = DEFAULT_MAX_EVENTS

    def __post_init__(self) -> None:
        for name in ("critical_disinformation", "critical_catalog", "high", "medium"):
            _probability(getattr(self, name), name)
        if self.critical_disinformation <= 0 or self.critical_catalog <= 0:
            raise ValidationError("a critical threshold must be above 0")
        if self.medium > self.high:
            raise ValidationError("the medium threshold must not exceed the high one")
        for name in ("critical_window", "dedup_window"):
            window = getattr(self, name)
            if not isinstance(window, timedelta) or not timedelta(0) < window <= MAX_WINDOW:
                raise ValidationError(f"{name} must be above 0 up to {MAX_WINDOW.days} days")
        if not _whole(self.retention_days, 0, MAX_RETENTION_DAYS):
            raise ValidationError(
                f"retention days must be a whole number from 0 to {MAX_RETENTION_DAYS}"
            )
        if not _whole(self.max_members, 1, MAX_MEMBERS_LIMIT):
            raise ValidationError(
                f"max members must be a whole number from 1 to {MAX_MEMBERS_LIMIT}"
            )
        if not _whole(self.max_events, 10, MAX_EVENTS_LIMIT):
            raise ValidationError(
                f"max events must be a whole number from 10 to {MAX_EVENTS_LIMIT}"
            )


# ---- the policy ----


def _label(value: Any) -> Label:
    return _coerce(Label, value, "label")


def qualifies(label: Any, confidence: Any, min_confidence: Any) -> bool:
    """AL-1: a risk label at or above the alert threshold (inclusive)."""
    chosen = _label(label)
    score = _probability(confidence, "confidence")
    threshold = _probability(min_confidence, "the alert threshold")
    return chosen in RISK_LABELS and score >= threshold


UNKNOWN_AGE = timedelta(seconds=10**10)  # "no idea how old": older than any window


def event_age(
    published_at: datetime, first_seen_at: datetime, *, date_known: bool = True
) -> timedelta:
    """How old the event was when this application first saw it; never negative. With
    `date_known` False (the feed gave no usable date, or one from the future) the age is
    `UNKNOWN_AGE`, so the event is never taken for a fresh one."""
    published, seen = _aware(published_at, "publication time"), _aware(first_seen_at, "time seen")
    if not date_known:
        return UNKNOWN_AGE
    return max(timedelta(0), seen - published)


def severity_for(
    label: Any,
    confidence: Any,
    *,
    catalog_status: str,
    event_age: timedelta,
    settings: AlertSettings,
) -> Severity:
    """AL-2: how severe a verdict is (it is assumed to qualify)."""
    chosen = _label(label)
    score = _probability(confidence, "confidence")
    deliberate = (
        chosen is Label.DELIBERATE_DISINFORMATION and score >= settings.critical_disinformation
    )
    contradicted = (
        catalog_status == CATALOG_MISMATCH
        and event_age <= settings.critical_window
        and score >= settings.critical_catalog
    )
    if deliberate or contradicted:
        return Severity.CRITICAL
    if score >= settings.high:
        return Severity.HIGH
    if score >= settings.medium:
        return Severity.MEDIUM
    return Severity.LOW


def assess(
    label: Any,
    confidence: Any,
    *,
    min_confidence: Any,
    catalog_status: str,
    event_age: timedelta,
    settings: AlertSettings,
) -> Severity | None:
    """The severity of the alert a verdict raises, or None when it raises none."""
    if not qualifies(label, confidence, min_confidence):
        return None
    return severity_for(
        label, confidence, catalog_status=catalog_status, event_age=event_age, settings=settings
    )


# ---- tags ----


def tags_for(
    topics: Sequence[str], signal_names: Sequence[str], catalog_status: str
) -> tuple[str, ...]:
    """What makes an alert filterable: the topics the filter found plus the notable evidence.
    `secretariat_attribution` marks a claim attributed to the secretariat (the Secretaria de
    Gestion del Riesgo de Emergencias y Desastres de Cali, formerly DAGRD) with no link to it."""
    names = set(signal_names)
    tags = {t for t in topics if t in {topic.value for topic in Topic}}
    if Topic.SECRETARIAT.value in tags and "official_attribution_unlinked" in names:
        tags.add(TAG_SECRETARIAT_ATTRIBUTION)
    if catalog_status == CATALOG_MISMATCH:
        tags.add(TAG_CATALOG_CONTRADICTION)
    if "obfuscated_text" in names:
        tags.add(TAG_OBFUSCATED)
    if names & {"prediction_claim", "prediction_specific_time"}:
        tags.add(TAG_PREDICTION)
    return tuple(sorted(tags))


def _check_tags(tags: Any) -> tuple[str, ...]:
    if (
        not isinstance(tags, tuple | list)
        or len(tags) > MAX_TAGS
        or not all(isinstance(t, str) and _TAG_RE.fullmatch(t) for t in tags)
    ):
        raise ValidationError("tags are short lowercase names")
    return tuple(sorted(set(tags)))


# ---- status machine (AL-5) ----


@dataclass(frozen=True)
class StatusChange:
    status: AlertStatus
    changed: bool


def apply_status(current: Any, target: Any) -> StatusChange:
    """Move an alert to `target`. The same status again is a no-op; a move that is not allowed
    is `InvalidTransition`; an unknown status is `ValidationError`."""
    now_at = _coerce(AlertStatus, current, "status")
    wanted = _coerce(AlertStatus, target, "status")
    if now_at is wanted:
        return StatusChange(now_at, changed=False)
    if (now_at, wanted) not in _TRANSITIONS:
        raise InvalidTransition(f"an alert cannot go from {now_at.value} to {wanted.value}")
    return StatusChange(wanted, changed=True)


# ---- evidence ----


@dataclass(frozen=True)
class Evidence:
    """The cross-check behind a verdict (AL-3): which stage decided, what the catalogs said, the
    signals, the model and prompt version when the model judged, and whether the verdict is still
    provisional (it waits for the model)."""

    catalog_status: str
    stage: str
    classifier_version: str
    signals: tuple[tuple[str, str, float], ...] = ()
    topics: tuple[str, ...] = ()
    settle_reason: str | None = None
    provisional: bool = False
    llm_model: str | None = None
    prompt_version: str | None = None
    duplicate_of: str | None = None
    near_of: str | None = None
    recycled_of: str | None = None
    source_credibility: int = 3
    truncated: bool = False

    def __post_init__(self) -> None:
        text_ok = all(
            isinstance(v, str) and 1 <= len(v) <= _VERSION_MAX
            for v in (self.stage, self.classifier_version)
        )
        optional_ok = all(
            v is None or (isinstance(v, str) and 1 <= len(v) <= _NAME_MAX and "\x00" not in v)
            for v in (self.settle_reason, self.llm_model, self.prompt_version)
        )
        if (
            not text_ok
            or not optional_ok
            or not isinstance(self.catalog_status, str)
            or not _STATUS_RE.fullmatch(self.catalog_status)
            or not isinstance(self.provisional, bool)
            or not isinstance(self.truncated, bool)
            or not _whole(self.source_credibility, 1, 5)
            or len(self.signals) > 20
            or len(self.topics) > 5
        ):
            raise ValidationError("the evidence of an alert is not valid")
        for link in (self.duplicate_of, self.near_of, self.recycled_of):
            if link is not None:
                validate_slug(link)

    def to_json(self) -> str:
        body = {
            "catalog_status": self.catalog_status,
            "stage": self.stage,
            "classifier_version": self.classifier_version,
            "signals": [list(s) for s in self.signals],
            "topics": list(self.topics),
            "settle_reason": self.settle_reason,
            "provisional": self.provisional,
            "llm_model": self.llm_model,
            "prompt_version": self.prompt_version,
            "duplicate_of": self.duplicate_of,
            "near_of": self.near_of,
            "recycled_of": self.recycled_of,
            "source_credibility": self.source_credibility,
            "truncated": self.truncated,
        }
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if len(text.encode("utf-8")) > MAX_EVIDENCE_BYTES:
            raise ValidationError("the evidence of an alert is too long")
        return text

    @classmethod
    def from_json(cls, text: str) -> "Evidence":
        try:
            body = json.loads(text)
            signals = tuple((str(n), str(lb), float(w)) for n, lb, w in body["signals"])
            return cls(
                catalog_status=body["catalog_status"],
                stage=body["stage"],
                classifier_version=body["classifier_version"],
                signals=signals,
                topics=tuple(body["topics"]),
                settle_reason=body["settle_reason"],
                provisional=body["provisional"],
                llm_model=body["llm_model"],
                prompt_version=body["prompt_version"],
                duplicate_of=body["duplicate_of"],
                near_of=body["near_of"],
                recycled_of=body["recycled_of"],
                source_credibility=body["source_credibility"],
                truncated=body["truncated"],
            )
        except (ValueError, TypeError, KeyError):
            raise ValidationError("stored evidence is invalid") from None


# ---- entities ----


def alert_id_for(item_id: str) -> str:
    """The id of the alert that the FIRST item to raise it gives: the same item always yields the
    same id, so a rebuild finds the alert it would have created."""
    digest = hashlib.sha256(validate_slug(item_id).encode("ascii")).hexdigest()
    return "alt_" + digest[:32]


def _printable(value: Any, what: str, low: int, high: int) -> str:
    if (
        not isinstance(value, str)
        or not low <= len(value) <= high
        or has_control_chars(value)
        or has_unsafe_chars(value)
    ):
        raise ValidationError(f"{what} must be {low} to {high} printable characters")
    return value


@dataclass(frozen=True)
class AlertMember:
    """One item behind an alert, with the verdict it was last evaluated on. `active` is whether
    that verdict still qualifies; the verdict snapshot (`label`, `confidence`, `stage`,
    `settled_at`) is what the sync compares against the stored classification."""

    alert_id: str
    item_id: str
    source_id: str
    source_name: str
    content_hash: str
    excerpt: str
    label: Label
    confidence: float
    stage: str
    rationale: str
    severity: Severity
    tags: tuple[str, ...]
    evidence: Evidence
    active: bool
    occurred_at: datetime
    linked_at: datetime
    event_age_s: int = 0
    settled_at: datetime | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        validate_slug(self.alert_id)
        validate_slug(self.item_id)
        validate_slug(self.source_id)
        _printable(self.source_name, "source name", 1, MAX_SOURCE_NAME_CHARS)
        if not isinstance(self.content_hash, str) or not _HASH_RE.fullmatch(self.content_hash):
            raise ValidationError("content hash must be 64 lowercase hex digits")
        _printable(self.excerpt, "excerpt", 0, MAX_EXCERPT_CHARS)
        set_(self, "label", _coerce(Label, self.label, "label"))
        set_(self, "severity", _coerce(Severity, self.severity, "severity"))
        set_(self, "confidence", _probability(self.confidence, "confidence"))
        if not isinstance(self.stage, str) or not 1 <= len(self.stage) <= _VERSION_MAX:
            raise ValidationError("stage must be short text")
        _printable(self.rationale, "rationale", 1, MAX_RATIONALE_CHARS)
        set_(self, "tags", _check_tags(self.tags))
        if not isinstance(self.evidence, Evidence) or not isinstance(self.active, bool):
            raise ValidationError("an alert member needs evidence and an active flag")
        set_(self, "occurred_at", _aware(self.occurred_at, "occurrence time"))
        set_(self, "linked_at", _aware(self.linked_at, "link time"))
        if not _whole(self.event_age_s, 0, 10**10):
            raise ValidationError("event age must be a whole number of seconds")
        if self.settled_at is not None:
            set_(self, "settled_at", _aware(self.settled_at, "settlement time"))


@dataclass(frozen=True)
class Alert:
    """AL-3. The fields from `label` to `evidence` are those of the leading member (the most
    severe active one); `occurrences` counts the active members plus the copies that were only
    counted (`overflow`); `tags` merge those of all. `downgraded_by_model` marks an alert whose
    provisional verdict the language model settled lower: it stays visible until a person looks."""

    id: str
    label: Label
    severity: Severity
    confidence: float
    rationale: str
    source_id: str
    source_name: str
    item_id: str
    excerpt: str
    tags: tuple[str, ...]
    evidence: Evidence
    status: AlertStatus
    occurrences: int
    first_seen_at: datetime
    last_seen_at: datetime
    created_at: datetime
    updated_at: datetime
    status_changed_at: datetime | None = None
    withdrawn_at: datetime | None = None
    withdrawn_reason: str | None = None
    linked_sources: tuple[tuple[str, str], ...] = field(default=())
    downgraded_by_model: bool = False
    overflow: int = 0
    events_dropped: int = 0

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        validate_slug(self.id)
        validate_slug(self.source_id)
        validate_slug(self.item_id)
        if not isinstance(self.downgraded_by_model, bool):
            raise ValidationError("downgraded_by_model is a flag")
        if not _whole(self.overflow, 0, 10**9) or not _whole(self.events_dropped, 0, 10**9):
            raise ValidationError("overflow and dropped events are whole numbers")
        set_(self, "label", _coerce(Label, self.label, "label"))
        set_(self, "severity", _coerce(Severity, self.severity, "severity"))
        set_(self, "status", _coerce(AlertStatus, self.status, "status"))
        set_(self, "confidence", _probability(self.confidence, "confidence"))
        _printable(self.rationale, "rationale", 1, MAX_RATIONALE_CHARS)
        _printable(self.source_name, "source name", 1, MAX_SOURCE_NAME_CHARS)
        _printable(self.excerpt, "excerpt", 0, MAX_EXCERPT_CHARS)
        set_(self, "tags", _check_tags(self.tags))
        if not isinstance(self.evidence, Evidence):
            raise ValidationError("an alert needs evidence")
        if not _whole(self.occurrences, 0, 10**9):
            raise ValidationError("occurrences must be a whole number")
        for name in ("first_seen_at", "last_seen_at", "created_at", "updated_at"):
            set_(self, name, _aware(getattr(self, name), name))
        for name in ("status_changed_at", "withdrawn_at"):
            if getattr(self, name) is not None:
                set_(self, name, _aware(getattr(self, name), name))
        if (self.withdrawn_at is None) != (self.withdrawn_reason is None):
            raise ValidationError("a withdrawn alert has a time and a reason, or neither")
        if self.withdrawn_reason is not None:
            _printable(self.withdrawn_reason, "withdrawn reason", 1, 200)

    @property
    def withdrawn(self) -> bool:
        return self.withdrawn_at is not None


@dataclass(frozen=True)
class AlertEvent:
    """One line of the audit trail of an alert: what happened, who did it and why.
    `from_absorbed` marks a line copied (with its original time) from an alert that was absorbed."""

    alert_id: str
    at: datetime
    kind: AlertEventKind
    actor: str
    reason: str
    before: str = ""
    after: str = ""
    from_absorbed: bool = False

    def __post_init__(self) -> None:
        validate_slug(self.alert_id)
        if not isinstance(self.from_absorbed, bool):
            raise ValidationError("from_absorbed is a flag")
        object.__setattr__(self, "at", _aware(self.at, "event time"))
        object.__setattr__(self, "kind", _coerce(AlertEventKind, self.kind, "event kind"))
        _printable(self.actor, "actor", 1, 100)
        _printable(self.reason, "reason", 1, 500)
        _printable(self.before, "before", 0, 100)
        _printable(self.after, "after", 0, 100)


def lead_key(member: AlertMember) -> tuple[int, float, datetime, str]:
    """Who leads an alert: the most severe, then the most confident, then the earliest, then the
    smallest item id (the least key leads)."""
    return (-severity_rank(member.severity), -member.confidence, member.occurred_at, member.item_id)


def _shown(lead: AlertMember) -> dict[str, Any]:
    return {
        "label": lead.label,
        "severity": lead.severity,
        "confidence": lead.confidence,
        "rationale": lead.rationale,
        "source_id": lead.source_id,
        "source_name": lead.source_name,
        "item_id": lead.item_id,
        "excerpt": lead.excerpt,
        "evidence": lead.evidence,
    }


def project(
    alert_id: str,
    members: Sequence[AlertMember],
    *,
    previous: Alert | None,
    now: datetime,
    overflow: int = 0,
) -> Alert:
    """The alert that `members` make: the most severe active member (then the most confident,
    then the earliest, then by item id) leads. With no active member the alert keeps what it
    showed and is withdrawn by the caller. `overflow` is the number of copies that were counted
    but are not stored as members. Nothing here depends on the order of `members`."""
    if not members:
        raise ValidationError("an alert needs at least one member")
    active = [m for m in members if m.active]
    pool = active or list(members)
    lead = min(pool, key=lead_key)
    tags = tuple(sorted({t for m in pool for t in m.tags}))[:MAX_TAGS]
    sources: dict[str, str] = {}
    for member in sorted(members, key=lambda m: (m.occurred_at, m.item_id)):
        sources.setdefault(member.source_id, member.source_name)
    base = {
        "id": alert_id,
        "tags": tags,
        "occurrences": len(active) + overflow,
        "overflow": overflow,
        "first_seen_at": min(m.occurred_at for m in members),
        "last_seen_at": max(m.occurred_at for m in members),
        "linked_sources": tuple(sorted(sources.items())),
        "updated_at": now,
    }
    if not active and previous is not None:
        # nothing qualifies any more: show what it showed, the caller withdraws it
        return replace(previous, **{**base, "tags": previous.tags})
    if previous is None:
        return Alert(**base, **_shown(lead), status=AlertStatus.NEW, created_at=now)
    return replace(previous, **base, **_shown(lead))


def merge_member(
    previous: Alert, lead: AlertMember | None, member: AlertMember, *, now: datetime
) -> Alert:
    """`previous` with one more ACTIVE member, without reading the others: the incremental form of
    `project` (same result as projecting all the members again, for the alert's own columns).
    `lead` is the member `previous` shows. A withdrawn alert (or one whose lead is gone) restarts
    from the new member alone, as `project` does when exactly one member is active."""
    restart = previous.withdrawn or lead is None or not lead.active
    sources = dict(previous.linked_sources)
    if member.source_id not in sources or member.occurred_at < previous.first_seen_at:
        sources[member.source_id] = member.source_name
    tags = set() if restart else set(previous.tags)
    tags |= set(member.tags)
    leads = restart or lead_key(member) < lead_key(lead)  # type: ignore[arg-type]
    return replace(
        previous,
        **(_shown(member) if leads else {}),
        tags=tuple(sorted(tags))[:MAX_TAGS],
        occurrences=(1 + previous.overflow) if restart else previous.occurrences + 1,
        first_seen_at=min(previous.first_seen_at, member.occurred_at),
        last_seen_at=max(previous.last_seen_at, member.occurred_at),
        linked_sources=tuple(sorted(sources.items())),
        updated_at=now,
    )


# ---- filters (AL-4) ----


@dataclass(frozen=True)
class AlertFilter:
    """What the list shows. Every field is optional; `status` may also be `withdrawn`. The
    list is always newest first. `until < since` is refused."""

    severity: Severity | None = None
    label: Label | None = None
    source_id: str | None = None
    status: AlertStatus | str | None = None
    tag: str | None = None
    since: datetime | None = None
    until: datetime | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        if self.severity is not None:
            set_(self, "severity", _coerce(Severity, self.severity, "severity"))
        if self.label is not None:
            chosen = _coerce(Label, self.label, "label")
            if chosen not in RISK_LABELS:
                raise ValidationError("an alert has one of the three risk labels")
            set_(self, "label", chosen)
        if self.source_id is not None:
            validate_slug(self.source_id)
        if self.status is not None and self.status != WITHDRAWN:
            set_(self, "status", _coerce(AlertStatus, self.status, "status"))
        if self.status == WITHDRAWN:
            set_(self, "status", WITHDRAWN)
        if self.tag is not None and not _TAG_RE.fullmatch(self.tag):
            raise ValidationError("a tag is a short lowercase name")
        for name in ("since", "until"):
            if getattr(self, name) is not None:
                set_(self, name, _aware(getattr(self, name), name))
        if self.since is not None and self.until is not None and self.until < self.since:
            raise ValidationError("the end of the date range is before its start")


_DAY_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)
_MAX_MOMENT_CHARS = 40


def parse_moment(text: Any, *, end: bool) -> datetime:
    """A bound of a date range as a person types it, in UTC: a day (`2026-03-01`: from its first
    instant, or with `end` through its last one) or a moment in ISO 8601 (`2026-03-01T10:30:00`,
    with or without an offset; with none it is UTC). Anything else is a `ValidationError`."""
    if not isinstance(text, str) or not text.strip() or len(text.strip()) > _MAX_MOMENT_CHARS:
        raise ValidationError("a date must be a day (2026-03-01) or a moment in ISO 8601")
    chosen = text.strip()
    try:
        if _DAY_RE.fullmatch(chosen):
            day = datetime.fromisoformat(chosen).replace(tzinfo=UTC)
            return day.replace(hour=23, minute=59, second=59, microsecond=999999) if end else day
        moment = datetime.fromisoformat(chosen)
    except ValueError:
        raise ValidationError("a date must be a day (2026-03-01) or a moment in ISO 8601") from None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return ensure_utc(moment)


def validate_page(limit: Any, offset: Any) -> tuple[int, int]:
    if not _whole(limit, 1, MAX_PAGE_SIZE):
        raise ValidationError(f"limit must be a whole number from 1 to {MAX_PAGE_SIZE}")
    if not _whole(offset, 0, MAX_OFFSET):
        raise ValidationError("offset must be a whole number from 0")
    return limit, offset


@dataclass(frozen=True)
class AlertPage:
    alerts: tuple[Alert, ...]
    total: int
    limit: int
    offset: int

    @property
    def has_next(self) -> bool:
        return self.offset + len(self.alerts) < self.total

    @property
    def has_previous(self) -> bool:
        return self.offset > 0


@dataclass(frozen=True)
class AlertCounts:
    """The figures behind the badge and `alerts stats`. `active` excludes withdrawn alerts; the
    pairs are sorted by name."""

    active: int
    new: int
    withdrawn: int
    by_status: tuple[tuple[str, int], ...] = ()
    by_severity: tuple[tuple[str, int], ...] = ()
    new_by_severity: tuple[tuple[str, int], ...] = ()
    downgraded: int = 0  # visible alerts the language model lowered or cleared (A11)
    retrying: int = 0  # items the alert step failed on and will try again
    parked: int = 0  # items it gave up on until the next rebuild (A14)


class SyncRef(NamedTuple):
    """An item the alert step has to look at, with the opaque cursor that continues the listing
    after it (the listing is chronological: fetch time, then item id)."""

    item_id: str
    cursor: str


@dataclass(frozen=True)
class ForgottenSignature:
    """The verdict an item had when its alert was forgotten by the retention purge. The item is
    not raised again while its verdict is the same; a changed or upgraded verdict clears it.
    An entry from before the signatures existed has none (every part None) and keeps suppressing."""

    label: str | None
    confidence: float | None
    settled_at: datetime | None

    def matches(self, label: Any, confidence: float, settled_at: datetime | None) -> bool:
        if self.label is None:
            return True
        return (
            str(getattr(label, "value", label)) == self.label
            and confidence == self.confidence
            and settled_at == self.settled_at
        )


@dataclass(frozen=True)
class SyncReport:
    """What one pass of the alert step did."""

    created: int = 0
    merged: int = 0
    updated: int = 0
    withdrawn: int = 0
    reactivated: int = 0
    unchanged: int = 0
    skipped: int = 0
    errors: int = 0
