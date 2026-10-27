"""Vocabulary of the classification cascade. Pure, no I/O.

The cascade (RC-1) runs the CPU stages in order: the topic filter, the rules, the recycled and
near-duplicate check, then (later units) the catalog cross-check and the LLM judgment. Each stage
may decide; `Stage` names the one that did.

Labels are exactly the three risk labels of RC-3 plus `no_risk`, `unclassified` and
`out_of_scope`. A risk label is only ever produced by evidence (a signal or a duplicate match):
the credibility of the source is a prior that moves a confidence, never a label (RC-8).

Everything here is deterministic (RC-10): no clock, no randomness, no I/O.
"""

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from social_tracker.classification.similarity import NUM_PERM
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError
from social_tracker.shared.ids import validate_slug
from social_tracker.shared.text import has_control_chars, has_unsafe_chars

MAX_RATIONALE_CHARS = 300  # RC-4: a short rationale
MAX_SIGNALS = 20
MAX_VERSION_CHARS = 64
_STATUS_RE = re.compile(r"[a-z_]{1,32}")
_HASH_RE = re.compile(r"[0-9a-f]{64}")


class Label(StrEnum):
    UNINTENTIONAL_RUMOR = "unintentional_rumor"
    DELIBERATE_DISINFORMATION = "deliberate_disinformation"
    RECYCLED_CONTENT = "recycled_content"
    NO_RISK = "no_risk"
    UNCLASSIFIED = "unclassified"
    OUT_OF_SCOPE = "out_of_scope"


RISK_LABELS = (
    Label.DELIBERATE_DISINFORMATION,
    Label.UNINTENTIONAL_RUMOR,
    Label.RECYCLED_CONTENT,
)


class Stage(StrEnum):
    TOPIC = "topic"
    RULES = "rules"
    DEDUP = "dedup"
    CATALOG = "catalog"
    LLM = "llm"


class Topic(StrEnum):
    EARTHQUAKE = "earthquake"
    SECRETARIAT = "secretariat"
    OTHER_DISASTER = "other_disaster"


class OutOfScopeReason(StrEnum):
    NO_TOPIC_MATCH = "no_topic_match"
    CRIME_PUBLIC_ORDER = "crime_public_order"
    EMPTY_TEXT = "empty_text"
    NON_SPANISH = "non_spanish"


@dataclass(frozen=True)
class TopicVerdict:
    """What the topic filter decided. `matched_terms` are the folded configuration terms found
    (never raw item text), sorted; empty when the item is out of scope. `score` is in [0, 1]."""

    in_scope: bool
    topics: tuple[Topic, ...]
    matched_terms: tuple[str, ...]
    score: float
    reason: OutOfScopeReason | None = None


@dataclass(frozen=True)
class SignalSpec:
    """What a rule signal argues for and how strongly by default."""

    label: Label
    weight: float


# The signals of the rules stage (RC-1 stage 2). The weight of each can be overridden in the topic
# configuration (`weights`); the names are fixed because the detectors are code. A weight is the
# confidence one signal alone gives its label; signals of one label combine as 1 - prod(1 - w).
# `obfuscated_text` argues for no label (UNCLASSIFIED): it is evidence of an evasion attempt that
# raises suspicion and the queue priority and never decides a label alone.
SIGNALS: dict[str, SignalSpec] = {
    "prediction_claim": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.45),
    "prediction_specific_time": SignalSpec(Label.DELIBERATE_DISINFORMATION, 0.35),
    "pseudo_authority": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.30),
    "virality_phrases": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.20),
    "censorship_bait": SignalSpec(Label.DELIBERATE_DISINFORMATION, 0.30),
    "all_caps": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.15),
    "repeated_exclamation": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.10),
    "unverifiable_sources": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.25),
    "official_attribution_unlinked": SignalSpec(Label.DELIBERATE_DISINFORMATION, 0.45),
    "official_link": SignalSpec(Label.NO_RISK, 0.50),
    "evacuation_claim": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.20),
    "casualty_claim": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.25),
    "recycled_marker": SignalSpec(Label.RECYCLED_CONTENT, 0.50),
    "old_year_reference": SignalSpec(Label.RECYCLED_CONTENT, 0.25),
    "news_report": SignalSpec(Label.NO_RISK, 0.60),
    "obfuscated_text": SignalSpec(Label.UNCLASSIFIED, 0.30),
}

# Risk signals that are only style or a passing mention: they argue for their label but do not, on
# their own, stop a clean report from being settled as `no_risk` (see `rules.combine`).
WEAK_SIGNALS = frozenset({"all_caps", "repeated_exclamation", "old_year_reference"})

# A signal that restates another one: the time of a prediction is part of the same claim, so the
# pair must not be counted as two opinions when looking for a conflict between labels.
COMPANION_SIGNALS = {"prediction_specific_time": "prediction_claim"}

# An unlinked attribution without an alarm word counts this share of its weight.
PLAIN_ATTRIBUTION_FACTOR = 0.55


@dataclass(frozen=True)
class Signal:
    """One piece of evidence found by the rules: its name (a key of `SIGNALS`), the label it
    argues for and its weight in (0, 1]. It carries no text of the item."""

    name: str
    label: Label
    weight: float

    def __post_init__(self) -> None:
        if self.name not in SIGNALS:
            raise ValidationError("unknown signal")
        if not isinstance(self.label, Label):
            raise ValidationError("a signal needs a label")
        if not _is_number(self.weight) or not 0.0 <= self.weight <= 1.0:
            raise ValidationError("a signal weight must be a number from 0 to 1")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _in_range(value: object, low: float, high: float) -> bool:
    return _is_number(value) and low <= value <= high  # type: ignore[operator]


def _whole(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


@dataclass(frozen=True)
class ClassifierSettings:
    """Thresholds of the cascade; the defaults are the product's (RC-5, RC-7).

    - `recycle_days`: a match to content first SEEN (by this application, never by a date the feed
      claims) strictly more than this many days apart is recycled; a match within it is a plain
      duplicate.
    - `near_duplicate_threshold`: estimated Jaccard similarity from which two texts are near
      duplicates (inclusive).
    - `min_near_tokens`: a text with fewer words than this is compared by identical content only;
      short headlines and templated bulletins are alike without being copies.
    - `candidate_limit`: near-duplicate candidates verified per item (1..1000); the most
      band-sharing ones first. Hitting it is counted in the report.
    - `escalate_low`/`escalate_high`: the confidence band [low, high) of a RISK label that goes to
      the LLM.
    - `inconclusive_below`: the evidence of the strongest label under this decides nothing (the
      item is `unclassified` and goes to the LLM). It equals `escalate_low` by default so the
      escalation is monotonic: all the evidence under `escalate_high` is escalated, none above.
    - `neutral_confidence`/`neutral_min_tokens`: an on-topic text of at least that many words with
      no signal at all is `no_risk` at that confidence and is not escalated.
    - `no_risk_settled_min`: a `no_risk` verdict resting on some evidence is settled from this
      confidence on; below it the item is inconclusive.
    - `conflict_min`/`conflict_margin`: two labels both at least `conflict_min` and closer than
      `conflict_margin` conflict.
    - `old_year_gap`: a year at least this many years before the item's own is an old reference.
    - `max_text_chars`: longer text is read at both ends (and flagged) before the stages see it.
    - `signature_retention_days`: how long near-duplicate signatures are kept.
    - `classification_retention_days`: how long the verdict of an item that no longer exists is
      kept.
    - `max_item_failures`: consecutive per-item errors after which an item is parked as
      `classification_failed`.
    """

    recycle_days: int = 7
    near_duplicate_threshold: float = 0.8
    min_near_tokens: int = 12
    candidate_limit: int = 500
    escalate_low: float = 0.4
    escalate_high: float = 0.8
    inconclusive_below: float = 0.4
    neutral_confidence: float = 0.6
    neutral_min_tokens: int = 6
    no_risk_settled_min: float = 0.5
    conflict_min: float = 0.3
    conflict_margin: float = 0.15
    old_year_gap: int = 2
    max_text_chars: int = 20_000
    signature_retention_days: int = 730
    classification_retention_days: int = 180
    max_item_failures: int = 3

    def __post_init__(self) -> None:
        valid = (
            _whole(self.recycle_days, 1, 3650)
            and _in_range(self.near_duplicate_threshold, 0.0, 1.0)
            and self.near_duplicate_threshold > 0
            and _whole(self.min_near_tokens, 1, 10_000)
            and _whole(self.candidate_limit, 1, 1000)
            and _in_range(self.escalate_low, 0.0, 1.0)
            and _in_range(self.escalate_high, 0.0, 1.0)
            and self.escalate_low <= self.escalate_high
            and _in_range(self.inconclusive_below, 0.0, 1.0)
            and _in_range(self.neutral_confidence, 0.0, 1.0)
            and _whole(self.neutral_min_tokens, 1, 10_000)
            and _in_range(self.no_risk_settled_min, 0.0, 1.0)
            and _in_range(self.conflict_min, 0.0, 1.0)
            and _in_range(self.conflict_margin, 0.0, 1.0)
            and self.conflict_margin > 0
            and _whole(self.old_year_gap, 1, 100)
            and _whole(self.max_text_chars, 1, 1_000_000)
            and _whole(self.signature_retention_days, 1, 3650)
            and _whole(self.classification_retention_days, 1, 3650)
            and _whole(self.max_item_failures, 1, 100)
        )
        if not valid:
            raise ValidationError("invalid classifier settings")


# ---- stored shapes ----


def signals_to_json(signals: tuple[Signal, ...]) -> str:
    """Signals as compact JSON, sorted by name: the text does not depend on discovery order."""
    rows = [
        {"name": s.name, "label": s.label.value, "weight": s.weight}
        for s in sorted(signals, key=lambda s: s.name)
    ]
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def signals_from_json(text: str) -> tuple[Signal, ...]:
    """The inverse of `signals_to_json`; anything that does not parse is a `ValidationError`."""
    try:
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise TypeError
        return tuple(Signal(r["name"], Label(r["label"]), r["weight"]) for r in rows)
    except (ValueError, TypeError, KeyError):
        raise ValidationError("stored signals are invalid") from None


def _flag(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{what} must be a bool")
    return value


def _coerce[E: StrEnum](enum_cls: type[E], value: Any, what: str) -> E:
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        raise ValidationError(f"unknown {what}") from None


@dataclass(frozen=True)
class Classification:
    """The cascade's verdict on one item (RC-4): label, confidence in [0, 1], the stage that
    decided and a short rationale, plus the evidence and the flags downstream stages need.

    - `topics`/`signals`: what the topic filter and the rules found (never item text).
    - `low_confidence`: below the alert threshold when classified (RC-11).
    - `needs_llm` and `priority`: queued for the LLM judgment (RC-7); the hourly cap is applied by
      the stage that consumes the queue.
    - `duplicate_of`: a copy, within the window, of an item already classified (its verdict is
      reused); `recycled_of`: matches content first seen/published longer ago (RC-5). Never both.
    - `truncated`: the text was cut before analysis. `scope_reason`: why an item is out of scope.
    - `catalog_status`: outcome of the official catalog cross-check (`unavailable` until it exists).
    - `source_credibility_used`: the 1..5 prior that adjusted the confidence (RC-8).
    """

    item_id: str
    label: Label
    confidence: float
    stage: Stage
    rationale: str
    classified_at: datetime
    classifier_version: str
    signals: tuple[Signal, ...] = ()
    topics: tuple[Topic, ...] = ()
    low_confidence: bool = False
    needs_llm: bool = False
    duplicate_of: str | None = None
    recycled_of: str | None = None
    truncated: bool = False
    catalog_status: str = "unavailable"
    priority: float = 0.0
    scope_reason: OutOfScopeReason | None = None
    source_credibility_used: int = 3

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        validate_slug(self.item_id)
        set_(self, "label", _coerce(Label, self.label, "label"))
        set_(self, "stage", _coerce(Stage, self.stage, "stage"))
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, int | float)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValidationError("confidence must be a number from 0 to 1")
        self._check_rationale()
        set_(self, "classified_at", ensure_utc(self.classified_at))
        if (
            not isinstance(self.classifier_version, str)
            or not 1 <= len(self.classifier_version) <= MAX_VERSION_CHARS
        ):
            raise ValidationError("classifier version must be short text")
        if not isinstance(self.signals, tuple) or len(self.signals) > MAX_SIGNALS:
            raise ValidationError(f"at most {MAX_SIGNALS} signals are kept")
        if not all(isinstance(s, Signal) for s in self.signals):
            raise ValidationError("signals must be Signal objects")
        set_(self, "topics", tuple(_coerce(Topic, t, "topic") for t in self.topics))
        for flag in ("low_confidence", "needs_llm", "truncated"):
            _flag(getattr(self, flag), flag)
        self._check_references()
        if not isinstance(self.catalog_status, str) or not _STATUS_RE.fullmatch(
            self.catalog_status
        ):
            raise ValidationError("catalog status must be a lowercase name")
        if not _is_number(self.priority) or not 0.0 <= self.priority <= 1.0:
            raise ValidationError("priority must be a number from 0 to 1")
        if not _whole(self.source_credibility_used, 1, 5):
            raise ValidationError("credibility must be a whole number from 1 to 5")

    def _check_rationale(self) -> None:
        text = self.rationale
        if (
            not isinstance(text, str)
            or not 1 <= len(text) <= MAX_RATIONALE_CHARS
            or has_control_chars(text)
            or has_unsafe_chars(text)
        ):
            raise ValidationError(
                f"rationale must be 1 to {MAX_RATIONALE_CHARS} printable characters"
            )

    def _check_references(self) -> None:
        for other in (self.duplicate_of, self.recycled_of):
            if other is not None and validate_slug(other) == self.item_id:
                raise ValidationError("an item cannot refer to itself")
        if self.duplicate_of is not None and self.recycled_of is not None:
            raise ValidationError("an item is a duplicate or recycled, not both")
        if self.needs_llm and self.duplicate_of is not None:
            raise ValidationError("a duplicate of a classified item is not escalated")
        if self.scope_reason is not None:
            object.__setattr__(
                self, "scope_reason", _coerce(OutOfScopeReason, self.scope_reason, "reason")
            )
        if (self.label is Label.OUT_OF_SCOPE) != (self.scope_reason is not None):
            raise ValidationError("only an out-of-scope item has a scope reason, and it needs one")
        if self.label is Label.OUT_OF_SCOPE and self.needs_llm:
            raise ValidationError("an out-of-scope item is never escalated")


@dataclass(frozen=True)
class ItemSignature:
    """What the near-duplicate check keeps of a classified item: its content hash, the time it
    counts from (first seen or published, whichever is earlier) and its MinHash signature. No
    text, so it can outlive the item itself."""

    item_id: str
    content_hash: str
    ref_at: datetime
    minhash: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_slug(self.item_id)
        if not isinstance(self.content_hash, str) or not _HASH_RE.fullmatch(self.content_hash):
            raise ValidationError("content hash must be 64 lowercase hex digits")
        object.__setattr__(self, "ref_at", ensure_utc(self.ref_at))
        if len(self.minhash) != NUM_PERM or not all(_whole(v, 0, 2**32 - 1) for v in self.minhash):
            raise ValidationError("a signature is 80 unsigned 32-bit values")
        object.__setattr__(self, "minhash", tuple(self.minhash))


@dataclass(frozen=True)
class ClassificationStats:
    """What `classify stats` reports. The pairs are sorted by name."""

    total: int
    by_label: tuple[tuple[str, int], ...]
    out_of_scope_by_reason: tuple[tuple[str, int], ...]
    needs_llm: int
    low_confidence: int
    duplicates: int
    recycled: int
    truncated: int
