"""Vocabulary of the classification cascade. Pure, no I/O.

The cascade (RC-1) runs the CPU stages in order: the topic filter, the rules, the recycled and
near-duplicate check, then (later units) the catalog cross-check and the LLM judgment. Each stage
may decide; `Stage` names the one that did.

Labels are exactly the three risk labels of RC-3 plus `no_risk`, `unclassified` and
`out_of_scope`. A risk label is only ever produced by evidence (a signal or a duplicate match):
the credibility of the source is a prior that moves a confidence, never a label (RC-8).

Everything here is deterministic (RC-10): no clock, no randomness, no I/O.
"""

from dataclasses import dataclass
from enum import StrEnum

from social_tracker.shared.errors import ValidationError


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
    "official_link": SignalSpec(Label.NO_RISK, 0.35),
    "evacuation_claim": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.20),
    "casualty_claim": SignalSpec(Label.UNINTENTIONAL_RUMOR, 0.25),
    "recycled_marker": SignalSpec(Label.RECYCLED_CONTENT, 0.50),
    "old_year_reference": SignalSpec(Label.RECYCLED_CONTENT, 0.25),
    "news_report": SignalSpec(Label.NO_RISK, 0.50),
}


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

    - `recycle_days`: a match to content first seen/published STRICTLY more than this many days
      earlier is recycled; a match within it is a plain duplicate.
    - `near_duplicate_threshold`: estimated Jaccard similarity from which two texts are near
      duplicates (inclusive).
    - `escalate_low`/`escalate_high`: the confidence band [low, high) that goes to the LLM.
    - `inconclusive_below`: a top evidence score under this decides nothing.
    - `conflict_min`/`conflict_margin`: two labels both at least `conflict_min` and closer than
      `conflict_margin` conflict.
    - `old_year_gap`: a year at least this many years before the item's own is an old reference.
    - `max_text_chars`: longer text is cut (and flagged) before the stages see it.
    - `signature_retention_days`: how long near-duplicate signatures are kept.
    """

    recycle_days: int = 7
    near_duplicate_threshold: float = 0.8
    escalate_low: float = 0.4
    escalate_high: float = 0.8
    inconclusive_below: float = 0.2
    conflict_min: float = 0.3
    conflict_margin: float = 0.15
    old_year_gap: int = 2
    max_text_chars: int = 20_000
    signature_retention_days: int = 730

    def __post_init__(self) -> None:
        valid = (
            _whole(self.recycle_days, 1, 3650)
            and _in_range(self.near_duplicate_threshold, 0.0, 1.0)
            and self.near_duplicate_threshold > 0
            and _in_range(self.escalate_low, 0.0, 1.0)
            and _in_range(self.escalate_high, 0.0, 1.0)
            and self.escalate_low <= self.escalate_high
            and _in_range(self.inconclusive_below, 0.0, 1.0)
            and _in_range(self.conflict_min, 0.0, 1.0)
            and _in_range(self.conflict_margin, 0.0, 1.0)
            and self.conflict_margin > 0
            and _whole(self.old_year_gap, 1, 100)
            and _whole(self.max_text_chars, 1, 1_000_000)
            and _whole(self.signature_retention_days, 1, 3650)
        )
        if not valid:
            raise ValidationError("invalid classifier settings")
