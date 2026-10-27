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
