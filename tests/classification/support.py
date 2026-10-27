"""The rules pipeline as the service runs it, for tests that look at one text at a time."""

from dataclasses import dataclass
from functools import cache

from social_tracker.classification.domain import ClassifierSettings, Label
from social_tracker.classification.rules import RuleOutcome, evaluate_rules, needs_escalation
from social_tracker.classification.text import prepare
from social_tracker.classification.topics import TopicConfig, TopicMatcher, TopicVerdict
from social_tracker.classification.yaml_io import default_topic_config

SETTINGS = ClassifierSettings()


@cache
def default() -> tuple[TopicConfig, TopicMatcher]:
    config = default_topic_config()
    return config, TopicMatcher(config)


@dataclass(frozen=True)
class Result:
    topic: TopicVerdict
    outcome: RuleOutcome
    escalated: bool

    @property
    def names(self) -> set[str]:
        return {s.name for s in self.outcome.signals}

    @property
    def label(self) -> Label:
        return self.outcome.label

    @property
    def confidence(self) -> float:
        return self.outcome.confidence


def run_rules(
    text: str,
    *,
    title: str = "",
    url: str | None = None,
    credibility: int = 3,
    year: int = 2026,
    config: TopicConfig | None = None,
    settings: ClassifierSettings = SETTINGS,
    source_official: bool = False,
) -> Result:
    """Topic filter, rules and the escalation decision of one text (it must be in scope)."""
    base_config, base_matcher = default()
    matcher = base_matcher if config is None else TopicMatcher(config)
    config = config or base_config
    prepared = prepare(title, text, max_chars=settings.max_text_chars)
    topic = matcher.evaluate_prepared(prepared)
    assert topic.in_scope, f"not in scope: {text!r}"
    outcome = evaluate_rules(
        prepared,
        matcher,
        config,
        credibility=credibility,
        reference_year=year,
        item_url=url,
        settings=settings,
        source_official=source_official,
    )
    escalated = needs_escalation(
        in_scope=True,
        confidence=outcome.confidence,
        inconclusive=outcome.inconclusive,
        conflicting=outcome.conflicting,
        duplicate_of_judged=False,
        settings=settings,
        label=outcome.label,
    )
    return Result(topic, outcome, escalated)
