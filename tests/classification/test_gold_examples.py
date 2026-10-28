"""The synthetic gold examples (`tests/data/gold_examples.yaml`): 65 labeled Spanish texts (and
one known gap, listed apart) that
state what the cascade should do, as the seed of the gold set of unit 3d.

Each example asserts the scope, the label (or the acceptable labels) and whether it is escalated.
The examples the cascade still gets wrong are listed apart, with the reason, and the test pins
that they are still wrong: fixing one fails the test until the example is moved to the main list.
"""

from collections import Counter
from pathlib import Path

import pytest
import yaml

from social_tracker.classification.domain import Label
from social_tracker.classification.text import prepare
from tests.classification.support import SETTINGS, default, run_rules

DATA = Path(__file__).parents[1] / "data" / "gold_examples.yaml"
GOLD = yaml.safe_load(DATA.read_text(encoding="utf-8"))
EXAMPLES = GOLD["examples"]
GAPS = GOLD.get("known_gaps", [])


def judge(example: dict):
    """The verdict the cascade gives an example: `(topic, result or None)`."""
    _, matcher = default()
    text, url = example["text"], example.get("url")
    topic = matcher.evaluate_prepared(prepare("", text, max_chars=SETTINGS.max_text_chars))
    if not topic.in_scope:
        return topic, None
    return topic, run_rules(text, url=url, credibility=example.get("credibility", 3))


def check(example: dict) -> list[str]:
    """What differs between the expectation and the cascade, as short sentences."""
    expect = example["expect"]
    topic, result = judge(example)
    problems = []
    if topic.in_scope != expect["in_scope"]:
        return [f"scope is {topic.in_scope}, expected {expect['in_scope']}"]
    if result is None:
        if expect.get("reason") and topic.reason.value != expect["reason"]:
            problems.append(f"reason {topic.reason.value}, expected {expect['reason']}")
        return problems
    labels = expect.get("label")
    if labels is not None:
        wanted = labels if isinstance(labels, list) else [labels]
        if result.label.value not in wanted:
            problems.append(f"label {result.label.value}, expected {wanted}")
    if "escalate" in expect and result.escalated != expect["escalate"]:
        problems.append(f"escalated {result.escalated}, expected {expect['escalate']}")
    low, high = expect.get("min_confidence"), expect.get("max_confidence")
    if low is not None and result.confidence < low:
        problems.append(f"confidence {result.confidence} under {low}")
    if high is not None and result.confidence > high:
        problems.append(f"confidence {result.confidence} over {high}")
    return problems


def test_the_gold_file_is_what_it_claims_to_be():
    assert GOLD["version"] == 1
    ids = [e["id"] for e in EXAMPLES]
    assert len(ids) == len(set(ids))
    assert len(EXAMPLES) == 65  # the README derives its count from this file: change both
    kinds = Counter(e["kind"] for e in EXAMPLES)
    for needed in (
        "official_report",
        "news_report",
        "prediction",
        "chain",
        "false_attribution",
        "recycled",
        "neutral",
        "complaint",
        "other_disaster",
        "english",
        "off_topic",
        "crime",
        "evasion",
        "laundering",
    ):
        assert kinds[needed] >= 2 or needed in {"crime"}, needed
    assert "SYNTHETIC" in DATA.read_text(encoding="utf-8")


def test_the_examples_do_not_reproduce_real_accounts_or_addresses():
    text = DATA.read_text(encoding="utf-8").lower()
    for fragment in ("@", "wa.me", "t.me/", "facebook.com", "twitter.com", "instagram.com"):
        assert fragment not in text.replace("@alertas", "").replace(
            "https://official-domain.example@", ""
        ), fragment


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["id"])
def test_the_cascade_does_what_the_example_expects(example: dict):
    assert check(example) == []


@pytest.mark.parametrize("gap", GAPS, ids=lambda g: g["id"])
def test_a_known_gap_is_still_a_gap(gap: dict):
    assert gap["reason"]
    assert check(gap) != [], "the cascade now gets this right: move it to `examples`"


def test_the_expectations_cover_every_label_the_rules_can_give():
    seen = set()
    for example in EXAMPLES:
        wanted = example["expect"].get("label")
        seen |= set(wanted if isinstance(wanted, list) else [wanted])
    seen.discard(None)
    assert {
        Label.NO_RISK.value,
        Label.UNINTENTIONAL_RUMOR.value,
        Label.DELIBERATE_DISINFORMATION.value,
        Label.RECYCLED_CONTENT.value,
        Label.UNCLASSIFIED.value,
    } <= seen
