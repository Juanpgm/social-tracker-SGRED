"""Recycled and near-duplicate content (RC-5, RC-7): stage three of the cascade."""

import math
from datetime import timedelta

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from social_tracker.classification.domain import (
    Classification,
    ClassifierSettings,
    ItemSignature,
    Label,
    Stage,
)
from social_tracker.classification.similarity import signature, similarity
from social_tracker.classification.text import tokens
from social_tracker.items.domain import ItemStatus
from tests.actors import WHO
from tests.classification.harness import (
    OFF_TOPIC,
    PREDICTION,
    PUBLISHED,
    REPORT,
    START,
    World,
    days,
    long_text,
    memory_world,
)


def three_sources(**options) -> World:
    world = memory_world(**options)
    world.ready("src_a")
    world.add_source("src_b")
    world.add_source("src_c")
    return world


def run(world: World, limit: int = 100):
    return world.service.classify_pending(limit)


def changed(text: str, words: int) -> str:
    """`text` with its `words` last words replaced by others."""
    parts = text.split()
    return " ".join(parts[: len(parts) - words] + [f"otra{n}" for n in range(words)])


# ---- duplicates within the window reuse the earlier verdict ----


def test_a_copy_within_the_window_reuses_the_earlier_verdict_and_is_not_escalated():
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a")
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(1))
    report = run(world)
    first, second = world.verdict(original), world.verdict(copy)
    assert first.needs_llm is True
    assert (second.label, second.confidence, second.signals) == (
        first.label,
        first.confidence,
        first.signals,
    )
    assert (second.stage, second.duplicate_of, second.recycled_of) == (
        Stage.DEDUP,
        original.id,
        None,
    )
    assert (second.needs_llm, second.priority) == (False, 0.0)
    assert report.needs_llm == 1
    assert original.id in second.rationale
    assert world.signatures.get(copy.id) is None  # only the original is compared against later


def test_later_copies_all_point_at_the_first_one():
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a")
    second = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(1))
    third = world.add("c", PREDICTION, source_id="src_c", published_at=PUBLISHED + days(2))
    run(world)
    assert world.verdict(second).duplicate_of == original.id
    assert world.verdict(third).duplicate_of == original.id


def test_a_copy_arriving_in_a_later_run_still_finds_the_original():
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a")
    run(world)
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(copy).duplicate_of == original.id


def test_a_copy_published_before_an_original_classified_earlier_is_a_duplicate_of_it():
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a", published_at=PUBLISHED + days(10))
    run(world)
    older = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED)
    run(world)
    assert world.verdict(older).duplicate_of == original.id  # never "recycled" from the future


def test_off_topic_text_is_never_a_duplicate_of_anything():
    world = three_sources()
    world.add("a", OFF_TOPIC, source_id="src_a")
    copy = world.add("b", OFF_TOPIC, source_id="src_b")
    run(world)
    assert world.verdict(copy).label is Label.OUT_OF_SCOPE


# ---- recycled content: older than the window ----


@pytest.mark.parametrize(
    ("age", "recycled"),
    [
        (timedelta(microseconds=1), False),
        (days(6) + timedelta(hours=23), False),
        (days(7) - timedelta(microseconds=1), False),
        (days(7), False),  # exactly the limit is still a duplicate
        (days(7) + timedelta(microseconds=1), True),
        (days(7) + timedelta(seconds=1), True),
        (days(8), True),
        (days(400), True),
    ],
)
def test_a_copy_is_recycled_only_strictly_after_seven_days(age, recycled):
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a")
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + age)
    run(world)
    verdict = world.verdict(copy)
    if recycled:
        assert verdict.label is Label.RECYCLED_CONTENT
        assert (verdict.recycled_of, verdict.duplicate_of) == (original.id, None)
    else:
        assert (verdict.duplicate_of, verdict.recycled_of) == (original.id, None)


def test_a_recycled_copy_references_the_original_with_a_confidence():
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a")
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(8))
    run(world)
    verdict = world.verdict(copy)
    assert (verdict.label, verdict.stage, verdict.recycled_of) == (
        Label.RECYCLED_CONTENT,
        Stage.DEDUP,
        original.id,
    )
    assert verdict.confidence == 0.9  # identical content, neutral credibility
    assert (verdict.needs_llm, verdict.low_confidence) == (False, False)
    assert original.id in verdict.rationale
    assert "identical" in verdict.rationale
    assert "8 days" in verdict.rationale
    assert world.signatures.get(copy.id) is None
    assert {s.name for s in verdict.signals} >= {"prediction_claim", "virality_phrases"}
    assert verdict.priority == 0.0  # settled, so not queued


@pytest.mark.parametrize(("credibility", "confidence"), [(1, 0.96), (3, 0.9), (5, 0.84)])
def test_the_credibility_prior_moves_the_confidence_of_a_recycled_item(credibility, confidence):
    world = memory_world()
    world.ready("src_a")
    world.add_source("src_b", credibility=credibility)
    world.add("a", PREDICTION, source_id="src_a")
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(8))
    run(world)
    assert world.verdict(copy).confidence == confidence
    assert world.verdict(copy).source_credibility_used == credibility


@pytest.mark.parametrize(("threshold", "low"), [(0.9, False), (0.91, True)])
def test_a_recycled_item_is_low_confidence_below_the_alert_threshold(threshold, low):
    world = three_sources()
    world.control.set_alert_min_confidence(threshold, **WHO)
    world.add("a", PREDICTION, source_id="src_a")
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(8))
    run(world)
    assert world.verdict(copy).low_confidence is low


@pytest.mark.parametrize(("threshold", "low"), [(0.56, False), (0.57, True)])
def test_a_duplicate_is_low_confidence_below_the_alert_threshold(threshold, low):
    world = three_sources()
    world.control.set_alert_min_confidence(threshold, **WHO)
    world.add("a", PREDICTION, source_id="src_a")
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(copy).low_confidence is low


def test_a_recycled_copy_the_cascade_cannot_settle_is_queued_with_a_priority():
    world = three_sources(settings=ClassifierSettings(near_duplicate_threshold=0.5))
    base = long_text(4)
    original = world.add("a", base, source_id="src_a")
    edited = world.add("b", changed(base, 12), source_id="src_b", published_at=PUBLISHED + days(9))
    run(world)
    verdict = world.verdict(edited)
    assert verdict.recycled_of == original.id
    assert 0.4 <= verdict.confidence < 0.8
    assert verdict.needs_llm is True
    # the confidence of the recycled evidence, and the topic score of "sismo" in "Cali": 0.6
    assert verdict.priority == round(0.6 * verdict.confidence + 0.3 * 0.6, 4)


def test_the_oldest_match_is_the_original_of_every_later_copy():
    world = three_sources()
    old = world.add("a", PREDICTION, source_id="src_a")
    world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(20))
    newest = world.add("c", PREDICTION, source_id="src_c", published_at=PUBLISHED + days(21))
    run(world)
    assert world.verdict(newest).recycled_of == old.id


# ---- near duplicates ----


def test_a_lightly_edited_old_article_is_recycled_with_a_similarity_based_confidence():
    world = three_sources()
    base, edited_text = long_text(1), changed(long_text(1), 3)
    original = world.add("a", base, source_id="src_a")
    edited = world.add("b", edited_text, source_id="src_b", published_at=PUBLISHED + days(9))
    run(world)
    verdict = world.verdict(edited)
    sim = similarity(signature(tokens(base)), signature(tokens(edited_text)))
    assert 0.8 <= sim < 1.0
    assert (verdict.label, verdict.recycled_of) == (Label.RECYCLED_CONTENT, original.id)
    assert verdict.confidence == round(0.9 * sim, 4)
    assert "similar" in verdict.rationale
    assert verdict.needs_llm is (0.4 <= verdict.confidence < 0.8)


def test_a_heavily_edited_text_is_not_a_near_duplicate():
    world = three_sources()
    base = long_text(2)
    world.add("a", base, source_id="src_a")
    other = world.add("b", changed(base, 40), source_id="src_b", published_at=PUBLISHED + days(9))
    run(world)
    verdict = world.verdict(other)
    assert (verdict.recycled_of, verdict.duplicate_of) == (None, None)
    assert verdict.stage is Stage.RULES
    assert world.signatures.get(other.id) is not None


def test_the_similarity_threshold_is_inclusive_and_configurable():
    base, edited = long_text(3), changed(long_text(3), 3)
    sim = similarity(signature(tokens(base)), signature(tokens(edited)))
    assert 0.5 < sim < 1.0

    def verdict_with(threshold: float):
        world = three_sources(settings=ClassifierSettings(near_duplicate_threshold=threshold))
        world.add("a", base, source_id="src_a")
        copy = world.add("b", edited, source_id="src_b", published_at=PUBLISHED + days(9))
        run(world)
        return world.verdict(copy)

    assert verdict_with(sim).recycled_of is not None  # exactly at the threshold: a match
    assert verdict_with(math.nextafter(sim, 2.0)).recycled_of is None  # just above: none
    assert verdict_with(sim - 0.05).recycled_of is not None


def test_a_threshold_of_one_still_matches_identical_content():
    world = three_sources(settings=ClassifierSettings(near_duplicate_threshold=1.0))
    original = world.add("a", PREDICTION, source_id="src_a")
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(8))
    run(world)
    assert world.verdict(copy).recycled_of == original.id


def test_the_window_is_configurable():
    world = three_sources(settings=ClassifierSettings(recycle_days=2))
    original = world.add("a", PREDICTION, source_id="src_a")
    inside = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(2))
    outside = world.add(
        "c", PREDICTION, source_id="src_c", published_at=PUBLISHED + days(2) + timedelta(seconds=1)
    )
    run(world)
    assert world.verdict(inside).duplicate_of == original.id
    assert world.verdict(outside).recycled_of == original.id


def test_a_recycled_copy_of_something_the_llm_already_judged_is_not_escalated_again():
    world = three_sources(settings=ClassifierSettings(near_duplicate_threshold=0.5))
    base = long_text(4)
    original = world.add("a", base, source_id="src_a")
    # What the LLM stage will leave behind: a judged verdict and the signature of the item.
    world.classifications.add(
        Classification(
            item_id=original.id,
            label=Label.UNINTENTIONAL_RUMOR,
            confidence=0.5,
            stage=Stage.LLM,
            rationale="judged by the language model",
            classified_at=START,
            classifier_version="test",
        )
    )
    world.signatures.add(
        ItemSignature(original.id, original.content_hash, PUBLISHED, signature(tokens(base)))
    )
    world.items.set_status(original.id, ItemStatus.CLASSIFIED)
    edited = world.add("b", changed(base, 12), source_id="src_b", published_at=PUBLISHED + days(9))
    run(world)
    verdict = world.verdict(edited)
    assert (verdict.recycled_of, verdict.needs_llm) == (original.id, False)
    assert 0.4 <= verdict.confidence < 0.8  # it would have been escalated if nobody had judged it


# ---- order independence (RC-10) ----

SPECS = [
    ("a1", PREDICTION, "src_a", 0),
    ("a2", PREDICTION, "src_b", 1),
    ("a3", PREDICTION, "src_c", 9),
    ("b1", long_text(5), "src_a", 2),
    ("b2", changed(long_text(5), 3), "src_b", 12),
    ("c1", OFF_TOPIC, "src_a", 0),
    ("d1", REPORT, "src_c", 3),
    ("e1", "", "src_a", 4),
]


def outcome(order: list[int]) -> dict:
    world = three_sources()
    added = {}
    for index in order:
        external, text, source, offset = SPECS[index]
        item = world.add(external, text, source_id=source, published_at=PUBLISHED + days(offset))
        added[external] = item
    run(world, 1000)
    results = {}
    for item in added.values():
        v = world.verdict(item)
        results[item.id] = (
            v.label,
            v.confidence,
            v.stage,
            v.duplicate_of,
            v.recycled_of,
            v.needs_llm,
            v.priority,
            v.rationale,
        )
    return results


CANONICAL = outcome(list(range(len(SPECS))))


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(order=st.permutations(range(len(SPECS))))
def test_the_results_do_not_depend_on_the_order_the_items_arrived_in(order):
    assert outcome(list(order)) == CANONICAL


def test_the_canonical_outcome_contains_duplicates_and_recycled_items_for_the_property_to_bite():
    kinds = {(v[3] is not None, v[4] is not None) for v in CANONICAL.values()}
    assert (True, False) in kinds
    assert (False, True) in kinds
    assert len(CANONICAL) == len(SPECS)
