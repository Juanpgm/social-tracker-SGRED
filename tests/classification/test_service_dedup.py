"""Recycled and near-duplicate content (RC-5, RC-7): stage three of the cascade."""

import hashlib
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
from social_tracker.classification.similarity import (
    numbers_digest,
    signature,
    similarity,
    text_digest,
)
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
    """`text` with its `words` last words replaced by others (no digits: a number would be a
    different report)."""
    parts = text.split()
    return " ".join(parts[: len(parts) - words] + [f"otra{chr(97 + n)}" for n in range(words)])


# A prediction long enough to be compared with near copies (18 words).
RUMOR = PREDICTION + ", alfa beta gama delta epsilon zeta eta theta"
WORDS = (
    "alfa", "beta", "gama", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda",
    "omega", "sigma", "tau", "upsilon", "fi", "chi", "psi",
)  # fmt: skip


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
    assert second.awaiting_original is True  # its original has not been judged yet
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


def test_a_copy_seen_earlier_than_an_original_classified_first_is_a_duplicate_in_the_window():
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a", published_at=PUBLISHED + days(3))
    run(world)
    older = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED)
    run(world)
    assert world.verdict(older).duplicate_of == original.id


def test_a_copy_seen_long_before_an_original_classified_first_is_recycled_not_a_duplicate():
    world = three_sources()
    original = world.add("a", RUMOR, source_id="src_a", published_at=PUBLISHED + days(10))
    run(world)
    older = world.add("b", RUMOR, source_id="src_b", published_at=PUBLISHED)
    run(world)
    verdict = world.verdict(older)
    assert (verdict.label, verdict.recycled_of, verdict.duplicate_of) == (
        Label.RECYCLED_CONTENT,
        original.id,
        None,
    )  # the gap counts in either direction: arriving late does not make it a copy "of the future"


def test_off_topic_text_is_never_a_duplicate_of_anything():
    world = three_sources()
    world.add("a", OFF_TOPIC, source_id="src_a")
    copy = world.add("b", OFF_TOPIC, source_id="src_b")
    run(world)
    assert world.verdict(copy).label is Label.OUT_OF_SCOPE


def test_a_duplicate_of_an_original_waiting_for_the_llm_is_linked_and_flagged_not_final():
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a")
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    first, second = world.verdict(original), world.verdict(copy)
    assert first.needs_llm is True
    assert (second.duplicate_of, second.awaiting_original) == (original.id, True)
    assert second.needs_llm is False  # it is never queued: the original's judgment settles both
    assert "waits for the LLM" in second.rationale
    assert world.service.stats().awaiting_original == 1


def test_a_duplicate_of_a_settled_original_is_not_flagged_as_waiting():
    world = three_sources()
    original = world.add("a", REPORT, source_id="src_a")
    copy = world.add("b", REPORT, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(original).needs_llm is False
    assert (world.verdict(copy).duplicate_of, world.verdict(copy).awaiting_original) == (
        original.id,
        False,
    )


def test_a_duplicate_of_a_judged_original_is_not_flagged_as_waiting():
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a")
    judged_base = judged(original, 0.77)
    world.classifications.add(judged_base)
    world.signatures.add(
        ItemSignature(
            original.id,
            original.content_hash,
            original.fetched_at,
            signature(tokens(PREDICTION)),
        )
    )
    world.items.set_status(original.id, ItemStatus.CLASSIFIED)
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    verdict = world.verdict(copy)
    assert (verdict.duplicate_of, verdict.awaiting_original) == (original.id, False)
    assert verdict.confidence == 0.77  # a judged confidence is the judge's, not a rules one


@pytest.mark.parametrize(("copy_credibility", "confidence"), [(1, 0.62), (3, 0.56), (5, 0.5)])
def test_a_duplicate_carries_the_credibility_prior_of_its_own_source(copy_credibility, confidence):
    world = memory_world()
    world.ready("src_a", credibility=5)  # the original's source: it lowered the confidence to 0.5
    world.add_source("src_b", credibility=copy_credibility)
    original = world.add("a", PREDICTION, source_id="src_a")
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(original).confidence == 0.5
    verdict = world.verdict(copy)
    assert (verdict.confidence, verdict.source_credibility_used) == (confidence, copy_credibility)


# ---- exact content only is reused (S1) ----


@pytest.mark.parametrize(
    ("original_text", "copy_text"),
    [
        ("Temblor de magnitud 3.2 en Cali a las 10:30 am, sin daños reportados", None),
        ("Temblor de magnitud 3.2 en Cali a las 10:30 am, sin daños reportados", "7.9"),
        ("Temblor de magnitud 3.2 en Cali a las 10:30 am, sin daños reportados", "3.3"),
    ],
)
def test_exact_text_is_a_duplicate_and_a_changed_number_is_not(original_text, copy_text):
    world = three_sources()
    long_original = original_text + " segun el reporte oficial de esta mañana en la ciudad de Cali"
    long_copy = long_original if copy_text is None else long_original.replace("3.2", copy_text)
    original = world.add("a", long_original, source_id="src_a")
    copy = world.add("b", long_copy, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    verdict = world.verdict(copy)
    if copy_text is None:
        assert verdict.duplicate_of == original.id
    else:
        assert verdict.duplicate_of is None
        assert verdict.near_of is None  # different numbers: not even a near copy
        assert verdict.stage is Stage.RULES  # the rules ran on its own text


QUAKE_REPORT = (
    "El Servicio Geológico Colombiano informó un sismo de magnitud 3.2 en Cali a las 10:30 "
    "sin reportes de daños ni heridos en la ciudad ni en los municipios vecinos"
)


@pytest.mark.parametrize(
    "changed_report",
    [
        QUAKE_REPORT.replace("3.2", "7.9"),  # the magnitude
        QUAKE_REPORT.replace("10:30", "11:45"),  # the time
        QUAKE_REPORT.replace("3.2", "3.3"),  # one digit
        QUAKE_REPORT + " y 12 réplicas",  # an extra number
        QUAKE_REPORT.replace("3.2 ", ""),  # a number fewer
    ],
)
@pytest.mark.parametrize("offset", [1, 30])  # inside the window and long after it
def test_a_report_with_a_changed_number_never_inherits_the_verdict_of_the_original(
    changed_report, offset
):
    world = three_sources()
    world.add("a", QUAKE_REPORT, source_id="src_a")
    copy = world.add("c", changed_report, source_id="src_b", published_at=PUBLISHED + days(offset))
    run(world)
    verdict = world.verdict(copy)
    assert verdict.duplicate_of is None
    assert verdict.recycled_of is None
    assert verdict.near_of is None
    assert verdict.stage is Stage.RULES
    assert world.signatures.get(copy.id) is not None


def test_a_near_copy_with_the_same_numbers_runs_its_own_rules_and_links_to_the_original():
    world = three_sources()
    original = world.add("a", QUAKE_REPORT, source_id="src_a")
    variant = QUAKE_REPORT.replace("sin reportes", "sin mayores reportes")
    near = world.add("b", variant, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    verdict = world.verdict(near)
    assert (verdict.near_of, verdict.duplicate_of, verdict.recycled_of) == (original.id, None, None)
    assert verdict.stage is Stage.RULES  # its own verdict, not the original's
    assert original.id in verdict.rationale
    assert world.signatures.get(near.id) is not None  # a candidate for later copies
    assert world.service.stats().near_duplicates == 1


def test_a_report_that_adds_a_panic_claim_is_judged_on_its_own_text():
    world = three_sources()
    world.add("a", QUAKE_REPORT, source_id="src_a")
    panic = QUAKE_REPORT + " pero hay 300 muertos y evacuen ya"
    near = world.add("b", panic, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    verdict = world.verdict(near)
    assert {"casualty_claim", "evacuation_claim"} <= {s.name for s in verdict.signals}
    assert verdict.label is not Label.NO_RISK


def test_the_same_text_with_other_case_accents_and_punctuation_is_an_exact_copy():
    world = three_sources()
    original = world.add("a", RUMOR, source_id="src_a")
    shouted = RUMOR.upper().replace(",", ";") + "!!"
    copy = world.add("b", shouted, source_id="src_b", published_at=PUBLISHED + days(1))
    assert copy.content_hash != original.content_hash
    run(world)
    assert world.verdict(copy).duplicate_of == original.id


# ---- time is first sight, never the date the feed claims (S2) ----


@pytest.mark.parametrize("claimed_age", [8, 15, 30, 400])
def test_a_backdated_publication_cannot_make_new_content_look_old_or_the_original_new(claimed_age):
    world = three_sources()
    original = world.add("a", RUMOR, source_id="src_a")
    run(world)
    # The copy is fetched a day later but CLAIMS to have been published long ago: still a
    # duplicate (one day apart in reality), never recycled.
    copy = world.add(
        "b",
        RUMOR,
        source_id="src_b",
        published_at=PUBLISHED - days(claimed_age),
        fetched_at=PUBLISHED + days(1),
    )
    run(world)
    verdict = world.verdict(copy)
    assert (verdict.duplicate_of, verdict.recycled_of) == (original.id, None)


@pytest.mark.parametrize("claimed_age", [8, 30])
def test_a_forged_early_original_does_not_make_the_real_report_recycled(claimed_age):
    world = three_sources()
    forged = world.add(
        "a",
        RUMOR,
        source_id="src_a",
        published_at=PUBLISHED - days(claimed_age),  # claims to be old; was seen just now
        fetched_at=PUBLISHED + days(1),
    )
    real = world.add("b", RUMOR, source_id="src_b", published_at=PUBLISHED, fetched_at=PUBLISHED)
    run(world)
    # the one seen first here (the real report) is the original; the forged one is its copy
    assert world.verdict(real).duplicate_of is None
    assert world.verdict(forged).duplicate_of == world.verdict(real).item_id


@pytest.mark.parametrize("claimed_future", [1, 30, 3000])
def test_a_future_publication_date_changes_nothing(claimed_future):
    world = three_sources()
    original = world.add("a", RUMOR, source_id="src_a")
    copy = world.add(
        "b",
        RUMOR,
        source_id="src_b",
        published_at=PUBLISHED + days(claimed_future),
        fetched_at=PUBLISHED + days(1),
    )
    run(world)
    assert world.verdict(copy).duplicate_of == original.id


def test_the_year_reference_uses_the_year_the_item_was_seen_not_the_year_it_claims():
    world = three_sources()
    text = "El terremoto de 2019 sacudió Cali"
    backdated = world.add(
        "a", text, source_id="src_a", published_at=PUBLISHED.replace(year=2019), fetched_at=START
    )
    run(world)
    assert "old_year_reference" in {s.name for s in world.verdict(backdated).signals}


# ---- short and templated texts (C1) ----


def test_a_short_headline_repeated_by_another_source_weeks_later_is_not_recycled():
    world = three_sources()
    original = world.add("a", "Temblor en Cali", source_id="src_a")
    later = world.add("b", "Temblor en Cali", source_id="src_b", published_at=PUBLISHED + days(40))
    run(world)
    verdict = world.verdict(later)
    assert (verdict.recycled_of, verdict.duplicate_of) == (None, None)
    assert world.signatures.get(later.id) is not None
    assert world.verdict(original).recycled_of is None


def test_a_short_headline_repeated_inside_the_window_is_still_a_duplicate():
    world = three_sources()
    original = world.add("a", "Temblor en Cali", source_id="src_a")
    copy = world.add("b", "Temblor en Cali", source_id="src_b", published_at=PUBLISHED + days(7))
    run(world)
    assert world.verdict(copy).duplicate_of == original.id


@pytest.mark.parametrize("words", [1, 5, 11])
def test_a_text_just_under_the_word_minimum_is_compared_by_identical_content_only(words):
    world = three_sources()
    text = " ".join(["sismo", *WORDS[: words - 1]])
    world.add("a", text, source_id="src_a")
    edited = text + "s"  # one character more: a near copy for any longer text
    near = world.add("b", edited, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(near).near_of is None


def test_a_text_of_exactly_the_word_minimum_can_be_a_near_copy():
    world = three_sources()
    text = " ".join(["sismo", *WORDS[:11]])  # 12 words
    original = world.add("a", text, source_id="src_a")
    near = world.add("b", text + "s", source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(near).near_of == original.id


def test_the_word_minimum_is_configurable():
    world = three_sources(settings=ClassifierSettings(min_near_tokens=30))
    text = " ".join(["sismo", *WORDS, *WORDS[:7]])  # 20 words
    world.add("a", text, source_id="src_a")
    near = world.add("b", text + "s", source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(near).near_of is None


# ---- candidates: the original is found among many (C2) ----


def colliders(world, probe_text: str, count: int) -> None:
    """`count` older signatures that share exactly one band with the probe's and nothing else."""
    words = tokens(probe_text)
    sign = signature(words)
    same_numbers = numbers_digest(words)
    for n in range(count):
        minhash = tuple(v if i < 5 else (v + 1 + n) % 2**32 for i, v in enumerate(sign))
        world.signatures.add(
            ItemSignature(
                f"itm_collider{n:05d}",
                f"{n:064x}",
                PUBLISHED - days(2) - timedelta(seconds=n),
                minhash,
                f"{n + 10**6:064x}",
                same_numbers,
                len(words),
            )
        )


@pytest.mark.parametrize("limit", [1, 5, 50])
def test_the_original_is_found_among_hundreds_of_colliding_signatures(limit):
    world = three_sources(settings=ClassifierSettings(candidate_limit=limit))
    base = long_text(21)
    original = world.add("orig", base, source_id="src_b", published_at=PUBLISHED - days(1))
    run(world)
    colliders(world, base, 260)  # all older than the original and all sharing one band
    near = world.add("near", changed(base, 2), source_id="src_c", published_at=PUBLISHED + days(30))
    run(world)
    verdict = world.verdict(near)
    assert (verdict.recycled_of, verdict.label) == (original.id, Label.RECYCLED_CONTENT)


def test_an_exact_copy_is_found_among_hundreds_of_colliding_signatures_with_any_limit():
    world = three_sources(settings=ClassifierSettings(candidate_limit=1))
    base = long_text(22)
    original = world.add("orig", base, source_id="src_b", published_at=PUBLISHED - days(1))
    run(world)
    colliders(world, base, 260)
    copy = world.add("copy", base, source_id="src_c", published_at=PUBLISHED + days(30))
    run(world)
    assert world.verdict(copy).recycled_of == original.id  # the exact lookup is never capped


def test_a_near_copy_is_found_when_a_decoy_with_other_numbers_ranks_first():
    # The decoy shares every band with the probe and is older, so it ranks first among the
    # candidates; its numbers differ, so it is rejected. With a candidate limit of one the real
    # original, ranked second, would never be examined.
    world = three_sources()
    base = "Sismo de magnitud 4.5 en Cali: " + long_text(30)
    decoy = base.replace("4.5", "6.1")
    world.add("decoy", decoy, source_id="src_a", published_at=PUBLISHED - days(3))
    original = world.add("orig", base, source_id="src_b", published_at=PUBLISHED - days(2))
    near = world.add("near", changed(base, 2), source_id="src_c", published_at=PUBLISHED + days(9))
    run(world)
    assert world.verdict(near).recycled_of == original.id


def test_the_candidate_limit_is_configurable_and_a_truncated_lookup_is_counted():
    world = three_sources(settings=ClassifierSettings(candidate_limit=2))
    base = long_text(31)
    for n in range(6):
        world.add(f"o{n}", changed(base, 5) + f" otra{chr(97 + n)}", source_id="src_a")
    run(world)
    probe = world.add("p", base, source_id="src_b", published_at=PUBLISHED + days(1))
    report = run(world)
    assert report.candidate_cap_hits >= 1
    assert world.verdict(probe) is not None


def test_an_untruncated_lookup_is_not_counted_as_a_cap_hit():
    world = three_sources()
    world.add("a", long_text(32), source_id="src_a")
    world.add("b", changed(long_text(32), 2), source_id="src_b", published_at=PUBLISHED + days(1))
    assert run(world).candidate_cap_hits == 0


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
    original = world.add("a", RUMOR, source_id="src_a")
    copy = world.add("b", RUMOR, source_id="src_b", published_at=PUBLISHED + age)
    run(world)
    verdict = world.verdict(copy)
    if recycled:
        assert verdict.label is Label.RECYCLED_CONTENT
        assert (verdict.recycled_of, verdict.duplicate_of) == (original.id, None)
    else:
        assert (verdict.duplicate_of, verdict.recycled_of) == (original.id, None)


def test_a_recycled_copy_references_the_original_with_a_confidence():
    world = three_sources()
    original = world.add("a", RUMOR, source_id="src_a")
    copy = world.add("b", RUMOR, source_id="src_b", published_at=PUBLISHED + days(8))
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
    world.add("a", RUMOR, source_id="src_a")
    copy = world.add("b", RUMOR, source_id="src_b", published_at=PUBLISHED + days(8))
    run(world)
    assert world.verdict(copy).confidence == confidence
    assert world.verdict(copy).source_credibility_used == credibility


@pytest.mark.parametrize(("threshold", "low"), [(0.9, False), (0.91, True)])
def test_a_recycled_item_is_low_confidence_below_the_alert_threshold(threshold, low):
    world = three_sources()
    world.control.set_alert_min_confidence(threshold, **WHO)
    world.add("a", RUMOR, source_id="src_a")
    copy = world.add("b", RUMOR, source_id="src_b", published_at=PUBLISHED + days(8))
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
    old = world.add("a", RUMOR, source_id="src_a")
    world.add("b", RUMOR, source_id="src_b", published_at=PUBLISHED + days(20))
    newest = world.add("c", RUMOR, source_id="src_c", published_at=PUBLISHED + days(21))
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
    assert (verdict.recycled_of, verdict.duplicate_of, verdict.near_of) == (None, None, None)
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
    original = world.add("a", RUMOR, source_id="src_a")
    copy = world.add("b", RUMOR, source_id="src_b", published_at=PUBLISHED + days(8))
    run(world)
    assert world.verdict(copy).recycled_of == original.id


def test_the_window_is_configurable():
    world = three_sources(settings=ClassifierSettings(recycle_days=2))
    original = world.add("a", RUMOR, source_id="src_a")
    inside = world.add("b", RUMOR, source_id="src_b", published_at=PUBLISHED + days(2))
    outside = world.add(
        "c", RUMOR, source_id="src_c", published_at=PUBLISHED + days(2) + timedelta(seconds=1)
    )
    run(world)
    assert world.verdict(inside).duplicate_of == original.id
    assert world.verdict(outside).recycled_of == original.id


def judged(original, confidence: float, **changes) -> Classification:
    """What the LLM stage will leave behind: a judged verdict."""
    return Classification(
        item_id=original.id,
        label=Label.UNINTENTIONAL_RUMOR,
        confidence=confidence,
        stage=Stage.LLM,
        rationale="judged by the language model",
        classified_at=START,
        classifier_version="test",
        **changes,
    )


def test_a_recycled_copy_of_something_the_llm_already_judged_is_not_escalated_again():
    world = three_sources(settings=ClassifierSettings(near_duplicate_threshold=0.5))
    base = long_text(4)
    original = world.add("a", base, source_id="src_a")
    world.classifications.add(judged(original, 0.5))
    words = tokens(base)
    world.signatures.add(
        ItemSignature(
            original.id,
            original.content_hash,
            original.fetched_at,
            signature(words),
            text_digest(words),
            numbers_digest(words),
            len(words),
        )
    )
    world.items.set_status(original.id, ItemStatus.CLASSIFIED)
    edited = world.add("b", changed(base, 12), source_id="src_b", published_at=PUBLISHED + days(9))
    run(world)
    verdict = world.verdict(edited)
    assert (verdict.recycled_of, verdict.needs_llm) == (original.id, False)
    assert 0.4 <= verdict.confidence < 0.8  # it would have been escalated if nobody had judged it


# ---- order (RC-10) ----

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
            v.near_of,
            v.needs_llm,
            v.priority,
            v.rationale,
        )
    return results


CANONICAL = outcome(list(range(len(SPECS))))


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(order=st.permutations(range(len(SPECS))))
def test_the_results_of_one_run_do_not_depend_on_the_order_the_items_were_stored_in(order):
    assert outcome(list(order)) == CANONICAL


def test_the_canonical_outcome_contains_duplicates_and_recycled_items_for_the_property_to_bite():
    kinds = {(v[3] is not None, v[4] is not None) for v in CANONICAL.values()}
    assert (True, False) in kinds
    assert (False, True) in kinds
    assert len(CANONICAL) == len(SPECS)


# Items of one story seen over two years, classified ONE AT A TIME in any order (the worst case of
# a feed that delivers late): who is the original depends on who was classified first, but the
# structure of the result must not.
STORY = [
    ("s0", 0),
    ("s1", 1),
    ("s2", 5),
    ("s3", 20),
    ("s4", 400),
    ("s5", 800),
]


def story_outcome(order: list[int]) -> dict:
    world = three_sources()
    text = RUMOR
    items = {}
    sources = ["src_a", "src_b", "src_c"]
    for index in order:
        name, offset = STORY[index]
        items[name] = world.add(
            name,
            text + "!" * (index + 1),  # the same words, so a distinct stored item per source
            source_id=sources[index % 3],
            published_at=PUBLISHED + days(offset),
        )
        run(world)
    return {name: world.verdict(item) for name, item in items.items()}, items


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(order=st.permutations(range(len(STORY))))
def test_property_one_story_has_one_original_and_every_gap_is_judged_in_either_direction(order):
    verdicts, items = story_outcome(list(order))
    by_id = {item.id: (name, item) for name, item in items.items()}
    originals = [
        name for name, v in verdicts.items() if v.duplicate_of is None and v.recycled_of is None
    ]
    assert len(originals) == 1  # exactly one item is not a copy of another
    offsets = dict(STORY)
    for name, verdict in verdicts.items():
        link = verdict.duplicate_of or verdict.recycled_of
        if link is None:
            continue
        other = by_id[link][0]
        gap = abs(offsets[name] - offsets[other])
        # a copy is recycled exactly when the two were first seen more than 7 days apart
        assert (verdict.recycled_of is not None) is (gap > 7), (name, other, gap)


def test_an_800_day_old_item_arriving_last_is_recycled_of_the_story_it_repeats():
    verdicts, items = story_outcome([0, 1, 2, 3, 4, 5])
    assert verdicts["s5"].recycled_of == items["s0"].id
    assert verdicts["s5"].label is Label.RECYCLED_CONTENT
    reversed_verdicts, reversed_items = story_outcome([5, 4, 3, 2, 1, 0])
    assert reversed_verdicts["s5"].duplicate_of is None  # it came first here: the original
    assert reversed_verdicts["s0"].recycled_of == reversed_items["s5"].id


# ---- the edges of the near-duplicate lookup ----


def plant(
    world,
    probe_text: str,
    name: str,
    *,
    shared: int,
    partial: int = 0,
    ago: int = 2,
    numbers: str | None = None,
    short: bool = False,
):
    """A stored original with a verdict whose signature shares `shared` whole bands with the
    probe's, then `partial` bands of which one row in five differs, and nothing else: its
    similarity to the probe is (5 * shared + 4 * partial) / 80."""
    words = tokens(probe_text)
    probe = signature(words)
    values = []
    for band in range(16):
        for row in range(5):
            i = band * 5 + row
            if band < shared:
                values.append(probe[i])
            elif band < shared + partial:
                values.append(probe[i] if row < 4 else (probe[i] + 1) % 2**32)
            else:
                values.append((probe[i] + 7 + i) % 2**32)
    item_id = f"itm_{name}"
    digest = hashlib.sha256(name.encode()).hexdigest()
    world.signatures.add(
        ItemSignature(
            item_id,
            digest,
            PUBLISHED - days(ago),
            tuple(values),
            None,
            numbers if numbers is not None else numbers_digest(words),
            3 if short else len(words),
        )
    )
    world.classifications.add(
        Classification(
            item_id=item_id,
            label=Label.NO_RISK,
            confidence=0.6,
            stage=Stage.RULES,
            rationale="planted original",
            classified_at=START,
            classifier_version="test",
        )
    )
    return item_id


def test_a_lookup_that_returns_exactly_the_limit_is_no_cap_hit_and_one_more_is():
    for count, hits in ((2, 0), (3, 1)):
        world = three_sources(settings=ClassifierSettings(candidate_limit=2))
        text = long_text(40)
        colliders(world, text, count)
        world.add("probe", text, source_id="src_a")
        assert run(world).candidate_cap_hits == hits, count


def test_the_most_similar_candidate_wins_even_when_it_ranks_after_a_weaker_one():
    world = three_sources(settings=ClassifierSettings(near_duplicate_threshold=0.5))
    text = long_text(41)
    plant(world, text, "ranked_first_weaker", shared=10)  # 10 bands, similarity 0.625
    best = plant(world, text, "ranked_second_stronger", shared=9, partial=7)  # 0.9125
    probe = world.add("probe", text, source_id="src_a", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(probe).near_of == best


def test_the_first_ranked_wins_over_a_weaker_one_ranked_after_it():
    world = three_sources(settings=ClassifierSettings(near_duplicate_threshold=0.5))
    text = long_text(42)
    best = plant(world, text, "ranked_first_stronger", shared=10, partial=6)  # 0.925
    plant(world, text, "ranked_second_weaker", shared=9, partial=0)  # 0.5625
    probe = world.add("probe", text, source_id="src_a", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(probe).near_of == best


def test_among_equally_similar_candidates_the_best_ranked_wins():
    world = three_sources(settings=ClassifierSettings(near_duplicate_threshold=0.5))
    text = long_text(43)
    older = plant(world, text, "same_older", shared=12, partial=2, ago=5)
    plant(world, text, "same_newer", shared=12, partial=2, ago=3)
    probe = world.add("probe", text, source_id="src_a", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(probe).near_of == older


def test_a_decoy_that_shares_every_band_but_has_other_numbers_cannot_hide_the_original():
    # The decoy ranks first (all 16 bands, older) and is refused for its numbers; the original
    # ranks second. With a candidate limit of one the original would never be examined.
    world = three_sources()
    text = "Sismo de magnitud 4.5 en Cali: " + long_text(44)
    plant(world, text, "decoy", shared=16, ago=4, numbers=hashlib.sha256(b"6.1").hexdigest())
    original = plant(world, text, "original", shared=14, partial=2, ago=3)
    probe = world.add("probe", text, source_id="src_a", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(probe).near_of == original


def test_a_stored_text_under_the_word_minimum_is_never_a_near_copy_of_a_longer_one():
    world = three_sources()
    short = " ".join(["sismo", *WORDS[:10]])  # 11 words
    world.add("a", short, source_id="src_a")
    longer = short + " omega"  # 12 words, the same numbers (none), nearly the same text
    near = world.add("b", longer, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert (world.verdict(near).near_of, world.verdict(near).recycled_of) == (None, None)


def test_a_short_text_does_not_match_an_identical_one_seen_much_later():
    world = three_sources()
    later = world.add("b", "Temblor en Cali", source_id="src_b", published_at=PUBLISHED + days(20))
    run(world)
    earlier = world.add("a", "Temblor en Cali", source_id="src_a", published_at=PUBLISHED)
    run(world)
    for item in (earlier, later):
        assert (world.verdict(item).duplicate_of, world.verdict(item).recycled_of) == (None, None)


def test_a_judged_original_is_never_waiting_even_if_its_queue_flag_is_set():
    world = three_sources()
    original = world.add("a", PREDICTION, source_id="src_a")
    world.classifications.add(judged(original, 0.77, needs_llm=True))
    world.signatures.add(
        ItemSignature(
            original.id, original.content_hash, original.fetched_at, signature(tokens(PREDICTION))
        )
    )
    world.items.set_status(original.id, ItemStatus.CLASSIFIED)
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(copy).awaiting_original is False


def test_a_judged_confidence_is_copied_untouched_whatever_the_two_credibilities():
    world = memory_world()
    world.ready("src_a", credibility=5)
    world.add_source("src_b", credibility=1)
    original = world.add("a", PREDICTION, source_id="src_a")
    world.classifications.add(judged(original, 0.77, source_credibility_used=5))
    world.signatures.add(
        ItemSignature(
            original.id, original.content_hash, original.fetched_at, signature(tokens(PREDICTION))
        )
    )
    world.items.set_status(original.id, ItemStatus.CLASSIFIED)
    copy = world.add("b", PREDICTION, source_id="src_b", published_at=PUBLISHED + days(1))
    run(world)
    assert world.verdict(copy).confidence == 0.77
