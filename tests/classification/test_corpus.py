"""Calibration over generated corpora: templated bulletins (C1) and a mixed day of feeds (C3).

The corpora are synthetic and seeded (see `corpus.py`); the shares of the mix are an assumption,
so the escalation RATE asserted here is a regression guard on the cascade's behavior for that
mix, not a measurement of the real feeds.
"""

from collections import Counter
from datetime import timedelta

import pytest

from social_tracker.classification.domain import ClassifierSettings, Label
from tests.classification.corpus import MIX, mixed_corpus, templated_bulletins
from tests.classification.harness import PUBLISHED, sqlite_world

BATCH = 500


def run_all(world, total: int) -> None:
    done = 0
    while done < total:
        report = world.service.classify_pending(BATCH)
        if report.classified + report.out_of_scope + report.skipped + report.errors == 0:
            break
        done += report.classified + report.out_of_scope


# ---- C1: templated bulletins are not copies of each other ----


def test_two_thousand_templated_bulletins_over_83_days_are_not_flagged_as_copies(tmp_path):
    world = sqlite_world(tmp_path / "templated.db")
    try:
        world.ready()
        bulletins = templated_bulletins(2000, days_span=83, seed=3)
        items = [
            world.add(f"t{n}", text, published_at=PUBLISHED + timedelta(hours=hours))
            for n, (text, hours) in enumerate(bulletins)
        ]
        run_all(world, len(items))
        verdicts = [world.verdict(i) for i in items]
        assert all(v is not None for v in verdicts)
        flagged = [v for v in verdicts if v.duplicate_of or v.recycled_of or v.near_of]
        copies = [v for v in verdicts if v.duplicate_of or v.recycled_of]
        assert len(copies) / len(items) < 0.01  # the requirement: ~0 recycled/duplicate labels
        assert len(flagged) / len(items) < 0.01
        assert Label.RECYCLED_CONTENT not in {v.label for v in verdicts}
    finally:
        world.close()


def test_a_short_headline_from_a_second_source_is_not_recycled(tmp_path):
    world = sqlite_world(tmp_path / "headline.db")
    try:
        world.ready("src_a")
        world.add_source("src_b")
        world.add("a", "Temblor en Cali", source_id="src_a")
        later = world.add(
            "b", "Temblor en Cali", source_id="src_b", published_at=PUBLISHED + timedelta(days=50)
        )
        run_all(world, 2)
        assert world.verdict(later).recycled_of is None
        assert world.verdict(later).duplicate_of is None
    finally:
        world.close()


# ---- C3: the mixed corpus is mostly settled ----


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    """2,000 items of the assumed mix, classified once for all the tests below."""
    world = sqlite_world(tmp_path_factory.mktemp("mixed") / "mixed.db")
    world.ready()
    kinds = {}
    for n, (text, kind) in enumerate(mixed_corpus(2000, seed=11)):
        item = world.add(f"m{n}", text, published_at=PUBLISHED + timedelta(minutes=n))
        kinds[item.id] = kind
    run_all(world, 2000)
    verdicts = {item_id: world.classifications.get(item_id) for item_id in kinds}
    yield kinds, verdicts
    world.close()


def test_the_mix_adds_up_and_expects_about_fourteen_percent_escalated():
    assert sum(k.share for k in MIX) == pytest.approx(1.0)
    assert sum(k.share for k in MIX if k.escalated) == pytest.approx(0.14, abs=0.001)


def test_every_item_of_the_corpus_has_a_verdict(mixed):
    kinds, verdicts = mixed
    assert len(verdicts) == 2000
    assert all(v is not None for v in verdicts.values())


def test_the_escalation_rate_of_the_whole_corpus_is_under_fifteen_percent(mixed):
    _, verdicts = mixed
    escalated = sum(v.needs_llm for v in verdicts.values())
    assert escalated / len(verdicts) < 0.15  # measured 13.0 % for this mix
    in_scope = [v for v in verdicts.values() if v.label is not Label.OUT_OF_SCOPE]
    assert escalated / len(in_scope) < 0.25  # measured 21.9 % of the in-scope items


def misjudged(kind, verdict) -> bool:
    """Scope or escalation different from what the kind implies (a copy of another item inherits
    or links, so it is judged by what it links to)."""
    if (verdict.label is Label.OUT_OF_SCOPE) == kind.in_scope:
        return True
    linked = verdict.near_of or verdict.duplicate_of or verdict.recycled_of
    return verdict.needs_llm != kind.escalated and not linked


def test_each_kind_is_escalated_as_expected(mixed):
    kinds, verdicts = mixed
    wrong: Counter = Counter()
    seen: Counter = Counter()
    for item_id, kind in kinds.items():
        seen[kind.name] += 1
        wrong[kind.name] += misjudged(kind, verdicts[item_id])
    for name, count in wrong.items():  # no kind may be systematically wrong
        assert count / seen[name] < 0.05, (name, count, seen[name])


def test_settled_kinds_get_the_label_one_would_expect(mixed):
    kinds, verdicts = mixed
    by_kind: dict[str, Counter] = {}
    for item_id, kind in kinds.items():
        by_kind.setdefault(kind.name, Counter())[verdicts[item_id].label] += 1
    for name in ("neutral_chatter", "secretariat_complaint", "news_report", "other_disaster_news"):
        counts = by_kind[name]
        assert counts[Label.NO_RISK] / sum(counts.values()) > 0.95, (name, counts)
    linked = by_kind["official_report_linked"]
    assert linked[Label.NO_RISK] / sum(linked.values()) > 0.95
    for name in ("off_topic", "crime", "english"):
        assert by_kind[name][Label.OUT_OF_SCOPE] == sum(by_kind[name].values())


def test_risky_kinds_are_not_settled_as_harmless(mixed):
    kinds, verdicts = mixed
    for item_id, kind in kinds.items():
        if kind.escalated and not (
            verdicts[item_id].near_of
            or verdicts[item_id].duplicate_of
            or verdicts[item_id].recycled_of
        ):
            assert verdicts[item_id].label is not Label.NO_RISK, (kind.name, item_id)


def test_the_settings_that_drive_the_rate_are_the_documented_ones():
    s = ClassifierSettings()
    assert (s.neutral_confidence, s.neutral_min_tokens, s.no_risk_settled_min) == (0.6, 6, 0.5)
    assert (s.inconclusive_below, s.escalate_low, s.escalate_high) == (0.4, 0.4, 0.8)
