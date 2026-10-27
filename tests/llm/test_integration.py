"""The real cascade feeding the escalation: what the classifier queues is what the judge gets."""

from datetime import timedelta

import pytest

from social_tracker.classification.domain import ClassifierSettings, Label, Stage
from social_tracker.classification.rules import needs_escalation
from social_tracker.llm.domain import LlmSettings
from tests.classification.harness import (
    ATTRIBUTION,
    BENIGN_QUAKE,
    OFF_TOPIC,
    PREDICTION,
    PUBLISHED,
    REPORT,
)
from tests.llm.harness import sqlite_llm
from tests.llm.support import response

SETTINGS = ClassifierSettings()


# ---- the boundaries of the escalation decision (RC-7) ----


@pytest.mark.parametrize(
    ("confidence", "label", "sent"),
    [
        (0.39, None, False),  # an unlabelled confidence under the band
        (0.4, None, True),  # the lower end is inside the band
        (0.79, None, True),
        (0.8, None, False),  # 0.8 is settled
        (0.79, Label.UNINTENTIONAL_RUMOR, True),
        (0.8, Label.UNINTENTIONAL_RUMOR, False),
        (0.3, Label.DELIBERATE_DISINFORMATION, True),  # a risk label has no lower bound
        (0.99, Label.DELIBERATE_DISINFORMATION, False),
        (0.79, Label.NO_RISK, False),  # a no_risk verdict has no band
        (0.1, Label.NO_RISK, False),
    ],
)
def test_the_confidence_band_that_goes_to_the_model(confidence, label, sent):
    got = needs_escalation(
        in_scope=True,
        confidence=confidence,
        inconclusive=False,
        conflicting=False,
        duplicate_of_judged=False,
        settings=SETTINGS,
        label=label,
    )
    assert got is sent


@pytest.mark.parametrize(
    ("changes", "sent"),
    [
        ({"in_scope": False, "confidence": 0.5}, False),
        ({"duplicate_of_judged": True, "confidence": 0.5}, False),
        ({"inconclusive": True, "confidence": 0.95}, True),
        ({"conflicting": True, "confidence": 0.95}, True),
    ],
)
def test_out_of_scope_and_judged_duplicates_never_go_and_doubts_always_do(changes, sent):
    base = {
        "in_scope": True,
        "confidence": 0.5,
        "inconclusive": False,
        "conflicting": False,
        "duplicate_of_judged": False,
        "settings": SETTINGS,
        "label": Label.UNINTENTIONAL_RUMOR,
    }
    assert needs_escalation(**{**base, **changes}) is sent


# ---- from the real classifier to the judge ----


def classified(tmp_path, texts, **settings):
    w = sqlite_llm(tmp_path / "e2e.db", LlmSettings(**settings))
    w.ready()
    ids = []
    for n, text in enumerate(texts):
        item = w.base.add(f"g{n}", text, published_at=PUBLISHED + timedelta(minutes=n))
        ids.append(item.id)
    w.base.service.classify_pending()
    return w, ids


def test_the_judge_gets_exactly_what_the_classifier_queued(tmp_path):
    w, ids = classified(tmp_path, [PREDICTION, ATTRIBUTION, REPORT, BENIGN_QUAKE, OFF_TOPIC])
    try:
        verdicts = {i: w.classifications.get(i) for i in ids}
        waiting = {i for i, v in verdicts.items() if v.needs_llm}
        assert waiting  # the sample has doubtful items...
        assert len(waiting) < len(ids)  # ...and settled ones
        report = w.escalate()
        assert {r.item_id for r in w.judge.requests} == waiting
        assert report.judged == len(waiting)
        for item_id, before in verdicts.items():
            after = w.classifications.get(item_id)
            if item_id in waiting:
                assert after.stage is Stage.LLM
                assert after.signals == before.signals  # the evidence stays
            else:
                assert after == before  # settled items are untouched
    finally:
        w.close()


def test_a_real_exact_copy_waits_for_its_original_and_takes_the_judgment(tmp_path):
    w = sqlite_llm(tmp_path / "copy.db")
    try:
        w.ready("src_a", credibility=5)
        w.base.add_source("src_b", credibility=1)
        first = w.base.add("g1", PREDICTION, source_id="src_a")
        second = w.base.add(
            "g2", PREDICTION, source_id="src_b", published_at=PUBLISHED + timedelta(hours=1)
        )
        w.base.service.classify_pending()
        original, copy = w.classifications.get(first.id), w.classifications.get(second.id)
        assert original.needs_llm, "the sample must be doubtful for this test to mean anything"
        assert copy.duplicate_of == first.id
        assert copy.awaiting_original is True
        w.judge.default = response(label=Label.DELIBERATE_DISINFORMATION, confidence=0.74)
        report = w.escalate()
        assert (report.judged, report.reused, w.judge.calls) == (1, 1, 1)
        settled = w.classifications.get(second.id)
        assert (settled.label, settled.confidence) == (Label.DELIBERATE_DISINFORMATION, 0.74)
        assert settled.awaiting_original is False
        assert settled.source_credibility_used == 1  # its own prior, not the original's 5
        assert w.classifications.get(first.id).source_credibility_used == 5
        assert w.classifications.stats().awaiting_original == 0
    finally:
        w.close()


def test_sending_back_a_reclassified_item_lets_it_be_judged_afresh(tmp_path):
    """`reclassify` forgets the verdict and with it the judgment and the bookkeeping."""
    from social_tracker.items.domain import ItemStatus

    w, ids = classified(tmp_path, [PREDICTION])
    try:
        (item_id,) = ids
        assert w.escalate().judged == 1
        assert w.escalations.get_judgment(item_id) is not None
        w.items.set_status(item_id, ItemStatus.CLASSIFICATION_FAILED)
        assert w.base.service.reclassify(ItemStatus.CLASSIFICATION_FAILED) == 1
        assert w.escalations.get_judgment(item_id) is None
        w.base.service.classify_pending()
        assert w.escalate().judged == 1
        assert w.judge.calls == 2
    finally:
        w.close()


def test_the_retention_purge_of_old_verdicts_takes_their_judgments_with_them(tmp_path):
    w, ids = classified(tmp_path, [PREDICTION])
    try:
        (item_id,) = ids
        w.escalate()
        w.items.purge_fetched_before(
            w.clock.now() + timedelta(days=3650),
            purge_queued_fetched_before=w.clock.now() + timedelta(days=3650),
        )
        w.clock.advance(timedelta(days=400))
        assert w.base.service.prune_history() == 1
        assert w.escalations.get_judgment(item_id) is None
    finally:
        w.close()
