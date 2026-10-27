"""Duplicates and recycled copies around an escalation, and the queue that drops its oldest."""

from datetime import timedelta

import pytest

from social_tracker.classification.domain import Classification, Label, Stage
from social_tracker.llm.domain import JudgeError, JudgeErrorKind, LlmSettings
from tests.llm.harness import LlmWorld, error, memory_llm
from tests.llm.support import NOW, response

COPIES = "src_copies"
MORE = "src_more"


def ready(w: LlmWorld) -> None:
    """Global analysis on, the news source and the source the copies come from (one source never
    repeats its own text)."""
    w.ready()
    w.ready(COPIES)
    w.ready(MORE)


def waiting_dup(
    w: LlmWorld,
    n: int,
    original: Classification,
    *,
    credibility: int = 3,
    same_text: bool = True,
    source: str = COPIES,
    **changes,
) -> Classification:
    """An exact copy that waits for its original (it took the original's provisional verdict).
    The text is the original's to the byte unless `same_text` is off."""
    text = w.items.get(original.item_id).text if same_text else None
    return w.seed(
        n,
        text=text,
        source=source,
        needs_llm=False,
        priority=0.0,
        duplicate_of=original.item_id,
        awaiting_original=True,
        stage=Stage.DEDUP,
        label=original.label,
        confidence=original.confidence,
        rationale=f"duplicate of {original.item_id}: reused its {original.label.value} verdict",
        source_credibility_used=credibility,
        **changes,
    )


# ---- duplicates waiting for their original ----


def test_waiting_duplicates_take_the_judged_verdict_with_their_own_credibility_and_a_note():
    w = memory_llm()
    ready(w)
    original = w.seed(1)
    low = waiting_dup(w, 2, original, credibility=1)
    high = waiting_dup(w, 3, original, credibility=5, source=MORE)
    w.judge.default = response(label=Label.RECYCLED_CONTENT, confidence=0.7)
    report = w.escalate()
    assert (report.judged, report.reused) == (1, 2)
    assert w.judge.calls == 1  # the copies cost no call
    for dup, credibility in ((low, 1), (high, 5)):
        after = w.verdict(dup)
        assert (after.label, after.confidence) == (Label.RECYCLED_CONTENT, 0.7)
        assert after.awaiting_original is False
        assert after.duplicate_of == original.item_id
        assert after.stage is Stage.DEDUP
        assert after.source_credibility_used == credibility
        assert original.item_id in after.rationale
        assert "reused" in after.rationale
        j = w.escalations.get_judgment(dup.item_id)
        assert (j.outcome, j.reused_from, j.label, j.confidence) == (
            "reused",
            original.item_id,
            "recycled_content",
            0.7,
        )
        assert j.llm_model == "st-gemma4-e4b"


def test_a_low_confidence_judgment_is_low_confidence_in_its_copies_too():
    w = memory_llm()
    ready(w)
    original = w.seed(1)
    dup = waiting_dup(w, 2, original)
    w.judge.default = response(confidence=0.45)
    w.escalate()
    assert w.verdict(dup).low_confidence is True


def test_copies_of_another_original_are_untouched():
    w = memory_llm()
    ready(w)
    original = w.seed(1, priority=0.9)
    other = w.seed(2, priority=0.1)  # still queued, but this run only reaches the first one
    stranger = waiting_dup(w, 3, other)
    w.escalate(limit=1)
    assert w.verdict(stranger).awaiting_original is True
    assert w.verdict(original).stage is Stage.LLM


def test_copies_are_settled_in_the_same_transaction_as_the_judgment():
    w = memory_llm()
    ready(w)
    original = w.seed(1)
    dup = waiting_dup(w, 2, original)
    real = w.escalations.add_judgment
    seen = []

    def spying(judgment):
        seen.append(w.uow.depth)
        return real(judgment)

    w.escalations.add_judgment = spying
    w.escalate()
    assert seen == [1, 1]  # the original's and the copy's records, both inside one transaction
    assert w.verdict(dup).awaiting_original is False


def test_an_original_judged_invalid_releases_its_copies_with_their_provisional_verdict():
    w = memory_llm()
    ready(w)
    original = w.seed(1)
    dup = waiting_dup(w, 2, original)
    w.prime_success()
    w.judge.script.extend([JudgeError(JudgeErrorKind.INVALID_OUTPUT)] * 2)
    report = w.escalate()
    assert (report.invalid, report.judged) == (1, 0)
    after = w.verdict(dup)
    assert after.awaiting_original is False
    assert (after.label, after.confidence) == (dup.label, dup.confidence)
    # the same treatment as the original: low confidence, settled as invalid
    assert (after.low_confidence, after.settle_reason) == (True, "invalid")
    j = w.escalations.get_judgment(dup.item_id)
    assert (j.outcome, j.label, j.reused_from) == ("reused", None, original.item_id)


def test_an_original_that_is_parked_releases_its_copies():
    w = memory_llm(LlmSettings(max_failures=1))
    ready(w)
    original = w.seed(1)
    dup = waiting_dup(w, 2, original)
    w.prime_success()
    w.judge.default = error("timeout")
    assert w.escalate().parked == 1
    after = w.verdict(dup)
    assert after.awaiting_original is False
    assert (after.low_confidence, after.settle_reason) == (True, "parked")
    assert w.escalations.get_judgment(dup.item_id) is None  # nothing was judged


def test_an_original_that_fails_for_now_keeps_its_copies_waiting():
    w = memory_llm()
    ready(w)
    original = w.seed(1)
    dup = waiting_dup(w, 2, original)
    w.judge.default = error("unreachable")
    w.escalate()
    assert w.verdict(dup).awaiting_original is True
    w.judge.default = None
    w.clock.advance(timedelta(hours=1))
    w.escalate()
    assert w.verdict(dup).awaiting_original is False


# ---- recycled copies of an item that is already judged ----


def test_a_recycled_copy_of_a_judged_original_is_not_sent_again():
    w = memory_llm()
    w.ready()
    judged = w.seed(1, needs_llm=False, priority=0.0, stage=Stage.LLM, label=Label.NO_RISK)
    recycled = w.seed(
        2, label=Label.RECYCLED_CONTENT, recycled_of=judged.item_id, stage=Stage.DEDUP
    )
    report = w.escalate()
    assert (w.judge.calls, report.reused) == (0, 1)
    after = w.verdict(recycled)
    assert after.needs_llm is False
    assert after.label is Label.RECYCLED_CONTENT  # the CPU verdict stands


def test_a_recycled_copy_of_an_original_that_was_not_judged_is_sent():
    w = memory_llm()
    w.ready()
    cpu_only = w.seed(1, needs_llm=False, priority=0.0, stage=Stage.RULES, label=Label.NO_RISK)
    recycled = w.seed(
        2, label=Label.RECYCLED_CONTENT, recycled_of=cpu_only.item_id, stage=Stage.DEDUP
    )
    w.escalate()
    assert [r.item_id for r in w.judge.requests] == [recycled.item_id]


# ---- the queue that drops its oldest ----


def seed_queue(w: LlmWorld, count: int) -> list[Classification]:
    return [w.seed(n, classified_at=NOW + timedelta(minutes=n)) for n in range(count)]


@pytest.mark.parametrize(("queued", "dropped"), [(2, 0), (3, 0), (4, 1), (5, 2)])
def test_the_oldest_beyond_the_queue_bound_are_dropped_and_counted(queued, dropped):
    w = memory_llm(LlmSettings(queue_max=3, hourly_cap=1))
    ready(w)
    stored = seed_queue(w, queued)
    report = w.escalate()
    assert report.dropped == dropped
    for gone in stored[:dropped]:
        after = w.verdict(gone)
        assert after.needs_llm is False
        assert after.low_confidence is True
        assert (after.label, after.stage) == (gone.label, gone.stage)  # the CPU result stays
        assert w.escalations.get_state(gone.item_id).state == "dropped"
    assert w.escalations.counts().dropped == dropped


def test_the_ones_kept_after_a_drop_are_the_newest():
    w = memory_llm(LlmSettings(queue_max=2, hourly_cap=10))
    ready(w)
    stored = seed_queue(w, 4)
    w.escalate()
    assert sorted(r.item_id for r in w.judge.requests) == sorted(s.item_id for s in stored[2:])


def test_a_dropped_original_releases_its_copies():
    w = memory_llm(LlmSettings(queue_max=1, hourly_cap=10))
    ready(w)
    oldest = w.seed(1, classified_at=NOW)
    w.seed(2, classified_at=NOW + timedelta(minutes=1))
    dup = waiting_dup(w, 3, oldest)
    w.escalate()
    assert w.verdict(dup).awaiting_original is False


def test_a_dropped_item_is_never_judged_later():
    w = memory_llm(LlmSettings(queue_max=1, hourly_cap=1))
    ready(w)
    stored = seed_queue(w, 2)
    w.escalate()
    w.clock.advance(timedelta(hours=2))
    w.escalate()
    assert stored[0].item_id not in [r.item_id for r in w.judge.requests]


def test_a_dry_run_drops_nothing():
    w = memory_llm(LlmSettings(queue_max=1))
    ready(w)
    stored = seed_queue(w, 3)
    report = w.escalate(dry_run=True)
    assert (report.dropped, report.pending_drops) == (0, 2)  # it says what it would drop
    assert all(w.verdict(s).needs_llm for s in stored)
