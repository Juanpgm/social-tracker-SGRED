"""The contract of the escalation repository, for the in-memory reference and for SQLite alike.

What every implementation must honour:
- the queue holds the items that still exist as `classified`, belong to an eligible source and
  wait for the judge, best first (priority, then oldest classified, then id), and leaves out the
  items that wait out a backoff, are parked or were dropped;
- the hourly budget is counted atomically and never goes past the cap;
- a judgment is stored once and never overwritten; bookkeeping is replaced in place;
- the lease is one slot that expires and that only its holder releases.
"""

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from social_tracker.classification.domain import Classification
from social_tracker.items.domain import ItemStatus
from social_tracker.llm.ports import (
    EscalationCounts,
    EscalationRepository,
    EscalationState,
    Judgment,
)
from tests.items.factories import make_item
from tests.llm.support import NOW, item_id, verdict

HOUR = datetime(2026, 3, 1, 12, tzinfo=UTC)


def surface(port: type) -> frozenset[str]:
    return frozenset(
        name for name, member in vars(port).items() if not name.startswith("_") and callable(member)
    )


def public(repo: object) -> set[str]:
    return {n for n in dir(repo) if not n.startswith("_") and callable(getattr(repo, n))}


def judgment(n: int | str = 1, **changes: Any) -> Judgment:
    fields: dict[str, Any] = {
        "item_id": item_id(n),
        "outcome": "judged",
        "label": "no_risk",
        "confidence": 0.9,
        "rationale": "reports a past event",
        "evidence": ("past tense", "official link"),
        "llm_model": "st-gemma4-e4b",
        "prompt_version": "judge-1",
        "raw_output": '{"label":"no_risk"}',
        "prior_label": "unintentional_rumor",
        "prior_confidence": 0.55,
        "prior_stage": "rules",
        "reused_from": None,
        "judged_at": NOW + timedelta(microseconds=123456),
    }
    fields.update(changes)
    return Judgment(**fields)


def state(n: int | str = 1, **changes: Any) -> EscalationState:
    fields: dict[str, Any] = {
        "item_id": item_id(n),
        "state": "escalation_failed",
        "attempts": 2,
        "item_failures": 1,
        "next_attempt_at": NOW + timedelta(minutes=5),
        "last_error": "llm_unreachable",
        "updated_at": NOW,
    }
    fields.update(changes)
    return EscalationState(**fields)


@dataclass
class Rig:
    repo: EscalationRepository
    classifications: Any
    items: Any

    def seed(
        self,
        n: int,
        *,
        source: str = "src_news",
        status: ItemStatus = ItemStatus.CLASSIFIED,
        **changes: Any,
    ) -> Classification:
        """An item with a stored verdict. `needs_llm` is on unless asked otherwise."""
        item = make_item(f"guid-{n}", source_id=source, text=f"text {n}")
        # the item id comes from the source and the external id: build the verdict for THAT id
        stored = verdict(item.id.removeprefix("itm_"), **changes)
        assert self.items.add_new([item]).new == 1
        assert self.items.set_status(item.id, status)
        assert self.classifications.add(stored)
        return stored


class EscalationRepositoryContract:
    def make_rig(self) -> Rig:
        raise NotImplementedError

    @pytest.fixture
    def rig(self) -> Rig:
        return self.make_rig()

    @pytest.fixture
    def repo(self, rig: Rig) -> EscalationRepository:
        return rig.repo

    def test_the_public_surface_is_exactly_the_port(self, repo):
        assert public(repo) == surface(EscalationRepository)

    # ---- the queue ----

    def test_the_queue_serves_priority_first_then_the_oldest_then_the_id(self, rig):
        late = NOW + timedelta(minutes=1)
        a = rig.seed(1, priority=0.2)
        b = rig.seed(2, priority=0.9, classified_at=late)
        c = rig.seed(3, priority=0.9)
        d = rig.seed(4, priority=0.9)
        entries = rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=10)
        assert [e.item_id for e in entries] == [
            *sorted([c.item_id, d.item_id]),
            b.item_id,
            a.item_id,
        ]
        assert (entries[0].priority, entries[0].classified_at) == (0.9, NOW)

    def test_the_queue_respects_the_limit(self, rig):
        for n in range(1, 6):
            rig.seed(n, priority=n / 10)
        got = rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=2)
        assert [e.priority for e in got] == [0.5, 0.4]

    def test_only_verdicts_that_wait_are_queued(self, rig):
        waiting = rig.seed(1)
        rig.seed(2, needs_llm=False, priority=0.0)
        assert [e.item_id for e in rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=9)] == [
            waiting.item_id
        ]

    @pytest.mark.parametrize("status", [ItemStatus.NEW, ItemStatus.OUT_OF_SCOPE])
    def test_an_item_that_is_not_classified_is_not_queued(self, rig, status):
        rig.seed(1, status=status)
        assert rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=9) == []
        rig.seed(2)  # the same call does find an item that is classified
        assert len(rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=9)) == 1

    def test_a_purged_item_is_not_queued(self, rig):
        stored = rig.seed(1)
        rig.items.purge_fetched_before(NOW + timedelta(days=3650))
        assert rig.items.get(stored.item_id) is None
        assert rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=9) == []
        assert rig.repo.queued_count() == 0

    def test_only_the_eligible_sources_are_queued(self, rig):
        mine = rig.seed(1, source="src_news")
        rig.seed(2, source="src_other")
        assert [e.item_id for e in rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=9)] == [
            mine.item_id
        ]
        assert rig.repo.queue(source_ids=set(), now=NOW, limit=9) == []
        assert len(rig.repo.queue(source_ids={"src_news", "src_other"}, now=NOW, limit=9)) == 2

    def test_a_backoff_keeps_an_item_out_until_its_moment_inclusive(self, rig):
        stored = rig.seed(1)
        wait = NOW + timedelta(seconds=1)
        rig.repo.put_state(replace(state(1), item_id=stored.item_id, next_attempt_at=wait))
        assert rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=9) == []
        at = NOW + timedelta(seconds=1)
        assert [e.item_id for e in rig.repo.queue(source_ids={"src_news"}, now=at, limit=9)] == [
            stored.item_id
        ]
        later = NOW + timedelta(seconds=2)
        assert len(rig.repo.queue(source_ids={"src_news"}, now=later, limit=9)) == 1

    @pytest.mark.parametrize("final", ["parked", "dropped"])
    def test_a_parked_or_dropped_item_is_not_queued(self, rig, final):
        stored = rig.seed(1)
        rig.repo.put_state(
            replace(state(1), item_id=stored.item_id, state=final, next_attempt_at=None)
        )
        assert rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=9) == []

    def test_a_failed_item_without_a_wait_is_queued(self, rig):
        stored = rig.seed(1)
        rig.repo.put_state(replace(state(1), item_id=stored.item_id, next_attempt_at=None))
        assert len(rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=9)) == 1

    @pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5])
    def test_a_bad_queue_limit_is_refused(self, rig, limit):
        with pytest.raises(Exception):  # noqa: B017,PT011 - ValidationError of either flavour
            rig.repo.queue(source_ids={"src_news"}, now=NOW, limit=limit)

    def test_the_queue_needs_an_aware_time(self, rig):
        with pytest.raises(Exception):  # noqa: B017,PT011
            rig.repo.queue(source_ids={"src_news"}, now=datetime(2026, 3, 1), limit=9)  # noqa: DTZ001

    def test_the_queued_count_ignores_backoff_and_source(self, rig):
        rig.seed(1, source="src_news")
        other = rig.seed(2, source="src_other")
        rig.seed(3, needs_llm=False, priority=0.0)
        rig.repo.put_state(replace(state(1), item_id=other.item_id))
        assert rig.repo.queued_count() == 2

    def test_the_oldest_waiting_items_come_first(self, rig):
        newest = rig.seed(1, classified_at=NOW + timedelta(hours=2))
        oldest = rig.seed(2, classified_at=NOW)
        middle = rig.seed(3, classified_at=NOW + timedelta(hours=1))
        rig.seed(4, needs_llm=False, priority=0.0, classified_at=NOW - timedelta(days=1))
        assert rig.repo.oldest_queued(2) == [oldest.item_id, middle.item_id]
        assert rig.repo.oldest_queued(9) == [oldest.item_id, middle.item_id, newest.item_id]

    @pytest.mark.parametrize("limit", [0, -3, True])
    def test_a_bad_oldest_limit_is_refused(self, rig, limit):
        with pytest.raises(Exception):  # noqa: B017,PT011
            rig.repo.oldest_queued(limit)

    # ---- the budget ----

    def test_the_hour_takes_calls_up_to_the_cap_and_no_more(self, repo):
        assert repo.calls_in_hour(HOUR) == 0
        for _ in range(3):
            assert repo.reserve_call(HOUR, 3) is True
        assert repo.reserve_call(HOUR, 3) is False
        assert repo.reserve_call(HOUR, 3) is False
        assert repo.calls_in_hour(HOUR) == 3

    @pytest.mark.parametrize(("cap", "granted"), [(199, 199), (200, 200), (201, 201)])
    def test_the_cap_of_the_product_is_exact_around_200(self, repo, cap, granted):
        got = sum(repo.reserve_call(HOUR, cap) for _ in range(cap + 5))
        assert got == granted
        assert repo.calls_in_hour(HOUR) == granted

    def test_a_new_hour_starts_from_zero_and_the_old_one_is_unchanged(self, repo):
        for _ in range(2):
            repo.reserve_call(HOUR, 2)
        assert repo.reserve_call(HOUR, 2) is False
        next_hour = HOUR + timedelta(hours=1)
        assert repo.reserve_call(next_hour, 2) is True
        assert repo.calls_in_hour(next_hour) == 1
        assert repo.calls_in_hour(HOUR) == 2

    def test_a_lowered_cap_applies_to_calls_already_counted(self, repo):
        for _ in range(5):
            repo.reserve_call(HOUR, 10)
        assert repo.reserve_call(HOUR, 5) is False
        assert repo.reserve_call(HOUR, 6) is True

    @pytest.mark.parametrize("cap", [0, -1, True])
    def test_a_bad_cap_is_refused(self, repo, cap):
        with pytest.raises(Exception):  # noqa: B017,PT011
            repo.reserve_call(HOUR, cap)

    def test_a_budget_hour_must_be_a_whole_hour(self, repo):
        with pytest.raises(Exception):  # noqa: B017,PT011
            repo.reserve_call(HOUR + timedelta(minutes=1), 3)

    # ---- bookkeeping ----

    def test_a_state_round_trips_and_is_replaced_in_place(self, repo):
        assert repo.get_state(item_id(1)) is None
        first = state(1)
        repo.put_state(first)
        assert repo.get_state(item_id(1)) == first
        second = state(1, attempts=3, next_attempt_at=None, last_error=None)
        repo.put_state(second)
        assert repo.get_state(item_id(1)) == second

    def test_clearing_a_state_forgets_it_and_is_idempotent(self, repo):
        repo.put_state(state(1))
        repo.clear_state(item_id(1))
        repo.clear_state(item_id(1))
        assert repo.get_state(item_id(1)) is None

    @pytest.mark.parametrize(
        "bad",
        [
            {"state": "waiting"},
            {"attempts": -1},
            {"attempts": True},
            {"item_failures": -1},
            {"last_error": "x" * 65},
            {"item_id": "X"},
        ],
    )
    def test_a_bad_state_is_refused(self, repo, bad):
        with pytest.raises(Exception):  # noqa: B017,PT011
            repo.put_state(state(1, **bad))

    # ---- judgments ----

    def test_a_judgment_round_trips_with_every_field(self, repo):
        stored = judgment(1)
        assert repo.add_judgment(stored) is True
        assert repo.get_judgment(item_id(1)) == stored

    def test_a_second_judgment_for_an_item_is_refused_and_changes_nothing(self, repo):
        first = judgment(1)
        repo.add_judgment(first)
        assert repo.add_judgment(judgment(1, label="deliberate_disinformation")) is False
        assert repo.get_judgment(item_id(1)) == first

    def test_an_invalid_and_a_reused_judgment_round_trip(self, repo):
        invalid = judgment(
            1, outcome="invalid", label=None, confidence=None, rationale=None, evidence=()
        )
        reused = judgment(2, outcome="reused", reused_from=item_id(1), raw_output=None)
        assert repo.add_judgment(invalid) is True
        assert repo.add_judgment(reused) is True
        assert repo.get_judgment(item_id(1)) == invalid
        assert repo.get_judgment(item_id(2)) == reused

    def test_an_unknown_item_has_no_judgment(self, repo):
        assert repo.get_judgment(item_id(9)) is None

    @pytest.mark.parametrize(
        "bad",
        [
            {"outcome": "maybe"},
            {"label": "unclassified"},
            {"confidence": 1.5},
            {"raw_output": "x" * 2049},
            {"item_id": "X"},
            {"evidence": "text"},
        ],
    )
    def test_a_bad_judgment_is_refused(self, repo, bad):
        with pytest.raises(Exception):  # noqa: B017,PT011
            repo.add_judgment(judgment(1, **bad))

    # ---- counts ----

    def test_the_counts_reflect_what_is_stored(self, rig):
        assert rig.repo.counts() == EscalationCounts(0, 0, 0, 0, 0, 0, 0)
        rig.seed(1)
        rig.seed(2)
        rig.repo.add_judgment(judgment(3))
        rig.repo.add_judgment(judgment(4))
        rig.repo.add_judgment(
            judgment(5, outcome="invalid", label=None, confidence=None, rationale=None)
        )
        rig.repo.add_judgment(judgment(6, outcome="reused", reused_from=item_id(3)))
        rig.repo.put_state(state(7))
        rig.repo.put_state(state(8, state="parked", next_attempt_at=None))
        rig.repo.put_state(state(9, state="dropped", next_attempt_at=None))
        rig.repo.put_state(state(10, state="dropped", next_attempt_at=None))
        assert rig.repo.counts() == EscalationCounts(
            judged=2, invalid=1, reused=1, failed_waiting=1, parked=1, dropped=2, queued=2
        )

    # ---- the lease ----

    def test_the_lease_is_one_slot(self, repo):
        until = NOW + timedelta(seconds=90)
        assert repo.try_lease("worker-a", NOW, until) is True
        assert repo.try_lease("worker-b", NOW, until) is False
        assert repo.try_lease("worker-a", NOW, until) is True  # the holder renews

    def test_the_lease_is_free_the_moment_it_expires(self, repo):
        until = NOW + timedelta(seconds=90)
        repo.try_lease("worker-a", NOW, until)
        just_before = until - timedelta(microseconds=1)
        assert repo.try_lease("worker-b", just_before, until + timedelta(seconds=90)) is False
        assert repo.try_lease("worker-b", until, until + timedelta(seconds=90)) is True

    def test_only_the_holder_releases_the_lease(self, repo):
        until = NOW + timedelta(seconds=90)
        repo.try_lease("worker-a", NOW, until)
        repo.release_lease("worker-b")
        assert repo.try_lease("worker-b", NOW, until) is False
        repo.release_lease("worker-a")
        assert repo.try_lease("worker-b", NOW, until) is True

    def test_releasing_a_free_lease_is_harmless(self, repo):
        repo.release_lease("nobody")
        assert repo.try_lease("worker-a", NOW, NOW + timedelta(seconds=1)) is True

    def test_a_lease_must_end_after_it_starts(self, repo):
        with pytest.raises(Exception):  # noqa: B017,PT011
            repo.try_lease("worker-a", NOW, NOW)

    # ---- forgetting ----

    def test_forgetting_items_removes_their_judgments_and_bookkeeping_only(self, repo):
        for n in (1, 2, 3):
            repo.add_judgment(judgment(n))
            repo.put_state(state(n))
        assert repo.forget([item_id(1), item_id(3), item_id(9)]) == 2
        assert repo.get_judgment(item_id(1)) is None
        assert repo.get_state(item_id(3)) is None
        assert repo.get_judgment(item_id(2)) is not None
        assert repo.get_state(item_id(2)) is not None
        assert repo.forget([]) == 0
