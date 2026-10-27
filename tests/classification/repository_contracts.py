"""Reusable contracts for the classification ports (in-memory fakes and SQLite alike).

What every implementation must honour (and these suites pin):
- ClassificationRepository: one verdict per item, never overwritten; a round trip loses nothing;
  `stats` counts what is stored.
- SignatureRepository: exact matches (content or text hash) come oldest first and are not capped
  by the band candidates; candidates are found by identical content hash or a shared band key,
  never include the item itself, come best first (most shared bands, then oldest), and pruning
  forgets their bands too.
- Verdict history: a verdict can be forgotten (alone, or in bulk when older than a cutoff), and the
  consecutive per-item errors of an item are counted until it is classified or forgotten.
- TopicConfigRepository: an append-only history numbered 1, 2, 3...; the latest is in force.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.classification.domain import (
    Classification,
    ItemSignature,
    Label,
    OutOfScopeReason,
    Signal,
    Stage,
    Topic,
)
from social_tracker.classification.ports import (
    ClassificationRepository,
    SignatureRepository,
    TopicConfigRepository,
)
from social_tracker.classification.similarity import BANDS, NUM_PERM, band_keys
from social_tracker.classification.topics import StoredTopicConfig, TopicConfig
from social_tracker.shared.errors import ValidationError

NOW = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def item(n: int | str) -> str:
    return f"itm_{n:0>8}" if isinstance(n, int) else f"itm_{n}"


def surface(port: type) -> frozenset[str]:
    return frozenset(
        name for name, member in vars(port).items() if not name.startswith("_") and callable(member)
    )


def public(repo: object) -> set[str]:
    return {n for n in dir(repo) if not n.startswith("_") and callable(getattr(repo, n))}


def verdict(n: int | str = 1, **changes) -> Classification:
    fields = {
        "item_id": item(n),
        "label": Label.UNINTENTIONAL_RUMOR,
        "confidence": 0.5875,
        "stage": Stage.RULES,
        "rationale": "rumor candidate: prediction_claim; topics: earthquake",
        "classified_at": NOW,
        "classifier_version": "cascade-2/topics-0",
        "signals": (
            Signal("prediction_claim", Label.UNINTENTIONAL_RUMOR, 0.45),
            Signal("unverifiable_sources", Label.UNINTENTIONAL_RUMOR, 0.25),
        ),
        "topics": (Topic.EARTHQUAKE, Topic.SECRETARIAT),
        **changes,
    }
    return Classification(**fields)


def out_of_scope(n: int | str, reason: OutOfScopeReason) -> Classification:
    return verdict(
        n,
        label=Label.OUT_OF_SCOPE,
        confidence=1.0,
        stage=Stage.TOPIC,
        rationale="out of scope",
        signals=(),
        topics=(),
        scope_reason=reason,
    )


class ClassificationRepositoryContract:
    def make_repo(self) -> ClassificationRepository:
        raise NotImplementedError

    @pytest.fixture
    def repo(self) -> ClassificationRepository:
        return self.make_repo()

    def test_the_public_surface_is_exactly_the_port(self, repo):
        assert public(repo) == surface(ClassificationRepository)

    def test_every_field_survives_a_round_trip(self, repo):
        original = verdict(
            "a" * 8,
            confidence=0.604,
            rationale="Sismo \U0001f30b en Cali: ñandú áéí",
            low_confidence=True,
            needs_llm=True,
            truncated=True,
            catalog_status="unavailable",
            priority=0.4321,
            source_credibility_used=5,
            classified_at=NOW + timedelta(microseconds=123456),
            classifier_version="cascade-2/topics-12",
        )
        assert repo.add(original) is True
        assert repo.get(original.item_id) == original

    def test_references_and_reasons_survive_a_round_trip(self, repo):
        duplicate = verdict(2, duplicate_of=item(1), stage=Stage.DEDUP)
        recycled = verdict(3, recycled_of=item(1), label=Label.RECYCLED_CONTENT, stage=Stage.DEDUP)
        gone = out_of_scope(4, OutOfScopeReason.CRIME_PUBLIC_ORDER)
        for c in (duplicate, recycled, gone):
            assert repo.add(c) is True
        assert repo.get(item(2)) == duplicate
        assert repo.get(item(3)) == recycled
        assert repo.get(item(4)) == gone

    def test_the_near_copy_link_and_a_waiting_duplicate_survive_a_round_trip(self, repo):
        near = verdict(2, near_of=item(1), needs_llm=True)
        waiting = verdict(3, duplicate_of=item(1), stage=Stage.DEDUP, awaiting_original=True)
        for c in (near, waiting):
            assert repo.add(c) is True
        assert repo.get(item(2)) == near
        assert repo.get(item(3)) == waiting
        assert repo.get(item(3)).awaiting_original is True

    def test_forgetting_a_verdict_lets_the_item_be_classified_again(self, repo):
        repo.add(verdict(1))
        assert repo.delete(item(1)) is True
        assert repo.get(item(1)) is None
        assert repo.delete(item(1)) is False
        assert repo.add(verdict(1, label=Label.NO_RISK, confidence=0.9)) is True
        assert repo.get(item(1)).label is Label.NO_RISK

    # ---- settling a verdict that waits for the LLM (unit 3c) ----

    def test_a_queued_verdict_is_settled_in_place_changing_only_what_a_judgment_changes(self, repo):
        queued = verdict(1, needs_llm=True, priority=0.6, low_confidence=False)
        repo.add(queued)
        settled = replace(
            queued,
            label=Label.NO_RISK,
            confidence=0.91,
            stage=Stage.LLM,
            rationale="the text only reports a past event",
            low_confidence=True,
            needs_llm=False,
            priority=0.0,
        )
        assert repo.settle(settled) is True
        assert repo.get(item(1)) == settled
        assert repo.stats().needs_llm == 0

    def test_a_settled_verdict_cannot_be_settled_again(self, repo):
        repo.add(verdict(1, needs_llm=True))
        first = replace(verdict(1), label=Label.NO_RISK, stage=Stage.LLM, needs_llm=False)
        second = replace(first, label=Label.DELIBERATE_DISINFORMATION)
        assert repo.settle(first) is True
        assert repo.settle(second) is False
        assert repo.get(item(1)).label is Label.NO_RISK

    def test_a_verdict_that_never_waited_is_not_settled(self, repo):
        plain = verdict(1, needs_llm=False)
        repo.add(plain)
        assert repo.settle(replace(plain, label=Label.NO_RISK)) is False
        assert repo.get(item(1)) == plain

    def test_an_unknown_item_is_not_settled(self, repo):
        assert repo.settle(verdict(9, needs_llm=False)) is False

    def test_a_waiting_duplicate_is_settled_and_stops_waiting(self, repo):
        waiting = verdict(2, duplicate_of=item(1), stage=Stage.DEDUP, awaiting_original=True)
        repo.add(waiting)
        done = replace(waiting, label=Label.NO_RISK, confidence=0.9, awaiting_original=False)
        assert repo.settle(done) is True
        assert repo.get(item(2)) == done
        assert repo.stats().awaiting_original == 0

    def test_only_a_verdict_object_can_settle(self, repo):
        with pytest.raises(ValidationError):
            repo.settle("nope")

    def test_the_duplicates_waiting_for_an_original_are_listed_by_id(self, repo):
        repo.add(verdict(1, needs_llm=True))
        for n in (4, 2, 3):
            repo.add(verdict(n, duplicate_of=item(1), stage=Stage.DEDUP, awaiting_original=True))
        repo.add(verdict(5, duplicate_of=item(1), stage=Stage.DEDUP))  # already settled
        repo.add(verdict(6, duplicate_of=item(9), stage=Stage.DEDUP, awaiting_original=True))
        assert [c.item_id for c in repo.awaiting(item(1))] == [item(2), item(3), item(4)]
        assert [c.item_id for c in repo.awaiting(item(9))] == [item(6)]
        assert repo.awaiting(item(7)) == []

    # ---- settling, reopening and releasing (the remediation round) ----

    def test_a_settlement_keeps_its_time_and_reason(self, repo):
        repo.add(verdict(1, needs_llm=True, priority=0.6))
        settled = replace(
            verdict(1),
            label=Label.NO_RISK,
            stage=Stage.LLM,
            needs_llm=False,
            priority=0.0,
            settled_at=NOW + timedelta(minutes=3, microseconds=7),
            settle_reason="llm",
        )
        assert repo.settle(settled) is True
        got = repo.get(item(1))
        assert got == settled
        assert (got.settled_at, got.settle_reason) == (settled.settled_at, "llm")

    def test_a_verdict_nobody_settled_has_no_settlement(self, repo):
        repo.add(verdict(1))
        got = repo.get(item(1))
        assert (got.settled_at, got.settle_reason) == (None, None)

    def test_a_copy_that_must_be_judged_on_its_own_is_requeued_by_settling_it(self, repo):
        repo.add(verdict(1, needs_llm=True))
        waiting = verdict(2, duplicate_of=item(1), stage=Stage.DEDUP, awaiting_original=True)
        repo.add(waiting)
        own = replace(
            waiting,
            duplicate_of=None,
            near_of=item(1),
            awaiting_original=False,
            needs_llm=True,
            priority=0.4,
        )
        assert repo.settle(own) is True
        got = repo.get(item(2))
        assert (got.duplicate_of, got.near_of, got.needs_llm, got.priority) == (
            None,
            item(1),
            True,
            0.4,
        )
        assert [c.item_id for c in repo.awaiting(item(1))] == []

    @pytest.mark.parametrize("reason", ["invalid", "parked"])
    def test_a_settled_verdict_can_be_reopened_only_if_it_was_parked(self, repo, reason):
        stamp = NOW + timedelta(minutes=1)
        repo.add(verdict(1, needs_llm=False, settled_at=stamp, settle_reason=reason))
        reopened = repo.reopen(item(1), 0.1)
        assert reopened is (reason == "parked")
        got = repo.get(item(1))
        if reason == "parked":
            assert (got.needs_llm, got.priority, got.settled_at, got.settle_reason) == (
                True,
                0.1,
                None,
                None,
            )
        else:
            assert (got.needs_llm, got.settle_reason) == (False, "invalid")

    def test_reopening_something_that_is_not_settled_changes_nothing(self, repo):
        repo.add(verdict(1, needs_llm=True, priority=0.8))
        repo.add(verdict(2, needs_llm=False))
        assert repo.reopen(item(1), 0.1) is False
        assert repo.reopen(item(2), 0.1) is False
        assert repo.reopen(item(9), 0.1) is False
        assert repo.get(item(1)).priority == 0.8
        assert repo.get(item(2)).needs_llm is False

    @pytest.mark.parametrize("priority", [-0.1, 1.1, True, "high"])
    def test_a_bad_reopen_priority_is_refused(self, repo, priority):
        stamp = NOW + timedelta(minutes=1)
        repo.add(verdict(1, needs_llm=False, settled_at=stamp, settle_reason="parked"))
        with pytest.raises(ValidationError):
            repo.reopen(item(1), priority)

    def test_forgetting_an_original_releases_its_waiting_copies_into_the_queue(self, repo):
        repo.add(verdict(1, needs_llm=True, priority=0.7))
        for n in (2, 3):
            repo.add(verdict(n, duplicate_of=item(1), stage=Stage.DEDUP, awaiting_original=True))
        repo.add(verdict(4, duplicate_of=item(1), stage=Stage.DEDUP))  # already settled
        assert repo.delete(item(1)) is True
        for n in (2, 3):
            got = repo.get(item(n))
            assert (got.awaiting_original, got.duplicate_of, got.needs_llm) == (False, None, True)
            assert got.priority == 0.7  # the priority its original had
        assert repo.get(item(4)).duplicate_of == item(1)  # a settled copy keeps its link
        assert repo.stats().awaiting_original == 0
        assert repo.stats().orphaned_awaiting == 0

    def test_forgetting_many_originals_releases_their_copies_with_a_floor_priority(self, repo):
        repo.add(verdict(1, needs_llm=False))  # an original that was settled meanwhile
        repo.add(verdict(2, needs_llm=True, priority=0.5))
        repo.add(verdict(3, duplicate_of=item(1), stage=Stage.DEDUP, awaiting_original=True))
        repo.add(verdict(4, duplicate_of=item(2), stage=Stage.DEDUP, awaiting_original=True))
        repo.add(verdict(5, duplicate_of=item(9), stage=Stage.DEDUP, awaiting_original=True))
        assert repo.delete_many([item(1), item(2), item(9)]) == 2
        assert repo.get(item(3)).priority == 0.1
        assert repo.get(item(4)).priority == 0.5
        assert repo.get(item(5)).needs_llm is True
        assert all(not repo.get(item(n)).awaiting_original for n in (3, 4, 5))

    def test_forgetting_a_copy_with_its_original_leaves_nothing_behind(self, repo):
        repo.add(verdict(1, needs_llm=True))
        repo.add(verdict(2, duplicate_of=item(1), stage=Stage.DEDUP, awaiting_original=True))
        assert repo.delete_many([item(1), item(2)]) == 2
        assert repo.get(item(2)) is None

    def test_statistics_do_not_count_a_copy_whose_original_is_gone_as_waiting(self, repo):
        repo.add(verdict(1, needs_llm=True))
        repo.add(verdict(2, duplicate_of=item(1), stage=Stage.DEDUP, awaiting_original=True))
        repo.add(verdict(3, duplicate_of=item(8), stage=Stage.DEDUP, awaiting_original=True))
        repo.add(verdict(4, needs_llm=False))
        repo.add(verdict(5, duplicate_of=item(4), stage=Stage.DEDUP, awaiting_original=True))
        stats = repo.stats()
        assert stats.awaiting_original == 1  # only the copy of an original that still waits
        assert stats.orphaned_awaiting == 2  # no original to wait for: released on the next pass

    def test_old_verdicts_are_listed_oldest_first_strictly_before_the_cutoff(self, repo):
        for n, days in ((1, -10), (2, -5), (3, -5), (4, 0)):
            repo.add(verdict(n, classified_at=NOW + timedelta(days=days)))
        cutoff = NOW - timedelta(days=5)
        assert repo.older_than(cutoff, 10) == [item(1)]  # exactly at the cutoff is kept
        assert repo.older_than(cutoff + timedelta(microseconds=1), 10) == [
            item(1),
            item(2),
            item(3),
        ]
        assert repo.older_than(NOW + timedelta(days=1), 2) == [item(1), item(2)]
        assert repo.older_than(NOW - timedelta(days=100), 10) == []

    def test_the_age_listing_can_be_paged(self, repo):
        for n in range(1, 6):
            repo.add(verdict(n, classified_at=NOW - timedelta(days=10 - n)))
        later = NOW + timedelta(days=1)
        assert repo.older_than(later, 2) == [item(1), item(2)]
        assert repo.older_than(later, 2, 2) == [item(3), item(4)]
        assert repo.older_than(later, 2, 4) == [item(5)]
        assert repo.older_than(later, 2, 9) == []

    @pytest.mark.parametrize("offset", [-1, True, 1.5, None, "1"])
    def test_a_bad_age_listing_offset_is_refused(self, repo, offset):
        with pytest.raises(ValidationError):
            repo.older_than(NOW, 5, offset)

    @pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5, None])
    def test_a_bad_age_listing_limit_is_refused(self, repo, limit):
        with pytest.raises(ValidationError):
            repo.older_than(NOW, limit)

    def test_the_age_listing_rejects_a_naive_cutoff(self, repo):
        with pytest.raises(ValidationError):
            repo.older_than(datetime(2026, 3, 1), 5)

    def test_verdicts_are_forgotten_in_bulk_and_the_count_is_exact(self, repo):
        for n in (1, 2, 3):
            repo.add(verdict(n))
        assert repo.delete_many([item(1), item(3), item(9)]) == 2
        assert repo.get(item(1)) is None
        assert repo.get(item(2)) is not None
        assert repo.delete_many([]) == 0
        assert repo.delete_many([item(1)]) == 0

    def test_consecutive_errors_are_counted_per_item_until_cleared_or_forgotten(self, repo):
        assert repo.note_failure(item(1), NOW) == 1
        assert repo.note_failure(item(1), NOW) == 2
        assert repo.note_failure(item(2), NOW) == 1
        assert repo.note_failure(item(1), NOW) == 3
        repo.clear_failure(item(1))
        repo.clear_failure(item(77))  # nothing to clear is fine
        assert repo.note_failure(item(1), NOW) == 1
        repo.add(verdict(2))
        repo.delete(item(2))  # forgetting a verdict forgets its counter
        assert repo.note_failure(item(2), NOW) == 1
        repo.delete_many([item(1)])
        assert repo.note_failure(item(1), NOW) == 1

    def test_a_failure_needs_an_aware_time(self, repo):
        with pytest.raises(ValidationError):
            repo.note_failure(item(1), datetime(2026, 3, 1))

    def test_a_second_verdict_for_an_item_is_refused_and_changes_nothing(self, repo):
        first = verdict(1, label=Label.UNINTENTIONAL_RUMOR)
        again = verdict(1, label=Label.NO_RISK, confidence=0.1)
        assert repo.add(first) is True
        assert repo.add(again) is False
        assert repo.get(item(1)) == first
        assert repo.stats().total == 1

    def test_an_unknown_item_has_no_verdict(self, repo):
        assert repo.get(item(99)) is None

    @pytest.mark.parametrize("bad", [None, "text", {"item_id": "x"}, 5])
    def test_only_classifications_are_stored(self, repo, bad):
        with pytest.raises(ValidationError):
            repo.add(bad)

    def test_an_empty_store_has_empty_statistics(self, repo):
        stats = repo.stats()
        assert stats.total == 0
        assert (stats.by_label, stats.out_of_scope_by_reason) == ((), ())
        assert (stats.needs_llm, stats.low_confidence) == (0, 0)
        assert (stats.duplicates, stats.recycled, stats.truncated) == (0, 0, 0)
        assert (stats.near_duplicates, stats.awaiting_original, stats.failed) == (0, 0, 0)

    def test_statistics_count_near_copies_and_waiting_duplicates(self, repo):
        repo.add(verdict(1, needs_llm=True))
        repo.add(verdict(2, near_of=item(1)))
        repo.add(verdict(3, duplicate_of=item(1), stage=Stage.DEDUP, awaiting_original=True))
        repo.add(verdict(4, duplicate_of=item(1), stage=Stage.DEDUP))
        stats = repo.stats()
        assert (stats.near_duplicates, stats.awaiting_original, stats.duplicates) == (1, 1, 2)

    def test_statistics_count_what_is_stored(self, repo):
        repo.add(verdict(1, needs_llm=True, low_confidence=True))
        repo.add(verdict(2, truncated=True))
        repo.add(verdict(3, label=Label.NO_RISK, duplicate_of=item(1), stage=Stage.DEDUP))
        repo.add(verdict(4, label=Label.RECYCLED_CONTENT, recycled_of=item(1), stage=Stage.DEDUP))
        repo.add(verdict(8, label=Label.RECYCLED_CONTENT, recycled_of=item(1), stage=Stage.DEDUP))
        repo.add(out_of_scope(5, OutOfScopeReason.NO_TOPIC_MATCH))
        repo.add(out_of_scope(6, OutOfScopeReason.NO_TOPIC_MATCH))
        repo.add(out_of_scope(7, OutOfScopeReason.CRIME_PUBLIC_ORDER))
        stats = repo.stats()
        assert stats.total == 8
        assert stats.by_label == (
            ("no_risk", 1),
            ("out_of_scope", 3),
            ("recycled_content", 2),
            ("unintentional_rumor", 2),
        )
        assert stats.out_of_scope_by_reason == (("crime_public_order", 1), ("no_topic_match", 2))
        assert (stats.needs_llm, stats.low_confidence) == (1, 1)
        assert (stats.duplicates, stats.recycled, stats.truncated) == (1, 2, 1)


def signature(
    n: int | str,
    *,
    content_hash: str | None = None,
    minhash=None,
    days: int = 0,
    **digests,
):
    return ItemSignature(
        item_id=item(n),
        content_hash=content_hash or f"{n:0>2}".encode().hex().ljust(64, "0")[:64],
        ref_at=NOW + timedelta(days=days),
        minhash=minhash if minhash is not None else tuple(range(NUM_PERM)),
        **digests,
    )


def disjoint(offset: int) -> tuple[int, ...]:
    return tuple(offset + i for i in range(NUM_PERM))


class SignatureRepositoryContract:
    def make_repo(self) -> SignatureRepository:
        raise NotImplementedError

    @pytest.fixture
    def repo(self) -> SignatureRepository:
        return self.make_repo()

    def find(self, repo, probe: ItemSignature, limit: int = 100, **changes):
        args = {
            "item_id": probe.item_id,
            "content_hash": probe.content_hash,
            "band_keys": band_keys(probe.minhash),
            "limit": limit,
            **changes,
        }
        return repo.find_candidates(**args)

    def test_the_public_surface_is_exactly_the_port(self, repo):
        assert public(repo) == surface(SignatureRepository)

    def test_a_signature_round_trips_and_is_stored_once(self, repo):
        original = signature(1, days=3)
        assert repo.add(original) is True
        assert repo.get(original.item_id) == original
        assert repo.add(signature(1, minhash=disjoint(5000))) is False
        assert repo.get(original.item_id) == original
        assert repo.get(item(99)) is None

    def test_the_digests_of_a_signature_survive_a_round_trip(self, repo):
        original = signature(1, text_hash="a" * 64, digits_hash="b" * 64, n_tokens=21)
        repo.add(original)
        got = repo.get(original.item_id)
        assert (got.text_hash, got.digits_hash, got.n_tokens) == ("a" * 64, "b" * 64, 21)
        old = signature(2)  # stored without digests
        repo.add(old)
        assert (repo.get(old.item_id).text_hash, repo.get(old.item_id).n_tokens) == (None, 0)

    def exact(self, repo, probe: ItemSignature, **changes):
        args = {
            "item_id": probe.item_id,
            "content_hash": probe.content_hash,
            "text_hash": probe.text_hash,
            **changes,
        }
        return repo.find_exact(**args)

    def test_an_exact_match_is_found_by_content_hash_or_by_text_hash(self, repo):
        repo.add(signature(1, content_hash="c" * 64, text_hash="1" * 64))
        repo.add(signature(2, content_hash="d" * 64, text_hash="2" * 64))
        by_content = signature(8, content_hash="c" * 64, text_hash="9" * 64)
        by_text = signature(9, content_hash="e" * 64, text_hash="2" * 64)
        neither = signature(7, content_hash="f" * 64, text_hash="8" * 64)
        assert [s.item_id for s in self.exact(repo, by_content)] == [item(1)]
        assert [s.item_id for s in self.exact(repo, by_text)] == [item(2)]
        assert self.exact(repo, neither) == []
        assert self.exact(repo, signature(6, content_hash="e" * 64, text_hash=None)) == []

    def test_an_exact_match_never_returns_the_item_itself_and_comes_oldest_first(self, repo):
        shared = "c" * 64
        for n, days in ((3, 2), (1, -9), (2, -9), (4, 0)):
            repo.add(signature(n, content_hash=shared, days=days))
        probe = signature(9, content_hash=shared)
        assert [s.item_id for s in self.exact(repo, probe)] == [item(1), item(2), item(4), item(3)]
        assert [s.item_id for s in self.exact(repo, probe, limit=1)] == [item(1)]
        mine = signature(1, content_hash=shared)
        assert item(1) not in [s.item_id for s in self.exact(repo, mine)]

    def test_an_exact_match_can_be_restricted_to_a_window_of_time_inclusive(self, repo):
        shared = "c" * 64
        for n, days in ((1, -10), (2, -7), (3, 0), (4, 7), (5, 10)):
            repo.add(signature(n, content_hash=shared, days=days))
        probe = signature(9, content_hash=shared)
        window = {
            "ref_from": NOW - timedelta(days=7),
            "ref_to": NOW + timedelta(days=7),
        }
        assert [s.item_id for s in self.exact(repo, probe, **window)] == [item(2), item(3), item(4)]
        only_from = self.exact(repo, probe, ref_from=NOW + timedelta(days=7))
        assert [s.item_id for s in only_from] == [item(4), item(5)]
        only_to = self.exact(repo, probe, ref_to=NOW - timedelta(days=10))
        assert [s.item_id for s in only_to] == [item(1)]

    @pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5, None])
    def test_a_bad_exact_limit_is_refused(self, repo, limit):
        with pytest.raises(ValidationError):
            self.exact(repo, signature(1), limit=limit)

    def test_an_exact_match_is_found_among_hundreds_of_band_collisions(self, repo):
        # 300 signatures that all collide with the probe on band 0; the original, older than all
        # of them and sharing nothing else but the hash, must still be the first exact match.
        base = tuple(range(NUM_PERM))
        collide = tuple(v if i < 5 else v + 50_000 for i, v in enumerate(base))
        for n in range(2, 302):
            repo.add(signature(n, minhash=collide, days=n, content_hash=f"{n:064x}"))
        repo.add(signature(1, minhash=disjoint(70_000), days=-30, content_hash="a" * 64))
        probe = signature(900, minhash=base, content_hash="a" * 64)
        assert [s.item_id for s in self.exact(repo, probe)] == [item(1)]

    def test_candidates_come_best_first_the_most_shared_bands_before_the_oldest(self, repo):
        base = tuple(range(NUM_PERM))
        one_band = tuple(v if i < 5 else v + 50_000 for i, v in enumerate(base))
        three_bands = tuple(v if i < 15 else v + 60_000 for i, v in enumerate(base))
        repo.add(signature(1, minhash=one_band, days=-20, content_hash="1" * 64))
        repo.add(signature(2, minhash=three_bands, days=5, content_hash="2" * 64))
        repo.add(signature(3, minhash=three_bands, days=-5, content_hash="3" * 64))
        repo.add(signature(4, minhash=base, days=9, content_hash="4" * 64))
        probe = signature(9, minhash=base, content_hash="9" * 64)
        order = [c.item_id for c in self.find(repo, probe)]
        assert order == [item(4), item(3), item(2), item(1)]
        assert [c.item_id for c in self.find(repo, probe, limit=1)] == [item(4)]
        assert [c.item_id for c in self.find(repo, probe, limit=2)] == [item(4), item(3)]

    def test_the_best_candidate_survives_hundreds_of_weaker_ones(self, repo):
        base = tuple(range(NUM_PERM))
        one_band = tuple(v if i < 5 else v + 50_000 for i, v in enumerate(base))
        for n in range(2, 262):  # 260 weak collisions, all older than the best one
            repo.add(signature(n, minhash=one_band, days=-n, content_hash=f"{n:064x}"))
        near = tuple(v if i < 70 else v + 60_000 for i, v in enumerate(base))
        repo.add(signature(1, minhash=near, days=0, content_hash="a" * 64))
        probe = signature(900, minhash=base, content_hash="b" * 64)
        assert [c.item_id for c in self.find(repo, probe, limit=1)] == [item(1)]

    def test_the_same_content_hash_is_a_candidate_even_with_an_unrelated_signature(self, repo):
        shared = "c" * 64
        repo.add(signature(1, content_hash=shared, minhash=disjoint(10_000)))
        probe = signature(2, content_hash=shared, minhash=disjoint(20_000))
        assert [c.item_id for c in self.find(repo, probe)] == [item(1)]

    def test_a_shared_band_makes_a_candidate_and_no_shared_band_does_not(self, repo):
        base = tuple(range(NUM_PERM))
        one_band = tuple(v if i < 5 else v + 50_000 for i, v in enumerate(base))  # band 0 only
        repo.add(signature(1, minhash=one_band))
        repo.add(signature(2, minhash=disjoint(90_000), content_hash="d" * 64))
        probe = signature(3, minhash=base, content_hash="e" * 64)
        assert [c.item_id for c in self.find(repo, probe)] == [item(1)]

    def test_a_signature_is_never_its_own_candidate(self, repo):
        mine = signature(1)
        repo.add(mine)
        assert self.find(repo, mine) == []
        repo.add(signature(2, content_hash=mine.content_hash))
        assert [c.item_id for c in self.find(repo, mine)] == [item(2)]

    def test_candidates_come_oldest_first_then_by_id_and_the_limit_applies(self, repo):
        for n, days in ((5, 0), (3, -2), (4, -2), (1, -9), (2, 4)):
            repo.add(signature(n, days=days))
        probe = signature(9, days=0)
        found = [c.item_id for c in self.find(repo, probe)]
        assert found == [item(1), item(3), item(4), item(5), item(2)]
        assert [c.item_id for c in self.find(repo, probe, limit=2)] == [item(1), item(3)]

    @pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5, None])
    def test_a_bad_limit_is_refused(self, repo, limit):
        with pytest.raises(ValidationError):
            self.find(repo, signature(1), limit=limit)

    @pytest.mark.parametrize("keys", [(), tuple(range(BANDS - 1)), tuple(range(BANDS + 1)), None])
    def test_the_band_keys_must_be_one_per_band(self, repo, keys):
        with pytest.raises(ValidationError):
            self.find(repo, signature(1), band_keys=keys)

    def test_pruning_forgets_strictly_older_signatures_and_their_bands(self, repo):
        old, edge, new = signature(1, days=-10), signature(2, days=-5), signature(3, days=0)
        for s in (old, edge, new):
            repo.add(s)
        cutoff = NOW - timedelta(days=5)
        assert repo.prune_before(cutoff) == 1
        assert repo.get(old.item_id) is None
        assert repo.get(edge.item_id) == edge  # exactly at the cutoff is kept
        probe = signature(9, content_hash="f" * 64)
        assert sorted(c.item_id for c in self.find(repo, probe)) == [item(2), item(3)]
        assert repo.prune_before(cutoff) == 0

    def test_a_pruned_item_can_be_signed_again(self, repo):
        s = signature(1, days=-10)
        repo.add(s)
        repo.prune_before(NOW)
        assert repo.add(s) is True

    def test_pruning_rejects_a_naive_cutoff(self, repo):
        with pytest.raises(ValidationError):
            repo.prune_before(datetime(2026, 3, 1))

    @pytest.mark.parametrize("bad", [None, "text", 5])
    def test_only_signatures_are_stored(self, repo, bad):
        with pytest.raises(ValidationError):
            repo.add(bad)


def config(**changes) -> TopicConfig:
    return TopicConfig.from_mapping({"version": 1, "earthquake": ["sismo"], **changes})


def stored(cfg: TopicConfig | None = None, *, actor="ana", reason="tune", minutes=0):
    return StoredTopicConfig(cfg or config(), NOW + timedelta(minutes=minutes), actor, reason)


class TopicConfigRepositoryContract:
    def make_repo(self) -> TopicConfigRepository:
        raise NotImplementedError

    @pytest.fixture
    def repo(self) -> TopicConfigRepository:
        return self.make_repo()

    def test_the_public_surface_is_exactly_the_port(self, repo):
        assert public(repo) == surface(TopicConfigRepository)

    def test_an_empty_history_has_no_latest_version(self, repo):
        assert repo.latest() is None
        assert repo.versions(10) == []

    def test_versions_are_numbered_from_one_and_the_latest_is_in_force(self, repo):
        assert repo.append(stored(config(), minutes=0)) == 1
        assert repo.append(stored(config(places=["Cali"]), minutes=1)) == 2
        assert repo.append(stored(config(places=["Cali", "Palmira"]), minutes=2)) == 3
        latest = repo.latest()
        assert latest.version == 3
        assert latest.config.places == ("cali", "palmira")

    def test_every_field_of_a_version_survives_a_round_trip(self, repo):
        cfg = config(
            secretariat=["Secretaría de Gestión del Riesgo", "DAGRD"],
            weights={"prediction_claim": 0.6},
            official_domains=["official-domain.example"],
            official_channel_hints=["@placeholder"],
        )
        original = StoredTopicConfig(
            cfg, NOW + timedelta(microseconds=7), "María Pérez", "añadir variantes \U0001f30b"
        )
        number = repo.append(original)
        got = repo.latest()
        assert got.version == number
        assert (got.config, got.created_at, got.actor, got.reason) == (
            cfg,
            original.created_at,
            original.actor,
            original.reason,
        )

    def test_history_lists_the_newest_first_up_to_the_limit(self, repo):
        for n in range(5):
            repo.append(stored(config(places=[f"place{n}"]), minutes=n))
        assert [v.version for v in repo.versions(10)] == [5, 4, 3, 2, 1]
        assert [v.version for v in repo.versions(2)] == [5, 4]

    @pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5, None])
    def test_a_bad_history_limit_is_refused(self, repo, limit):
        with pytest.raises(ValidationError):
            repo.versions(limit)

    @pytest.mark.parametrize("bad", [None, "text", {"config": 1}, 5])
    def test_only_stored_configurations_are_appended(self, repo, bad):
        with pytest.raises(ValidationError):
            repo.append(bad)
        assert repo.latest() is None

    def test_a_configuration_larger_than_the_cap_is_refused(self, repo):
        huge = config(**{"places": [f"lugar{n:04d} " + "ñ" * 70 for n in range(500)]})
        many = TopicConfig.from_mapping(
            {
                **huge.to_mapping(),
                "earthquake": [f"term{n:04d} " + "ñ" * 70 for n in range(500)],
                "secretariat": [f"sec{n:04d} " + "ñ" * 70 for n in range(500)],
                "other_disaster": [f"dis{n:04d} " + "ñ" * 70 for n in range(500)],
                "official_bodies": [f"bod{n:04d} " + "ñ" * 70 for n in range(500)],
                "exclude": [f"exc{n:04d} " + "ñ" * 70 for n in range(500)],
                "official_channel_hints": [f"@h{n:04d}" + "漢" * 70 for n in range(500)],
            }
        )
        with pytest.raises(ValidationError):
            repo.append(stored(many))
        assert repo.latest() is None
