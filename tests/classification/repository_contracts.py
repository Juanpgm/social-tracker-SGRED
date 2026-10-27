"""Reusable contracts for the classification ports (in-memory fakes and SQLite alike).

What every implementation must honour (and these suites pin):
- ClassificationRepository: one verdict per item, never overwritten; a round trip loses nothing;
  `stats` counts what is stored.
- SignatureRepository: candidates are found by identical content hash or a shared band key, never
  include the item itself, come oldest first, and pruning forgets their bands too.
- TopicConfigRepository: an append-only history numbered 1, 2, 3...; the latest is in force.
"""

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
        "classifier_version": "cascade-1/topics-0",
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
            classifier_version="cascade-1/topics-12",
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


def signature(n: int | str, *, content_hash: str | None = None, minhash=None, days: int = 0):
    return ItemSignature(
        item_id=item(n),
        content_hash=content_hash or f"{n:0>2}".encode().hex().ljust(64, "0")[:64],
        ref_at=NOW + timedelta(days=days),
        minhash=minhash if minhash is not None else tuple(range(NUM_PERM)),
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
