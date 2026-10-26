"""Reusable contract for any `SourceRepository` implementation.

Subclass it (name the subclass `Test...`) and implement `make_repo`. The in-memory fake
uses it now; the SQLite repository must pass the very same suite in unit 1c.
"""

import threading
from datetime import timedelta
from typing import Any

import pytest

from social_tracker.shared.errors import ConflictError, NotFoundError
from social_tracker.sources.domain import Source, parse_source
from social_tracker.sources.ports import SourceRepository
from tests.sources.factories import T0, payload

TRIALS = 25


def make_source(source_id: str = "src_one", name: str = "Some Source", **extra: Any) -> Source:
    return parse_source(payload("news_rss", id=source_id, name=name, **extra))


class SourceRepositoryContract:
    def make_repo(self) -> SourceRepository:
        raise NotImplementedError

    @pytest.fixture
    def repo(self) -> SourceRepository:
        return self.make_repo()

    # ---- add / get ----

    def test_add_then_get_round_trips(self, repo: SourceRepository):
        source = make_source()
        repo.add(source)
        assert repo.get("src_one") == source

    def test_every_source_type_round_trips(self, repo: SourceRepository):
        sources = [
            parse_source(payload("news_rss", id="s_news", name="n", keywords=["a", "b"])),
            parse_source(payload("instagram", id="s_ig", name="i")),
            parse_source(payload("live_stream", id="s_live", name="l")),
            parse_source(payload("recorded_audio", id="s_audio", name="r")),
        ]
        for source in sources:
            repo.add(source)
        for source in sources:
            assert repo.get(source.id) == source

    def test_unicode_round_trip(self, repo: SourceRepository):
        source = make_source(name="Café 新聞 \U0001f4f0")
        repo.add(source)
        assert repo.get("src_one").name == "Café 新聞 \U0001f4f0"

    def test_timestamps_survive_as_aware_utc(self, repo: SourceRepository):
        repo.add(make_source())
        got = repo.get("src_one")
        assert got.created_at == T0
        assert got.created_at.utcoffset() == timedelta(0)

    def test_add_duplicate_id_conflicts(self, repo: SourceRepository):
        repo.add(make_source("src_one", "First"))
        with pytest.raises(ConflictError) as info:
            repo.add(make_source("src_one", "Second"))
        assert info.value.code == "conflict"
        assert repo.get("src_one").name == "First"

    @pytest.mark.parametrize("other", ["same name", "SAME NAME", "  Same Name  ", "sAmE nAmE\t"])
    def test_add_duplicate_name_differing_by_case_or_whitespace_conflicts(
        self, repo: SourceRepository, other: str
    ):
        repo.add(make_source("src_one", "Same Name"))
        with pytest.raises(ConflictError):
            repo.add(make_source("src_two", other))
        assert [s.id for s in repo.list()] == ["src_one"]

    def test_add_duplicate_name_differing_by_unicode_composition_conflicts(
        self, repo: SourceRepository
    ):
        repo.add(make_source("src_one", "Café"))
        with pytest.raises(ConflictError):
            repo.add(make_source("src_two", "Café"))

    def test_distinct_names_coexist(self, repo: SourceRepository):
        repo.add(make_source("src_one", "Radio A"))
        repo.add(make_source("src_two", "Radio B"))
        assert len(repo.list()) == 2

    def test_get_missing_raises_not_found(self, repo: SourceRepository):
        with pytest.raises(NotFoundError) as info:
            repo.get("nope")
        assert info.value.code == "not_found"

    def test_get_empty_id_raises_not_found(self, repo: SourceRepository):
        with pytest.raises(NotFoundError):
            repo.get("")

    # ---- aliasing: the store owns its copy ----

    def test_tampering_with_a_returned_source_cannot_change_the_stored_copy(
        self, repo: SourceRepository
    ):
        repo.add(make_source())
        got = repo.get("src_one")
        object.__setattr__(got, "credibility", 1)  # bypasses the frozen guard on purpose
        assert repo.get("src_one").credibility == 3

    def test_tampering_with_the_added_source_cannot_change_the_stored_copy(
        self, repo: SourceRepository
    ):
        source = make_source()
        repo.add(source)
        object.__setattr__(source, "name", "Hacked")
        assert repo.get("src_one").name == "Some Source"
        assert repo.find_by_name_ci("hacked") is None
        assert repo.find_by_name_ci("some source") is not None

    def test_tampering_with_the_updated_source_cannot_change_the_stored_copy(
        self, repo: SourceRepository
    ):
        repo.add(make_source())
        changed = make_source(credibility=5)
        repo.update(changed)
        object.__setattr__(changed, "credibility", 1)
        assert repo.get("src_one").credibility == 5

    def test_tampering_with_listed_and_found_sources_cannot_change_the_store(
        self, repo: SourceRepository
    ):
        repo.add(make_source())
        object.__setattr__(repo.list()[0], "credibility", 1)
        found = repo.find_by_name_ci("some source")
        assert found is not None
        object.__setattr__(found, "credibility", 2)
        assert repo.get("src_one").credibility == 3

    def test_each_read_returns_an_independent_object(self, repo: SourceRepository):
        repo.add(make_source())
        assert repo.get("src_one") is not repo.get("src_one")

    def test_with_changes_results_never_alias_the_stored_source(self, repo: SourceRepository):
        repo.add(make_source())
        changed = repo.get("src_one").with_changes(credibility=4)
        assert repo.get("src_one").credibility == 3
        repo.update(changed)
        assert repo.get("src_one").credibility == 4

    # ---- list ----

    def test_list_is_empty_initially(self, repo: SourceRepository):
        assert repo.list() == []

    def test_list_is_sorted_by_id(self, repo: SourceRepository):
        for source_id in ["src_c", "src_a", "src_b"]:
            repo.add(make_source(source_id, f"name {source_id}"))
        assert [s.id for s in repo.list()] == ["src_a", "src_b", "src_c"]

    def test_mutating_the_returned_list_does_not_affect_the_store(self, repo: SourceRepository):
        repo.add(make_source())
        repo.list().clear()
        assert len(repo.list()) == 1

    # ---- find_by_name_ci ----

    @pytest.mark.parametrize("query", ["Some Source", "some source", "  SOME SOURCE  "])
    def test_find_by_name_ci_matches_ignoring_case_and_whitespace(
        self, repo: SourceRepository, query: str
    ):
        repo.add(make_source())
        found = repo.find_by_name_ci(query)
        assert found is not None
        assert found.id == "src_one"

    def test_find_by_name_ci_returns_none_when_absent(self, repo: SourceRepository):
        repo.add(make_source())
        assert repo.find_by_name_ci("other") is None
        assert repo.find_by_name_ci("") is None

    def test_find_by_name_ci_on_empty_repo(self, repo: SourceRepository):
        assert repo.find_by_name_ci("anything") is None

    # ---- update ----

    def test_update_persists_changes(self, repo: SourceRepository):
        repo.add(make_source())
        changed = make_source(name="Renamed", credibility=5, enabled=True)
        repo.update(changed)
        got = repo.get("src_one")
        assert (got.name, got.credibility, got.enabled) == ("Renamed", 5, True)

    def test_update_missing_raises_not_found(self, repo: SourceRepository):
        with pytest.raises(NotFoundError):
            repo.update(make_source("src_ghost", "Ghost"))
        assert repo.list() == []

    def test_update_to_another_sources_name_conflicts_and_changes_nothing(
        self, repo: SourceRepository
    ):
        repo.add(make_source("src_one", "Alpha"))
        repo.add(make_source("src_two", "Beta"))
        with pytest.raises(ConflictError):
            repo.update(make_source("src_two", " ALPHA "))
        assert repo.get("src_two").name == "Beta"

    def test_update_may_change_the_case_of_its_own_name(self, repo: SourceRepository):
        repo.add(make_source("src_one", "alpha"))
        repo.update(make_source("src_one", "ALPHA"))
        assert repo.get("src_one").name == "ALPHA"

    def test_rename_frees_the_old_name(self, repo: SourceRepository):
        repo.add(make_source("src_one", "Alpha"))
        repo.update(make_source("src_one", "Gamma"))
        repo.add(make_source("src_two", "alpha"))
        assert repo.find_by_name_ci("alpha").id == "src_two"  # type: ignore[union-attr]
        assert repo.find_by_name_ci("gamma").id == "src_one"  # type: ignore[union-attr]

    # ---- delete ----

    def test_delete_removes_and_frees_id_and_name(self, repo: SourceRepository):
        repo.add(make_source("src_one", "Alpha"))
        repo.delete("src_one")
        assert repo.list() == []
        assert repo.find_by_name_ci("alpha") is None
        repo.add(make_source("src_one", "Alpha"))  # id and name reusable
        assert len(repo.list()) == 1

    def test_delete_missing_raises_not_found(self, repo: SourceRepository):
        with pytest.raises(NotFoundError):
            repo.delete("nope")

    def test_delete_twice_raises_not_found_the_second_time(self, repo: SourceRepository):
        repo.add(make_source())
        repo.delete("src_one")
        with pytest.raises(NotFoundError):
            repo.delete("src_one")

    def test_delete_does_not_touch_other_sources(self, repo: SourceRepository):
        repo.add(make_source("src_one", "Alpha"))
        repo.add(make_source("src_two", "Beta"))
        repo.delete("src_one")
        assert [s.id for s in repo.list()] == ["src_two"]

    # ---- concurrency ----

    def test_two_threads_creating_the_same_name_exactly_one_wins(self, repo: SourceRepository):
        lock = threading.Lock()
        for trial in range(TRIALS):
            name = f"Contended {trial}"
            barrier = threading.Barrier(2)
            outcomes: list[object] = []

            def worker(source_id: str, name: str = name, barrier=barrier, outcomes=outcomes):
                source = make_source(source_id, name)
                barrier.wait(timeout=5)
                try:
                    repo.add(source)
                    result: object = "won"
                except ConflictError:
                    result = "lost"
                except Exception as exc:  # pragma: no cover - surfaced by the assert below
                    result = exc
                with lock:
                    outcomes.append(result)

            threads = [
                threading.Thread(target=worker, args=(f"src_a{trial}",)),
                threading.Thread(target=worker, args=(f"src_b{trial}",)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            assert sorted(map(str, outcomes)) == ["lost", "won"], outcomes
            matches = [s for s in repo.list() if s.name == name]
            assert len(matches) == 1

    def test_two_threads_creating_the_same_id_exactly_one_wins(self, repo: SourceRepository):
        barrier = threading.Barrier(2)
        outcomes: list[str] = []
        lock = threading.Lock()

        def worker(name: str):
            source = make_source("src_one", name)
            barrier.wait(timeout=5)
            try:
                repo.add(source)
                result = "won"
            except ConflictError:
                result = "lost"
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=worker, args=(n,)) for n in ("First", "Second")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert sorted(outcomes) == ["lost", "won"]
        assert len(repo.list()) == 1
