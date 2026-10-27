"""Collector service: hostile failure text, the item cap, and recovery details."""

from datetime import timedelta

from social_tracker.collector.domain import FetchError, FetchErrorKind, FetchResponse
from social_tracker.collector.service import CollectOutcome
from social_tracker.control.domain import CollectionState
from tests.collector.harness import FEED, rss


class LeakyFetchError(FetchError):
    """A failure whose message, against the rules, carries a URL with a secret and a newline."""

    def __init__(self) -> None:
        super().__init__(FetchErrorKind.TIMEOUT)
        self.message = "boom https://news.example.org/rss.xml?token=hunter2\nnext line"


def test_even_a_misbehaving_fetcher_cannot_put_a_url_or_a_secret_into_the_last_error(world):
    world.ready()
    world.fetcher.then(LeakyFetchError())
    result = world.collector.collect_once("src_news")
    assert result.error == "boom <url> next line"
    stored = world.control_repo.get_fetch_status("src_news").last_error
    assert stored == "boom <url> next line"
    audit = " ".join(e.detail for e in world.events())
    assert "hunter2" not in audit
    assert "news.example.org" not in audit


def test_entries_beyond_the_cap_are_counted_in_the_result(world):
    world.ready()
    many = rss(*[(f"g{i}", f"title {i}", f"body {i}") for i in range(505)])
    world.fetcher.then(many)
    result = world.collector.collect_once("src_news")
    assert (result.new, result.discarded) == (500, {"over_limit": 5})
    assert world.items.count() == 500


def test_recovering_an_interrupted_collection_keeps_the_last_successful_fetch(world):
    world.ready()
    world.collector.collect_once("src_news")
    fetched_at = world.clock.now()
    world.clock.advance(timedelta(minutes=11))
    world.control.start_collection("src_news", actor="system:worker", reason="before the crash")
    assert world.collector.recover_interrupted(force=True) == ["src_news"]
    status = world.control_repo.get_fetch_status("src_news")
    assert (status.last_fetch_at, status.last_error) == (fetched_at, "interrupted collection")


def test_recovery_opens_no_write_transaction_when_nothing_was_interrupted(world):
    world.ready("src_a")
    world.add_source("src_b")
    world.add_source("src_off", enable=False)
    opened = []
    original = world.uow.transaction

    def counting():
        opened.append(1)
        return original()

    world.uow.transaction = counting
    assert world.collector.recover_interrupted() == []
    assert opened == []
    assert world.state("src_a").state is CollectionState.IDLE


def test_the_result_of_a_not_eligible_source_carries_no_counts(world):
    world.ready()
    world.fetcher.then(FetchError(FetchErrorKind.TIMEOUT))
    world.collector.collect_once("src_news")
    result = world.collector.collect_once("src_news")  # still backing off
    assert result.outcome is CollectOutcome.SKIPPED_NOT_ELIGIBLE
    assert (result.new, result.duplicate, result.error) == (0, 0, None)
    assert dict(result.discarded) == {}


def test_a_source_switched_off_in_the_registry_discards_the_unit_even_if_its_state_lags(world):
    world.ready()

    def switch_the_flag_only(request):
        current = world.sources_repo.get("src_news")
        world.sources_repo.update(current.with_changes({"enabled": False}))
        return rss(("a", "t", "b"))

    world.fetcher.then(switch_the_flag_only)
    assert world.collector.collect_once("src_news").outcome is CollectOutcome.DISCARDED
    assert world.items.count() == 0


def test_a_failure_keeps_the_time_of_the_last_successful_fetch_and_the_validators(world):
    world.ready()
    world.fetcher.then(
        FetchResponse(FEED, etag='"v1"', last_modified="Sun, 01 Mar 2026 07:00:00 GMT")
    )
    world.collector.collect_once("src_news")
    fetched_at = world.clock.now()
    world.clock.advance(timedelta(minutes=11))
    world.fetcher.then(FetchError(FetchErrorKind.TIMEOUT))
    world.collector.collect_once("src_news")
    status = world.control_repo.get_fetch_status("src_news")
    assert (status.last_fetch_at, status.last_error) == (fetched_at, "timeout")
    assert (status.etag, status.last_modified) == ('"v1"', "Sun, 01 Mar 2026 07:00:00 GMT")
