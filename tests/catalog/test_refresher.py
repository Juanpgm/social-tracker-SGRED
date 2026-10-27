"""The catalog refresher: fetch, parse, write one catalog at a time, never crash, back off.

Network failures (timeout, DNS, 4xx, 429 with Retry-After, 5xx then 200, truncated body, a redirect
off the allowlist, a downgrade, an oversized body), no partial writes, the claim that keeps two
refreshers apart, dry runs, and a writer that can enter while a fetch is in flight.
"""

import threading
from datetime import timedelta

import pytest

from social_tracker.catalog.domain import CatalogSettings, CatalogSource
from social_tracker.catalog.service import CatalogRefresher, Outcome
from social_tracker.collector.domain import FetchError, FetchErrorKind
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import FixedClock
from tests.catalog.support import NOW, QUAKE, ScriptedFetcher, emsc_payload, usgs_payload

USGS, EMSC = CatalogSource.USGS, CatalogSource.EMSC
WIDE = (NOW - timedelta(days=30), NOW + timedelta(days=1))


class World:
    def __init__(self, tmp_path, settings=None):
        self.db = Database(tmp_path / "c.db", busy_timeout_ms=200)
        self.repo = SqliteCatalogRepository(self.db)
        self.clock = FixedClock(NOW)
        self.fetcher = ScriptedFetcher()
        self.fetcher.on("usgs", usgs_payload({"id": "us1"}))
        self.fetcher.on("emsc", emsc_payload({"id": "em1"}))
        self.refresher = CatalogRefresher(
            self.fetcher,
            self.repo,
            SqliteUnitOfWork(self.db),
            self.clock,
            settings or CatalogSettings(),
            jitter=lambda base: 0.0,
        )

    def ids(self) -> list[str]:
        return sorted(e.external_id for e in self.repo.events_between(*WIDE))

    def outcome(self, report, source):
        return next(r for r in report.results if r.source is source)


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.db.close()


# ---- the good path ----


def test_a_refresh_stores_both_catalogs_and_records_the_good_time(world):
    report = world.refresher.refresh()
    assert [r.outcome for r in report.results] == [Outcome.REFRESHED, Outcome.REFRESHED]
    assert world.ids() == ["em1", "us1"]
    for source in (USGS, EMSC):
        state = world.repo.get_state(source)
        assert state.last_success_at == NOW
        assert state.consecutive_failures == 0
    assert len(world.fetcher.calls_to("usgs")) == 2  # regional and global
    assert len(world.fetcher.calls_to("emsc")) == 2


def test_the_regional_and_the_global_answers_are_merged_without_repeats(world):
    world.fetcher.on("usgs", usgs_payload({"id": "same", "mag": 6.0}, {"id": "a"}), "regional")
    world.fetcher.on("usgs", usgs_payload({"id": "same", "mag": 6.0}, {"id": "b"}), "global")
    report = world.refresher.refresh()
    assert world.outcome(report, USGS).new == 3
    assert [e for e in world.ids() if e in ("same", "a", "b")] == ["a", "b", "same"]


def test_an_empty_answer_is_a_good_refresh_with_nothing_to_store(world):
    world.fetcher.on("usgs", usgs_payload())
    report = world.refresher.refresh()
    assert world.outcome(report, USGS).outcome is Outcome.REFRESHED
    assert world.outcome(report, USGS).new == 0
    assert world.repo.get_state(USGS).last_success_at == NOW


def test_a_revised_magnitude_updates_the_event_in_place(world):
    world.fetcher.on("usgs", usgs_payload({"id": "us1", "mag": 5.0}))
    world.refresher.refresh()
    world.clock.advance(timedelta(minutes=11))
    world.fetcher.on("usgs", usgs_payload({"id": "us1", "mag": 5.6}))
    report = world.refresher.refresh()
    assert (world.outcome(report, USGS).new, world.outcome(report, USGS).updated) == (0, 1)
    (stored,) = [e for e in world.repo.events_between(*WIDE) if e.source is USGS]
    assert stored.magnitude == 5.6


def test_the_report_counts_what_was_dropped(world):
    world.fetcher.on("usgs", usgs_payload({"id": "ok"}, {"id": "bad", "mag": None}))
    report = world.refresher.refresh()
    assert world.outcome(report, USGS).discarded == 2  # the bad one, in both requests


# ---- schedule ----


def test_a_catalog_refreshed_recently_is_not_due_until_its_interval_passes(world):
    world.refresher.refresh()
    calls = len(world.fetcher.calls)
    report = world.refresher.refresh()
    assert {r.outcome for r in report.results} == {Outcome.NOT_DUE}
    assert len(world.fetcher.calls) == calls
    world.clock.advance(timedelta(seconds=599))
    assert {r.outcome for r in world.refresher.refresh().results} == {Outcome.NOT_DUE}
    world.clock.advance(timedelta(seconds=1))
    assert {r.outcome for r in world.refresher.refresh().results} == {Outcome.REFRESHED}


def test_force_ignores_the_interval(world):
    world.refresher.refresh()
    report = world.refresher.refresh(force=True)
    assert {r.outcome for r in report.results} == {Outcome.REFRESHED}


# ---- failures: one catalog never takes the other down, nothing is half-written ----

FAILURES = {
    "timeout": FetchError(FetchErrorKind.TIMEOUT),
    "dns": FetchError(FetchErrorKind.DNS),
    "connect": FetchError(FetchErrorKind.CONNECT),
    "404": FetchError.http(404),
    "403": FetchError.http(403),
    "500": FetchError.http(500),
    "503": FetchError.http(503),
    "429": FetchError.http(429),
    "too_large": FetchError(FetchErrorKind.TOO_LARGE),
    "decode": FetchError(FetchErrorKind.DECODE),
    "off_allowlist_redirect": FetchError(FetchErrorKind.HOST_NOT_ALLOWED),
    "downgrade": FetchError(FetchErrorKind.INSECURE_REDIRECT),
    "blocked_address": FetchError(FetchErrorKind.BLOCKED_ADDRESS),
    "too_many_redirects": FetchError(FetchErrorKind.TOO_MANY_REDIRECTS),
    "protocol": FetchError(FetchErrorKind.PROTOCOL),
}


@pytest.mark.parametrize("name", FAILURES)
def test_a_failing_catalog_is_recorded_and_the_other_is_unaffected(world, name):
    world.fetcher.on("usgs", FAILURES[name])
    report = world.refresher.refresh()
    failed = world.outcome(report, USGS)
    assert failed.outcome is Outcome.FAILED
    assert failed.error == FAILURES[name].message
    assert world.outcome(report, EMSC).outcome is Outcome.REFRESHED
    assert world.ids() == ["em1"]
    state = world.repo.get_state(USGS)
    assert state.consecutive_failures == 1
    assert state.last_success_at is None
    assert state.last_error == FAILURES[name].message
    assert state.claimed_until is None


@pytest.mark.parametrize(
    "payload",
    [b"{", b"not json", b"<html>503</html>", b'{"features": "x"}', b"\xff\xfe", b"[]"],
    ids=["truncated", "text", "html", "bad-shape", "bad-utf8", "list"],
)
def test_an_unreadable_answer_is_a_failure_not_a_crash(world, payload):
    world.fetcher.on("emsc", payload)
    report = world.refresher.refresh()
    assert world.outcome(report, EMSC).outcome is Outcome.FAILED
    assert world.outcome(report, EMSC).error == "unreadable answer"
    assert "em1" not in world.ids()
    assert world.outcome(report, USGS).outcome is Outcome.REFRESHED


def test_a_half_answered_catalog_stores_nothing_of_it(world):
    world.fetcher.on("usgs", usgs_payload({"id": "regional-only"}), "regional")
    world.fetcher.on("usgs", FetchError.http(502), "global")
    report = world.refresher.refresh()
    assert world.outcome(report, USGS).outcome is Outcome.FAILED
    assert "regional-only" not in world.ids()
    assert world.repo.count_by_source().get(USGS, 0) == 0


def test_an_unexpected_exception_in_the_fetcher_is_a_failure_too(world):
    world.fetcher.on("usgs", RuntimeError("boom with a secret token"))
    report = world.refresher.refresh()
    failed = world.outcome(report, USGS)
    assert failed.outcome is Outcome.FAILED
    assert failed.error == "unexpected error"
    assert "secret" not in repr(report)
    assert "secret" not in (world.repo.get_state(USGS).last_error or "")


def test_a_fetch_without_a_body_is_not_trusted(world):
    class NotModified:
        def fetch(self, request):
            from social_tracker.collector.domain import FetchResponse

            return FetchResponse(b"", not_modified=True)

    world.refresher._fetcher = NotModified()
    report = world.refresher.refresh()
    assert {r.outcome for r in report.results} == {Outcome.FAILED}


# ---- backoff ----


def test_failures_back_off_exponentially_up_to_the_cap(world):
    world.fetcher.on("usgs", FetchError(FetchErrorKind.TIMEOUT))
    waits = []
    for _ in range(9):
        world.refresher.refresh(force=True)
        state = world.repo.get_state(USGS)
        waits.append((state.next_attempt_at - NOW).total_seconds())
    assert waits == [60, 120, 240, 480, 960, 1920, 3600, 3600, 3600]
    assert world.repo.get_state(USGS).consecutive_failures == 9


def test_a_failed_catalog_is_not_asked_again_before_its_retry_time(world):
    world.fetcher.on("usgs", FetchError.http(503))
    world.refresher.refresh()
    before = len(world.fetcher.calls_to("usgs"))
    world.clock.advance(timedelta(seconds=59))
    report = world.refresher.refresh()
    assert world.outcome(report, USGS).outcome is Outcome.SKIPPED_BACKOFF
    assert len(world.fetcher.calls_to("usgs")) == before
    world.clock.advance(timedelta(seconds=1))
    assert world.outcome(world.refresher.refresh(), USGS).outcome is Outcome.FAILED
    assert len(world.fetcher.calls_to("usgs")) > before


def test_a_5xx_then_a_good_answer_recovers_and_forgets_the_failures(world):
    world.fetcher.on("usgs", FetchError.http(503))
    world.refresher.refresh()
    world.clock.advance(timedelta(seconds=60))
    world.fetcher.on("usgs", usgs_payload({"id": "back"}))
    report = world.refresher.refresh()
    assert world.outcome(report, USGS).outcome is Outcome.REFRESHED
    state = world.repo.get_state(USGS)
    assert (state.consecutive_failures, state.next_attempt_at, state.last_error) == (0, None, None)
    assert "back" in world.ids()


def test_retry_after_longer_than_the_backoff_wins_and_is_capped(world):
    world.fetcher.on("usgs", FetchError.http(429, retry_after_s=900.0))
    world.refresher.refresh()
    assert world.repo.get_state(USGS).next_attempt_at == NOW + timedelta(seconds=900)
    world.fetcher.on("usgs", FetchError.http(429, retry_after_s=10 * 86400.0))
    world.refresher.refresh(force=True)
    assert world.repo.get_state(USGS).next_attempt_at == NOW + timedelta(hours=6)
    world.fetcher.on("usgs", FetchError.http(429, retry_after_s=5.0))
    world.refresher.refresh(force=True)
    assert world.repo.get_state(USGS).next_attempt_at == NOW + timedelta(seconds=240)


def test_force_also_ignores_a_backoff(world):
    world.fetcher.on("usgs", FetchError.http(503))
    world.refresher.refresh()
    world.fetcher.on("usgs", usgs_payload({"id": "manual"}))
    report = world.refresher.refresh(force=True)
    assert world.outcome(report, USGS).outcome is Outcome.REFRESHED


# ---- dry run ----


def test_a_dry_run_fetches_and_reports_but_writes_nothing(world):
    report = world.refresher.refresh(dry_run=True)
    assert {r.outcome for r in report.results} == {Outcome.DRY_RUN}
    assert world.outcome(report, USGS).new == 1
    assert world.ids() == []
    assert world.repo.all_states() == []
    assert len(world.fetcher.calls) == 4


def test_a_dry_run_failure_is_reported_and_leaves_no_state(world):
    world.fetcher.on("usgs", FetchError(FetchErrorKind.TIMEOUT))
    report = world.refresher.refresh(dry_run=True)
    assert world.outcome(report, USGS).outcome is Outcome.FAILED
    assert world.repo.all_states() == []


# ---- two refreshers, one writer ----


def test_a_second_refresher_while_the_first_fetches_does_not_fetch(world, tmp_path):
    other_db = Database(tmp_path / "c.db", busy_timeout_ms=200)
    other = CatalogRefresher(
        world.fetcher,
        SqliteCatalogRepository(other_db),
        SqliteUnitOfWork(other_db),
        world.clock,
        jitter=lambda base: 0.0,
    )
    inner = {}

    def while_fetching(url: str) -> bytes:
        if "usgs.gov" in url and "report" not in inner:
            inner["report"] = other.refresh(force=True)
        return usgs_payload({"id": "us1"})

    world.fetcher.on("usgs", while_fetching)
    try:
        world.refresher.refresh()
    finally:
        other_db.close()
    second = next(r for r in inner["report"].results if r.source is USGS)
    assert second.outcome is Outcome.SKIPPED_BUSY
    assert len(world.fetcher.calls_to("usgs")) == 2  # the first one's two requests only
    assert world.ids() == ["em1", "us1"]


def test_an_expired_claim_does_not_block_a_refresh_forever(world):
    world.repo.try_claim(USGS, NOW - timedelta(hours=1), NOW - timedelta(minutes=1))
    assert world.outcome(world.refresher.refresh(), USGS).outcome is Outcome.REFRESHED


def test_a_claim_still_running_blocks_even_a_forced_refresh(world):
    world.repo.try_claim(USGS, NOW, NOW + timedelta(minutes=2))
    report = world.refresher.refresh(force=True)
    assert world.outcome(report, USGS).outcome is Outcome.SKIPPED_BUSY
    assert world.outcome(report, EMSC).outcome is Outcome.REFRESHED
    assert world.fetcher.calls_to("usgs") == []


def test_the_claim_is_released_after_every_outcome(world):
    world.fetcher.on("emsc", FetchError.http(500))
    world.refresher.refresh()
    assert all(s.claimed_until is None for s in world.repo.all_states())


# ---- a writer can enter while the network is slow ----


def test_another_writer_can_commit_while_a_fetch_is_in_flight(world, tmp_path):
    written = []

    def slow_answer(url: str) -> bytes:
        if "usgs.gov" in url and not written:
            other = Database(tmp_path / "c.db", busy_timeout_ms=100)
            try:
                with other.transaction() as conn:
                    conn.execute(
                        "INSERT INTO control_events (at, kind, detail) VALUES (?, 'probe', '{}')",
                        (NOW.isoformat(),),
                    )
                written.append(True)
            finally:
                other.close()
        return usgs_payload({"id": "us1"})

    world.fetcher.on("usgs", slow_answer)
    report = world.refresher.refresh()
    assert written == [True]  # no busy error: the refresh holds no lock while it fetches
    assert world.outcome(report, USGS).outcome is Outcome.REFRESHED


def test_concurrent_refreshers_on_threads_store_every_event_once(tmp_path):
    results = []
    path = tmp_path / "t.db"
    barrier = threading.Barrier(4)

    def run() -> None:
        with Database(path, busy_timeout_ms=5000) as db:
            fetcher = (
                ScriptedFetcher()
                .on("usgs", usgs_payload({"id": "u"}))
                .on("emsc", emsc_payload({"id": "e"}))
            )
            refresher = CatalogRefresher(
                fetcher, SqliteCatalogRepository(db), SqliteUnitOfWork(db), FixedClock(NOW)
            )
            barrier.wait()
            results.append(refresher.refresh(force=True))

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with Database(path) as db:
        rows = SqliteCatalogRepository(db).events_between(*WIDE)
    assert sorted(e.external_id for e in rows) == ["e", "u"]
    assert len(results) == 4


# ---- housekeeping ----


def test_the_event_a_catalog_reports_is_the_one_stored(world):
    world.refresher.refresh()
    (event,) = [e for e in world.repo.events_between(*WIDE) if e.source is USGS]
    assert (event.origin_time, event.latitude, event.longitude) == (QUAKE, 3.5, -76.5)
