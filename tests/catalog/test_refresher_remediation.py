"""The refresher after the adversarial review: an empty answer is a quiet day (R2), the stop flag
(R14), the claim that is always released, the race between the read and the claim, answers cut
short, and events a catalog no longer lists.
"""

import logging
from datetime import timedelta

import pytest

from social_tracker.catalog.domain import CatalogEvent, CatalogSettings, CatalogSource
from social_tracker.catalog.service import CatalogRefresher, Outcome
from social_tracker.collector.domain import FetchError, FetchErrorKind
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import StorageError
from tests.catalog.support import NOW, QUAKE, ScriptedFetcher, emsc_payload, usgs_payload

USGS, EMSC = CatalogSource.USGS, CatalogSource.EMSC
WIDE = (NOW - timedelta(days=30), NOW + timedelta(days=1))


class World:
    def __init__(self, tmp_path, settings=None, *, should_stop=None, repo_wrapper=None):
        self.db = Database(tmp_path / "c.db", busy_timeout_ms=200)
        self.repo = SqliteCatalogRepository(self.db)
        self.clock = FixedClock(NOW)
        self.fetcher = ScriptedFetcher()
        self.fetcher.on("usgs", usgs_payload({"id": "us1"}))
        self.fetcher.on("emsc", emsc_payload({"id": "em1"}))
        self.refresher = CatalogRefresher(
            self.fetcher,
            repo_wrapper(self.repo) if repo_wrapper else self.repo,
            SqliteUnitOfWork(self.db),
            self.clock,
            settings or CatalogSettings(),
            jitter=lambda base: 0.0,
            should_stop=should_stop,
        )

    def ids(self, source=None):
        found = self.repo.events_between(*WIDE)
        return sorted(e.external_id for e in found if source in (None, e.source))

    def outcome(self, report, source):
        return next(r for r in report.results if r.source is source)


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.db.close()


def put(world, id, source=USGS, at=QUAKE, magnitude=5.1):
    event = CatalogEvent(source, id, at, 3.5, -76.5, magnitude, "Cali")
    world.repo.upsert_events([event], NOW)


# ---- R2: an empty answer is a quiet day ----


@pytest.mark.parametrize("empty", [b"", b"  \n", b"\xef\xbb\xbf"], ids=["empty", "blank", "bom"])
@pytest.mark.parametrize("source", [USGS, EMSC])
@pytest.mark.parametrize("kind", ["regional", "global"])
def test_one_empty_answer_does_not_discard_its_good_sibling(world, source, kind, empty):
    sibling = "global" if kind == "regional" else "regional"
    world.fetcher.on(source.value, empty, kind)
    payload = usgs_payload if source is USGS else emsc_payload
    world.fetcher.on(source.value, payload({"id": "kept"}), sibling)
    report = world.refresher.refresh()
    result = world.outcome(report, source)
    assert result.outcome is Outcome.REFRESHED
    assert "kept" in world.ids(source)
    assert world.repo.get_state(source).consecutive_failures == 0
    assert world.repo.get_state(source).next_attempt_at is None


@pytest.mark.parametrize("source", [USGS, EMSC])
def test_two_empty_answers_are_a_good_refresh_with_nothing_to_store(world, source):
    world.fetcher.on(source.value, b"")
    result = world.outcome(world.refresher.refresh(), source)
    assert (result.outcome, result.new, result.discarded) == (Outcome.REFRESHED, 0, 0)
    assert world.repo.get_state(source).last_success_at == NOW


def test_a_non_empty_unreadable_answer_is_still_a_failure(world):
    world.fetcher.on("usgs", b"<html>", "regional")
    assert world.outcome(world.refresher.refresh(), USGS).outcome is Outcome.FAILED


# ---- R14: the stop flag ----


def test_a_stop_requested_before_the_refresh_fetches_nothing(tmp_path):
    w = World(tmp_path, should_stop=lambda: True)
    try:
        report = w.refresher.refresh()
        assert {r.outcome for r in report.results} == {Outcome.STOPPED}
        assert w.fetcher.calls == []
        assert w.repo.all_states() == []
    finally:
        w.db.close()


def test_a_stop_between_the_two_requests_of_a_catalog_leaves_no_trace(tmp_path):
    flag = []
    w = World(tmp_path, should_stop=lambda: bool(flag))
    try:
        w.fetcher.on("usgs", lambda url: (flag.append(1), usgs_payload({"id": "us1"}))[1])
        report = w.refresher.refresh()
        assert w.outcome(report, USGS).outcome is Outcome.STOPPED
        assert w.outcome(report, EMSC).outcome is Outcome.STOPPED
        assert len(w.fetcher.calls) == 1  # the second request of USGS was never sent
        assert w.ids() == []
        state = w.repo.get_state(USGS)
        assert (state.claimed_until, state.consecutive_failures, state.next_attempt_at) == (
            None,
            0,
            None,
        )
        assert state.last_success_at is None
    finally:
        w.db.close()


def test_a_stop_after_the_first_catalog_keeps_it_and_skips_the_second(tmp_path):
    flag = []
    w = World(tmp_path, should_stop=lambda: bool(flag))
    try:
        w.fetcher.on("usgs", usgs_payload({"id": "us1"}))
        original_record = w.repo.record_success

        def record_then_stop(source, at, **kw):
            original_record(source, at, **kw)
            flag.append(1)

        w.repo.record_success = record_then_stop
        report = w.refresher.refresh()
        assert w.outcome(report, USGS).outcome is Outcome.REFRESHED
        assert w.outcome(report, EMSC).outcome is Outcome.STOPPED
        assert w.fetcher.calls_to("emsc") == []
        assert w.ids() == ["us1"]
    finally:
        w.db.close()


def test_without_a_stop_check_nothing_changes(world):
    assert {r.outcome for r in world.refresher.refresh().results} == {Outcome.REFRESHED}


# ---- the claim is always released ----


def test_a_keyboard_interrupt_during_the_fetch_still_releases_the_claim(world):
    def interrupted(url):
        raise KeyboardInterrupt

    world.fetcher.on("usgs", interrupted)
    with pytest.raises(KeyboardInterrupt):
        world.refresher.refresh()
    assert world.repo.get_state(USGS).claimed_until is None


def test_a_failing_failure_record_still_releases_the_claim(tmp_path):
    class Broken:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def record_failure(self, *args, **kwargs):
            raise StorageError("database error: SQLITE_BUSY")

    w = World(tmp_path, repo_wrapper=Broken)
    try:
        w.fetcher.on("usgs", FetchError(FetchErrorKind.TIMEOUT))
        report = w.refresher.refresh()
        assert w.outcome(report, USGS).outcome is Outcome.FAILED
        assert w.repo.get_state(USGS).claimed_until is None
    finally:
        w.db.close()


def test_a_failing_write_still_releases_the_claim(tmp_path):
    class Broken:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def upsert_events(self, *args, **kwargs):
            raise StorageError("database error: disk full")

    w = World(tmp_path, repo_wrapper=Broken)
    try:
        report = w.refresher.refresh()
        assert w.outcome(report, USGS).outcome is Outcome.FAILED
        assert all(s.claimed_until is None for s in w.repo.all_states())
    finally:
        w.db.close()


# ---- the race between the read of the state and the claim ----


def test_a_refresh_that_finished_while_we_were_claiming_is_not_repeated(tmp_path):
    class Racing:
        """Another refresher completes the catalog right before our claim succeeds."""

        def __init__(self, inner):
            self._inner = inner
            self.raced = False

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def try_claim(self, source, now, until):
            if not self.raced:
                self.raced = True
                self._inner.record_success(source, now)
            return self._inner.try_claim(source, now, until)

    w = World(tmp_path, repo_wrapper=Racing)
    try:
        report = w.refresher.refresh()
        assert w.outcome(report, USGS).outcome is Outcome.NOT_DUE
        assert w.fetcher.calls_to("usgs") == []
        assert w.repo.get_state(USGS).claimed_until is None
    finally:
        w.db.close()


def test_force_does_not_honour_the_race_check(tmp_path):
    class Racing:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def try_claim(self, source, now, until):
            self._inner.record_success(source, now)
            return self._inner.try_claim(source, now, until)

    w = World(tmp_path, repo_wrapper=Racing)
    try:
        assert w.outcome(w.refresher.refresh(force=True), USGS).outcome is Outcome.REFRESHED
    finally:
        w.db.close()


# ---- an answer that reached the limit may be cut short ----


def three(prefix):
    return [{"id": f"{prefix}{n}", "mag": 5.0 + n / 10} for n in range(3)]


def test_an_answer_that_reaches_max_events_is_marked_truncated(tmp_path, caplog):
    w = World(tmp_path, CatalogSettings(max_events=3))
    try:
        w.fetcher.on("usgs", usgs_payload(*three("a")), "regional")
        with caplog.at_level(logging.WARNING, logger="social_tracker.catalog"):
            report = w.refresher.refresh()
        result = w.outcome(report, USGS)
        assert (result.outcome, result.truncated, result.new) == (Outcome.REFRESHED, True, 4)
        assert w.repo.get_state(USGS).truncated is True
        assert w.repo.get_state(EMSC).truncated is False
        assert "cut short" in caplog.text
        assert "a0" not in caplog.text
    finally:
        w.db.close()


@pytest.mark.parametrize(("count", "truncated"), [(2, False), (3, True), (4, True)])
def test_the_truncation_boundary_is_the_limit(tmp_path, count, truncated):
    w = World(tmp_path, CatalogSettings(max_events=3))
    try:
        quakes = [{"id": f"a{n}", "mag": 5.0} for n in range(count)]
        w.fetcher.on("usgs", usgs_payload(*quakes), "global")
        w.refresher.refresh()
        assert w.repo.get_state(USGS).truncated is truncated
    finally:
        w.db.close()


def test_the_next_complete_answer_clears_the_mark(tmp_path):
    w = World(tmp_path, CatalogSettings(max_events=3))
    try:
        w.fetcher.on("usgs", usgs_payload(*three("a")), "regional")
        w.refresher.refresh()
        w.clock.advance(timedelta(minutes=11))
        w.fetcher.on("usgs", usgs_payload({"id": "a0"}), "regional")
        w.refresher.refresh()
        assert w.repo.get_state(USGS).truncated is False
    finally:
        w.db.close()


def test_a_dry_run_reports_the_truncation_and_writes_nothing(tmp_path):
    w = World(tmp_path, CatalogSettings(max_events=3))
    try:
        w.fetcher.on("usgs", usgs_payload(*three("a")), "regional")
        result = w.outcome(w.refresher.refresh(dry_run=True), USGS)
        assert (result.outcome, result.truncated) == (Outcome.DRY_RUN, True)
        assert w.repo.all_states() == []
    finally:
        w.db.close()


# ---- events a catalog no longer lists ----


def test_an_event_the_catalog_dropped_from_the_covered_window_is_removed(world):
    world.refresher.refresh()
    put(world, "gone", at=NOW - timedelta(hours=3))
    world.clock.advance(timedelta(minutes=11))
    report = world.refresher.refresh()
    assert world.outcome(report, USGS).removed == 1
    assert world.ids(USGS) == ["us1"]
    assert world.ids(EMSC) == ["em1"]


def test_the_revised_events_are_updated_not_removed(world):
    world.refresher.refresh()
    world.clock.advance(timedelta(minutes=11))
    world.fetcher.on("usgs", usgs_payload({"id": "us1", "mag": 5.9}))
    result = world.outcome(world.refresher.refresh(), USGS)
    assert (result.updated, result.removed) == (1, 0)


def test_the_edge_of_the_covered_window_is_included_and_one_second_older_is_kept(world):
    start = NOW - timedelta(hours=CatalogSettings().lookback_hours)
    put(world, "edge", at=start)
    put(world, "older", at=start - timedelta(seconds=1))
    put(world, "newest", at=NOW)
    world.refresher.refresh()
    assert world.ids(USGS) == ["older", "us1"]


def test_an_event_of_the_other_catalog_is_never_removed_by_this_one(world):
    put(world, "em-only", source=EMSC, at=NOW - timedelta(hours=2))
    world.fetcher.on("emsc", RuntimeError("down"))
    world.refresher.refresh()
    assert "em-only" in world.ids(EMSC)
    world.fetcher.on("emsc", emsc_payload({"id": "em1"}))
    world.clock.advance(timedelta(hours=1))
    world.refresher.refresh(force=True)
    assert "em-only" not in world.ids(EMSC)


def test_a_failed_refresh_removes_nothing(world):
    put(world, "keep", at=NOW - timedelta(hours=3))
    world.fetcher.on("usgs", FetchError.http(503))
    world.refresher.refresh()
    assert "keep" in world.ids(USGS)


def test_a_half_answered_catalog_removes_nothing(world):
    put(world, "keep", at=NOW - timedelta(hours=3))
    world.fetcher.on("usgs", usgs_payload({"id": "x"}), "regional")
    world.fetcher.on("usgs", FetchError.http(502), "global")
    world.refresher.refresh()
    assert world.ids(USGS) == ["keep"]


def test_a_truncated_answer_removes_nothing(tmp_path):
    w = World(tmp_path, CatalogSettings(max_events=3))
    try:
        put(w, "keep", at=NOW - timedelta(hours=3))
        w.fetcher.on("usgs", usgs_payload(*three("a")), "regional")
        result = w.outcome(w.refresher.refresh(), USGS)
        assert result.removed == 0
        assert "keep" in w.ids(USGS)
    finally:
        w.db.close()


def test_an_empty_answer_removes_nothing_because_it_proves_too_little(world):
    put(world, "keep", at=NOW - timedelta(hours=3))
    world.fetcher.on("usgs", b"")
    result = world.outcome(world.refresher.refresh(), USGS)
    assert (result.outcome, result.removed) == (Outcome.REFRESHED, 0)
    assert "keep" in world.ids(USGS)


def test_a_dry_run_counts_what_would_be_removed_and_removes_nothing(world):
    put(world, "gone", at=NOW - timedelta(hours=3))
    result = world.outcome(world.refresher.refresh(dry_run=True), USGS)
    assert (result.outcome, result.removed) == (Outcome.DRY_RUN, 1)
    assert "gone" in world.ids(USGS)


def test_the_removal_and_the_state_are_one_transaction(tmp_path):
    class Broken:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def record_success(self, *args, **kwargs):
            raise StorageError("database error: SQLITE_BUSY")

    w = World(tmp_path, repo_wrapper=Broken)
    try:
        put(w, "gone", at=NOW - timedelta(hours=3))
        report = w.refresher.refresh()
        assert w.outcome(report, USGS).outcome is Outcome.FAILED
        assert "gone" in w.ids(USGS)  # rolled back with everything else
        assert "us1" not in w.ids(USGS)
    finally:
        w.db.close()
