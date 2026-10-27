"""The SQLite catalog cache: upserts, range reads, purge, refresh state and the claim."""

import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.catalog.domain import CatalogEvent, CatalogSource, RefreshState
from social_tracker.catalog.ports import CatalogRepository
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.db import Database, load_migrations
from social_tracker.shared.errors import StorageError

T = datetime(2026, 10, 27, 17, 0, tzinfo=UTC)
LATER = T + timedelta(minutes=10)


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "c.db") as database:
        yield database


@pytest.fixture
def repo(db):
    return SqliteCatalogRepository(db)


def event(id="us1", source=CatalogSource.USGS, minutes=0, magnitude=5.0, place="Cali", **kw):
    return CatalogEvent(
        source,
        id,
        T + timedelta(minutes=minutes),
        kw.get("latitude", 3.4),
        kw.get("longitude", -76.5),
        magnitude,
        place,
    )


def test_it_satisfies_the_port(repo):
    assert isinstance(repo, CatalogRepository)


def test_the_migration_is_the_sixth_and_creates_the_two_tables(db):
    assert [m.version for m in load_migrations()][-1] >= 6
    names = {r[0] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"catalog_events", "catalog_refresh_state"} <= names


# ---- events ----


def test_new_events_are_stored_and_read_back_by_time(repo, db):
    with db.transaction():
        report = repo.upsert_events([event("a", minutes=-5), event("b", minutes=5)], LATER)
    assert (report.new, report.updated, report.unchanged) == (2, 0, 0)
    found = repo.events_between(T - timedelta(minutes=10), T + timedelta(minutes=10))
    assert [e.external_id for e in found] == ["a", "b"]
    assert found[0] == event("a", minutes=-5)


def test_a_revision_of_the_same_event_updates_it_in_place(repo, db):
    with db.transaction():
        repo.upsert_events([event("a", magnitude=5.0, place="Old place")], T)
    with db.transaction():
        report = repo.upsert_events([event("a", magnitude=5.6, place="New place")], LATER)
    assert (report.new, report.updated, report.unchanged) == (0, 1, 0)
    (stored,) = repo.events_between(T - timedelta(hours=1), T + timedelta(hours=1))
    assert (stored.magnitude, stored.place) == (5.6, "New place")
    assert repo.count_by_source() == {CatalogSource.USGS: 1}


def test_the_same_event_again_is_unchanged_and_does_not_touch_the_row(repo, db):
    with db.transaction():
        repo.upsert_events([event("a")], T)
    with db.transaction():
        report = repo.upsert_events([event("a")], LATER)
    assert (report.new, report.updated, report.unchanged) == (0, 0, 1)
    fetched = db.query("SELECT fetched_at FROM catalog_events")[0][0]
    assert fetched.startswith("2026-10-27T17:00:00")


def test_the_same_id_in_two_catalogs_is_two_rows(repo, db):
    with db.transaction():
        repo.upsert_events([event("x"), event("x", source=CatalogSource.EMSC)], T)
    assert repo.count_by_source() == {CatalogSource.USGS: 1, CatalogSource.EMSC: 1}


def test_an_empty_batch_writes_nothing(repo, db):
    with db.transaction():
        report = repo.upsert_events([], T)
    assert (report.new, report.updated, report.unchanged) == (0, 0, 0)
    assert repo.count_by_source() == {}


def test_the_time_range_includes_both_edges_and_excludes_beyond(repo, db):
    edges = [event("lo", minutes=-60), event("hi", minutes=60), event("out1", minutes=-61),
             event("out2", minutes=61)]  # fmt: skip
    with db.transaction():
        repo.upsert_events(edges, T)
    ids = [
        e.external_id
        for e in repo.events_between(T - timedelta(minutes=60), T + timedelta(minutes=60))
    ]
    assert ids == ["lo", "hi"]
    assert repo.events_between(T + timedelta(hours=5), T + timedelta(hours=6)) == []


def test_events_come_back_in_time_then_id_order(repo, db):
    with db.transaction():
        repo.upsert_events([event("b"), event("a"), event("c", minutes=-1)], T)
    ids = [
        e.external_id for e in repo.events_between(T - timedelta(hours=1), T + timedelta(hours=1))
    ]
    assert ids == ["c", "a", "b"]


def test_a_reversed_range_is_empty(repo, db):
    with db.transaction():
        repo.upsert_events([event("a")], T)
    assert repo.events_between(T + timedelta(hours=1), T - timedelta(hours=1)) == []


def test_hostile_text_is_stored_as_data(repo, db):
    nasty = "'); DROP TABLE catalog_events;-- <script>alert(1)</script>"
    with db.transaction():
        repo.upsert_events([event("a", place=nasty)], T)
    (stored,) = repo.events_between(T - timedelta(hours=1), T + timedelta(hours=1))
    assert stored.place == nasty
    assert repo.count_by_source() == {CatalogSource.USGS: 1}


def test_a_failed_transaction_leaves_nothing_behind(repo, db):
    with pytest.raises(RuntimeError), db.transaction():
        repo.upsert_events([event("a"), event("b")], T)
        raise RuntimeError("boom")
    assert repo.count_by_source() == {}


def test_purge_removes_only_events_strictly_before_the_cutoff(repo, db):
    with db.transaction():
        repo.upsert_events(
            [event("old", minutes=-120), event("edge", minutes=-60), event("new")], T
        )
        removed = repo.purge_before(T - timedelta(minutes=60))
    assert removed == 1
    ids = [e.external_id for e in repo.events_between(T - timedelta(days=1), T + timedelta(days=1))]
    assert ids == ["edge", "new"]


def test_purge_on_an_empty_cache_removes_nothing(repo, db):
    with db.transaction():
        assert repo.purge_before(T) == 0


def test_the_schema_refuses_rows_the_domain_would_refuse(db):
    bad = [
        ("usgs", "a", "2026-10-27T17:00:00+00:00", 95.0, 0.0, 5.0, "", T.isoformat()),
        ("usgs", "a", "2026-10-27T17:00:00+00:00", 0.0, 190.0, 5.0, "", T.isoformat()),
        ("usgs", "a", "2026-10-27T17:00:00+00:00", 0.0, 0.0, 11.0, "", T.isoformat()),
        ("sgc", "a", "2026-10-27T17:00:00+00:00", 0.0, 0.0, 5.0, "", T.isoformat()),
        ("usgs", "", "2026-10-27T17:00:00+00:00", 0.0, 0.0, 5.0, "", T.isoformat()),
        ("usgs", "a", "yesterday", 0.0, 0.0, 5.0, "", T.isoformat()),
        ("usgs", "a", "2026-10-27T17:00:00+00:00", 0.0, 0.0, 5.0, "x" * 201, T.isoformat()),
    ]
    for row in bad:
        with pytest.raises(StorageError), db.transaction() as conn:
            conn.execute(
                "INSERT INTO catalog_events (source, external_id, origin_time, latitude, "
                "longitude, magnitude, place, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )


def test_a_tampered_row_is_a_storage_error_not_a_half_valid_event(repo, db):
    with db.transaction() as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")  # as a tampered file would be
        conn.execute(
            "INSERT INTO catalog_events (source, external_id, origin_time, latitude, longitude, "
            "magnitude, place, fetched_at) VALUES ('usgs', 'bad id', ?, 0, 0, 5, '', ?)",
            (T.isoformat(), T.isoformat()),
        )
        conn.execute("PRAGMA ignore_check_constraints = OFF")
    with pytest.raises(StorageError):
        repo.events_between(T - timedelta(hours=1), T + timedelta(hours=1))


# ---- refresh state ----


def test_there_is_no_state_before_the_first_refresh(repo):
    assert repo.get_state(CatalogSource.USGS) is None
    assert repo.all_states() == []


def test_a_success_records_the_time_and_clears_failures_and_the_claim(repo, db):
    with db.transaction():
        repo.record_failure(CatalogSource.USGS, T, T + timedelta(minutes=2), "timeout")
        assert repo.try_claim(CatalogSource.USGS, T, T + timedelta(minutes=2))
        repo.record_success(CatalogSource.USGS, LATER)
    state = repo.get_state(CatalogSource.USGS)
    assert state == RefreshState(CatalogSource.USGS, last_success_at=LATER, last_attempt_at=LATER)


def test_failures_count_up_and_keep_the_last_good_time(repo, db):
    with db.transaction():
        repo.record_success(CatalogSource.EMSC, T)
        repo.record_failure(CatalogSource.EMSC, LATER, LATER + timedelta(minutes=1), "http 503")
        repo.record_failure(CatalogSource.EMSC, LATER, LATER + timedelta(minutes=2), "timeout")
    state = repo.get_state(CatalogSource.EMSC)
    assert state.last_success_at == T
    assert state.consecutive_failures == 2
    assert state.next_attempt_at == LATER + timedelta(minutes=2)
    assert state.last_error == "timeout"
    assert state.claimed_until is None


def test_all_states_come_back_sorted_by_source(repo, db):
    with db.transaction():
        repo.record_success(CatalogSource.USGS, T)
        repo.record_success(CatalogSource.EMSC, T)
    assert [s.source for s in repo.all_states()] == [CatalogSource.EMSC, CatalogSource.USGS]


def test_only_one_claim_at_a_time_and_an_expired_claim_can_be_retaken(repo, db):
    until = T + timedelta(minutes=2)
    with db.transaction():
        assert repo.try_claim(CatalogSource.USGS, T, until) is True
        assert repo.try_claim(CatalogSource.USGS, T + timedelta(seconds=30), until) is False
        assert repo.try_claim(CatalogSource.EMSC, T, until) is True  # another catalog
        assert repo.try_claim(CatalogSource.USGS, until, until + timedelta(minutes=2)) is True


def test_a_released_claim_can_be_taken_again(repo, db):
    with db.transaction():
        assert repo.try_claim(CatalogSource.USGS, T, T + timedelta(minutes=5))
        repo.release_claim(CatalogSource.USGS)
        assert repo.try_claim(CatalogSource.USGS, T, T + timedelta(minutes=5))


def test_two_connections_racing_for_a_claim_get_exactly_one(tmp_path):
    path = tmp_path / "race.db"
    with Database(path) as seed:
        seed.query("SELECT 1")
    wins: list[bool] = []
    barrier = threading.Barrier(6)

    def contender() -> None:
        with Database(path, busy_timeout_ms=5000) as conn_db:
            local = SqliteCatalogRepository(conn_db)
            barrier.wait()
            with conn_db.transaction():
                wins.append(local.try_claim(CatalogSource.USGS, T, T + timedelta(minutes=5)))

    threads = [threading.Thread(target=contender) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(wins) == [False] * 5 + [True]


def test_a_tampered_state_row_is_a_storage_error(repo, db):
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO catalog_refresh_state (source, consecutive_failures) VALUES ('usgs', 0)"
        )
    assert repo.get_state(CatalogSource.USGS).consecutive_failures == 0
    with pytest.raises((sqlite3.Error, StorageError)), db.transaction() as conn:
        conn.execute("UPDATE catalog_refresh_state SET last_success_at = 'never'")
