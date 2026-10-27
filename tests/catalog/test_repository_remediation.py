"""The SQLite catalog cache after the review: deleting events a catalog no longer lists and the
"answer was cut short" mark of a refresh."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.catalog.domain import CatalogEvent, CatalogSource, RefreshState
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.db import Database, load_migrations
from social_tracker.shared.errors import StorageError, ValidationError

T = datetime(2026, 10, 27, 17, 0, tzinfo=UTC)
USGS, EMSC = CatalogSource.USGS, CatalogSource.EMSC


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "c.db") as database:
        yield database


@pytest.fixture
def repo(db):
    return SqliteCatalogRepository(db)


def event(id, source=USGS, minutes=0):
    return CatalogEvent(source, id, T + timedelta(minutes=minutes), 3.4, -76.5, 5.0, "Cali")


def ids(repo, source=None):
    found = repo.events_between(T - timedelta(days=2), T + timedelta(days=2))
    return sorted(e.external_id for e in found if source in (None, e.source))


def test_the_migration_adds_the_truncated_mark(db):
    assert [m.version for m in load_migrations()][-1] >= 7
    columns = {row[1] for row in db.query("PRAGMA table_info(catalog_refresh_state)")}
    assert "last_truncated" in columns


# ---- delete_events ----


def test_delete_events_removes_only_the_named_events_of_that_source(repo, db):
    with db.transaction():
        repo.upsert_events([event("a"), event("b"), event("a", EMSC)], T)
        removed = repo.delete_events(USGS, ["a"])
    assert removed == 1
    assert ids(repo, USGS) == ["b"]
    assert ids(repo, EMSC) == ["a"]


@pytest.mark.parametrize("names", [[], ["missing"], ["missing", "missing"]])
def test_deleting_nothing_or_unknown_events_removes_nothing(repo, db, names):
    with db.transaction():
        repo.upsert_events([event("a")], T)
        assert repo.delete_events(USGS, names) == 0
    assert ids(repo) == ["a"]


def test_a_large_delete_is_not_limited_by_the_sql_variable_cap(repo, db):
    many = [event(f"id{n}", minutes=n % 300) for n in range(2000)]
    with db.transaction():
        repo.upsert_events(many, T)
        assert repo.delete_events(USGS, [e.external_id for e in many]) == 2000
    assert ids(repo) == []


def test_a_hostile_id_is_bound_as_data(repo, db):
    with db.transaction():
        repo.upsert_events([event("a")], T)
        assert repo.delete_events(USGS, ["a' OR '1'='1", "%", "_"]) == 0
    assert ids(repo) == ["a"]


def test_delete_events_refuses_an_unknown_source(repo):
    with pytest.raises(ValueError, match="not a valid"):
        repo.delete_events("sgc", ["a"])


def test_delete_events_rolls_back_with_its_transaction(repo, db):
    with db.transaction():
        repo.upsert_events([event("a")], T)

    def delete_then_fail():
        with db.transaction():
            repo.delete_events(USGS, ["a"])
            raise RuntimeError

    with pytest.raises(RuntimeError):
        delete_then_fail()
    assert ids(repo) == ["a"]


# ---- the truncated mark ----


def test_a_success_records_whether_the_answer_was_cut_short(repo, db):
    with db.transaction():
        repo.record_success(USGS, T, truncated=True)
    assert repo.get_state(USGS).truncated is True
    with db.transaction():
        repo.record_success(USGS, T + timedelta(minutes=10))
    assert repo.get_state(USGS).truncated is False


def test_a_failure_keeps_the_last_truncated_mark(repo, db):
    with db.transaction():
        repo.record_success(USGS, T, truncated=True)
        repo.record_failure(USGS, T, T + timedelta(minutes=1), "x")
    state = repo.get_state(USGS)
    assert (state.truncated, state.consecutive_failures) == (True, 1)


def test_a_claim_keeps_the_truncated_mark(repo, db):
    with db.transaction():
        repo.record_success(USGS, T, truncated=True)
    assert repo.try_claim(USGS, T, T + timedelta(minutes=2))
    assert repo.get_state(USGS).truncated is True


def test_the_state_defaults_to_not_truncated():
    assert RefreshState(USGS).truncated is False
    assert RefreshState(USGS, truncated=True).truncated is True
    for bad in (1, "yes", None):
        with pytest.raises(ValidationError):
            RefreshState(USGS, truncated=bad)


def test_a_truncated_column_outside_zero_and_one_is_refused_by_the_database(repo, db):
    with db.transaction():
        repo.record_success(USGS, T)
    with pytest.raises((sqlite3.Error, StorageError)), db.transaction() as conn:
        conn.execute("UPDATE catalog_refresh_state SET last_truncated = 7")
