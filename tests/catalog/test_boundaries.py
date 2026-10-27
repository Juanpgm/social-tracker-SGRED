"""Exact edges of the parser and the cached check, pinned from both sides (found by a mutation
sweep)."""

import json
from datetime import timedelta

import pytest

from social_tracker.catalog.domain import CatalogEvent, CatalogSettings, CatalogSource
from social_tracker.catalog.parsers import (
    MAX_AGE,
    MAX_FEATURES,
    MAX_FUTURE,
    CatalogFormatError,
    parse_catalog,
)
from social_tracker.catalog.service import CachedCatalogCheck
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.db import Database
from social_tracker.shared.clock import FixedClock
from tests.catalog.support import NOW, usgs_payload

USGS, EMSC = CatalogSource.USGS, CatalogSource.EMSC


def parse(*quakes):
    return parse_catalog(USGS, usgs_payload(*quakes), NOW)


# ---- the time tolerances of the parser ----


def test_an_origin_exactly_at_the_future_tolerance_is_kept_and_a_second_beyond_is_not():
    assert len(parse({"time": NOW + MAX_FUTURE}).events) == 1
    with pytest.raises(CatalogFormatError):  # the only feature is dropped: nothing readable
        parse({"time": NOW + MAX_FUTURE + timedelta(seconds=1)})


def test_an_origin_exactly_at_the_age_limit_is_kept_and_a_second_older_is_not():
    assert len(parse({"time": NOW - MAX_AGE}).events) == 1
    with pytest.raises(CatalogFormatError):
        parse({"time": NOW - MAX_AGE - timedelta(seconds=1)})


def test_the_number_of_features_is_limited_exactly():
    quakes = [{"id": f"us{n}"} for n in range(MAX_FEATURES)]
    assert len(parse(*quakes).events) == MAX_FEATURES
    with pytest.raises(CatalogFormatError):
        parse(*quakes, {"id": "one-too-many"})


def test_a_json_constant_anywhere_makes_the_whole_payload_unreadable():
    document = json.loads(usgs_payload({"id": "good"}))
    text = json.dumps(document)[:-2] + ', {"properties": {"mag": NaN}}]}'
    with pytest.raises(CatalogFormatError):
        parse_catalog(USGS, text.encode(), NOW)
    with pytest.raises(CatalogFormatError):
        parse_catalog(
            USGS, json.dumps(document).replace('"mag": 5.1', '"mag": Infinity').encode(), NOW
        )


def test_a_place_of_exactly_the_limit_is_kept_whole_and_a_longer_one_is_cut():
    assert len(parse({"place": "p" * 200}).events[0].place) == 200
    assert len(parse({"place": "p" * 201}).events[0].place) == 200


# ---- the cached check ----


@pytest.fixture
def world(tmp_path):
    db = Database(tmp_path / "b.db")
    repo = SqliteCatalogRepository(db)
    yield repo, CachedCatalogCheck(repo, FixedClock(NOW), CatalogSettings())
    db.close()


def seed(repo, *, usgs_age, emsc_age):
    event = CatalogEvent(USGS, "e1", NOW - timedelta(minutes=3), 3.5, -76.5, 6.0, "x")
    repo.upsert_events([event], NOW)
    repo.record_success(USGS, NOW - usgs_age)
    repo.record_success(EMSC, NOW - emsc_age)


def test_the_coverage_of_the_cache_is_that_of_the_most_recently_refreshed_catalog(world):
    repo, check = world
    seed(repo, usgs_age=timedelta(0), emsc_age=timedelta(hours=1))
    published = NOW - timedelta(minutes=1)
    # 42 hours back: the window opens 48 h and 1 minute before the newest refresh
    assert check.check("Sismo de magnitud 6.0 hace 42 horas", published).status == "no_claim"
    # 41 hours back: the window opens inside what both catalogs cover
    assert check.check("Sismo de magnitud 6.0 hace 41 horas", published).status == "mismatch"
