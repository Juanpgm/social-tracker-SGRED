"""The cached check after the adversarial review: breaking news (R1), the place of the claim (R3),
extreme publication times (S1) and the settings that carry them.
"""

from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.catalog.domain import CatalogEvent, CatalogSettings, CatalogSource
from social_tracker.catalog.service import CachedCatalogCheck
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.db import Database
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import ValidationError
from tests.catalog.support import NOW

USGS, EMSC = CatalogSource.USGS, CatalogSource.EMSC
BREAKING = "Sismo de magnitud 7.5 en Cali hace un momento"


class World:
    def __init__(self, tmp_path, settings=None):
        self.db = Database(tmp_path / "c.db")
        self.repo = SqliteCatalogRepository(self.db)
        self.clock = FixedClock(NOW)
        self.check = CachedCatalogCheck(self.repo, self.clock, settings or CatalogSettings())

    def add(self, id="e1", source=USGS, at=NOW, magnitude=3.1, lat=3.5, lon=-76.5):
        event = CatalogEvent(source, id, at, lat, lon, magnitude, "Cali")
        self.repo.upsert_events([event], NOW)

    def fresh(self, usgs=NOW, emsc=NOW):
        self.repo.record_success(USGS, usgs)
        self.repo.record_success(EMSC, emsc)

    def status(self, text, published):
        return self.check.check(text, published).status


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.db.close()


# ---- R1 ----


def test_breaking_news_the_cache_has_not_ingested_yet_is_unavailable(world):
    published = NOW - timedelta(minutes=2)
    world.fresh()
    world.add(at=NOW - timedelta(hours=3))  # only old, small events: the real quake is not in yet
    assert world.status(BREAKING, published) == "unavailable"


def test_breaking_news_next_to_a_small_recent_event_is_unavailable(world):
    published = NOW - timedelta(minutes=2)
    world.fresh()
    world.add(at=NOW - timedelta(minutes=10), magnitude=2.9)
    assert world.status(BREAKING, published) == "unavailable"


def test_breaking_news_the_cache_already_has_is_consistent(world):
    published = NOW - timedelta(minutes=2)
    world.fresh()
    world.add(at=NOW - timedelta(minutes=3), magnitude=7.4)
    assert world.status(BREAKING, published) == "consistent"


def test_an_old_post_about_a_quake_that_never_happened_is_a_mismatch(world):
    published = NOW - timedelta(hours=2)
    world.fresh()
    world.add(at=NOW - timedelta(hours=1, minutes=50), magnitude=2.9)
    assert world.status(BREAKING.replace("hace un momento", "hoy"), published) == "mismatch"


def test_the_grace_boundary_in_the_cached_check(world):
    grace = timedelta(seconds=CatalogSettings().ingest_grace_s)
    world.fresh()
    world.add(at=NOW - timedelta(hours=5), magnitude=2.9)
    edge = NOW - grace
    assert world.status(BREAKING, edge) == "mismatch"
    assert world.status(BREAKING, edge + timedelta(seconds=1)) == "unavailable"


def test_the_oldest_refresh_of_the_two_catalogs_bounds_what_is_covered(world):
    world.fresh(usgs=NOW, emsc=NOW - timedelta(hours=1))
    world.add(at=NOW - timedelta(hours=5), magnitude=2.9)
    published = NOW - timedelta(minutes=45)  # newer than EMSC's last refresh minus the grace
    assert world.status(BREAKING, published) == "unavailable"
    assert world.status(BREAKING, NOW - timedelta(hours=2)) == "mismatch"


def test_the_grace_is_configurable_and_validated(tmp_path):
    w = World(tmp_path, CatalogSettings(ingest_grace_s=0))
    try:
        w.fresh()
        w.add(at=NOW - timedelta(hours=5), magnitude=2.9)
        assert w.status(BREAKING, NOW - timedelta(minutes=1)) == "mismatch"
    finally:
        w.db.close()
    assert CatalogSettings(ingest_grace_s=86_400).ingest_grace_s == 86_400
    for bad in (-1, 86_401, True, 1.5, "60", None):
        with pytest.raises(ValidationError):
            CatalogSettings(ingest_grace_s=bad)


# ---- R3 ----


def test_a_claim_about_japan_is_not_compared_with_a_local_micro_event(world):
    world.fresh()
    world.add(at=NOW - timedelta(hours=3), magnitude=2.8)
    published = NOW - timedelta(hours=2)
    assert world.status("Sismo de magnitud 6.5 en Japón", published) == "unavailable"


def test_a_claim_about_a_big_event_elsewhere_is_compared_with_the_global_events(world):
    world.fresh()
    world.add(id="jp", at=NOW - timedelta(hours=2, minutes=5), magnitude=6.4, lat=35.7, lon=139.7)
    published = NOW - timedelta(hours=2)
    assert world.status("Sismo de magnitud 6.5 en Japón", published) == "consistent"


def test_the_regional_floor_follows_the_settings(tmp_path):
    w = World(tmp_path, CatalogSettings(min_magnitude=4.0))
    try:
        w.fresh()
        w.add(at=NOW - timedelta(hours=5), magnitude=7.0)
        published = NOW - timedelta(hours=2)
        assert w.status("Sismo de magnitud 3.5 en Cali", published) == "unavailable"
        assert w.status("Sismo de magnitud 4.0 en Cali", published) == "mismatch"
    finally:
        w.db.close()


# ---- S1 ----


@pytest.mark.parametrize(
    "published",
    [
        datetime(9999, 12, 31, 23, 59, tzinfo=UTC),
        datetime(1, 1, 1, 0, 0, tzinfo=UTC),
        datetime(1, 1, 1, 0, 30, tzinfo=UTC),
        datetime.max.replace(tzinfo=UTC),
        datetime.min.replace(tzinfo=UTC),
    ],
)
def test_an_extreme_publication_time_never_raises(world, published):
    world.fresh()
    world.add()
    assert world.status("sismo de magnitud 5.5 en Cali hoy", published) in {
        "no_claim",
        "unavailable",
    }
    assert world.status("sismo M5.5 a las 3 pm", published) in {"no_claim", "unavailable"}
