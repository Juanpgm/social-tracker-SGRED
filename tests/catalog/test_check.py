"""The cached cross-check: reads only the local cache, and says `unavailable` rather than guess."""

from datetime import timedelta

import pytest

from social_tracker.catalog.domain import CatalogEvent, CatalogSettings, CatalogSource
from social_tracker.catalog.service import CachedCatalogCheck, CatalogCache
from social_tracker.classification.ports import CatalogCheck
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.db import Database
from social_tracker.shared.clock import FixedClock
from tests.catalog.support import NOW

USGS, EMSC = CatalogSource.USGS, CatalogSource.EMSC
PUB = NOW - timedelta(minutes=1)
CLAIM = "Sismo de magnitud 6.2 en Cali hace un momento"


class World:
    def __init__(self, tmp_path, settings=None):
        self.db = Database(tmp_path / "c.db")
        self.repo = SqliteCatalogRepository(self.db)
        self.clock = FixedClock(NOW)
        self.settings = settings or CatalogSettings()
        self.check = CachedCatalogCheck(self.repo, self.clock, self.settings)

    def add(self, id="e1", source=USGS, minutes=-3, magnitude=6.2, lat=3.5, lon=-76.5):
        event = CatalogEvent(
            source, id, PUB + timedelta(minutes=minutes), lat, lon, magnitude, "Cali"
        )
        self.repo.upsert_events([event], NOW)

    def fresh(self, *sources, at=None):
        for source in sources or (USGS, EMSC):
            self.repo.record_success(source, at or NOW)

    def status(self, text=CLAIM, published=PUB) -> str:
        return self.check.check(text, published).status


@pytest.fixture
def world(tmp_path):
    w = World(tmp_path)
    yield w
    w.db.close()


def test_it_satisfies_the_classifier_port(world):
    assert isinstance(world.check, CatalogCheck)


# ---- unavailable: never a guess ----


def test_a_cache_never_refreshed_is_unavailable(world):
    assert world.status() == "unavailable"
    world.add()  # events without any good refresh on record are not trusted either
    assert world.status() == "unavailable"


def test_a_fresh_refresh_with_no_event_at_all_is_unavailable(world):
    world.fresh()
    assert world.status() == "unavailable"


def test_a_purged_cache_is_unavailable(world):
    world.fresh()
    world.add()
    assert world.status() == "consistent"
    world.repo.purge_before(NOW + timedelta(days=1))
    assert world.status() == "unavailable"


@pytest.mark.parametrize(
    ("age_s", "expected"),
    [(0, "consistent"), (7199, "consistent"), (7200, "consistent"), (7201, "unavailable")],
)
def test_the_cache_goes_stale_just_after_max_age(world, age_s, expected):
    world.fresh(at=NOW - timedelta(seconds=age_s))
    world.add(minutes=-3)
    assert world.status() == expected


def test_a_stale_cache_is_unavailable_even_for_a_text_with_no_claim(world):
    world.fresh(at=NOW - timedelta(days=1))
    world.add()
    assert world.status("se sintió un temblor") == "unavailable"


def test_without_a_claim_there_is_nothing_to_check(world):
    world.fresh()
    world.add()
    assert world.status("se sintió un temblor fuerte en Cali") == "no_claim"


def test_when_one_catalog_is_stale_an_absent_event_proves_nothing(world):
    world.fresh(USGS)
    world.fresh(EMSC, at=NOW - timedelta(days=1))
    world.add(id="far", minutes=-400)
    assert world.status() == "unavailable"


def test_when_one_catalog_is_stale_a_found_event_still_decides(world):
    world.fresh(USGS)
    world.fresh(EMSC, at=NOW - timedelta(days=1))
    world.add(magnitude=6.0)
    assert world.status() == "consistent"
    world.add(id="e2", magnitude=3.0, minutes=-2)
    assert world.status() == "mismatch"


# ---- consistent and mismatch ----


def test_a_claim_matching_the_catalog_is_consistent(world):
    world.fresh()
    world.add(magnitude=6.0)
    assert world.status() == "consistent"


@pytest.mark.parametrize(
    ("event", "expected"), [(5.21, "consistent"), (5.2, "mismatch"), (7.3, "mismatch")]
)
def test_a_difference_of_a_whole_degree_is_a_mismatch(world, event, expected):
    world.fresh()
    world.add(magnitude=event)
    assert world.status() == expected


def test_no_event_in_the_window_with_both_catalogs_fresh_is_a_mismatch(world):
    world.fresh()
    world.add(minutes=-60 * 7)  # seven hours before: outside the six hour window
    assert world.status() == "mismatch"


def test_an_event_far_from_the_region_below_the_global_floor_does_not_count(world):
    world.fresh()
    world.add(magnitude=5.4, lat=35.7, lon=139.7)
    world.add(id="near", magnitude=6.0, lat=-40.0, lon=-70.0, minutes=-1)
    assert world.status("Sismo de magnitud 5.4 hace un momento") == "consistent"  # the 6.0 one


def test_an_earthquake_felt_elsewhere_counts_from_the_global_floor(world):
    world.fresh()
    world.add(magnitude=7.4, lat=-33.0, lon=-71.6)
    assert world.status("Terremoto de magnitud 7.4 hace un momento") == "consistent"


def test_the_same_quake_in_two_catalogs_counts_as_one_and_the_closest_magnitude_wins(world):
    world.fresh()
    world.add(id="u", source=USGS, magnitude=6.1)
    world.add(id="e", source=EMSC, magnitude=7.3, minutes=-3)
    assert world.status("Sismo de magnitud 7.4 hace un momento") == "consistent"


def test_a_time_of_day_claim_is_compared_at_that_time(world):
    world.fresh()
    world.add(minutes=-120, magnitude=6.0)  # two hours before the publication
    text = "Sismo de magnitud 6.0 a las 11:00 am en Cali"  # 16:00 UTC, publication 17:59 UTC
    assert world.status(text) == "consistent"
    assert (
        world.status("Sismo de magnitud 6.0 hace 1 hora", published=PUB + timedelta(hours=9))
        == "no_claim"
    )


def test_a_claim_about_an_old_date_is_not_checked(world):
    world.fresh()
    world.add()
    assert world.status("Sismo de magnitud 6.2 del 12 de agosto de 2019") == "no_claim"


def test_a_claim_older_than_the_cache_reaches_is_not_checked(world):
    world.fresh()
    world.add()
    text = "Sismo de magnitud 6.2 hace 3 días"
    assert world.status(text) == "no_claim"


def test_a_publication_date_in_the_future_is_not_checked(world):
    world.fresh()
    world.add()
    assert world.status(CLAIM, published=NOW + timedelta(days=3)) == "no_claim"


# ---- it only reads ----


def test_a_check_writes_nothing(world):
    world.fresh()
    world.add()
    before = [tuple(r) for r in world.db.query("SELECT * FROM catalog_events")]
    states = [tuple(r) for r in world.db.query("SELECT * FROM catalog_refresh_state")]
    for text in (CLAIM, "nada", "Sismo de magnitud 3.0 hace un momento"):
        world.check.check(text, PUB)
    assert [tuple(r) for r in world.db.query("SELECT * FROM catalog_events")] == before
    assert [tuple(r) for r in world.db.query("SELECT * FROM catalog_refresh_state")] == states


def test_the_check_has_no_way_to_reach_the_network():
    import inspect

    from social_tracker.catalog import service

    params = inspect.signature(CachedCatalogCheck.__init__).parameters
    assert set(params) == {"self", "repo", "clock", "settings"}
    assert "fetch" not in inspect.getsource(CachedCatalogCheck).lower().replace("fetched", "")
    assert service.CachedCatalogCheck is CachedCatalogCheck


def test_the_same_inputs_give_the_same_answer(world):
    world.fresh()
    world.add()
    assert {world.status() for _ in range(5)} == {"consistent"}


@pytest.mark.parametrize(
    "text",
    [
        "x" * 500_000,
        "magnitud 6.2 " * 10_000,
        "mаgnitud 6.2",  # a Cyrillic letter
        "magnitud ٦.٢",  # Arabic-Indic digits
        "magnitud 77",
        "",
        "\x00" * 100,
    ],
    ids=["huge", "repeated", "homoglyph", "arabic-digits", "absurd", "empty", "nul"],
)
def test_hostile_text_never_raises_and_never_becomes_a_mismatch_by_itself(world, text):
    world.fresh()
    world.add()
    assert world.status(text) in {"consistent", "no_claim"}


# ---- the cache service ----


def test_the_status_counts_events_and_flags_what_is_stale(world):
    world.fresh(USGS)
    world.add()
    world.add(id="em", source=EMSC)
    status = CatalogCache(world.repo, world.clock, world.settings).status()
    by_source = {s.source: s for s in status.sources}
    assert (by_source[USGS].events, by_source[USGS].stale) == (1, False)
    assert (by_source[EMSC].events, by_source[EMSC].stale) == (1, True)
    assert status.available is True
    assert status.complete is False
    assert status.newest_event_at == PUB - timedelta(minutes=3)


def test_the_status_of_an_empty_cache(world):
    status = CatalogCache(world.repo, world.clock, world.settings).status()
    assert status.available is False
    assert status.newest_event_at is None
    assert all(s.stale and s.last_success_at is None for s in status.sources)


def test_purge_forgets_only_events_older_than_the_retention(world):
    cache = CatalogCache(world.repo, world.clock, world.settings)
    world.add(id="old", minutes=-60 * 24 * 15)
    world.add(id="edge", minutes=-60 * 24 * 14 + 5)
    world.add(id="new")
    assert cache.purge_expired() == 1
    kept = world.repo.events_between(NOW - timedelta(days=60), NOW)
    assert [e.external_id for e in kept] == ["edge", "new"]
    assert cache.purge_expired() == 0
