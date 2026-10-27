"""Catalog settings from the environment and the wiring of the cache, the check and the worker."""

from datetime import timedelta
from pathlib import Path

import pytest

from social_tracker.bootstrap import (
    CATALOG_CENTER_ENV,
    CATALOG_GLOBAL_MIN_MAGNITUDE_ENV,
    CATALOG_INGEST_GRACE_ENV,
    CATALOG_INTERVAL_ENV,
    CATALOG_MAX_AGE_ENV,
    CATALOG_MIN_MAGNITUDE_ENV,
    CATALOG_RADIUS_ENV,
    DB_PATH_ENV,
    build_catalog,
    build_container,
    build_worker,
    parse_catalog_settings,
)
from social_tracker.catalog.domain import CATALOG_HOSTS, CatalogSettings
from social_tracker.catalog.service import CachedCatalogCheck, CatalogCache
from social_tracker.collector.http_fetcher import HttpFetcher
from social_tracker.shared.errors import ValidationError
from tests.catalog.support import ScriptedFetcher


def test_no_variable_means_the_defaults():
    assert parse_catalog_settings({}) == CatalogSettings()
    blank = {name: "  " for name in (CATALOG_CENTER_ENV, CATALOG_RADIUS_ENV, CATALOG_INTERVAL_ENV)}
    assert parse_catalog_settings(blank) == CatalogSettings()


def test_every_variable_is_read():
    env = {
        CATALOG_CENTER_ENV: "4.711, -74.0721",
        CATALOG_RADIUS_ENV: "150.5",
        CATALOG_MIN_MAGNITUDE_ENV: "3",
        CATALOG_GLOBAL_MIN_MAGNITUDE_ENV: "6.0",
        CATALOG_INTERVAL_ENV: "900",
        CATALOG_MAX_AGE_ENV: "3600",
        CATALOG_INGEST_GRACE_ENV: "900",
    }
    settings = parse_catalog_settings(env)
    assert (settings.center_latitude, settings.center_longitude) == (4.711, -74.0721)
    assert settings.radius_km == 150.5
    assert (settings.min_magnitude, settings.global_min_magnitude) == (3.0, 6.0)
    assert (settings.interval_s, settings.max_age_s) == (900, 3600)
    assert settings.ingest_grace_s == 900


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (CATALOG_CENTER_ENV, "3.4516"),
        (CATALOG_CENTER_ENV, "3.4516,-76.5,9"),
        (CATALOG_CENTER_ENV, "abc,def"),
        (CATALOG_CENTER_ENV, "91,0"),
        (CATALOG_CENTER_ENV, "0,181"),
        (CATALOG_CENTER_ENV, "nan,0"),
        (CATALOG_CENTER_ENV, "1e2,0"),
        (CATALOG_RADIUS_ENV, "0"),
        (CATALOG_RADIUS_ENV, "-5"),
        (CATALOG_RADIUS_ENV, "inf"),
        (CATALOG_RADIUS_ENV, "300 km"),
        (CATALOG_RADIUS_ENV, "25000"),
        (CATALOG_MIN_MAGNITUDE_ENV, "11"),
        (CATALOG_MIN_MAGNITUDE_ENV, "x"),
        (CATALOG_GLOBAL_MIN_MAGNITUDE_ENV, "99"),
        (CATALOG_INTERVAL_ENV, "10"),
        (CATALOG_INTERVAL_ENV, "6.5"),
        (CATALOG_INTERVAL_ENV, "-600"),
        (CATALOG_INTERVAL_ENV, "999999999999"),
        (CATALOG_MAX_AGE_ENV, "30"),
        (CATALOG_INGEST_GRACE_ENV, "-1"),
        (CATALOG_INGEST_GRACE_ENV, "86401"),
        (CATALOG_INGEST_GRACE_ENV, "1.5"),
        (CATALOG_INGEST_GRACE_ENV, "abc"),
    ],
)
def test_an_invalid_value_names_the_variable_and_does_not_echo_the_value(name, value):
    with pytest.raises(ValidationError) as caught:
        parse_catalog_settings({name: value})
    assert name in caught.value.message
    if len(value) > 3:
        assert value not in caught.value.message


def test_the_limits_of_the_valid_ranges_are_accepted():
    env = {CATALOG_CENTER_ENV: "-90,-180", CATALOG_INTERVAL_ENV: "60", CATALOG_MAX_AGE_ENV: "60"}
    settings = parse_catalog_settings(env)
    assert (settings.center_latitude, settings.center_longitude) == (-90.0, -180.0)
    assert settings.interval_s == 60


def test_the_ingest_grace_accepts_zero_and_a_day_and_nothing_beyond():
    assert parse_catalog_settings({CATALOG_INGEST_GRACE_ENV: "0"}).ingest_grace_s == 0
    assert parse_catalog_settings({CATALOG_INGEST_GRACE_ENV: "86400"}).ingest_grace_s == 86_400


def test_a_max_age_under_the_interval_is_refused():
    with pytest.raises(ValidationError):
        parse_catalog_settings({CATALOG_INTERVAL_ENV: "900", CATALOG_MAX_AGE_ENV: "600"})


def test_the_container_has_the_cache_service_and_the_cached_check(tmp_path: Path):
    container = build_container({DB_PATH_ENV: str(tmp_path / "a.db")})
    try:
        assert isinstance(container.catalog, CatalogCache)
        assert isinstance(container.classifier._catalog, CachedCatalogCheck)
        assert container.catalog.status().available is False
        assert container.catalog.purge_expired() == 0
    finally:
        container.close()


def test_a_bad_catalog_setting_stops_the_start_before_the_database_is_opened(tmp_path: Path):
    path = tmp_path / "never.db"
    with pytest.raises(ValidationError):
        build_container({DB_PATH_ENV: str(path), CATALOG_RADIUS_ENV: "0"})
    assert not path.exists()


def test_the_catalog_container_builds_a_refresher_without_the_worker_lock(tmp_path: Path):
    env = {DB_PATH_ENV: str(tmp_path / "a.db")}
    fake = ScriptedFetcher()
    built = build_catalog(env, fetcher=fake)
    second = build_catalog(env, fetcher=fake)  # two at once: there is no lock to refuse it
    try:
        assert built.fetcher is fake
        assert built.refresher is not second.refresher
    finally:
        second.close()
        built.close()


def test_the_default_catalog_fetcher_is_confined_to_the_two_hosts_over_https(tmp_path: Path):
    built = build_catalog({DB_PATH_ENV: str(tmp_path / "a.db")})
    try:
        fetcher = built.fetcher
        assert isinstance(fetcher, HttpFetcher)
        assert fetcher._allowed_hosts == CATALOG_HOSTS
        assert fetcher._https_only is True
        assert fetcher._allowed_ports == frozenset({443})
        assert fetcher._private_hosts == frozenset()
    finally:
        built.close()


def test_the_private_host_allowance_of_the_feeds_never_reaches_the_catalog_fetcher(tmp_path: Path):
    env = {DB_PATH_ENV: str(tmp_path / "a.db"), "SOCIAL_TRACKER_ALLOW_PRIVATE_HOSTS": "localhost"}
    built = build_catalog(env)
    try:
        assert built.fetcher._private_hosts == frozenset()
    finally:
        built.close()


def test_the_worker_refreshes_the_catalogs_with_its_own_confined_fetcher(tmp_path: Path):
    built = build_worker({DB_PATH_ENV: str(tmp_path / "w.db")})
    try:
        assert built.worker._catalog is not None
        assert isinstance(built.fetcher, HttpFetcher)
        assert built.fetcher._allowed_hosts is None  # the feed fetcher is not the catalog one
    finally:
        built.close()


def test_a_replaced_feed_fetcher_leaves_the_catalogs_unfetched(tmp_path: Path):
    """Tests and tooling that replace the feed fetcher must never reach the real catalogs."""
    built = build_worker({DB_PATH_ENV: str(tmp_path / "w.db")}, fetcher=ScriptedFetcher())
    try:
        assert built.worker._catalog is None
    finally:
        built.close()


def test_the_catalog_fetch_limits_are_bounded():
    from social_tracker.bootstrap import CATALOG_FETCH_LIMITS

    assert CATALOG_FETCH_LIMITS.max_bytes <= 8 * 1024 * 1024
    assert CATALOG_FETCH_LIMITS.deadline_s <= 30
    assert timedelta(seconds=CATALOG_FETCH_LIMITS.deadline_s) < timedelta(seconds=120)


def test_the_hourly_housekeeping_purges_the_catalog_even_when_earlier_steps_fail():
    from social_tracker.bootstrap import _purge_everything

    calls = []

    class Fails:
        def purge_expired(self):
            raise RuntimeError("items")

        def prune_history(self):
            raise RuntimeError("history")

    class CatalogSpy:
        def purge_expired(self):
            calls.append("catalog purged")

    fake = type("C", (), {"items": Fails(), "classifier": Fails(), "catalog": CatalogSpy()})()
    with pytest.raises(RuntimeError):
        _purge_everything(fake)()
    assert calls == ["catalog purged"]
