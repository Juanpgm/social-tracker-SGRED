"""The requests the refresher sends: only the two catalogs, over https, one regional and one
global per catalog, built from the settings and from nothing the network can influence."""

from urllib.parse import parse_qs, urlsplit

import pytest

from social_tracker.catalog.domain import CATALOG_HOSTS, CatalogSettings, CatalogSource
from social_tracker.catalog.queries import catalog_requests
from tests.catalog.support import NOW


def params(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


@pytest.mark.parametrize("source", list(CatalogSource))
def test_each_catalog_gets_a_regional_and_a_global_request_on_its_own_host(source):
    urls = catalog_requests(source, CatalogSettings(), NOW)
    assert len(urls) == 2
    hosts = {urlsplit(u).hostname for u in urls}
    assert len(hosts) == 1
    assert hosts <= CATALOG_HOSTS
    assert all(urlsplit(u).scheme == "https" for u in urls)
    assert all(urlsplit(u).path == "/fdsnws/event/1/query" for u in urls)


def test_the_usgs_requests_carry_the_region_the_floors_and_the_window():
    regional, world = (
        params(u) for u in catalog_requests(CatalogSource.USGS, CatalogSettings(), NOW)
    )
    assert regional["format"] == "geojson"
    assert (regional["latitude"], regional["longitude"]) == ("3.4516", "-76.532")
    assert regional["maxradiuskm"] == "300"
    assert regional["minmagnitude"] == "2.5"
    assert regional["starttime"] == "2026-10-25T18:00:00"
    assert regional["endtime"] == "2026-10-27T18:00:00"
    assert regional["limit"] == "500"
    assert world["minmagnitude"] == "5.5"
    assert "latitude" not in world
    assert "maxradiuskm" not in world


def test_the_emsc_requests_use_degrees_for_the_radius():
    regional, world = (
        params(u) for u in catalog_requests(CatalogSource.EMSC, CatalogSettings(), NOW)
    )
    assert regional["format"] == "json"
    assert (regional["lat"], regional["lon"]) == ("3.4516", "-76.532")
    assert float(regional["maxradius"]) == pytest.approx(300 / 111.195, abs=0.001)
    assert regional["minmag"] == "2.5"
    assert regional["start"] == "2026-10-25T18:00:00"
    assert world["minmag"] == "5.5"
    assert "lat" not in world


def test_the_region_and_floors_follow_the_settings():
    settings = CatalogSettings(
        center_latitude=-12.5, center_longitude=179.9, radius_km=150.0, min_magnitude=3.0,
        global_min_magnitude=6.5, lookback_hours=24, max_events=50,
    )  # fmt: skip
    regional, world = (params(u) for u in catalog_requests(CatalogSource.USGS, settings, NOW))
    assert (regional["latitude"], regional["longitude"]) == ("-12.5", "179.9")
    assert regional["maxradiuskm"] == "150"
    assert regional["minmagnitude"] == "3"
    assert world["minmagnitude"] == "6.5"
    assert regional["starttime"] == "2026-10-26T18:00:00"
    assert regional["limit"] == "50"


def test_nothing_names_the_colombian_survey_or_any_other_host():
    for source in CatalogSource:
        for url in catalog_requests(source, CatalogSettings(), NOW):
            assert "sgc" not in url.lower()
            assert "gov.co" not in url


def test_the_requests_are_deterministic():
    first = catalog_requests(CatalogSource.USGS, CatalogSettings(), NOW)
    assert catalog_requests(CatalogSource.USGS, CatalogSettings(), NOW) == first


def test_a_naive_clock_is_refused():
    from datetime import datetime

    with pytest.raises(Exception, match="timezone"):
        catalog_requests(CatalogSource.USGS, CatalogSettings(), datetime(2026, 1, 1))
