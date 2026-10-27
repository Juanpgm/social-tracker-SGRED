"""Vocabulary of the catalog cache: events, settings, refresh state."""

from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.catalog.domain import (
    CATALOG_HOSTS,
    CatalogEvent,
    CatalogSettings,
    CatalogSource,
    RefreshState,
)
from social_tracker.classification.catalog_match import MatchSettings
from social_tracker.shared.errors import ValidationError

T = datetime(2026, 10, 27, 17, 0, tzinfo=UTC)


def event(**changes):
    base = {
        "source": CatalogSource.USGS,
        "external_id": "us7000abcd",
        "origin_time": T,
        "latitude": 3.4,
        "longitude": -76.5,
        "magnitude": 5.1,
        "place": "10 km N of Cali, Colombia",
    }
    return CatalogEvent(**{**base, **changes})


def test_the_two_catalogs_and_their_only_hosts():
    assert {s.value for s in CatalogSource} == {"usgs", "emsc"}
    assert frozenset({"earthquake.usgs.gov", "www.seismicportal.eu"}) == CATALOG_HOSTS


def test_an_event_is_converted_to_the_point_the_matcher_reads():
    point = event().to_point()
    assert (point.source, point.magnitude, point.origin_time) == ("usgs", 5.1, T)


@pytest.mark.parametrize(
    "changes",
    [
        {"external_id": ""},
        {"external_id": "x" * 65},
        {"external_id": "has space"},
        {"external_id": "semi;colon"},
        {"external_id": "quote'"},
        {"external_id": 17},
        {"source": "sgc"},
        {"magnitude": float("nan")},
        {"magnitude": 11.0},
        {"latitude": 91.0},
        {"longitude": -181.0},
        {"origin_time": datetime(2026, 10, 27, 17, 0)},
        {"place": "x" * 201},
        {"place": "bell\x07"},
    ],
)
def test_an_invalid_event_is_refused(changes):
    with pytest.raises(ValidationError):
        event(**changes)


def test_the_source_accepts_its_text_value_and_a_place_may_be_empty():
    assert event(source="emsc").source is CatalogSource.EMSC
    assert event(place="").place == ""


def test_the_default_settings_are_the_product_ones_centered_on_cali():
    s = CatalogSettings()
    assert (s.center_latitude, s.center_longitude, s.radius_km) == (3.4516, -76.5320, 300.0)
    assert (s.min_magnitude, s.global_min_magnitude) == (2.5, 5.5)
    assert s.interval_s == 600
    assert s.time_window == timedelta(hours=6)
    assert s.max_age_s >= s.interval_s
    assert isinstance(s.match_settings(), MatchSettings)
    assert s.match_settings().radius_km == 300.0


@pytest.mark.parametrize(
    "changes",
    [
        {"center_latitude": 91.0},
        {"center_longitude": 181.0},
        {"radius_km": 0.0},
        {"radius_km": 25000.0},
        {"radius_km": float("nan")},
        {"min_magnitude": -3.0},
        {"min_magnitude": 11.0},
        {"global_min_magnitude": 11.0},
        {"interval_s": 0},
        {"interval_s": 59},
        {"interval_s": True},
        {"max_age_s": 100},  # below the interval
        {"lookback_hours": 11},  # under twice the window
        {"lookback_hours": 24 * 31},
        {"retention_days": 0},
        {"max_events": 0},
        {"max_events": 20001},
        {"utc_offset_hours": 15},
    ],
)
def test_invalid_settings_are_refused(changes):
    with pytest.raises(ValidationError):
        CatalogSettings(**changes)


def test_settings_at_their_limits_are_accepted():
    assert CatalogSettings(interval_s=60, max_age_s=60).interval_s == 60
    assert CatalogSettings(center_latitude=-90.0, center_longitude=180.0).center_latitude == -90.0


def test_a_refresh_state_validates_its_counters_and_times():
    state = RefreshState(CatalogSource.EMSC, last_success_at=T, consecutive_failures=2)
    assert state.source is CatalogSource.EMSC
    with pytest.raises(ValidationError):
        RefreshState(CatalogSource.EMSC, consecutive_failures=-1)
    with pytest.raises(ValidationError):
        RefreshState(CatalogSource.EMSC, last_success_at=datetime(2026, 1, 1))
    with pytest.raises(ValidationError):
        RefreshState(CatalogSource.EMSC, last_error="x" * 201)
