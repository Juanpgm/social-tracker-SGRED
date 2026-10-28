"""The settings the remediation added to the environment: the caps of an alert and the local time
zone, and the container that carries them."""

from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from social_tracker.alerts.domain import AlertSettings
from social_tracker.bootstrap import (
    ALERT_MAX_EVENTS_ENV,
    ALERT_MAX_MEMBERS_ENV,
    DB_PATH_ENV,
    TIMEZONE_ENV,
    build_container,
    parse_alert_settings,
)
from social_tracker.shared.errors import ValidationError


def test_the_caps_default_to_the_products():
    settings = parse_alert_settings({})
    assert (settings.max_members, settings.max_events) == (500, 200)
    assert parse_alert_settings({ALERT_MAX_MEMBERS_ENV: " ", ALERT_MAX_EVENTS_ENV: ""}) == (
        AlertSettings()
    )


def test_the_caps_can_be_set():
    settings = parse_alert_settings({ALERT_MAX_MEMBERS_ENV: "50", ALERT_MAX_EVENTS_ENV: "30"})
    assert (settings.max_members, settings.max_events) == (50, 30)


@pytest.mark.parametrize("value", ["0", "-1", "5001", "1.5", "abc", "1e2", "٣", "50 items"])
def test_a_bad_member_cap_is_refused_naming_the_variable(value):
    with pytest.raises(ValidationError) as caught:
        parse_alert_settings({ALERT_MAX_MEMBERS_ENV: value})
    assert ALERT_MAX_MEMBERS_ENV in caught.value.message


@pytest.mark.parametrize("value", ["9", "2001", "-1", "x", "20.5"])
def test_a_bad_event_cap_is_refused_naming_the_variable(value):
    with pytest.raises(ValidationError) as caught:
        parse_alert_settings({ALERT_MAX_EVENTS_ENV: value})
    assert ALERT_MAX_EVENTS_ENV in caught.value.message


@pytest.mark.parametrize(("value", "bound"), [("1", True), ("5000", True), ("10", True)])
def test_the_bounds_are_inclusive(value, bound):
    assert parse_alert_settings({ALERT_MAX_MEMBERS_ENV: value}).max_members == int(value)
    assert bound


def test_the_container_carries_the_zone_and_the_default_is_bogota(tmp_path: Path):
    container = build_container({DB_PATH_ENV: str(tmp_path / "z.db")})
    try:
        assert container.timezone == ZoneInfo("America/Bogota")
    finally:
        container.close()


def test_the_zone_comes_from_the_environment(tmp_path: Path):
    container = build_container(
        {DB_PATH_ENV: str(tmp_path / "z.db"), TIMEZONE_ENV: "Europe/Madrid"}
    )
    try:
        assert container.timezone == ZoneInfo("Europe/Madrid")
    finally:
        container.close()


def test_a_bad_zone_stops_the_start_before_the_database_is_opened(tmp_path: Path):
    with pytest.raises(ValidationError) as caught:
        build_container({DB_PATH_ENV: str(tmp_path / "x.db"), TIMEZONE_ENV: "Mars/Phobos"})
    assert TIMEZONE_ENV in caught.value.message
    assert not (tmp_path / "x.db").exists()
