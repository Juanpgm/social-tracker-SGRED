"""The local time zone of the operator: where the dates they type are read and the times they see
are drawn. Storage and the policy stay in UTC."""

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from social_tracker.shared.errors import ValidationError
from social_tracker.shared.timezone import (
    DEFAULT_TIMEZONE,
    format_compact,
    format_iso,
    format_local,
    parse_timezone,
)

NOON_UTC = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def test_the_default_is_bogota():
    assert DEFAULT_TIMEZONE == "America/Bogota"
    assert parse_timezone("") == ZoneInfo("America/Bogota")
    assert parse_timezone("   ") == ZoneInfo("America/Bogota")
    assert parse_timezone(None) == ZoneInfo("America/Bogota")


@pytest.mark.parametrize("name", ["UTC", "Europe/Madrid", "America/Bogota", "Asia/Kolkata"])
def test_a_known_zone_is_accepted(name):
    assert parse_timezone(name) == ZoneInfo(name)
    assert parse_timezone(f"  {name}\n") == ZoneInfo(name)  # an env var may carry white space


@pytest.mark.parametrize(
    "bad",
    [
        "Mars/Phobos",
        "bogota",
        "../etc/passwd",
        "/etc/localtime",
        "a" * 500,
        "America/Bog\x00ota",
        "‮America/Bogota",
        "UTC+5",
    ],
)
def test_an_unknown_or_hostile_zone_is_refused_with_a_plain_message(bad):
    with pytest.raises(ValidationError) as caught:
        parse_timezone(bad)
    assert "time zone" in caught.value.message
    assert "passwd" not in caught.value.message
    assert len(caught.value.message) < 200


def test_a_time_is_drawn_in_the_local_zone_with_its_offset():
    assert format_local(NOON_UTC, ZoneInfo("America/Bogota")) == "2026-03-01 07:00 -05"
    assert format_local(NOON_UTC, ZoneInfo("UTC")) == "2026-03-01 12:00 UTC"


def test_seconds_can_be_asked_for():
    moment = NOON_UTC + timedelta(seconds=7)
    assert format_local(moment, ZoneInfo("UTC"), seconds=True) == "2026-03-01 12:00:07 UTC"


def test_summer_time_is_followed():
    summer = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    assert format_local(summer, ZoneInfo("Europe/Madrid")) == "2026-07-01 14:00 CEST"
    assert format_local(NOON_UTC, ZoneInfo("Europe/Madrid")) == "2026-03-01 13:00 CET"


def test_the_day_can_change_with_the_zone():
    late = datetime(2026, 3, 1, 3, 30, tzinfo=UTC)
    assert format_local(late, ZoneInfo("America/Bogota")).startswith("2026-02-28 22:30")


def test_a_time_with_another_offset_is_converted():
    other = datetime(2026, 3, 1, 13, 0, tzinfo=timezone(timedelta(hours=1)))
    assert format_local(other, ZoneInfo("UTC")) == "2026-03-01 12:00 UTC"


def test_a_naive_time_is_refused_not_guessed():
    with pytest.raises(ValidationError):
        format_local(datetime(2026, 3, 1, 12, 0), ZoneInfo("UTC"))


# ---- the compact and the full ISO forms the CLI prints ----


@pytest.mark.parametrize(
    ("zone", "expected"),
    [
        ("UTC", "2026-03-01T12:00Z"),
        ("America/Bogota", "2026-03-01T07:00-05"),
        ("Europe/Madrid", "2026-03-01T13:00+01"),
        ("Asia/Kolkata", "2026-03-01T17:30+05:30"),
        ("America/St_Johns", "2026-03-01T08:30-03:30"),
    ],
)
def test_the_compact_form_has_the_offset_the_zone_has(zone, expected):
    assert format_compact(NOON_UTC, ZoneInfo(zone)) == expected


def test_the_full_iso_form_has_seconds_and_the_offset():
    moment = NOON_UTC + timedelta(seconds=9)
    assert format_iso(moment, ZoneInfo("America/Bogota")) == "2026-03-01T07:00:09-05:00"
    assert format_iso(moment, ZoneInfo("UTC")) == "2026-03-01T12:00:09+00:00"


def test_both_refuse_a_naive_time():
    naive = datetime(2026, 3, 1, 12, 0)
    for fn in (format_compact, format_iso):
        with pytest.raises(ValidationError):
            fn(naive, ZoneInfo("UTC"))
