"""ControlEvent value object and settings key/value rules."""

from datetime import UTC, datetime

import pytest

from social_tracker.control.domain import (
    DEFAULT_SETTINGS,
    SETTING_GLOBAL_ANALYSIS_ENABLED,
    ControlEvent,
    validate_setting,
)
from social_tracker.shared.errors import ValidationError

AT = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def test_valid_event_with_defaults():
    event = ControlEvent(at=AT, kind="source_enabled", source_id="src_one")
    assert event.detail == ""
    assert event.at == AT


def test_global_event_has_no_source():
    assert ControlEvent(at=AT, kind="paused").source_id is None


@pytest.mark.parametrize("kind", ["k", "k" * 64])
def test_kind_boundaries_accepted(kind: str):
    assert ControlEvent(at=AT, kind=kind).kind == kind


@pytest.mark.parametrize("kind", ["", "   ", "k" * 65, None, 5])
def test_bad_kind_rejected(kind):
    with pytest.raises(ValidationError, match="kind"):
        ControlEvent(at=AT, kind=kind)  # type: ignore[arg-type]


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        ControlEvent(at=datetime(2026, 3, 1), kind="paused")


@pytest.mark.parametrize("source_id", ["", "Bad Id", "ab", 5])
def test_bad_source_id_rejected(source_id):
    with pytest.raises(ValidationError):
        ControlEvent(at=AT, kind="paused", source_id=source_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("detail", [None, 5, "d" * 2001])
def test_bad_detail_rejected(detail):
    with pytest.raises(ValidationError, match="detail"):
        ControlEvent(at=AT, kind="paused", detail=detail)  # type: ignore[arg-type]


def test_detail_boundary_and_unicode_accepted():
    assert len(ControlEvent(at=AT, kind="paused", detail="d" * 2000).detail) == 2000
    assert ControlEvent(at=AT, kind="paused", detail="新聞").detail == "新聞"


def test_event_is_immutable():
    event = ControlEvent(at=AT, kind="paused")
    with pytest.raises(AttributeError):
        event.kind = "other"  # type: ignore[misc]


def test_global_analysis_defaults_to_false():
    assert DEFAULT_SETTINGS[SETTING_GLOBAL_ANALYSIS_ENABLED] == "false"


@pytest.mark.parametrize("key", ["a", "k" * 100])
def test_setting_boundaries_accepted(key: str):
    assert validate_setting(key, "v") == (key, "v")


@pytest.mark.parametrize("key", ["", "  ", "k" * 101, None, 5])
def test_bad_setting_key_rejected(key):
    with pytest.raises(ValidationError, match="key"):
        validate_setting(key, "v")


@pytest.mark.parametrize("value", [None, 1, True, b"x", "v" * 1001])
def test_bad_setting_value_rejected(value):
    with pytest.raises(ValidationError, match="value"):
        validate_setting("k", value)


def test_empty_setting_value_is_allowed():
    assert validate_setting("k", "") == ("k", "")
