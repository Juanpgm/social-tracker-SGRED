"""ControlEvent value object and settings key/value rules."""

from datetime import UTC, datetime

import pytest

from social_tracker.control.domain import (
    DEFAULT_SETTINGS,
    MAX_EVENT_DETAIL,
    MAX_EVENT_KIND,
    MAX_SETTING_KEY,
    MAX_SETTING_VALUE,
    SETTING_GLOBAL_ANALYSIS_ENABLED,
    ControlEvent,
    validate_setting,
    validate_setting_key,
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


CONTROL_KEYS = [
    "\x00",
    "\x00k",
    "k\x00",
    "k\x00k",
    "k\x1fk",
    "k\x7fk",
    "k\x80k",
    "k\x85k",
    "k\x9fk",
    "k\tk",
    "k\nk",
]


@pytest.mark.parametrize("kind", CONTROL_KEYS)
def test_kind_rejects_control_characters_anywhere(kind: str):
    with pytest.raises(ValidationError, match="kind"):
        ControlEvent(at=AT, kind=kind)


@pytest.mark.parametrize("key", CONTROL_KEYS)
def test_setting_key_rejects_control_characters_anywhere(key: str):
    with pytest.raises(ValidationError, match="key"):
        validate_setting(key, "v")
    with pytest.raises(ValidationError, match="key"):
        validate_setting_key(key)


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


@pytest.mark.parametrize("size", [MAX_SETTING_VALUE - 1, MAX_SETTING_VALUE])
def test_setting_value_boundary_accepted(size: int):
    assert validate_setting("k", "v" * size) == ("k", "v" * size)


def test_setting_value_over_the_boundary_rejected():
    with pytest.raises(ValidationError, match="value"):
        validate_setting("k", "v" * (MAX_SETTING_VALUE + 1))


def test_documented_limits():
    assert (MAX_SETTING_KEY, MAX_SETTING_VALUE) == (100, 1000)
    assert (MAX_EVENT_KIND, MAX_EVENT_DETAIL) == (64, 2000)


@pytest.mark.parametrize("value", ["true", "false"])
def test_global_analysis_accepts_only_lowercase_booleans(value: str):
    assert validate_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, value) == (
        SETTING_GLOBAL_ANALYSIS_ENABLED,
        value,
    )


@pytest.mark.parametrize(
    "value", ["", "TRUE", "False", "1", "0", "yes", " true", "true\n", "maybe"]
)
def test_global_analysis_rejects_everything_else(value: str):
    with pytest.raises(ValidationError, match=SETTING_GLOBAL_ANALYSIS_ENABLED):
        validate_setting(SETTING_GLOBAL_ANALYSIS_ENABLED, value)


def test_other_settings_are_free_form_strings():
    assert validate_setting("other_key", "maybe") == ("other_key", "maybe")


def test_blank_event_kind_and_detail_rules_differ():
    # `kind` must be non-blank; `detail` may be empty or blank.
    assert ControlEvent(at=AT, kind="paused", detail="   ").detail == "   "
    with pytest.raises(ValidationError, match="kind"):
        ControlEvent(at=AT, kind="   ")


# ---- hidden characters ----------------------------------------------------------------

HIDDEN = ["\ud800x", "x\udfff", "a​b", "‮", "a b", "a b", "a﻿b"]


@pytest.mark.parametrize("value", HIDDEN, ids=ascii)
def test_event_kind_and_detail_reject_surrogates_and_format_characters(value: str):
    with pytest.raises(ValidationError, match="kind"):
        ControlEvent(at=AT, kind=value)
    with pytest.raises(ValidationError, match="detail"):
        ControlEvent(at=AT, kind="paused", detail=value)


@pytest.mark.parametrize("value", HIDDEN, ids=ascii)
def test_setting_keys_and_values_reject_surrogates_and_format_characters(value: str):
    with pytest.raises(ValidationError, match="key"):
        validate_setting_key(value)
    with pytest.raises(ValidationError, match="value"):
        validate_setting("note", value)


def test_accents_combining_marks_and_emoji_remain_allowed_in_events_and_settings():
    text = "Mañana café \U0001f600"
    assert ControlEvent(at=AT, kind="paused", detail=text).detail == text
    assert validate_setting("note", text) == ("note", text)
