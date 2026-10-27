"""Alert threshold parsing and the actor/reason fields of control actions."""

import math
from decimal import Decimal
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from social_tracker.control.domain import (
    DEFAULT_ALERT_MIN_CONFIDENCE,
    DEFAULT_SETTINGS,
    MAX_ACTOR,
    MAX_REASON,
    SETTING_ALERT_MIN_CONFIDENCE,
    format_confidence,
    parse_confidence,
    validate_actor,
    validate_confidence,
    validate_reason,
    validate_setting,
)
from social_tracker.shared.errors import ValidationError

TINY = math.nextafter(0.0, 1.0)
JUST_BELOW_ONE = math.nextafter(1.0, 0.0)
CANARY = "canary-" + "value-42"


def test_the_default_threshold_is_exactly_one_half_and_is_in_the_default_settings():
    assert DEFAULT_ALERT_MIN_CONFIDENCE == 0.5
    assert DEFAULT_SETTINGS[SETTING_ALERT_MIN_CONFIDENCE] == "0.5"
    assert SETTING_ALERT_MIN_CONFIDENCE == "alert_min_confidence"


# ---- validate_confidence (typed values) ----------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 0.25, TINY, JUST_BELOW_ONE, 0, 1])
def test_valid_thresholds_are_accepted_as_floats(value: float):
    result = validate_confidence(value)
    assert result == value
    assert type(result) is float


def test_negative_zero_is_normalised_to_positive_zero():
    assert math.copysign(1.0, validate_confidence(-0.0)) == 1.0


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        -0.1,
        1.01,
        -TINY,
        math.nextafter(1.0, 2.0),
        2,
        -1,
        "abc",
        "0.5",
        True,
        False,
        None,
        [],
        Decimal("0.5"),
        1j,
        b"0.5",
    ],
    ids=repr,
)
def test_invalid_thresholds_are_rejected(value: Any):
    with pytest.raises(ValidationError) as caught:
        validate_confidence(value)
    assert caught.value.fields == (SETTING_ALERT_MIN_CONFIDENCE,)


# ---- format / parse (stored strings) --------------------------------------------------


@pytest.mark.parametrize(
    ("value", "text"),
    [(0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0"), (0.25, "0.25"), (1e-05, "0.00001")],
)
def test_format_uses_plain_decimal_notation(value: float, text: str):
    assert format_confidence(value) == text


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, TINY, JUST_BELOW_ONE, 0.1, 1 / 3])
def test_format_then_parse_returns_the_same_float(value: float):
    text = format_confidence(value)
    assert "e" not in text.lower()
    assert len(text) <= 400
    assert parse_confidence(text) == value


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_any_valid_float_survives_format_and_parse(value: float):
    assert parse_confidence(format_confidence(value)) == value


@pytest.mark.parametrize(
    ("text", "value"),
    [("0.5", 0.5), ("0", 0.0), ("1", 1.0), ("1.0", 1.0), ("0.50", 0.5), ("0.000001", 1e-06)],
)
def test_parse_accepts_plain_decimals_in_range(text: str, value: float):
    assert parse_confidence(text) == value


@pytest.mark.parametrize(
    "text",
    [
        "",
        " 0.5",
        "0.5 ",
        "0.5\n",
        "+0.5",
        "-0.1",
        "-0",
        "1.01",
        "2",
        "10",
        "0.5e0",
        "1e-3",
        ".5",
        "5.",
        "nan",
        "NaN",
        "inf",
        "-inf",
        "abc",
        "0,5",
        "0.5.1",
        "٠.٥",  # Arabic-Indic digits
        "0." + "0" * 500,
        None,
        0.5,
        True,
        b"0.5",
    ],
    ids=repr,
)
def test_parse_rejects_anything_that_is_not_a_plain_decimal_in_range(text: Any):
    with pytest.raises(ValidationError) as caught:
        parse_confidence(text)
    assert caught.value.fields == (SETTING_ALERT_MIN_CONFIDENCE,)


def test_the_error_for_a_bad_stored_value_does_not_echo_it():
    with pytest.raises(ValidationError) as caught:
        parse_confidence(f"{CANARY}")
    assert CANARY not in caught.value.message


@pytest.mark.parametrize("value", ["0.7", "0", "1", "0.0"])
def test_validate_setting_accepts_a_valid_threshold(value: str):
    assert validate_setting(SETTING_ALERT_MIN_CONFIDENCE, value) == (
        SETTING_ALERT_MIN_CONFIDENCE,
        value,
    )


@pytest.mark.parametrize("value", ["abc", "1.5", "-0.1", "", "nan", "0.5e0", " 0.5"])
def test_validate_setting_rejects_an_invalid_threshold(value: str):
    with pytest.raises(ValidationError):
        validate_setting(SETTING_ALERT_MIN_CONFIDENCE, value)


def test_other_settings_are_not_affected_by_the_threshold_rule():
    assert validate_setting("some_other_setting", "abc") == ("some_other_setting", "abc")


# ---- actor / reason -------------------------------------------------------------------

FIELDS = [(validate_actor, "actor", MAX_ACTOR), (validate_reason, "reason", MAX_REASON)]


@pytest.mark.parametrize(("check", "field", "limit"), FIELDS)
@pytest.mark.parametrize("value", ["operator", "José ✓", "a", "with: punctuation; ok."])
def test_valid_text_is_returned_unchanged(check, field, limit, value):
    assert check(value) == value


@pytest.mark.parametrize(("check", "field", "limit"), FIELDS)
def test_text_of_exactly_the_limit_is_accepted_and_one_more_is_rejected(check, field, limit):
    assert check("x" * limit) == "x" * limit
    with pytest.raises(ValidationError) as caught:
        check("x" * (limit + 1))
    assert caught.value.fields == (field,)


@pytest.mark.parametrize(("check", "field", "limit"), FIELDS)
@pytest.mark.parametrize(
    "value",
    ["", " ", "   ", " ", None, 5, b"x", ["x"], "a\x00b", "a\nb", "a\tb", "a\x7fb", "a\x85b"],
    ids=repr,
)
def test_blank_non_text_and_control_character_values_are_rejected(check, field, limit, value):
    with pytest.raises(ValidationError) as caught:
        check(value)
    assert caught.value.fields == (field,)


@pytest.mark.parametrize(("check", "field", "limit"), FIELDS)
def test_errors_never_echo_the_offending_value(check, field, limit):
    with pytest.raises(ValidationError) as caught:
        check(CANARY + "\x00")
    assert CANARY not in caught.value.message


def test_the_stored_threshold_text_limit_is_inclusive():
    at_limit = "0." + "0" * 398
    assert len(at_limit) == 400
    assert parse_confidence(at_limit) == 0.0
    with pytest.raises(ValidationError):
        parse_confidence(at_limit + "0")


def test_the_documented_limits_are_pinned():
    assert (MAX_ACTOR, MAX_REASON) == (100, 500)
