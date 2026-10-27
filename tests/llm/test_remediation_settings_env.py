"""The settings the remediation round added, read from the environment."""

import pytest

from social_tracker.bootstrap import (
    LLM_LOAD_TIMEOUT_ENV,
    LLM_NUM_CTX_ENV,
    LLM_PARKED_TTL_ENV,
    LLM_PASS_BUDGET_ENV,
    LLM_SOURCE_SHARE_ENV,
    parse_llm_settings,
)
from social_tracker.llm.domain import LlmSettings
from social_tracker.shared.errors import ValidationError


def test_the_defaults_are_the_documented_ones():
    s = parse_llm_settings({})
    assert (s.load_deadline_s, s.num_ctx, s.pass_budget_s) == (180.0, 8192, 120.0)
    assert (s.parked_ttl_s, s.source_share) == (86_400, 0.25)
    assert s == LlmSettings()


def test_every_new_variable_is_read():
    env = {
        LLM_LOAD_TIMEOUT_ENV: "300",
        LLM_NUM_CTX_ENV: "16384",
        LLM_PASS_BUDGET_ENV: "45",
        LLM_PARKED_TTL_ENV: "7200",
        LLM_SOURCE_SHARE_ENV: "0.4",
    }
    s = parse_llm_settings(env)
    assert (s.load_deadline_s, s.num_ctx, s.pass_budget_s) == (300.0, 16384, 45.0)
    assert (s.parked_ttl_s, s.source_share) == (7200, 0.4)


def test_blank_values_mean_the_default():
    blank = {
        LLM_LOAD_TIMEOUT_ENV: " ",
        LLM_NUM_CTX_ENV: "",
        LLM_PASS_BUDGET_ENV: " ",
        LLM_PARKED_TTL_ENV: "",
        LLM_SOURCE_SHARE_ENV: " ",
    }
    assert parse_llm_settings(blank) == LlmSettings()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (LLM_LOAD_TIMEOUT_ENV, "0"),
        (LLM_LOAD_TIMEOUT_ENV, "1801"),
        (LLM_LOAD_TIMEOUT_ENV, "2.5"),
        (LLM_LOAD_TIMEOUT_ENV, "3m"),
        (LLM_NUM_CTX_ENV, "2559"),
        (LLM_NUM_CTX_ENV, "262145"),
        (LLM_NUM_CTX_ENV, "8k"),
        (LLM_PASS_BUDGET_ENV, "0"),
        (LLM_PASS_BUDGET_ENV, "86401"),
        (LLM_PASS_BUDGET_ENV, "-5"),
        (LLM_PARKED_TTL_ENV, "59"),
        (LLM_PARKED_TTL_ENV, "2592001"),
        (LLM_PARKED_TTL_ENV, "a day"),
        (LLM_SOURCE_SHARE_ENV, "0"),
        (LLM_SOURCE_SHARE_ENV, "0.0"),
        (LLM_SOURCE_SHARE_ENV, "1.01"),
        (LLM_SOURCE_SHARE_ENV, "2"),
        (LLM_SOURCE_SHARE_ENV, "-0.2"),
        (LLM_SOURCE_SHARE_ENV, "0.12345"),
        (LLM_SOURCE_SHARE_ENV, "a quarter"),
        (LLM_SOURCE_SHARE_ENV, "nan"),
    ],
)
def test_an_invalid_value_is_refused_naming_the_variable(name, value):
    with pytest.raises(ValidationError) as caught:
        parse_llm_settings({name: value})
    assert name in str(caught.value)


@pytest.mark.parametrize(
    ("name", "value", "field", "expected"),
    [
        (LLM_LOAD_TIMEOUT_ENV, "1", "load_deadline_s", 1.0),
        (LLM_LOAD_TIMEOUT_ENV, "1800", "load_deadline_s", 1800.0),
        (LLM_NUM_CTX_ENV, "2560", "num_ctx", 2560),
        (LLM_NUM_CTX_ENV, "262144", "num_ctx", 262144),
        (LLM_PASS_BUDGET_ENV, "1", "pass_budget_s", 1.0),
        (LLM_PASS_BUDGET_ENV, "86400", "pass_budget_s", 86400.0),
        (LLM_PARKED_TTL_ENV, "60", "parked_ttl_s", 60),
        (LLM_PARKED_TTL_ENV, "2592000", "parked_ttl_s", 2_592_000),
        (LLM_SOURCE_SHARE_ENV, "1", "source_share", 1.0),
        (LLM_SOURCE_SHARE_ENV, "0.0001", "source_share", 0.0001),
        (LLM_SOURCE_SHARE_ENV, "0.5", "source_share", 0.5),
    ],
)
def test_the_ends_of_every_range_are_accepted(name, value, field, expected):
    assert getattr(parse_llm_settings({name: value}), field) == expected
