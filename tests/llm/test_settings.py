"""LlmSettings: the only http (non-https) address of the application is the loopback one."""

import pytest

from social_tracker.llm.domain import (
    DEFAULT_BASE_URL,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_MODEL,
    LlmSettings,
    parse_loopback_url,
)
from social_tracker.shared.errors import ValidationError


@pytest.mark.parametrize(
    ("url", "host", "port"),
    [
        ("http://127.0.0.1:11434", "127.0.0.1", 11434),
        ("http://127.0.0.1:11434/", "127.0.0.1", 11434),
        ("http://localhost:11434", "localhost", 11434),
        ("HTTP://LocalHost:8080", "localhost", 8080),
        ("http://127.0.0.1", "127.0.0.1", 80),
        ("http://127.0.0.1:65535", "127.0.0.1", 65535),
        ("http://127.0.0.1:1", "127.0.0.1", 1),
    ],
)
def test_a_loopback_address_is_accepted(url: str, host: str, port: int):
    endpoint = parse_loopback_url(url)
    assert (endpoint.host, endpoint.port) == (host, port)


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:11434",  # the one http exception is plain http on loopback
        "http://example.org:11434",
        "http://127.0.0.2:11434",
        "http://0.0.0.0:11434",
        "http://10.0.0.5:11434",
        "http://[::1]:11434",
        "http://127.0.0.1.evil.example:11434",
        "http://localhost.evil.example:11434",
        "http://evil.example/127.0.0.1",
        "http://user:pw@127.0.0.1:11434",
        "http://127.0.0.1@evil.example",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?x=1",
        "http://127.0.0.1:11434#frag",
        "http://127.0.0.1:0",
        "http://127.0.0.1:65536",
        "http://127.0.0.1:abc",
        "http://127.0.0.1:011434x",
        "http://127.1:11434",
        "http://2130706433:11434",
        " http://127.0.0.1:11434",
        "http://127.0.0.1:11434\n",
        "http://127.0.0.1 :11434",
        "ftp://127.0.0.1",
        "127.0.0.1:11434",
        "",
    ],
)
def test_anything_else_is_refused(url: str):
    with pytest.raises(ValidationError):
        parse_loopback_url(url)


@pytest.mark.parametrize("value", [None, 11434, b"http://127.0.0.1", ["http://127.0.0.1"]])
def test_a_non_text_address_is_refused(value: object):
    with pytest.raises(ValidationError):
        parse_loopback_url(value)  # type: ignore[arg-type]


def test_a_name_that_may_resolve_elsewhere_still_connects_to_the_literal_loopback_address():
    assert parse_loopback_url("http://localhost:11434").connect_host == "127.0.0.1"
    assert parse_loopback_url("http://127.0.0.1:9").connect_host == "127.0.0.1"
    assert parse_loopback_url("http://localhost:9").url == "http://127.0.0.1:9"


def test_the_defaults_are_the_products():
    s = LlmSettings()
    assert s.base_url == DEFAULT_BASE_URL == "http://127.0.0.1:11434"
    assert s.model == DEFAULT_MODEL == "st-gemma4-e4b"
    assert s.fallback_model == DEFAULT_FALLBACK_MODEL == "st-qwen35-4b"
    assert (s.keep_alive_s, s.call_deadline_s, s.hourly_cap) == (300, 60.0, 200)


def test_settings_refuse_a_non_loopback_base_url():
    with pytest.raises(ValidationError):
        LlmSettings(base_url="http://example.org:11434")


@pytest.mark.parametrize(
    "changes",
    [
        {"model": ""},
        {"model": "bad model"},
        {"model": "x" * 101},
        {"model": "-leading"},
        {"model": "a;rm"},
        {"fallback_model": "has space"},
        {"keep_alive_s": 0},
        {"keep_alive_s": 86_401},
        {"call_deadline_s": 0},
        {"call_deadline_s": 601},
        {"call_deadline_s": float("nan")},
        {"call_deadline_s": float("inf")},
        {"hourly_cap": 0},
        {"hourly_cap": 100_001},
        {"hourly_cap": True},
        {"queue_max": 0},
        {"batch_limit": 0},
        {"max_response_bytes": 1023},
        {"max_prompt_chars": 99},
        {"max_output_tokens": 0},
        {"retry_base_s": 0},
        {"retry_cap_s": 1, "retry_base_s": 2},
        {"max_failures": 0},
    ],
)
def test_invalid_numbers_and_names_are_refused(changes: dict):
    with pytest.raises(ValidationError):
        LlmSettings(**changes)


def test_the_fallback_may_be_switched_off():
    assert LlmSettings(fallback_model=None).fallback_model is None


@pytest.mark.parametrize("model", ["st-gemma4-e4b", "library/model:7b-q4_K_M", "a", "A.b_c-d"])
def test_ordinary_model_names_are_accepted(model: str):
    assert LlmSettings(model=model).model == model
