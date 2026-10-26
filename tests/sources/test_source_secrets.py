"""G-2 / SR secrets: credentials never travel inside source configuration."""

import time

import pytest

from social_tracker.shared.errors import ValidationError
from social_tracker.sources.domain import parse_source
from tests.sources.factories import CANARY, EMBEDDED_CREDENTIALS_URL, payload


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "token",
        "cookie",
        "session",
        "secret",
        "api_key",
        "apikey",
        "api-key",
        "my_Api_Key",
        "API_KEY",
        "Password",
        "auth_token",
        "sessionid",
        "cookies",
        "client_secret",
        "secret_ref_extra",
        "Secret_Ref",
        "SECRET_REF",
        "secret_ref ",
    ],
)
def test_inline_secret_keys_are_rejected_at_top_level(key: str):
    with pytest.raises(ValidationError, match="inline secret"):
        parse_source(payload("news_rss", **{key: "hunter2"}))


@pytest.mark.parametrize("key", ["password", "my_Api_Key", "Token", "cookie"])
def test_inline_secret_keys_are_rejected_when_nested(key: str):
    with pytest.raises(ValidationError, match="inline secret"):
        parse_source(payload("news_rss", extra={"deep": [{"x": {key: "v"}}]}))


def test_nested_secret_inside_list_of_dicts_is_rejected():
    with pytest.raises(ValidationError, match="inline secret"):
        parse_source(payload("news_rss", keywords=[{"api_key": "v"}]))


def test_exact_secret_ref_key_is_never_flagged_as_inline_secret():
    # Nested `secret_ref` passes the secret scan (it is then rejected as an unknown key).
    with pytest.raises(ValidationError) as info:
        parse_source(payload("news_rss", extra={"secret_ref": "x"}))
    assert "inline secret" not in info.value.message


def test_secret_ref_top_level_is_allowed():
    assert parse_source(payload("instagram", secret_ref="vault_key")).secret_ref == "vault_key"


def test_innocent_keys_are_not_flagged():
    with pytest.raises(ValidationError) as info:
        parse_source(payload("news_rss", description="hello"))
    assert "inline secret" not in info.value.message
    assert "description" in info.value.message


def test_error_message_never_echoes_the_secret_value():
    with pytest.raises(ValidationError) as info:
        parse_source(payload("news_rss", **{"pass" + "word": CANARY}))
    assert CANARY not in info.value.message
    assert "password" in info.value.message


def test_very_long_offending_key_is_truncated_in_the_message():
    with pytest.raises(ValidationError) as info:
        parse_source(payload("news_rss", **{"password" + "x" * 10_000: 1}))
    assert len(info.value.message) < 500


def test_self_referencing_input_does_not_hang():
    loop: dict = {}
    loop["me"] = loop
    with pytest.raises(ValidationError):
        parse_source(payload("news_rss", extra=loop))


def test_deeply_nested_input_does_not_overflow_the_stack():
    deep: dict = {}
    cursor = deep
    for _ in range(50_000):
        cursor["n"] = {}
        cursor = cursor["n"]
    cursor["password"] = "x"
    with pytest.raises(ValidationError, match="inline secret"):
        parse_source(payload("news_rss", extra=deep))


def test_one_megabyte_key_is_scanned_quickly():
    start = time.perf_counter()
    with pytest.raises(ValidationError):
        parse_source(payload("news_rss", **{"k" * 1_000_000: 1}))
    assert time.perf_counter() - start < 2.0


def test_url_with_embedded_credentials_is_rejected():
    with pytest.raises(ValidationError, match="credentials"):
        parse_source(payload("news_rss", feed_url=EMBEDDED_CREDENTIALS_URL))
