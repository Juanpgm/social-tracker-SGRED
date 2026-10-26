"""G-2 / SR secrets: credentials never travel inside source configuration.

`secret_ref` (a name pointing at a secret stored elsewhere) is the ONLY way to reference a
secret. Every other key is rejected by `extra="forbid"`: the model has no dict-typed fields,
so no key can hide below the top level. Nothing here depends on a key-name blacklist.
"""

import pytest

from social_tracker.shared.errors import ValidationError
from social_tracker.sources.domain import parse_source
from tests.sources.factories import EMBEDDED_CREDENTIALS_URL, payload

TYPES = ["news_rss", "instagram", "live_stream", "recorded_audio"]

SECRETISH_KEYS = [
    "password",
    "Password",
    "auth",
    "bearer",
    "credentials",
    "private_key",
    "token",
    "auth_token",
    "cookie",
    "session",
    "secret",
    "api_key",
    "API_KEY",
    "client_secret",
    "secret_ref_extra",
    "Secret_Ref",
    "secret_ref ",
    "pаssword",  # Cyrillic a
    "pass​word",  # zero-width space
    "lives",
    "include_lives",
    "live",
]


@pytest.mark.parametrize("type_", TYPES)
@pytest.mark.parametrize("key", SECRETISH_KEYS)
def test_unknown_top_level_keys_are_rejected(type_: str, key: str):
    with pytest.raises(ValidationError):
        parse_source(payload(type_, **{key: "hunter2"}))


@pytest.mark.parametrize(
    "extra",
    [
        {"deep": [{"x": {"password": "v"}}]},
        {"password": "v"},
        {1: "v"},
        [{"token": "v"}],
    ],
)
def test_nested_values_under_unknown_keys_are_rejected(extra):
    with pytest.raises(ValidationError):
        parse_source(payload("news_rss", extra=extra))


def test_non_string_top_level_key_is_rejected():
    with pytest.raises(ValidationError):
        parse_source({**payload("news_rss"), 5: "v"})


def test_secret_like_dict_in_keywords_is_rejected():
    with pytest.raises(ValidationError, match="keywords"):
        parse_source(payload("news_rss", keywords=[{"api_key": "v"}]))


def test_secret_ref_nested_is_just_an_unknown_key():
    with pytest.raises(ValidationError):
        parse_source(payload("news_rss", extra={"secret_ref": "x"}))


def test_secret_ref_top_level_is_allowed():
    assert parse_source(payload("instagram", secret_ref="vault_key")).secret_ref == "vault_key"


def test_url_with_embedded_credentials_is_rejected():
    with pytest.raises(ValidationError, match="credentials"):
        parse_source(payload("news_rss", feed_url=EMBEDDED_CREDENTIALS_URL))
