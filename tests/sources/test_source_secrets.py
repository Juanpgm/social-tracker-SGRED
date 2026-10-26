"""G-2 / SR secrets: credentials never travel inside source configuration.

`secret_ref` (a name pointing at a secret stored elsewhere) is the ONLY way to reference a
secret. Every other key is rejected by `extra="forbid"`: the model has no dict-typed fields,
so no key can hide below the top level. Nothing here depends on a key-name blacklist.
"""

import time

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


# ---- credentials in the URL query string / fragment -----------------------------------------

SECRET_QUERY_NAMES = [
    "password",
    "passwd",
    "pwd",
    "token",
    "access_token",
    "oauth_token",
    "secret",
    "client_secret",
    "api_key",
    "api-key",
    "apikey",
    "ApiKey",
    "X-Api-Key",
    "auth",
    "AUTH",
    "authorization",
    "signature",
    "X-Amz-Signature",
    "sig",
    "session",
    "sessionid",
    "cookie",
    "access_key",
    "accesskey",
    "access-key",
]


@pytest.mark.parametrize("name", SECRET_QUERY_NAMES)
@pytest.mark.parametrize("field", ["feed_url", "stream_url"])
def test_secret_looking_query_parameters_are_rejected(name: str, field: str):
    type_ = "news_rss" if field == "feed_url" else "live_stream"
    with pytest.raises(ValidationError, match=field) as info:
        parse_source(payload(type_, **{field: f"https://x.org/p?{name}=abc123"}))
    assert "abc123" not in info.value.message
    assert name not in info.value.message


@pytest.mark.parametrize(
    "query",
    [
        "a=1&token=x",  # not the first parameter
        "a=1;token=x",  # semicolon separator
        "token",  # name only, no value
        "token=",  # empty value still names a secret slot
        "%74oken=x",  # percent-encoded name
        "to%6Ben=x",
        "%2574oken=x",  # double-encoded
        "api%5Fkey=x",
        "api%2Dkey=x",
        "ｔoken=x",  # fullwidth letter, NFKC-folds to 't'
        "Token=x",
        "a=b&&c=d&password=x",
    ],
)
def test_encoded_and_positional_secret_parameters_are_rejected(query: str):
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(payload("news_rss", feed_url=f"https://x.org/p?{query}"))


def test_secret_in_fragment_is_rejected_too():
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(payload("news_rss", feed_url="https://x.org/p#access_token=abc"))


def test_secret_query_on_streaming_scheme_is_rejected():
    with pytest.raises(ValidationError, match="stream_url"):
        parse_source(payload("live_stream", stream_url="rtmp://x.org/live?token=abc"))


@pytest.mark.parametrize(
    "query",
    [
        "author=bob",  # contains 'auth' but is not a secret name
        "design=1",  # contains 'sig' but is not a secret name
        "category=news&page=2",
        "q=token",  # a secret-ish VALUE is fine, only names matter
        "feed=1#top",
        "",
        "a=1&&",
        "%ZZ=1",  # malformed escape must not crash
        "%FF%FE=1",  # invalid utf-8 bytes must not crash
        "v2sig",  # digits stay inside the word
        "authority",
    ],
)
def test_innocent_query_parameters_are_accepted(query: str):
    url = f"https://x.org/p?{query}"
    assert parse_source(payload("news_rss", feed_url=url)).feed_url == url


@pytest.mark.parametrize("name", ["x_auth", "x.auth", "x-sig", "oauth", "My Pwd", "a%20pwd"])
def test_short_secret_words_match_as_whole_tokens(name: str):
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(payload("news_rss", feed_url=f"https://x.org/p?{name}=1"))


def test_fragment_anchor_without_value_is_not_a_parameter():
    url = "https://x.org/docs#token"
    assert parse_source(payload("news_rss", feed_url=url)).feed_url == url


def test_url_without_query_is_accepted():
    assert parse_source(payload("news_rss", feed_url="https://x.org/token/rss")).feed_url


def test_long_query_is_checked_quickly_and_correctly():
    start = time.perf_counter()
    noise = "&".join(f"k{i}=v" for i in range(150))
    with pytest.raises(ValidationError, match="feed_url"):
        parse_source(payload("news_rss", feed_url=f"https://x.org/p?{noise}&token=x"))
    assert time.perf_counter() - start < 1.0


def test_percent_run_pathological_input_is_fast():
    start = time.perf_counter()
    url = "https://x.org/p?" + "%25" * 600
    assert parse_source(payload("news_rss", feed_url=url)).feed_url == url
    assert time.perf_counter() - start < 1.0
