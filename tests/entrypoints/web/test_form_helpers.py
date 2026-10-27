"""The pure helpers behind the edit and busy forms: address masking and version parsing."""

from datetime import UTC, datetime

import pytest

from social_tracker.entrypoints.web.forms import (
    is_plain_url,
    keep_masked_urls,
    mask_url,
    parse_version,
    without_unvalidated_secrets,
)
from social_tracker.sources.domain import parse_source
from tests.sources.factories import payload

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org",
        "https://example.org:8443/a/b.xml",
        "rtmp://example.org/live",
        "icecast://radio.example.org:8000/mount",
        "https://[2001:db8::1]/feed",
    ],
)
def test_a_plain_address_has_a_known_scheme_and_a_host_and_nothing_else(url: str):
    assert is_plain_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "example.org",
        "ftp://example.org/x",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://",
        "https:///path",
        "https://user:pw@example.org/",
        "https://user@example.org/",
        "https://example.org/?a=1",
        "https://example.org/#frag",
        "https://[::1/",
        "not a url",
    ],
)
def test_anything_that_may_hold_a_credential_or_is_not_an_address_is_not_plain(url: str):
    assert not is_plain_url(url)


def test_masking_replaces_the_query_and_drops_the_fragment_but_keeps_the_rest():
    assert mask_url("https://example.org:8443/live.mp3?a=1&b=2#x") == (
        "https://example.org:8443/live.mp3?***"
    )
    assert mask_url("https://u:p@example.org/x?k=v") == "https://example.org/x?***"


@pytest.mark.parametrize("url", ["https://example.org/x", "https://example.org/x#frag", "", "x"])
def test_an_address_without_a_query_is_left_alone(url: str):
    assert mask_url(url) == url


def test_an_unparseable_address_is_returned_as_it_is():
    assert mask_url("https://[::1/x?a=1") == "https://[::1/x?a=1"


def stream(url: str):
    return parse_source(
        payload("live_stream", id="src_one", stream_url=url, created_at=NOW, updated_at=NOW)
    )


def test_the_mask_of_the_stored_address_restores_it_and_nothing_else_does():
    stored = stream("https://example.org/live.mp3?station=7")
    mask = "https://example.org/live.mp3?***"
    assert keep_masked_urls(stored, {"stream_url": mask})["stream_url"] == stored.stream_url
    assert (
        keep_masked_urls(stored, {"stream_url": f"  {mask}  "})["stream_url"] == stored.stream_url
    )
    for other in ("https://example.org/live.mp3", "https://example.org/other.mp3?***", ""):
        assert keep_masked_urls(stored, {"stream_url": other})["stream_url"] == other


def test_other_fields_and_sources_without_a_query_pass_through_untouched():
    plain = stream("https://example.org/live.mp3")
    form = {"name": "Radio", "stream_url": "https://example.org/live.mp3?***"}
    assert keep_masked_urls(plain, form) == form
    stored = stream("https://example.org/live.mp3?station=7")
    out = keep_masked_urls(stored, {"name": "Radio", "language": "es"})
    assert out == {"name": "Radio", "language": "es"}


def test_the_input_form_is_never_modified():
    stored = stream("https://example.org/live.mp3?station=7")
    form = {"stream_url": "https://example.org/live.mp3?***"}
    keep_masked_urls(stored, form)
    assert form == {"stream_url": "https://example.org/live.mp3?***"}


def test_unvalidated_secrets_are_dropped_and_harmless_values_kept():
    form = {
        "name": "N",
        "feed_url": "https://example.org/a.xml",
        "watch_dir": "/srv/in",
        "secret_ref": "env:X",
    }
    kept = without_unvalidated_secrets("recorded_audio", form)
    assert kept == {"name": "N", "feed_url": "https://example.org/a.xml", "watch_dir": "/srv/in"}
    leaky = without_unvalidated_secrets("recorded_audio", {**form, "feed_url": "https://a/?k=v"})
    assert "feed_url" not in leaky
    insta = without_unvalidated_secrets("instagram", {"handle": "h", "secret_ref": "env:X"})
    assert insta == {"handle": "h"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-01-02T03:04:05+00:00", NOW),
        ("2026-01-02T03:04:05.123456+00:00", NOW.replace(microsecond=123456)),
        ("2026-01-02T04:04:05+01:00", NOW),
        ("2026-01-02T03:04:05Z", NOW),
    ],
)
def test_a_version_with_an_offset_is_parsed(raw: str, expected: datetime):
    assert parse_version(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "yesterday",
        "2026-01-02T03:04:05",
        "2026-01-02",
        "2026-13-45T00:00:00+00:00",
        "x" * 65,
    ],
)
def test_a_missing_naive_or_malformed_version_is_none(raw: str | None):
    assert parse_version(raw) is None
