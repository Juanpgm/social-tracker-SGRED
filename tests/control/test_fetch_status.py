"""FetchStatus: what the collector remembers about a source, and error sanitizing."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from social_tracker.control.domain import MAX_FETCH_ERROR, FetchStatus, sanitize_error
from social_tracker.shared.errors import ValidationError

AT = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def test_a_fetch_status_defaults_to_nothing_known():
    status = FetchStatus("src_one")
    assert (status.last_fetch_at, status.last_error, status.etag, status.last_modified) == (
        None,
        None,
        None,
        None,
    )


def test_the_time_is_normalized_to_utc():
    status = FetchStatus(
        "src_one", datetime(2026, 3, 1, 3, 0, tzinfo=timezone(timedelta(hours=-5)))
    )
    assert status.last_fetch_at == AT
    assert status.last_fetch_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_id": "Bad Id"},
        {"last_fetch_at": datetime(2026, 3, 1)},
        {"last_fetch_at": "2026-03-01"},
        {"last_error": ""},
        {"last_error": "e" * (MAX_FETCH_ERROR + 1)},
        {"last_error": "line\nbreak"},
        {"last_error": "hidden​char"},
        {"last_error": 5},
        {"etag": ""},
        {"etag": "e" * 513},
        {"last_modified": "m" * 101},
    ],
    ids=lambda k: next(iter(k)) + "-" + str(next(iter(k.values())))[:10],
)
def test_an_invalid_fetch_status_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        FetchStatus(**{"source_id": "src_one", **kwargs})


def test_an_error_exactly_at_the_limit_is_accepted():
    assert len(FetchStatus("src_one", last_error="e" * MAX_FETCH_ERROR).last_error) == (
        MAX_FETCH_ERROR
    )


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("http 503", "http 503"),
        ("timeout\n  while\tfetching", "timeout while fetching"),
        (
            "failed for https://user:pw@example.org/feed?token=abc123 today",
            "failed for <url> today",
        ),
        ("see HTTP://EXAMPLE.ORG/x and http://a.b/c", "see <url> and <url>"),
        ("secret \x00\x1b[31m bytes", "secret [31m bytes"),
        ("bad​hidden", "badhidden"),
        ("   ", "unknown error"),
        ("", "unknown error"),
        ("x" * 1000, "x" * MAX_FETCH_ERROR),
    ],
)
def test_error_text_is_made_safe_before_it_is_stored(raw, clean):
    assert sanitize_error(raw) == clean


def test_a_sanitized_error_is_always_a_valid_stored_error():
    nasty = "‮\x00" + "https://h/?password=hunter2 " * 100
    FetchStatus("src_one", last_error=sanitize_error(nasty))
    assert "hunter2" not in sanitize_error(nasty)
