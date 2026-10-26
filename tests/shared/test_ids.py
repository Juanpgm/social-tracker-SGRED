"""Slug helpers: ASCII lowercase, digits, underscore; 3-64 chars; no regex pitfalls."""

import time

import pytest

from social_tracker.shared.errors import ValidationError
from social_tracker.shared.ids import SLUG_MAX, SLUG_MIN, is_valid_slug, validate_slug


@pytest.mark.parametrize(
    "slug",
    ["abc", "a" * 64, "a_b", "123", "___", "feed_01"],
)
def test_valid_slugs(slug: str):
    assert is_valid_slug(slug)
    assert validate_slug(slug) == slug


@pytest.mark.parametrize(
    "slug",
    [
        "",
        "ab",
        "a" * 65,
        "ABC",
        "abC",
        "a-b",
        "a b",
        " abc",
        "abc ",
        "abc\n",  # `$` would accept a trailing newline; fullmatch must not
        "\nabc",
        "café",
        "ａｂｃ",  # fullwidth latin letters
        "١٢٣",  # arabic-indic digits
        "a.b",
    ],
)
def test_invalid_slugs(slug: str):
    assert not is_valid_slug(slug)
    with pytest.raises(ValidationError, match="slug"):
        validate_slug(slug)


@pytest.mark.parametrize("bad", [None, 123, b"abc", ["abc"]])
def test_non_string_is_invalid_not_a_crash(bad):
    assert not is_valid_slug(bad)
    with pytest.raises(ValidationError):
        validate_slug(bad)


def test_bounds_constants():
    assert (SLUG_MIN, SLUG_MAX) == (3, 64)


def test_one_megabyte_input_is_rejected_quickly():
    start = time.perf_counter()
    assert not is_valid_slug("a" * 1_000_000)
    assert not is_valid_slug("a" * 1_000_000 + "-")
    assert time.perf_counter() - start < 1.0
