"""Item model and pure helpers: normalization, hashing, truncation, retention, search terms."""

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from social_tracker.items.domain import (
    DEFAULT_RETENTION_DAYS,
    MAX_RAW_PAYLOAD_BYTES,
    MAX_TEXT_BYTES,
    Item,
    ItemKind,
    ItemStatus,
    bounded_raw_payload,
    content_hash,
    item_id,
    normalize_text,
    retention_cutoff,
    search_terms,
    truncate_utf8,
    validate_retention_days,
)
from social_tracker.shared.errors import ValidationError

NOW = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def make(**overrides) -> Item:
    fields = {
        "id": item_id("src_news", "guid-1"),
        "source_id": "src_news",
        "source_name": "City News",
        "kind": ItemKind.ARTICLE,
        "external_id": "guid-1",
        "parent_id": None,
        "url": "https://example.org/a",
        "title": "Quake felt in Cali",
        "author": "Ana",
        "text": "A magnitude 4.2 tremor.",
        "language": None,
        "published_at": NOW,
        "fetched_at": NOW + timedelta(minutes=1),
        "content_hash": content_hash("Quake felt in Cali", "A magnitude 4.2 tremor."),
        "flags": (),
        "status": ItemStatus.NEW,
        "raw_payload": {"guid": "guid-1"},
    }
    fields.update(overrides)
    return Item(**fields)


# ---- item ----


def test_a_valid_item_keeps_its_fields_and_normalizes_time_to_utc():
    item = make(published_at=datetime(2026, 3, 1, 3, 0, tzinfo=timezone(timedelta(hours=-5))))
    assert item.published_at == NOW
    assert item.published_at.utcoffset() == timedelta(0)
    assert item.kind is ItemKind.ARTICLE
    assert item.status is ItemStatus.NEW
    assert item.title == "Quake felt in Cali"


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_id": "Bad Id"},
        {"source_name": ""},
        {"source_name": "x" * 101},
        {"kind": "video"},
        {"status": "weird"},
        {"external_id": ""},
        {"external_id": "e" * 513},
        {"url": "javascript:alert(1)"},
        {"url": "ftp://example.org/x"},
        {"url": "https://user:pw@example.org/x"},
        {"url": "https://example.org/" + "a" * 2100},
        {"title": "t" * 1001},
        {"text": "x" * (MAX_TEXT_BYTES + 1)},
        {"text": "é" * (MAX_TEXT_BYTES // 2 + 1)},
        {"content_hash": "abc"},
        {"content_hash": "Z" * 64},
        {"published_at": datetime(2026, 3, 1)},
        {"fetched_at": "2026-03-01"},
        {"flags": ("Bad Flag",)},
        {"raw_payload": ["not", "a", "mapping"]},
        {"language": "x" * 36},
        {"parent_id": ""},
    ],
    ids=lambda o: next(iter(o)) + "-" + str(next(iter(o.values())))[:12],
)
def test_an_invalid_item_is_rejected(overrides):
    with pytest.raises(ValidationError):
        make(**overrides)


def test_text_exactly_at_the_byte_limit_is_accepted():
    assert len(make(text="a" * MAX_TEXT_BYTES).text) == MAX_TEXT_BYTES


def test_item_ids_are_deterministic_per_source_and_external_id():
    assert item_id("src_a", "g1") == item_id("src_a", "g1")
    assert item_id("src_a", "g1") != item_id("src_b", "g1")
    assert item_id("src_a", "g1") != item_id("src_a", "g2")
    with pytest.raises(ValidationError):  # a separator inside the source id would be ambiguous
        item_id("src_a\0ab", "c")


# ---- normalize_text ----


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  hello   world \t again ", "hello world again"),
        ("line one\r\nline two", "line one\nline two"),
        ("a\n\n\n\n\nb", "a\n\nb"),
        ("nul\x00and\x07bell", "nul and bell"),
        ("café", "café"),
        ("tab\tseparated nbsp", "tab separated nbsp"),
        ("sismo \U0001f30b en Cali áéí", "sismo \U0001f30b en Cali áéí"),
        (
            "family \U0001f468‍\U0001f469‍\U0001f467",
            "family \U0001f468‍\U0001f469‍\U0001f467",
        ),
        ("   \n \t  ", ""),
        ("", ""),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent():
    once = normalize_text("  a\r\n\r\n\r\n\r\nb\x00 c  ")
    assert once == "a\n\nb c"
    assert normalize_text(once) == once


# ---- truncate_utf8 ----


def test_truncate_leaves_text_within_the_limit_alone():
    assert truncate_utf8("hello", 10) == ("hello", False)
    assert truncate_utf8("hello", 5) == ("hello", False)


def test_truncate_cuts_at_the_byte_limit():
    assert truncate_utf8("hello world", 5) == ("hello", True)


def test_truncate_never_splits_a_multibyte_character():
    assert truncate_utf8("aé", 2) == ("a", True)
    assert truncate_utf8("a\U0001f30b", 4) == ("a", True)
    cut, truncated = truncate_utf8("\U0001f30b" * 10, 10)
    assert truncated is True
    assert cut == "\U0001f30b" * 2


# ---- content_hash ----


def test_hash_ignores_case_whitespace_and_unicode_form():
    base = content_hash("Quake felt in Cali", "A magnitude 4.2 tremor.")
    assert content_hash("  quake  FELT in cali", "a magnitude\n4.2   tremor.") == base
    assert content_hash("café", "x") == content_hash("café", "x")


def test_hash_differs_for_different_content_and_separates_title_from_text():
    assert content_hash("a", "b") != content_hash("a", "c")
    assert content_hash("ab", "") != content_hash("a", "b")
    assert len(content_hash("a", "b")) == 64


# ---- bounded raw payload ----


def test_a_small_payload_is_kept_as_is():
    payload = {"guid": "g", "description": "<p>hi</p>"}
    assert bounded_raw_payload(payload) == (payload, False)


def test_an_oversize_payload_is_shrunk_deterministically_to_fit():
    payload = {"guid": "g", "description": "x" * 500_000, "title": "t" * 10}
    shrunk, truncated = bounded_raw_payload(payload)
    assert truncated is True
    assert len(json.dumps(shrunk, ensure_ascii=False).encode()) <= MAX_RAW_PAYLOAD_BYTES
    assert shrunk["guid"] == "g"
    assert shrunk["title"] == "t" * 10
    assert shrunk["description"].startswith("xxx")
    assert bounded_raw_payload(payload) == (shrunk, True)


def test_multibyte_payloads_are_measured_in_bytes():
    shrunk, truncated = bounded_raw_payload({"description": "\U0001f30b" * 100_000})
    assert truncated is True
    assert len(json.dumps(shrunk, ensure_ascii=False).encode()) <= MAX_RAW_PAYLOAD_BYTES


# ---- retention ----


def test_the_default_retention_is_thirty_days():
    assert DEFAULT_RETENTION_DAYS == 30


def test_cutoff_is_now_minus_the_retention_days():
    assert retention_cutoff(NOW, 30) == NOW - timedelta(days=30)
    assert retention_cutoff(NOW, 1) == NOW - timedelta(days=1)


@pytest.mark.parametrize("days", [0, 1, 30, 3650])
def test_valid_retention_days(days):
    assert validate_retention_days(days) == days


@pytest.mark.parametrize("days", [-1, 3651, 1.5, "30", None, True])
def test_invalid_retention_days(days):
    with pytest.raises(ValidationError):
        validate_retention_days(days)


def test_a_zero_retention_has_no_cutoff():
    with pytest.raises(ValidationError):
        retention_cutoff(NOW, 0)


# ---- search terms ----


@pytest.mark.parametrize(
    ("query", "terms"),
    [
        ("sismo Cali", ("sismo", "cali")),
        ("  SISMO   ", ("sismo",)),
        ("terremoto: magnitud 6.5", ("terremoto", "magnitud", "6", "5")),
        ('"quoted phrase"', ("quoted", "phrase")),
        ("a AND b OR NOT c NEAR/3 d", ("a", "and", "b", "or", "not", "c", "near", "3", "d")),
        ("pre* -minus col:umn (grouped)", ("pre", "minus", "col", "umn", "grouped")),
        ("Árbol ñandú", ("arbol", "nandu")),
        ("snake_case", ("snake", "case")),
        ("", ()),
        ("   ", ()),
        ('"" * - ^ ( ) :', ()),
        ("\U0001f30b", ()),
    ],
)
def test_search_terms_reduce_any_text_to_plain_lowercase_words(query, terms):
    assert search_terms(query) == terms


def test_search_terms_are_bounded():
    many = " ".join(f"w{i}" for i in range(100))
    assert len(search_terms(many)) == 16
    assert search_terms("x" * 500) == ("x" * 64,)


@pytest.mark.parametrize("bad", [None, 5, b"bytes", ["a"]])
def test_search_terms_reject_non_text(bad):
    with pytest.raises(ValidationError):
        search_terms(bad)
