"""Text normalization: how both the topic filter and the rules see an item's text."""

import pytest

from social_tracker.classification.text import fold, prepare, tokens


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SISMO", "sismo"),  # case
        ("Sísmo", "sismo"),  # accent
        ("sísmo", "sismo"),  # decomposed accent (i + combining acute) -- NFC/NFD insensitive
        ("movimiento telúrico", "movimiento telurico"),
        ("¡Sismo!", "sismo"),  # punctuation
        ("sismo,temblor;replica", "sismo temblor replica"),
        ("ＳＩＳＭＯ", "sismo"),  # NFKC: full-width letters
        ("ﬁrme", "firme"),  # NFKC: ligature
        ("sis​mo", "sismo"),  # zero width space hides the word from a naive match
        ("sis‍mo", "sismo"),  # zero width joiner
        ("﻿sismo⁠", "sismo"),  # BOM, word joiner
        ("sis­mo", "sismo"),  # soft hyphen
        ("Straße", "strasse"),  # casefold, not lower
        ("año", "ano"),  # the tilde of n is an accent too: documented conflation
        ("magnitud 4.2", "magnitud 4 2"),
        ("🌋 volcán 🔥", "volcan"),  # emoji are not words
        ("  \n\t ", ""),
        ("", ""),
        ("!!! ??? ...", ""),
        ("snake_case", "snake case"),
    ],
)
def test_fold(raw, expected):
    assert fold(raw) == expected


@pytest.mark.parametrize("raw", ["Sísmo en CALI!", "ＳＩＳＭＯ  ​", "a_b-c", "ñandú 🌋", ""])
def test_fold_is_idempotent(raw):
    assert fold(fold(raw)) == fold(raw)


def test_tokens_split_the_folded_text():
    assert tokens("¡Sismo en Cali!") == ("sismo", "en", "cali")
    assert tokens("") == ()


def test_prepare_joins_title_and_text_and_keeps_the_views_in_step():
    prepared = prepare("Sismo", "En CALI hoy", max_chars=100)
    assert prepared.plain == "sismo en cali hoy"
    assert prepared.tokens == ("sismo", "en", "cali", "hoy")
    assert "Sismo" in prepared.raw  # the raw view keeps the case
    assert "CALI" in prepared.raw
    assert prepared.truncated is False
    assert prepared.empty is False


def test_prepare_without_a_title_does_not_start_with_a_blank_line():
    assert prepare("", "texto", max_chars=100).raw == "texto"


@pytest.mark.parametrize(("title", "text"), [("", ""), ("  ", "\n"), ("!!", "...")])
def test_prepare_reports_text_without_words_as_empty(title, text):
    assert prepare(title, text, max_chars=100).empty is True


def test_prepare_truncates_at_the_character_bound_and_says_so():
    text = "a" * 50 + " sismo"
    cut = prepare("", text, max_chars=50)
    assert cut.truncated is True
    assert cut.tokens == ("a" * 50,)  # the word after the bound is not seen
    exact = prepare("", "a" * 50, max_chars=50)
    assert exact.truncated is False  # exactly at the bound is not a truncation
    assert prepare("", "a" * 51, max_chars=50).truncated is True


def test_truncation_is_deterministic_and_never_splits_a_character():
    text = "😀" * 30
    first = prepare("", text, max_chars=7)
    assert first == prepare("", text, max_chars=7)
    assert first.raw == "😀" * 7
