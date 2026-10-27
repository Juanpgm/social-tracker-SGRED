"""Evasion by spelling and by padding: what the text views make of hostile input (S5, S6)."""

import time

import pytest

from social_tracker.classification.text import fold, prepare


def seen(text: str) -> set[str]:
    """The words of the plain and of the decoded view."""
    prepared = prepare("", text, max_chars=20_000)
    return set(prepared.tokens) | set(prepared.decoded_tokens)


# ---- the confusables skeleton ----


@pytest.mark.parametrize(
    "text",
    [
        "ѕіѕmо",  # Cyrillic s, i, s, o: only the m is Latin
        "sіsmo",  # one Cyrillic i
        "tеmblor",  # Cyrillic e
        "tеmblоr",  # Cyrillic e and o
        "ΤΕΜΒLΟR",  # Greek capitals mixed in
        "sιsmο",  # Greek iota and omicron
    ],
)
def test_lookalike_letters_fold_to_the_latin_word(text):
    assert fold(text) in {"sismo", "temblor"}


@pytest.mark.parametrize("filler", ["ㅤ", "⠀", "ᅟ", "ᅠ", "ﾠ", "឴"])
def test_blank_letters_inside_a_word_do_not_split_it(filler):
    assert fold(f"tem{filler}blor") == "temblor"
    assert fold(f"{filler}sismo{filler}") == "sismo"


def test_fold_stays_idempotent_with_the_skeleton():
    for text in ("ѕіѕmо ΤΕΜΒLΟR", "tㅤe⠀mblor", "SÍSMO ｓｉｓｍｏ"):
        assert fold(fold(text)) == fold(text)


# ---- separators and digits for letters ----


@pytest.mark.parametrize(
    "text",
    [
        "t.e.m.b.l.o.r",
        "t e m b l o r",
        "t-e-m-b-l-o-r",
        "T_E_M_B_L_O_R",
        "t*e*m*b*l*o*r",
    ],
)
def test_a_word_spelled_letter_by_letter_is_joined_in_the_decoded_view(text):
    prepared = prepare("", text, max_chars=1000)
    assert "temblor" in prepared.decoded_tokens
    assert "separators" in prepared.obfuscation
    assert "temblor" not in prepared.tokens  # the plain view stays faithful


@pytest.mark.parametrize(
    ("text", "word"),
    [("t3mblor", "temblor"), ("s1smo", "sismo"), ("sism0", "sismo"), ("t3rr3m0t0", "terremoto")],
)
def test_digits_for_letters_inside_a_word_are_read_as_letters(text, word):
    assert word in seen(text)


@pytest.mark.parametrize("text", ["covid19", "mp3", "5g", "iphone15", "m4", "sismo 4.2", "3d"])
def test_ordinary_tokens_with_digits_are_not_decoded(text):
    prepared = prepare("", text, max_chars=1000)
    assert prepared.decoded_tokens == prepared.tokens


def test_short_spelled_runs_and_initials_are_not_joined():
    prepared = prepare("", "S O S ayuda", max_chars=1000)
    assert prepared.decoded_tokens == prepared.tokens
    assert prepared.obfuscation == ()


# ---- the obfuscation report ----


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("tеmblor", "mixed_script"),
        ("sismoㅤ fuerte", "invisible"),
        ("t.e.m.b.l.o.r", "separators"),
        ("t3mblor", "leet"),
        ("ѕеѕо", "mixed_script"),  # a whole word made only of lookalikes
    ],
)
def test_the_kind_of_obfuscation_is_reported(text, kind):
    assert kind in prepare("", text, max_chars=1000).obfuscation


@pytest.mark.parametrize(
    "text",
    [
        "Mañana habrá un temblor en Cali",
        "El sismo de magnitud 4.2 sacudió Cali. Привет, мир",  # Russian is not a lookalike word
        "ΑΘΗΝΑ capital de Grecia",  # Greek that is not a Latin lookalike
        "sismo‍👨‍👩 familia",
    ],
)
def test_plain_text_in_any_script_is_not_obfuscation(text):
    assert prepare("", text, max_chars=1000).obfuscation == ()


def test_the_decoded_view_equals_the_plain_one_for_clean_text():
    prepared = prepare("Sismo", "En Cali hoy", max_chars=1000)
    assert prepared.decoded == prepared.plain
    assert len(prepared.views) == 1


def test_both_views_are_offered_when_they_differ():
    prepared = prepare("", "t3mblor", max_chars=1000)
    assert [text for text, _ in prepared.views] == ["t3mblor", "temblor"]


# ---- truncation reads both ends (S6) ----


def test_a_rumor_after_padding_is_still_read():
    text = "x " * 10001 + "Mañana habrá un temblor"
    prepared = prepare("", text, max_chars=20_000)
    assert prepared.truncated is True
    assert "temblor" in prepared.tokens


def test_a_rumor_before_padding_is_still_read():
    text = "Mañana habrá un temblor " + "relleno " * 10_000
    prepared = prepare("", text, max_chars=20_000)
    assert prepared.truncated
    assert "temblor" in prepared.tokens


def test_the_middle_of_a_long_text_is_the_part_that_is_not_read():
    text = "inicio " + "z" * 30_000 + " medio sismo " + "z" * 30_000 + " final"
    prepared = prepare("", text, max_chars=20_000)
    assert prepared.truncated is True
    assert {"inicio", "final"} <= set(prepared.tokens)
    assert "sismo" not in prepared.tokens


def test_the_read_text_never_exceeds_the_bound_plus_the_seam():
    prepared = prepare("", "a" * 100_000, max_chars=1000)
    assert len(prepared.raw) == 1001  # two halves and one newline between them


@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_a_text_exactly_at_the_bound_is_not_truncated(size):
    assert prepare("", "a" * (50 + size), max_chars=50).truncated is (size > 0)


def test_the_seam_is_deterministic_and_keeps_characters_whole():
    text = "😀" * 30
    first = prepare("", text, max_chars=8)
    assert first == prepare("", text, max_chars=8)
    assert first.raw == "😀" * 4 + "\n" + "😀" * 4


# ---- cost ----


def test_a_megabyte_of_adversarial_text_is_prepared_quickly():
    chunk = "t.e.m.b.l.o.r ѕіѕmо t3mblor ㅤ⠀ " + "x " * 200 + "!" * 50 + " "
    hostile = chunk * (1_000_000 // len(chunk) + 1)
    started = time.perf_counter()
    prepared = prepare("", hostile, max_chars=20_000)
    assert time.perf_counter() - started < 0.5
    assert prepared.truncated is True
