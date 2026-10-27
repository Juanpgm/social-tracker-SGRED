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


@pytest.mark.parametrize("filler", ["\u3164", "\u2800", "\u115f", "\u1160", "\uffa0", "\u17b4"])
def test_blank_letters_inside_a_word_do_not_split_it(filler):
    assert fold(f"tem{filler}blor") == "temblor"
    assert fold(f"{filler}sismo{filler}") == "sismo"


def test_fold_stays_idempotent_with_the_skeleton():
    for text in ("ѕіѕmо ΤΕΜΒLΟR", "t\u3164e\u2800mblor", "SÍSMO ｓｉｓｍｏ"):
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
        ("sismo\u3164 fuerte", "invisible"),
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
    chunk = "t.e.m.b.l.o.r ѕіѕmо t3mblor \u3164\u2800 " + "x " * 200 + "!" * 50 + " "
    hostile = chunk * (1_000_000 // len(chunk) + 1)
    started = time.perf_counter()
    prepared = prepare("", hostile, max_chars=20_000)
    assert time.perf_counter() - started < 0.5
    assert prepared.truncated is True


# ---- the edges of the decoding ----


def test_a_word_spelled_with_exactly_the_longest_joinable_run_is_joined_and_one_more_is_not():
    longest = "a b c d e f g h i j k l m n o"  # 15 letters
    joined = prepare("", longest, max_chars=1000)
    assert joined.decoded_tokens == ("abcdefghijklmno",)
    too_long = prepare("", longest + " p", max_chars=1000)
    assert too_long.decoded_tokens == tuple("abcdefghijklmnop")  # left as it is...
    assert "separators" in too_long.obfuscation  # ...but still reported


def test_a_run_of_exactly_the_minimum_length_is_joined_and_one_fewer_is_not():
    assert prepare("", "s i s m", max_chars=100).decoded_tokens == ("sism",)
    assert prepare("", "s i s", max_chars=100).decoded_tokens == ("s", "i", "s")


@pytest.mark.parametrize(
    ("word", "decoded"),
    [("m4s", "m4s"), ("m4sa", "masa"), ("a1", "a1"), ("t3mb", "temb"), ("abc11d", "abc11d")],
)
def test_digits_are_read_as_letters_only_in_a_word_with_at_least_three_letters(word, decoded):
    assert prepare("", word, max_chars=100).decoded_tokens == (decoded,)


def test_only_a_digit_between_two_letters_is_reported_as_obfuscation():
    assert "leet" in prepare("", "t3mblor", max_chars=100).obfuscation
    ending = prepare("", "sism0", max_chars=100)
    assert ending.decoded_tokens == ("sismo",)  # still decoded for matching...
    assert "leet" not in ending.obfuscation  # ...but a digit at the end is not a disguise


def test_separate_digits_are_not_a_spelled_word():
    prepared = prepare("", "magnitud 4 5 6 7 8", max_chars=100)
    assert prepared.decoded_tokens == prepared.tokens
    assert prepared.obfuscation == ()


def test_a_short_word_made_only_of_lookalikes_is_not_reported_but_a_longer_one_is():
    assert prepare("", "\u0441\u043e\u0440", max_chars=100).obfuscation == ()  # 3 letters
    assert "mixed_script" in prepare("", "\u0441\u043e\u0440\u0441", max_chars=100).obfuscation
