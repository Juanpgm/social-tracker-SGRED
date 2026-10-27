"""Shared text helpers."""

import pytest

from social_tracker.shared.text import (
    escape_unprintable,
    has_control_chars,
    has_unsafe_chars,
    is_blank_text,
)


@pytest.mark.parametrize("code", [*range(0, 32), 127, *range(128, 160)])
def test_every_c0_c1_control_and_delete_is_detected(code: int):
    char = chr(code)
    assert has_control_chars(char)
    assert has_control_chars(f"a{char}b")
    assert has_control_chars(f"{char}ab")
    assert has_control_chars(f"ab{char}")


@pytest.mark.parametrize("code", [32, 33, 126, 160, 161, 173, 0x2028, 0x200B, 0xFEFF])
def test_printable_boundaries_are_not_control_characters(code: int):
    # 160 (NBSP) and 161 sit right after the C1 block; the rest are format/separator characters.
    assert not has_control_chars(chr(code))


@pytest.mark.parametrize("text", ["", "plain", "café 新聞 \U0001f4f0", "a b"])
def test_ordinary_text_has_no_control_characters(text: str):
    assert not has_control_chars(text)


# ---- hidden characters: lone surrogates, format characters (Cf), line/paragraph separators ----

UNSAFE_CHARS = [
    "\ud800",  # lone high surrogate
    "\udfff",  # lone low surrogate
    "\u200b",  # zero width space
    "\u200c",
    "\u200d",  # zero width joiner
    "\u202e",  # right-to-left override
    "\u2066",  # bidi isolate
    "\u2028",  # line separator (Zl)
    "\u2029",  # paragraph separator (Zp)
    "\ufeff",  # byte order mark / zero width no-break space
    "\u00ad",  # soft hyphen
    "\U000e0001",  # language tag
]


@pytest.mark.parametrize("char", UNSAFE_CHARS)
def test_hidden_characters_are_unsafe_anywhere(char: str):
    assert has_unsafe_chars(char)
    assert has_unsafe_chars(f"ab{char}")
    assert has_unsafe_chars(f"{char}ab")
    assert has_unsafe_chars(f"a{char}b")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain",
        "Mañana, ¿cómo estás? Él dijo: «sí»",
        "cafe\u0301",  # combining acute accent (Mn) stays allowed
        "\u0928\u092e\u0938\u094d\u0924\u0947",  # Devanagari with combining signs
        "noticias \U0001f4f0 \U0001f600",
        "a\u00a0b",
        "\ufe0f",  # variation selector-16 is a nonspacing mark (Mn)
        "\u2027",  # hyphenation point (Po): next to the separators but harmless
    ],
)
def test_ordinary_text_marks_and_emoji_are_not_unsafe(text: str):
    assert not has_unsafe_chars(text)


def test_a_hidden_character_after_a_combining_sequence_is_still_found():
    assert not has_unsafe_chars("e\u0301")
    assert has_unsafe_chars("e\u0301\u200b")


@pytest.mark.parametrize(
    "text", ["", " ", "\t \n", "\u200b", "\u200b \u202e \ufeff", "\u00a0", "\u2028"]
)
def test_text_with_nothing_visible_is_blank_even_when_it_has_format_characters(text: str):
    assert is_blank_text(text)


@pytest.mark.parametrize("text", ["a", " a ", "\u200ba\u200b", "\u00e9", "\U0001f600", "e\u0301"])
def test_text_with_a_visible_character_is_not_blank(text: str):
    assert not is_blank_text(text)


# ---- escaping text for a terminal ----


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ("\x1b[31mred", "\\x1b[31mred"),
        ("a\nb", "a\\x0ab"),
        ("a\rb", "a\\x0db"),
        ("a\x00b", "a\\x00b"),
        ("a\x7fb", "a\\x7fb"),
        ("\x9b31m", "\\x9b31m"),  # a C1 control sequence introducer
        ("a\u202eb", "a\\u202eb"),  # bidi override
        ("a\u2028b", "a\\u2028b"),
        ("a\ufeffb", "a\\ufeffb"),
        ("a\ud800b", "a\\ud800b"),  # lone surrogate
        ("\U000e0041", "\\U000e0041"),  # tag character
    ],
)
def test_control_and_hidden_characters_are_shown_as_escapes(raw: str, escaped: str):
    assert escape_unprintable(raw) == escaped


@pytest.mark.parametrize(
    "text", ["", "plain/path.db", "C:\\Users\\x\\a.db", "café 新聞 \U0001f4f0", "a b", "e\u0301"]
)
def test_ordinary_text_is_left_alone(text: str):
    assert escape_unprintable(text) == text


def test_escaped_text_contains_nothing_a_terminal_would_act_on():
    escaped = escape_unprintable("".join(chr(code) for code in range(160)))
    assert escaped.isascii()
    assert not has_control_chars(escaped)
