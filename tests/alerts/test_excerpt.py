"""The excerpt an alert keeps of an item (AL-11): short, plain, free of control and hidden
characters, and free of anything that looks like a secret."""

import time
import unicodedata

import pytest

from social_tracker.alerts.domain import MAX_EXCERPT_CHARS
from social_tracker.alerts.excerpt import HIDDEN_RE, clean_name, make_excerpt

ELLIPSIS = chr(0x2026)
ZWSP, ZWJ, BOM, SHY = chr(0x200B), chr(0x200D), chr(0xFEFF), chr(0xAD)
RLO, PDF, LRI, PDI = chr(0x202E), chr(0x202C), chr(0x2066), chr(0x2069)
LINE_SEP = chr(0x2028)
SIREN = chr(0x1F6A8)
INVERTED_BANG = chr(0xA1)


# ---- the 500-character bound ----


@pytest.mark.parametrize("length", [0, 1, 499, 500])
def test_text_up_to_the_limit_is_kept_whole(length):
    text = "z" * length
    assert make_excerpt("", text) == text


@pytest.mark.parametrize("length", [501, 502, 1000, 100_000])
def test_longer_text_is_cut_to_the_limit_and_says_so(length):
    excerpt = make_excerpt("", "z" * length)
    assert len(excerpt) == MAX_EXCERPT_CHARS
    assert excerpt.endswith(ELLIPSIS)
    assert excerpt[:-1] == "z" * (MAX_EXCERPT_CHARS - 1)


def test_cutting_is_idempotent():
    once = make_excerpt("", "word " * 400)
    assert make_excerpt("", once) == once


def test_the_title_comes_first_when_there_is_one():
    assert make_excerpt("Quake", "felt in Cali") == "Quake | felt in Cali"
    assert make_excerpt("", "felt in Cali") == "felt in Cali"
    assert make_excerpt("Quake", "") == "Quake"
    assert make_excerpt("  ", "\n\t ") == ""


# ---- control and hidden characters ----


def test_nul_and_other_control_characters_are_removed():
    assert make_excerpt("", "a\x00b\x1b[31mc\x7fd\x85e") == "ab[31mcde"


def test_line_breaks_and_tabs_become_one_space():
    assert make_excerpt("", "one\r\ntwo\n\n\nthree\tfour   five") == "one two three four five"


def test_right_to_left_overrides_and_zero_width_characters_are_removed():
    hostile = f"pre{RLO}evil{PDF} {ZWSP}zero{ZWJ}width{BOM}{LRI}x{PDI}{SHY}soft{LINE_SEP}sep"
    assert make_excerpt("", hostile) == "preevil zerowidthxsoftsep"


def test_a_lone_surrogate_cannot_reach_the_store():
    out = make_excerpt("", "a" + chr(0xD800) + "b")
    assert out == "ab"
    out.encode("utf-8")  # must be encodable


def test_accents_and_emoji_survive():
    text = f"Sismo en Cali: {INVERTED_BANG}fuerte! {SIREN}"
    assert make_excerpt("", text) == text


def test_html_is_kept_as_text_and_escaped_only_when_it_is_drawn():
    assert make_excerpt("", "<script>alert(1)</script> & co") == "<script>alert(1)</script> & co"


def test_a_hostile_megabyte_is_handled_fast_and_bounded():
    start = time.monotonic()
    out = make_excerpt("t" * 1000, (ZWSP * 200_000) + ("x" * 5_000_000))
    assert time.monotonic() - start < 2
    assert len(out) <= MAX_EXCERPT_CHARS
    assert out.startswith("ttt")


def test_many_hidden_characters_before_the_text_do_not_hide_the_text():
    assert make_excerpt("", ZWSP * 5000 + "visible words") == "visible words"


def test_the_hidden_character_class_matches_unicode_exactly():
    kept_as_spaces = {"\t", "\n", "\r", "\x0b", "\x0c"}
    wrong = []
    for code in range(0x110000):
        char = chr(code)
        hidden = unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        expected = hidden and char not in kept_as_spaces
        if bool(HIDDEN_RE.fullmatch(char)) != expected:
            wrong.append(hex(code))
    assert wrong == []


# ---- secrets ----


@pytest.mark.parametrize(
    ("raw", "gone"),
    [
        ("Cookie: sessionid=abc123def; csrftoken=zzz999", "abc123def"),
        ("set-cookie: auth=SECRETVALUE; Path=/", "SECRETVALUE"),
        ("Authorization: " + "Bearer " + "abcdef0123456789", "abcdef0123456789"),
        ("send header Authorization: Basic dXNlcjpwYXNz now", "dXNlcjpwYXNz"),
        ("password=hunter2 and more", "hunter2"),
        ('api_key="K-12345 67890"', "12345"),
        ("access_token: ya29.A0ARrdaM", "ya29"),
        ("token=abc.def.ghi", "abc.def.ghi"),
        ("got " + "Bearer " + "ABCDEFGH12345678" + " today", "ABCDEFGH12345678"),
        ("jwt " + "eyJhbGciOi" + ".eyJ" + "zdWIiOiIx" + ".SflKxwRJSMeKKF2QT4 end", "eyJhbGciOi"),
        ("sid d41d8cd98f00b204e9800998ecf8427e end", "d41d8cd98f00b204e9800998ecf8427e"),
    ],
)
def test_secrets_are_redacted(raw, gone):
    out = make_excerpt("", raw)
    assert gone not in out
    assert "[redacted]" in out


def test_a_link_loses_its_credentials_query_and_fragment():
    raw = (
        "mira https://" + "user:pw" + "@news.example/a/b?token=abc&x=1#frag y http://h.example?q=1"
    )
    out = make_excerpt("", raw)
    assert out == "mira https://news.example/a/b y http://h.example"


def test_ordinary_text_is_not_mistaken_for_a_secret():
    text = "El token de acceso al estadio y la sesion de ayer: nada que ocultar. 123456"
    assert make_excerpt("", text) == text


# ---- the source name snapshot ----


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("City News", "City News"),
        (f"A\x00B{RLO}C", "ABC"),
        ("  padded  ", "padded"),
        ("", "src_fallback"),
        (ZWSP * 2, "src_fallback"),
        ("x" * 300, "x" * 100),
    ],
)
def test_the_source_name_is_cleaned_and_never_empty(name, expected):
    assert clean_name(name, "src_fallback") == expected
