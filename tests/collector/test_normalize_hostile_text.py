"""Hostile text in entries: invisible and bidi characters, and markup that hides the rest.

The stored title, text and author carry no bidi control and no zero-width character (they could
reorder what a reader sees, spoof a link text or split a keyword so no search finds it); the entry
exactly as received stays in `raw_payload`. An element that swallows the rest of the entry (an
unclosed `<script>`) is not silent: the item says so with the flag `content_dropped`.
"""

from datetime import UTC, datetime

import pytest

from social_tracker.collector.feed import FeedEntry
from social_tracker.collector.normalize import (
    CONTENT_DROPPED_CHARS,
    normalize_entries,
    strip_html_report,
)
from social_tracker.items.domain import neutralize_invisible

FETCHED = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)

BIDI = [
    "‪",
    "‫",
    "‬",
    "‭",
    "‮",
    "⁦",
    "⁧",
    "⁨",
    "⁩",
    "‎",
    "‏",
    "؜",
]
ZERO_WIDTH = ["​", "⁠", "﻿", "­", "᠎", "⁡", "⁤"]


def one(**fields) -> object:
    entry = FeedEntry(
        title=fields.get("title", "Title"),
        link=None,
        guid="g",
        published="Sun, 01 Mar 2026 08:00:00 GMT",
        body_html=fields.get("body_html", "Body"),
        author=fields.get("author"),
        raw={"guid": "g", "title": fields.get("title", "Title"), "description": "raw"},
    )
    report = normalize_entries([entry], source_id="src_news", source_name="N", fetched_at=FETCHED)
    assert len(report.items) == 1, report
    return report.items[0]


# ---- bidi and zero-width characters ----


@pytest.mark.parametrize("char", BIDI + ZERO_WIDTH, ids=lambda c: f"U+{ord(c):04X}")
def test_invisible_characters_are_removed_from_title_text_and_author(char):
    item = one(title=f"sis{char}mo", body_html=f"<p>ha{char}lo</p>", author=f"An{char}a")
    assert (item.title, item.text, item.author) == ("sismo", "halo", "Ana")


def test_a_right_to_left_override_cannot_reorder_what_is_stored():
    item = one(title="Invoice ‮gnp.exe", body_html="‮⁦txt⁩")
    assert item.title == "Invoice gnp.exe"
    assert item.text == "txt"


def test_the_original_entry_is_kept_untouched_in_the_raw_payload():
    item = one(title="sis​mo ‮")
    assert item.raw_payload["title"] == "sis​mo ‮"


def test_the_content_hash_ignores_the_hidden_characters_so_obfuscated_copies_dedup():
    plain, hidden = one(title="Sismo", body_html="en Cali"), one(title="Sis​mo", body_html="en‮ Cali")
    assert plain.content_hash == hidden.content_hash


@pytest.mark.parametrize("joiner", ["‌", "‍"], ids=["ZWNJ", "ZWJ"])
def test_a_joiner_between_two_latin_letters_is_removed(joiner):
    assert one(title=f"sis{joiner}mo").title == "sismo"


@pytest.mark.parametrize(
    "text",
    [
        "family \U0001f468‍\U0001f469‍\U0001f467",  # emoji ZWJ sequence
        "\U0001f3f3️‍\U0001f308",  # rainbow flag
        "می‌خواهم",  # Persian: ZWNJ between letters of the script
        "क्‍ष",  # Devanagari: ZWJ in a conjunct
    ],
)
def test_joiners_that_belong_to_the_script_or_to_an_emoji_are_kept(text):
    assert one(title=text).title == text


@pytest.mark.parametrize(
    "text",
    ["x\u200d\u65e5\u672c", "\u65e5\u200dx", "1\u200d\U0001f468", "\U0001f468\u200cz"],
    ids=["latin-cjk", "cjk-latin", "digit-emoji", "emoji-latin"],
)
def test_a_joiner_with_only_one_ascii_neighbour_is_kept(text):
    assert one(title=text).title == text


def test_a_run_of_joiners_counts_once():
    assert one(title="\U0001f468\u200d\u200d\u200c\U0001f469").title == (
        "\U0001f468\u200d\U0001f469"
    )
    assert neutralize_invisible("a\u200d\u200d\u200db") == "ab"


def test_joiners_at_the_edges_or_next_to_spaces_are_removed():
    assert one(title="‍abc‌").title == "abc"
    assert one(title="a ‍ b").title == "a b"
    assert one(title="1‍2").title == "12"


def test_ordinary_text_accents_and_emoji_are_untouched():
    text = "Ñandú é \U0001f30b 日本語 العربية"
    assert one(title=text, body_html=text, author="Ana Pérez").text == text


def test_neutralize_invisible_is_idempotent_and_total():
    hostile = "a‮​b‍c" + "".join(BIDI + ZERO_WIDTH)
    once = neutralize_invisible(hostile)
    assert once == "abc"
    assert neutralize_invisible(once) == once
    assert neutralize_invisible("") == ""


def test_an_entry_made_only_of_hidden_characters_is_empty():
    entry = FeedEntry(
        title="​‮", link=None, guid="g", published=None, body_html="﻿", author=None, raw={}
    )
    report = normalize_entries([entry], source_id="src_news", source_name="N", fetched_at=FETCHED)
    assert report.items == ()
    assert report.discarded == {"empty": 1}


# ---- markup that swallows the rest ----


def test_an_unclosed_script_marks_the_item_and_keeps_what_came_before_it():
    item = one(body_html="<p>Real news</p><script>" + "var x = 1;" * 5)
    assert item.text == "Real news"
    assert "content_dropped" in item.flags


@pytest.mark.parametrize("tag", ["script", "style", "noscript", "template", "iframe", "object"])
def test_every_dropped_element_left_open_is_flagged(tag):
    item = one(body_html=f"before <{tag}>hidden after")
    assert item.text == "before"
    assert "content_dropped" in item.flags


def test_a_properly_closed_script_is_dropped_without_a_flag_when_it_is_small():
    item = one(body_html="a<script>alert(1)</script>b")
    assert item.text == "ab"
    assert "content_dropped" not in item.flags


@pytest.mark.parametrize(
    ("size", "flagged"),
    [
        (CONTENT_DROPPED_CHARS - 1, False),
        (CONTENT_DROPPED_CHARS, True),
        (CONTENT_DROPPED_CHARS + 1, True),
    ],
)
def test_a_closed_element_flags_the_item_once_it_dropped_enough(size, flagged):
    item = one(body_html="keep<style>" + "x" * size + "</style> this")
    assert item.text == "keep this"
    assert ("content_dropped" in item.flags) is flagged


def test_a_flagged_title_flags_the_item_too():
    item = one(title="Headline<script>" + "x" * 10, body_html="Body")
    assert item.title == "Headline"
    assert "content_dropped" in item.flags


def test_nested_dropped_elements_are_counted_and_balanced():
    item = one(body_html="a<script>1<style>2</style>3</script>b")
    assert item.text == "ab"
    assert "content_dropped" not in item.flags


def test_a_stray_closing_tag_does_not_flag_anything():
    item = one(body_html="a</script>b</style>")
    assert item.text == "ab"
    assert "content_dropped" not in item.flags


def test_the_report_counts_what_was_dropped_and_whether_an_element_was_left_open():
    text, dropped, unclosed = strip_html_report("ok<script>12345</script>fine<style>abc")
    assert (text, dropped, unclosed) == ("okfine", 8, True)
    assert strip_html_report("plain text") == ("plain text", 0, False)
    assert strip_html_report("") == ("", 0, False)


def test_the_flag_does_not_depend_on_the_text_that_is_left():
    item = one(body_html="<script>" + "x" * 10)  # nothing visible remains, the title does
    assert item.title == "Title"
    assert "content_dropped" in item.flags
