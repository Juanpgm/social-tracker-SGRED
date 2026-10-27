"""From parsed feed entries to normalized items (CI-2, CI-8, CI-9)."""

import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

from social_tracker.collector.feed import FeedEntry
from social_tracker.collector.normalize import (
    NormalizeReport,
    normalize_entries,
    strip_html,
)
from social_tracker.items.domain import (
    MAX_RAW_PAYLOAD_BYTES,
    MAX_TEXT_BYTES,
    ItemKind,
    ItemStatus,
    content_hash,
    item_id,
)

FETCHED = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def entry(**overrides) -> FeedEntry:
    fields = {
        "title": "Sismo en Cali",
        "link": "https://example.org/1",
        "guid": "g-1",
        "published": "Sun, 01 Mar 2026 08:00:00 GMT",
        "body_html": "<p>Un <b>temblor</b> sacudio la ciudad.</p>",
        "author": "Ana",
        "raw": {"guid": "g-1", "description": "<p>Un <b>temblor</b> sacudio la ciudad.</p>"},
    }
    fields.update(overrides)
    return FeedEntry(**fields)


def normalize_one(**overrides) -> NormalizeReport:
    return normalize_entries(
        [entry(**overrides)], source_id="src_news", source_name="City News", fetched_at=FETCHED
    )


def only_item(**overrides):
    report = normalize_one(**overrides)
    assert len(report.items) == 1, report
    return report.items[0]


# ---- the normalized record ----


def test_an_entry_becomes_a_full_normalized_item():
    item = only_item()
    assert item.id == item_id("src_news", "g-1")
    assert (item.source_id, item.source_name) == ("src_news", "City News")
    assert (item.kind, item.status) == (ItemKind.ARTICLE, ItemStatus.NEW)
    assert item.external_id == "g-1"
    assert item.url == "https://example.org/1"
    assert item.title == "Sismo en Cali"
    assert item.text == "Un temblor sacudio la ciudad."
    assert item.author == "Ana"
    assert item.language is None
    assert item.published_at == datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
    assert item.fetched_at == FETCHED
    assert item.content_hash == content_hash(item.title, item.text)
    assert item.flags == ()
    assert item.raw_payload["description"] == "<p>Un <b>temblor</b> sacudio la ciudad.</p>"


def test_the_report_counts_what_was_discarded_and_keeps_the_order():
    entries = [
        entry(guid="a", title="first", body_html="one"),
        entry(guid="b", title="  ", body_html="<p>&nbsp;</p>"),
        entry(guid="c", title="third", body_html="three"),
        entry(guid="d", title="", body_html=""),
    ]
    report = normalize_entries(
        entries, source_id="src_news", source_name="City News", fetched_at=FETCHED, over_limit=7
    )
    assert [i.external_id for i in report.items] == ["a", "c"]
    assert report.discarded == {"empty": 2, "over_limit": 7}


def test_nothing_discarded_gives_an_empty_counter():
    assert normalize_one().discarded == {}
    assert normalize_entries([], source_id="src_news", source_name="N", fetched_at=FETCHED) == (
        NormalizeReport(items=(), discarded={})
    )


# ---- html is turned into inert text ----


@pytest.mark.parametrize(
    ("html", "text"),
    [
        ("<p>one</p><p>two</p>", "one\n\ntwo"),
        ("line<br>break<br/>again", "line\nbreak\nagain"),
        ("<ul><li>a</li><li>b</li></ul>", "a\nb"),
        ("a &amp; b &lt; c &gt; d &quot;e&quot; &#233; &eacute;", 'a & b < c > d "e" é é'),
        ("<script>alert(1)</script>visible", "visible"),
        ("<style>p{color:red}</style>visible", "visible"),
        ('<img src=x onerror="alert(1)">after', "after"),
        ('<a href="javascript:alert(1)">click</a>', "click"),
        ("<iframe src='http://evil.invalid'>frame</iframe>text", "text"),
        ("<!-- hidden comment -->shown", "shown"),
        ("<![CDATA[ x ]]>after", "after"),
        ("<p>unclosed <b>bold", "unclosed bold"),
        ("2 < 3 and 5 > 4", "2 < 3 and 5 > 4"),
        ("plain text, no markup", "plain text, no markup"),
        ("<p>café \U0001f30b</p>", "café \U0001f30b"),
        ("", ""),
    ],
)
def test_strip_html_keeps_only_visible_text(html, text):
    assert strip_html(html) == text


def test_script_content_never_reaches_the_item_but_stays_in_the_raw_payload():
    raw = "<p>safe</p><script>steal()</script>"
    item = only_item(body_html=raw, raw={"description": raw})
    assert item.text == "safe"
    assert "steal" not in item.text + item.title
    assert item.raw_payload["description"] == raw


def test_html_in_the_title_is_stripped_and_kept_on_one_line():
    item = only_item(title="<b>Alerta</b>\n  en <i>Cali</i> &amp; Yumbo")
    assert item.title == "Alerta en Cali & Yumbo"


def test_broken_nested_script_markup_never_leaks_script_content():
    assert "alert" not in strip_html("<<script>script>alert(1)<</script>/script>")


def test_pathological_html_is_bounded_in_size_and_time():
    nasty = "<a " * 200_000 + "<!--" * 100_000
    started = time.monotonic()
    result = strip_html(nasty)
    assert len(result) <= len(nasty)
    assert time.monotonic() - started < 5


# ---- ids ----


def test_the_guid_is_the_external_id():
    assert only_item(guid="urn:uuid:42").external_id == "urn:uuid:42"


def test_a_missing_guid_falls_back_to_a_stable_hash_of_link_and_title():
    first = only_item(guid=None)
    again = only_item(guid=None)
    assert first.external_id.startswith("h:")
    assert first.external_id == again.external_id
    assert "external_id_synthesized" in first.flags
    assert only_item(guid=None, link="https://example.org/other").external_id != first.external_id
    assert only_item(guid=None, title="Another title").external_id != first.external_id


def test_the_fallback_id_survives_without_link_and_title():
    item = only_item(guid=None, link=None, title="", body_html="just a body")
    assert item.external_id.startswith("h:")
    assert only_item(guid=None, link=None, title="", body_html="other body").external_id != (
        item.external_id
    )


@pytest.mark.parametrize("guid", ["   ", ""])
def test_a_blank_guid_counts_as_missing(guid):
    assert "external_id_synthesized" in only_item(guid=guid).flags


def test_an_oversize_guid_is_replaced_by_its_hash():
    item = only_item(guid="g" * 600)
    assert item.external_id.startswith("h:")
    assert len(item.external_id) <= 512
    assert only_item(guid="g" * 600).external_id == item.external_id
    assert only_item(guid="g" * 599).external_id != item.external_id


# ---- dates ----


@pytest.mark.parametrize(
    ("published", "expected"),
    [
        ("Sun, 01 Mar 2026 08:00:00 GMT", datetime(2026, 3, 1, 8, 0, tzinfo=UTC)),
        ("Sun, 01 Mar 2026 03:00:00 -0500", datetime(2026, 3, 1, 8, 0, tzinfo=UTC)),
        ("Sun, 01 Mar 2026 08:00:00 +0000", datetime(2026, 3, 1, 8, 0, tzinfo=UTC)),
        ("2026-03-01T08:00:00Z", datetime(2026, 3, 1, 8, 0, tzinfo=UTC)),
        ("2026-03-01T03:00:00-05:00", datetime(2026, 3, 1, 8, 0, tzinfo=UTC)),
        ("2026-03-01T08:00:00", datetime(2026, 3, 1, 8, 0, tzinfo=UTC)),  # naive means UTC
        ("2026-03-01T08:00:00.250Z", datetime(2026, 3, 1, 8, 0, 0, 250000, tzinfo=UTC)),
    ],
)
def test_publication_dates_in_feed_formats_become_utc(published, expected):
    item = only_item(published=published)
    assert item.published_at == expected
    assert "published_at_missing" not in item.flags


@pytest.mark.parametrize("published", [None, "", "   ", "yesterday", "32 Foo 2026", "0000-00-00"])
def test_a_missing_or_unreadable_date_falls_back_to_the_fetch_time_and_is_flagged(published):
    item = only_item(published=published)
    assert item.published_at == FETCHED
    assert "published_at_missing" in item.flags


def test_a_date_in_the_future_is_clamped_to_the_fetch_time_and_flagged():
    later = (FETCHED + timedelta(days=3)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    item = only_item(published=later)
    assert item.published_at == FETCHED
    assert "published_at_future" in item.flags
    soon = only_item(published=FETCHED.astimezone(timezone(timedelta(hours=-5))).isoformat())
    assert soon.published_at == FETCHED
    assert "published_at_future" not in soon.flags


# ---- empty and missing content ----


@pytest.mark.parametrize(
    ("title", "body"),
    [("", ""), ("   ", "\n\t "), ("", "<p></p>"), ("<b></b>", "<br><br>"), ("", "&nbsp;&#160;")],
)
def test_items_without_visible_content_are_discarded_and_counted(title, body):
    report = normalize_one(title=title, body_html=body)
    assert report.items == ()
    assert report.discarded == {"empty": 1}


def test_an_entry_with_only_a_title_or_only_a_body_is_kept():
    assert only_item(title="Just a headline", body_html="").text == ""
    body_only = only_item(title="", body_html="Only a body")
    assert (body_only.title, body_only.text) == ("", "Only a body")


# ---- limits ----


def test_oversize_text_is_cut_at_the_byte_limit_and_flagged_deterministically():
    body = "a" * (MAX_TEXT_BYTES + 5000)
    item = only_item(body_html=body)
    assert len(item.text.encode()) == MAX_TEXT_BYTES
    assert "truncated" in item.flags
    assert only_item(body_html=body).text == item.text
    assert item.content_hash == content_hash(item.title, item.text)


def test_text_exactly_at_the_limit_is_not_truncated():
    item = only_item(body_html="a" * MAX_TEXT_BYTES)
    assert len(item.text) == MAX_TEXT_BYTES
    assert "truncated" not in item.flags


def test_truncation_never_splits_a_character():
    item = only_item(body_html="\U0001f30b" * (MAX_TEXT_BYTES // 4 + 100))
    assert item.text == "\U0001f30b" * (MAX_TEXT_BYTES // 4)
    assert "truncated" in item.flags


def test_an_oversize_title_is_cut_and_flagged():
    item = only_item(title="t" * 1500)
    assert item.title == "t" * 1000
    assert "truncated" in item.flags


def test_an_oversize_raw_payload_is_shrunk_and_flagged():
    item = only_item(raw={"guid": "g-1", "description": "d" * 400_000})
    assert len(item.raw_payload["description"]) < 400_000
    assert item.raw_payload["guid"] == "g-1"
    assert "raw_truncated" in item.flags
    assert len(str(item.raw_payload).encode()) <= MAX_RAW_PAYLOAD_BYTES


# ---- text fidelity ----


def test_accents_and_emoji_survive_normalization():
    item = only_item(
        title="Sísmo \U0001f30b en Cali",
        body_html="<p>Ñandú, canción, \U0001f468‍\U0001f469‍\U0001f467</p>",
    )
    assert item.title == "Sísmo \U0001f30b en Cali"
    assert item.text == "Ñandú, canción, \U0001f468‍\U0001f469‍\U0001f467"


def test_decomposed_accents_are_composed():
    assert only_item(title="Café").title == "Café"


# ---- link and author ----


@pytest.mark.parametrize(
    "link",
    [
        None,
        "",
        "javascript:alert(1)",
        "data:text/html,<script>1</script>",
        "ftp://example.org/x",
        "file:///etc/passwd",
        "https://" + "user:secret" + "@example.org/x",
        "not a url",
        "https://example.org/" + "a" * 3000,
    ],
)
def test_an_unusable_or_dangerous_link_is_dropped_but_the_item_is_kept(link):
    item = only_item(link=link)
    assert item.url is None
    assert item.title == "Sismo en Cali"


def test_a_link_with_surrounding_whitespace_is_trimmed():
    assert only_item(link="  https://example.org/a  ").url == "https://example.org/a"


def test_the_author_is_one_clean_line_and_optional():
    assert only_item(author="  Ana \n Pérez  ").author == "Ana Pérez"
    assert only_item(author=None).author is None
    assert only_item(author="  ").author is None
    assert len(only_item(author="x" * 500).author) == 200
