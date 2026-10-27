"""The short how-to per source type (UI-4) and the document it mirrors."""

import re
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from social_tracker.sources.domain import (
    CHUNK_SECONDS_MAX,
    CHUNK_SECONDS_MIN,
    INSTAGRAM_MIN_POLL_S,
    NEWS_RSS_MIN_POLL_S,
    RECORDED_AUDIO_MIN_POLL_S,
    SOURCE_TYPES,
    InstagramSource,
    LiveStreamSource,
    NewsRssSource,
    RecordedAudioSource,
)
from tests.entrypoints.web.conftest import Web, visible_text

DOC = Path(__file__).resolve().parents[3] / "docs" / "howto.md"


def section(page: str, source_type: str) -> str:
    found = re.search(rf'<section id="{source_type}".*?</section>', page, re.DOTALL)
    assert found, source_type
    return visible_text(found.group())


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_every_type_has_what_to_enter_an_example_and_its_limits(web: Web, source_type: str):
    text = section(web.client.get("/howto").text, source_type)
    for heading in ("What to enter", "Example", "Limits"):
        assert heading in text, (source_type, heading)
    assert f'href="/sources/new/{source_type}"' in web.client.get("/howto").text


def test_instagram_is_described_as_best_effort_with_local_credentials(web: Web):
    text = section(web.client.get("/howto").text, "instagram")
    assert "best-effort" in text
    assert "dedicated account" in text
    assert "no Lives" in text
    assert "stay on this machine" in text
    assert "never tracked by git" in text
    assert "env:IG_MAIN" in text  # an example reference, never a credential


def test_the_formats_each_type_expects_are_named(web: Web):
    page = web.client.get("/howto").text
    assert "direct audio stream" in section(page, "live_stream")
    assert "RSS or Atom" in section(page, "news_rss")
    recorded = section(page, "recorded_audio")
    assert "feed" in recorded
    assert "directory" in recorded


def default_of(model: type, field: str) -> Any:
    return model.model_fields[field].default


NEWS_DEFAULT = default_of(NewsRssSource, "poll_interval_s")
INSTAGRAM_DEFAULT = default_of(InstagramSource, "poll_interval_s")
RECORDED_DEFAULT = default_of(RecordedAudioSource, "poll_interval_s")
CHUNK_DEFAULT = default_of(LiveStreamSource, "chunk_seconds")
LANGUAGE_DEFAULT = default_of(LiveStreamSource, "language")

# The whole-number facts each section states: the minimum, the default and nothing else.
EXPECTED_NUMBERS = {
    "news_rss": {NEWS_RSS_MIN_POLL_S, NEWS_DEFAULT},
    "instagram": {INSTAGRAM_MIN_POLL_S, INSTAGRAM_DEFAULT},
    "live_stream": {CHUNK_SECONDS_MIN, CHUNK_SECONDS_MAX, CHUNK_DEFAULT},
    "recorded_audio": {RECORDED_AUDIO_MIN_POLL_S, RECORDED_DEFAULT},
}


def test_the_defaults_come_from_the_domain_and_are_what_the_code_says():
    # If a default changes in the domain these literals fail first, then the how-to follows.
    assert (NEWS_DEFAULT, INSTAGRAM_DEFAULT, RECORDED_DEFAULT, CHUNK_DEFAULT) == (
        600,
        1800,
        1800,
        30,
    )
    assert LANGUAGE_DEFAULT == "es"


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_each_section_states_exactly_the_numbers_the_domain_enforces(web: Web, source_type: str):
    text = section(web.client.get("/howto").text, source_type)
    shown = {int(n) for n in re.findall(r"(?<![\w.])\d+(?![\w.])", text)}
    assert shown == EXPECTED_NUMBERS[source_type], text


def test_the_poll_limits_read_minimum_then_default_with_the_right_numbers(web: Web):
    page = web.client.get("/howto").text
    news = section(page, "news_rss")
    assert f"at most every {NEWS_RSS_MIN_POLL_S} seconds (default {NEWS_DEFAULT})" in news
    insta = section(page, "instagram")
    assert f"at most every {INSTAGRAM_MIN_POLL_S} seconds (default {INSTAGRAM_DEFAULT})" in insta
    recorded = section(page, "recorded_audio")
    assert (
        f"at most every {RECORDED_AUDIO_MIN_POLL_S} second (default {RECORDED_DEFAULT})" in recorded
    )
    live = section(page, "live_stream")
    assert (
        f"chunks of {CHUNK_SECONDS_MIN} to {CHUNK_SECONDS_MAX} seconds (default {CHUNK_DEFAULT})"
        in live
    )
    assert f"The language is a tag such as es (default {LANGUAGE_DEFAULT})" in live


def example_codes(html: str, source_type: str) -> list[str]:
    found = re.search(rf'<section id="{source_type}".*?</section>', html, re.DOTALL)
    assert found
    example = re.search(r"<h3>Example</h3>\s*<p>(.*?)</p>", found.group(), re.DOTALL)
    assert example, source_type
    return [c for c in re.findall(r"<code>(.*?)</code>", example.group(1)) if c]


def form_for(source_type: str, codes: list[str]) -> list[dict[str, str]]:
    """The add form(s) a reader would fill in with the example values, one per alternative."""
    base = {"id": f"howto_{source_type}", "name": f"How-to {source_type}", "credibility": "3"}
    if source_type == "news_rss":
        return [{**base, "feed_url": codes[0]}]
    if source_type == "instagram":
        return [{**base, "handle": codes[0], "secret_ref": codes[1]}]
    if source_type == "live_stream":
        return [{**base, "stream_url": codes[0]}]
    return [
        {
            **base,
            "id": f"howto_{n}",
            "name": f"How-to {n}",
            "feed_url" if c.startswith("http") else "watch_dir": c,
        }
        for n, c in enumerate(codes)
    ]


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_every_example_is_accepted_by_the_domain_through_the_add_form(web: Web, source_type: str):
    codes = example_codes(web.client.get("/howto").text, source_type)
    assert codes, source_type
    for form in form_for(source_type, codes):
        response = web.post(f"/sources/new/{source_type}", form)
        assert response.status_code == 303, (source_type, form, visible_text(response.text)[:300])
        with closing(web.services()) as services:
            assert services.sources.get(form["id"]).type == source_type


def test_the_recorded_audio_example_offers_both_origins(web: Web):
    codes = example_codes(web.client.get("/howto").text, "recorded_audio")
    assert [c.startswith("http") for c in codes] == [True, False]


def test_the_live_stream_example_is_a_direct_audio_address_not_a_playlist(web: Web):
    (url,) = example_codes(web.client.get("/howto").text, "live_stream")
    assert url.endswith(".mp3")
    assert "m3u8" not in url
    assert "hls" not in url.lower()
    page = section(web.client.get("/howto").text, "live_stream")
    assert "Icecast" in page
    assert "Playlists such as HLS (.m3u8) are not a direct stream" in page


def test_the_page_is_linked_from_the_navigation_and_has_no_outbound_actions(web: Web):
    page = web.client.get("/howto").text
    assert 'href="/howto"' in page
    assert "<form" not in page  # a reading page: nothing to submit


def doc_section(text: str, source_type: str) -> str:
    found = re.search(rf"^## `{source_type}`.*?(?=^## |\Z)", text, re.DOTALL | re.MULTILINE)
    assert found, source_type
    return found.group()


@pytest.mark.parametrize("source_type", SOURCE_TYPES)
def test_the_markdown_document_states_the_same_numbers_and_examples_as_the_page(
    web: Web, source_type: str
):
    mine = doc_section(DOC.read_text(encoding="utf-8"), source_type)
    shown = {int(n) for n in re.findall(r"(?<![\w.])\d+(?![\w.])", mine)}
    assert shown == EXPECTED_NUMBERS[source_type], mine
    example = re.search(r"\*\*Example\.\*\*(.*?)\n\n", mine, re.DOTALL)
    assert example
    in_doc = re.findall(r"`([^`]+)`", example.group(1))
    assert in_doc == example_codes(web.client.get("/howto").text, source_type)


def test_the_markdown_document_keeps_the_cautions_of_the_page():
    text = DOC.read_text(encoding="utf-8")
    for source_type in SOURCE_TYPES:
        assert f"`{source_type}`" in text
    for phrase in (
        "best-effort",
        "dedicated account",
        "no Lives",
        "direct audio stream",
        "Icecast",
    ):
        assert phrase in text
    assert "Playlists such as HLS (`.m3u8`) are not a direct stream" in text
    assert "https://example.org/live.m3u8" not in text
