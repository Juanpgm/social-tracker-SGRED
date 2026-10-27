"""The short how-to per source type (UI-4) and the document it mirrors."""

import re
from pathlib import Path

import pytest

from social_tracker.sources.domain import (
    CHUNK_SECONDS_MAX,
    CHUNK_SECONDS_MIN,
    INSTAGRAM_MIN_POLL_S,
    NEWS_RSS_MIN_POLL_S,
    SOURCE_TYPES,
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


def test_the_limits_shown_are_the_ones_the_domain_enforces(web: Web):
    page = web.client.get("/howto").text
    assert str(NEWS_RSS_MIN_POLL_S) in section(page, "news_rss")
    assert str(INSTAGRAM_MIN_POLL_S) in section(page, "instagram")
    live = section(page, "live_stream")
    assert str(CHUNK_SECONDS_MIN) in live
    assert str(CHUNK_SECONDS_MAX) in live


def test_the_page_is_linked_from_the_navigation_and_has_no_outbound_actions(web: Web):
    page = web.client.get("/howto").text
    assert 'href="/howto"' in page
    assert "<form" not in page  # a reading page: nothing to submit


def test_the_markdown_document_covers_the_same_types_and_limits():
    text = DOC.read_text(encoding="utf-8")
    for source_type in SOURCE_TYPES:
        assert f"`{source_type}`" in text
    for number in (NEWS_RSS_MIN_POLL_S, INSTAGRAM_MIN_POLL_S, CHUNK_SECONDS_MIN, CHUNK_SECONDS_MAX):
        assert str(number) in text
    for phrase in ("best-effort", "dedicated account", "no Lives", "direct audio stream"):
        assert phrase in text
