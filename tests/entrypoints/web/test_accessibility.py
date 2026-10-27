"""Accessibility basics and the offline-safe client script (CSP forbids inline script)."""

import re

import pytest

from tests.entrypoints.web.conftest import Web

PAGES = [
    "/",
    "/sources/new",
    "/sources/new/news_rss",
    "/sources/new/instagram",
    "/sources/new/live_stream",
    "/sources/new/recorded_audio",
    "/sources/src_one/edit",
    "/sources/src_one/delete",
    "/howto",
    "/nope",
]


@pytest.mark.parametrize("path", PAGES)
def test_every_page_has_a_language_a_skip_link_a_main_landmark_and_a_status_region(
    web: Web, path: str
):
    web.seed()
    page = web.client.get(path).text
    assert '<html lang="en">' in page
    assert 'class="skip" href="#main"' in page
    assert '<main id="main">' in page
    assert re.search(r'<div id="live-region" role="status" aria-live="polite">', page)
    assert "<h1>" in page


@pytest.mark.parametrize("path", PAGES)
def test_every_visible_input_has_a_label_and_no_page_uses_inline_script_or_handlers(
    web: Web, path: str
):
    web.seed()
    page = web.client.get(path).text
    inputs = re.findall(r'<(?:input|textarea)\b[^>]*\bid="([^"]+)"', page)
    for field_id in inputs:
        assert f'for="{field_id}"' in page, (path, field_id)
    assert not re.search(r"\son[a-z]+\s*=", page), path
    for script in re.findall(r"<script\b[^>]*>", page):
        assert "src=" in script, (path, script)


def test_actions_are_real_buttons_and_links_not_clickable_divs(web: Web):
    web.seed()
    page = web.client.get("/").text
    assert page.count("<button") >= 3  # global switch, enable, analysis
    assert 'role="button"' not in page
    assert "tabindex" not in page  # nothing is pulled out of the natural focus order


def test_buttons_name_the_source_they_act_on_for_screen_readers(web: Web):
    web.seed(name="City News")
    page = web.client.get("/").text
    assert 'aria-label="Enable City News"' in page
    assert 'aria-label="Edit City News"' in page
    assert 'aria-label="Delete City News"' in page


def test_invalid_fields_are_flagged_and_point_to_their_message(web: Web):
    response = web.post(
        "/sources/new/news_rss", {"id": "city_news", "name": "", "credibility": "9", "feed_url": ""}
    )
    page = response.text
    for field in ("name", "credibility", "feed_url"):
        assert f'id="f-{field}"' in page
        assert re.search(
            rf'id="f-{field}"[^>]*aria-invalid="true"[^>]*aria-describedby="err-{field}"', page
        )
        assert f'id="err-{field}"' in page


def test_the_client_script_reports_an_unreachable_server_in_the_status_region(web: Web):
    page = web.client.get("/").text
    assert '<script src="/static/app.js" defer></script>' in page
    script = web.client.get("/static/app.js").text
    assert "htmx:sendError" in script
    assert "live-region" in script
    assert "cannot be reached" in script
    assert "innerHTML" not in script  # text only, never markup
    assert not re.search(r"https?://", script)
