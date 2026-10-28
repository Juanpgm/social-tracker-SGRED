"""The UI in a real browser: htmx, the live region and the one-click guards (opt in: -m browser).

TestClient proves what the server sends; only a browser proves what the page does with it.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from playwright.sync_api import expect

from social_tracker.bootstrap import DB_PATH_ENV
from tests.entrypoints.web.browser.conftest import Live, flush, settled

pytestmark = pytest.mark.browser

ENABLE = "button[aria-label='Enable City News']"
DISABLE = "button[aria-label='Disable City News']"
REGION = "#live-region"


def open_list(page: Any, live: Live) -> None:
    page.goto(live.url + "/")
    page.wait_for_selector("#row-src_one")


def region_text(page: Any) -> str:
    return page.inner_text(REGION).strip()


def csp_violations(page: Any) -> list[str]:
    return [p for p in page.problems if "Content Security Policy" in p or "Refused to" in p]


@contextmanager
def write_lock(live: Live) -> Iterator[None]:
    holder = sqlite3.connect(live.env[DB_PATH_ENV], isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        yield
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def hold_posts(page: Any, pattern: str) -> list[Any]:
    """Keep the POSTs matching `pattern` in flight until the test lets them go."""
    held: list[Any] = []
    page.route(
        pattern,
        lambda route: held.append(route) if route.request.method == "POST" else route.continue_(),
    )
    return held


def posts_to(page: Any, fragment: str) -> list[str]:
    seen: list[str] = []
    page.on(
        "request",
        lambda r: seen.append(r.url) if r.method == "POST" and fragment in r.url else None,
    )
    return seen


# ---- a double click ----


def test_a_double_click_on_enable_is_one_request_and_one_audit_event(page: Any, live: Live):
    live.seed(name="City News")
    open_list(page, live)
    sent = posts_to(page, "/enable")
    page.dblclick(ENABLE)
    page.wait_for_selector(DISABLE)
    settled(page)  # no request is left in flight: a second one would have been seen
    assert len(sent) == 1
    assert len(live.events("source_enabled")) == 1
    assert len(live.events("source_disabled")) == 0
    assert csp_violations(page) == []


def test_the_row_button_is_disabled_while_its_request_is_in_flight(page: Any, live: Live):
    live.seed(name="City News")
    open_list(page, live)
    held = hold_posts(page, "**/sources/src_one/enable")
    page.click(ENABLE)
    page.wait_for_selector(f"{ENABLE}[disabled]")  # nothing else can be sent from this button
    page.locator(ENABLE).click(force=True, timeout=1000)  # a click on a disabled button is nothing
    flush(page)  # the click has been handled; a request it started would be held by now
    assert len(held) == 1
    held[0].continue_()
    page.wait_for_selector(DISABLE)
    assert len(live.events("source_enabled")) == 1


def test_the_submit_button_is_disabled_as_soon_as_a_once_only_form_is_submitted(
    page: Any, live: Live
):
    page.goto(live.url + "/sources/new/news_rss")
    page.fill("#f-id", "city_news")
    page.fill("#f-name", "City News")
    page.fill("#f-feed_url", "https://example.org/rss.xml")
    # Keep the page where it is, so the state right after the submit can be looked at.
    page.evaluate("document.addEventListener('submit', e => e.preventDefault(), true)")
    assert page.locator("button[type=submit]").is_enabled()
    page.click("button[type=submit]")
    page.wait_for_selector("button[type=submit][disabled]")
    assert live.events("source_added") == []  # nothing was sent: only the guard was exercised


def test_an_answer_that_is_not_a_row_leaves_the_row_alone_and_is_announced(page: Any, live: Live):
    live.seed(name="City News")
    open_list(page, live)
    token = page.locator("input[name=csrf_token]").first.get_attribute("value")
    page.evaluate("() => { window.__row = document.getElementById('row-src_one'); }")
    page.evaluate(
        """([token]) => htmx.ajax('POST', '/analysis', {
            target: '#row-src_one', swap: 'outerHTML', values: {csrf_token: token, on: 'maybe'}
        })""",
        [token],
    )
    page.wait_for_selector(f"{REGION}:has-text('not valid')")  # the 400 fragment's message
    assert page.evaluate("window.__row === document.getElementById('row-src_one')")
    assert page.locator("tbody tr").count() == 1
    assert page.locator("main h1", has_text="Problem").count() == 0
    assert csp_violations(page) == []


def test_a_busy_answer_corrects_a_row_that_was_already_out_of_date(page: Any, live: Live):
    live.seed(name="City News")
    open_list(page, live)  # the page shows "Enable"...
    live.enable("src_one")  # ...but another operator enabled it meanwhile
    with write_lock(live):
        page.click(ENABLE)
        page.wait_for_selector(f"{REGION}:has-text('database is busy')")
        page.wait_for_selector(DISABLE)  # the row now shows what the server holds
    assert page.locator(ENABLE).count() == 0


# ---- the server restarted under an open page ----


def test_a_toggle_after_a_server_restart_reloads_the_page_and_the_next_click_works(
    page: Any, live: Live
):
    live.seed(name="City News")
    open_list(page, live)
    live.restart()  # the page still holds a token of the previous process
    with page.expect_navigation():  # HX-Refresh reloads the page
        page.click(ENABLE)
    page.wait_for_selector("#row-src_one")
    assert page.locator("#row-src_one").count() == 1  # the row was not replaced by an error page
    assert page.locator("main h1", has_text="Problem").count() == 0
    assert page.locator("tbody tr").count() == 1
    assert page.locator(ENABLE).count() == 1  # nothing changed: still disabled
    assert live.events("source_enabled") == []
    page.click(ENABLE)
    page.wait_for_selector(DISABLE)
    assert len(live.events("source_enabled")) == 1
    assert csp_violations(page) == []


# ---- a busy database ----


def test_a_busy_database_leaves_the_true_row_and_says_so_then_it_works(page: Any, live: Live):
    live.seed(name="City News")
    open_list(page, live)
    with write_lock(live):
        page.click(ENABLE)
        page.wait_for_selector(f"{REGION}:has-text('database is busy')")
        assert page.locator(ENABLE).count() == 1  # the row still offers the same action
        assert page.locator("#row-src_one td").nth(3).inner_text().strip() == "no"
        assert page.locator("tbody tr").count() == 1
        assert page.locator(f"{ENABLE}[disabled]").count() == 0  # and the button is usable again
    assert live.events("source_enabled") == []
    page.click(ENABLE)
    page.wait_for_selector(DISABLE)
    assert "is enabled" in region_text(page)
    assert len(live.events("source_enabled")) == 1


# ---- announcements ----


def test_the_status_text_is_swapped_into_the_live_region_not_over_it(page: Any, live: Live):
    live.seed(name="City News")
    open_list(page, live)
    page.evaluate(
        """() => {
            const region = document.getElementById('live-region');
            window.__region = region;
            window.__mutations = 0;
            new MutationObserver(list => { window.__mutations += list.length; })
              .observe(region, {childList: true, subtree: true, characterData: true});
        }"""
    )
    page.click(ENABLE)
    page.wait_for_selector(DISABLE)
    page.wait_for_selector(f"{REGION}:has-text('is enabled')")
    assert page.evaluate("window.__mutations") > 0  # the region's content changed in place
    assert page.evaluate("window.__region === document.getElementById('live-region')")
    region = page.locator(REGION)
    assert region.get_attribute("role") == "status"
    assert region.get_attribute("aria-live") == "polite"
    assert region_text(page) == "City News is enabled."
    page.click(DISABLE)
    page.wait_for_selector(ENABLE)
    assert region_text(page) == "City News is disabled."  # replaced, not appended
    assert page.evaluate("window.__region === document.getElementById('live-region')")


# ---- the add form ----


def test_double_clicking_add_sends_the_form_once(page: Any, live: Live):
    page.goto(live.url + "/sources/new/news_rss")
    page.fill("#f-id", "city_news")
    page.fill("#f-name", "City News")
    page.fill("#f-feed_url", "https://example.org/rss.xml")
    sent = posts_to(page, "/sources/new/news_rss")
    page.dblclick("button[type=submit]")
    page.wait_for_url(lambda url: "notice=added" in url)
    settled(page)
    assert len(sent) == 1
    assert len(live.events("source_added")) == 1


def test_the_add_button_is_given_back_after_going_back_in_history(page: Any, live: Live):
    page.goto(live.url + "/sources/new/news_rss")
    page.fill("#f-id", "city_news")
    page.fill("#f-name", "City News")
    page.fill("#f-feed_url", "https://example.org/rss.xml")
    page.click("button[type=submit]")
    page.wait_for_url(lambda url: "notice=added" in url)
    page.go_back()
    page.wait_for_selector("form[data-once]")
    expect(page.locator("form[data-once] button[type=submit]")).to_be_enabled()


# ---- plumbing ----


def test_a_lost_connection_is_said_in_the_live_region(page: Any, live: Live):
    live.seed(name="City News")
    open_list(page, live)
    live.stop()
    page.click(ENABLE)
    page.wait_for_selector(f"{REGION}:has-text('cannot be reached')")
    assert page.locator(ENABLE).count() == 1
