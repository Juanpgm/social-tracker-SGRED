"""The alerts page in a real browser (opt in: -m browser): htmx swaps the row and the badge, a
double click is one request, stale clicks are corrected, and hostile text stays text.

TestClient proves what the server sends; only a browser proves what the page does with it.
"""

from contextlib import closing
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from social_tracker.alerts.domain import AlertFilter
from social_tracker.bootstrap import DB_PATH_ENV, build_container
from social_tracker.classification.domain import Classification, Label, Signal, Stage, Topic
from social_tracker.persistence.classification_repo import SqliteClassificationRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from tests.entrypoints.web.browser.conftest import Live, settled
from tests.items.factories import NOW, make_item

pytestmark = pytest.mark.browser

HOSTILE = '<img src=x onerror="window.__pwned = 1"><script>window.__pwned = 2</script>'


def plant(live: Live, name: str, *, confidence: float = 0.6, hours: int = 0, **item: Any) -> None:
    made = make_item(
        name,
        text=item.pop("text", f"Texto {name}: manana habra un temblor fuerte en Cali"),
        published_at=NOW + timedelta(hours=hours),
        **item,
    )
    verdict = Classification(
        item_id=made.id,
        label=Label.UNINTENTIONAL_RUMOR,
        confidence=confidence,
        stage=Stage.RULES,
        rationale="prediction_claim",
        classified_at=NOW,
        classifier_version="cascade-2/topics-0",
        signals=(Signal("prediction_claim", Label.UNINTENTIONAL_RUMOR, 0.45),),
        topics=(Topic.EARTHQUAKE,),
    )
    with Database(Path(live.env[DB_PATH_ENV])) as db:
        SqliteItemRepository(db).add_new([made])
        SqliteClassificationRepository(db).add(verdict)


def raise_alerts(live: Live) -> list[Any]:
    with closing(build_container(live.env)) as services:
        services.alerts.refresh()
        return list(services.alerts.list(AlertFilter(), limit=200).alerts)


def trail_length(live: Live, alert_id: str) -> int:
    with closing(build_container(live.env)) as services:
        return len(services.alerts.detail(alert_id).events)


def open_alerts(page: Any, live: Live, query: str = "") -> None:
    page.goto(live.url + "/alerts" + query)
    page.wait_for_selector("h1")


def test_marking_seen_swaps_the_row_and_the_badge_without_a_reload(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    plant(live, "b", hours=1)
    first, _ = raise_alerts(live)
    open_alerts(page, live)
    assert page.inner_text("#alerts-badge").strip() == "2"
    page.click(f"#alert-{first.id} button:has-text('Mark seen')")
    page.wait_for_selector(f"#alert-{first.id} td:text-is('seen')")
    assert page.inner_text("#alerts-badge").strip() == "1"
    assert page.locator(f"#alert-{first.id} button").all_inner_texts() == ["Dismiss"]
    assert "marked as seen" in page.inner_text("#live-region")
    assert page.url.endswith("/alerts")  # no navigation happened
    assert page.problems == []


def test_a_double_click_is_one_request_and_one_audit_entry(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    (alert,) = raise_alerts(live)
    open_alerts(page, live)
    sent: list[str] = []
    page.on("request", lambda r: sent.append(r.url) if r.method == "POST" else None)
    page.dblclick(f"#alert-{alert.id} button:has-text('Mark seen')")
    page.wait_for_selector(f"#alert-{alert.id} td:text-is('seen')")
    settled(page)  # no request is left in flight: a second one would have been seen
    assert len(sent) == 1
    assert trail_length(live, alert.id) == 2  # created, status_changed


def test_dismiss_then_reopen_in_place(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    (alert,) = raise_alerts(live)
    open_alerts(page, live)
    page.click(f"#alert-{alert.id} button:has-text('Dismiss')")
    page.wait_for_selector(f"#alert-{alert.id} button:has-text('Reopen')")
    assert page.inner_text("#alerts-badge").strip() == ""
    page.click(f"#alert-{alert.id} button:has-text('Reopen')")
    page.wait_for_selector(f"#alert-{alert.id} button:has-text('Dismiss')")
    assert trail_length(live, alert.id) == 3


def test_a_stale_click_is_corrected_by_the_server_not_believed(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    (alert,) = raise_alerts(live)
    open_alerts(page, live)
    with closing(build_container(live.env)) as services:  # another tab dismissed it meanwhile
        services.alerts.set_status(alert.id, "dismissed", actor="other", reason="not relevant")
    page.click(f"#alert-{alert.id} button:has-text('Dismiss')")
    page.wait_for_selector(f"#alert-{alert.id} button:has-text('Reopen')")
    assert "already dismissed" in page.inner_text("#live-region")


def test_an_alert_that_was_purged_meanwhile_asks_for_a_reload(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    (alert,) = raise_alerts(live)
    open_alerts(page, live)
    db = Database(Path(live.env[DB_PATH_ENV]))
    with db.transaction() as conn:
        conn.execute("DELETE FROM alerts WHERE id = ?", (alert.id,))
    db.close()
    with page.expect_navigation():
        page.click(f"#alert-{alert.id} button:has-text('Mark seen')")
    assert page.locator(f"#alert-{alert.id}").count() == 0


def test_the_filters_work_as_a_plain_form_and_keep_their_values(page: Any, live: Live):
    live.seed()
    plant(live, "a", confidence=0.55)
    plant(live, "b", confidence=0.9, hours=1)
    raise_alerts(live)
    open_alerts(page, live)
    page.select_option("#f-severity", "high")
    page.click("button:has-text('Filter')")
    page.wait_for_url("**/alerts?*severity=high*")
    assert page.locator("tbody tr").count() == 1
    assert page.input_value("#f-severity") == "high"
    page.click("text=Clear the filters")
    page.wait_for_url("**/alerts")
    assert page.locator("tbody tr").count() == 2


def test_the_detail_page_opens_from_the_list_and_shows_the_trail(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    (alert,) = raise_alerts(live)
    open_alerts(page, live)
    page.click(f"#alert-{alert.id} a:has-text('Details')")
    page.wait_for_url(f"**/alerts/{alert.id}")
    assert "Trail" in page.inner_text("main")
    page.click("#alert-status button:has-text('Mark seen')")
    page.wait_for_selector("#alert-status strong:text-is('seen')")
    assert page.locator("#alert-status button").all_inner_texts() == ["Dismiss"]


def test_hostile_text_is_shown_as_text_and_runs_nothing(page: Any, live: Live):
    live.seed()
    plant(live, "x", title=HOSTILE, text=HOSTILE + " temblor")
    (alert,) = raise_alerts(live)
    open_alerts(page, live)
    assert page.evaluate("window.__pwned") is None
    assert "<script>" in page.inner_text(f"#alert-{alert.id} .excerpt")
    assert page.locator("img").count() == 0
    page.goto(live.url + f"/alerts/{alert.id}")
    assert page.evaluate("window.__pwned") is None
    assert page.locator("main img, main script").count() == 0
    assert page.evaluate("window.__pwned") is None


def test_the_page_loads_nothing_from_outside(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    raise_alerts(live)
    requested: list[str] = []
    page.on("request", lambda r: requested.append(r.url))
    open_alerts(page, live)
    settled(page)
    assert requested
    assert all(url.startswith(live.url) for url in requested), requested


def test_the_keyboard_can_reach_and_press_a_mark(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    (alert,) = raise_alerts(live)
    open_alerts(page, live)
    page.focus(f"#alert-{alert.id} button:has-text('Mark seen')")
    page.keyboard.press("Enter")
    page.wait_for_selector(f"#alert-{alert.id} td:text-is('seen')")
    assert trail_length(live, alert.id) == 2


def test_the_server_going_away_says_so_in_the_status_region(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    (alert,) = raise_alerts(live)
    open_alerts(page, live)
    live.stop()
    page.click(f"#alert-{alert.id} button:has-text('Mark seen')")
    page.wait_for_selector("#live-region:has-text('cannot be reached')")


def test_the_header_and_the_new_filters_work_in_the_browser(page: Any, live: Live):
    live.seed()
    plant(live, "a")
    raise_alerts(live)
    open_alerts(page, live)
    header = page.inner_text(".monitor")
    assert "Analysis: off" in header
    assert "last worker pass: never" in header
    assert "nothing is being analysed" in header
    page.check("#f-hide")
    page.select_option("#f-when", "first_seen")
    page.click("form.filters button[type=submit]")
    page.wait_for_url(lambda url: "hide_provisional=1" in url and "when=first_seen" in url)
    assert page.is_checked("#f-hide")
    assert page.input_value("#f-when") == "first_seen"
    assert page.problems == []
