"""Marking alerts from the page (AL-5, UI-6, UI-7): htmx fragments that always show the truth, a
plain form that works without script, idempotent clicks, and every refusal drawn without losing
the row."""

import re
from contextlib import closing
from urllib.parse import urlencode

import pytest

from social_tracker.alerts.domain import AlertEventKind
from tests.entrypoints.web.alerts_support import events_of, plant, raise_alerts
from tests.entrypoints.web.conftest import Web, visible_text
from tests.entrypoints.web.test_htmx_errors import assert_fragment
from tests.entrypoints.web.test_resilience import impatient, token_of, write_lock

HTMX = {"HX-Request": "true"}
FORM = {"Content-Type": "application/x-www-form-urlencoded"}


@pytest.fixture
def one(web: Web) -> tuple[Web, str]:
    web.seed("src_one")
    plant(web, "a", confidence=0.6)
    (alert,) = raise_alerts(web)
    return web, alert.id


def row_html(fragment: str, alert_id: str) -> str:
    found = re.search(rf'<tr id="alert-{alert_id}".*?</tr>', fragment, re.DOTALL)
    assert found, fragment[:400]
    return found.group()


def buttons(html: str) -> list[str]:
    return re.findall(r"<button[^>]*>([^<]+)</button>", html)


def trail(web: Web, alert_id: str) -> list[str]:
    return [e.kind.value for e in events_of(web, alert_id)]


# ---- the row buttons follow the status ----


def test_a_new_alert_offers_seen_and_dismiss(one):
    web, alert_id = one
    row = row_html(web.client.get("/alerts").text, alert_id)
    assert buttons(row) == ["Mark seen", "Dismiss"]


# ---- htmx: the row comes back as it is now ----


def test_marking_seen_redraws_the_row_with_the_buttons_of_its_new_state(one):
    web, alert_id = one
    response = web.post_htmx(f"/alerts/{alert_id}/seen", {"view": "list"})
    assert response.status_code == 200
    body = response.text
    assert "<!doctype" not in body.lower()
    assert "<html" not in body.lower()
    row = row_html(body, alert_id)
    assert ">seen<" in row
    assert buttons(row) == ["Dismiss"]
    assert 'hx-swap-oob="innerHTML:#live-region"' in body
    assert "marked as seen" in visible_text(body)
    assert re.search(r'<span id="alerts-badge"[^>]*hx-swap-oob="true"[^>]*hidden', body)
    assert trail(web, alert_id) == ["created", "status_changed"]


def test_dismissing_and_reopening_walk_the_row_through_its_states(one):
    web, alert_id = one
    dismissed = web.post_htmx(f"/alerts/{alert_id}/dismiss", {"view": "list"}).text
    assert buttons(row_html(dismissed, alert_id)) == ["Reopen"]
    reopened = web.post_htmx(f"/alerts/{alert_id}/reopen", {"view": "list"}).text
    row = row_html(reopened, alert_id)
    assert ">seen<" in row
    assert buttons(row) == ["Dismiss"]
    assert trail(web, alert_id) == ["created", "status_changed", "status_changed"]


def test_the_audit_names_the_web_ui_as_actor_and_gives_a_reason(one):
    web, alert_id = one
    web.post_htmx(f"/alerts/{alert_id}/seen", {"view": "list"})
    last = events_of(web, alert_id)[-1]
    assert (last.kind, last.actor, last.reason) == (
        AlertEventKind.STATUS_CHANGED,
        "web:local",
        "via web UI",
    )
    web.post_htmx(
        f"/alerts/{alert_id}/dismiss", {"view": "list", "reason": "checked with the desk"}
    )
    assert events_of(web, alert_id)[-1].reason == "checked with the desk"


def test_the_badge_counts_the_alerts_that_are_still_new(web: Web):
    web.seed("src_one")
    for name in "abc":
        plant(web, name, hours="abc".index(name))
    alerts = raise_alerts(web)
    body = web.post_htmx(f"/alerts/{alerts[0].id}/seen", {"view": "list"}).text
    assert re.search(r'<span id="alerts-badge"[^>]*>2</span>', body)


def test_the_detail_view_redraws_the_status_panel_not_a_row(one):
    web, alert_id = one
    body = web.post_htmx(f"/alerts/{alert_id}/seen", {"view": "detail"}).text
    assert '<div id="alert-status"' in body
    assert "<tr" not in body
    assert "seen" in visible_text(body)
    assert buttons(body.split('id="alert-status"')[1]) == ["Dismiss"]


# ---- idempotence ----


def test_a_double_click_changes_once_and_the_second_answer_is_calm(one):
    web, alert_id = one
    first = web.post_htmx(f"/alerts/{alert_id}/seen", {"view": "list"})
    second = web.post_htmx(f"/alerts/{alert_id}/seen", {"view": "list"})
    assert (first.status_code, second.status_code) == (200, 200)
    assert "already seen" in visible_text(second.text)
    assert ">seen<" in row_html(second.text, alert_id)
    assert trail(web, alert_id).count("status_changed") == 1


def test_a_stale_click_redraws_the_true_state_instead_of_the_clicked_one(one):
    web, alert_id = one
    with closing(web.services()) as services:  # someone else dismissed it
        services.alerts.set_status(alert_id, "dismissed", actor="other", reason="not relevant")
    response = web.post_htmx(f"/alerts/{alert_id}/dismiss", {"view": "list"})
    assert response.status_code == 200
    assert "already dismissed" in visible_text(response.text)
    assert buttons(row_html(response.text, alert_id)) == ["Reopen"]


def test_an_illegal_move_is_refused_and_the_row_shows_what_it_really_is(one):
    web, alert_id = one
    response = web.post_htmx(f"/alerts/{alert_id}/reopen", {"view": "list"})  # nobody looked yet
    assert response.status_code == 409
    assert "<html" not in response.text.lower()
    assert "can be reopened" in visible_text(response.text)
    assert buttons(row_html(response.text, alert_id)) == ["Mark seen", "Dismiss"]
    assert trail(web, alert_id) == ["created"]


# ---- plain forms, no script ----


def test_without_script_a_mark_redirects_back_with_a_notice(one):
    web, alert_id = one
    response = web.post(f"/alerts/{alert_id}/seen", {"view": "list"})
    assert response.status_code == 303
    assert response.headers["location"].startswith("/alerts?")
    page = web.client.get(response.headers["location"])
    assert "Alert marked as seen" in visible_text(page.text)
    assert ">seen<" in row_html(page.text, alert_id)


def test_the_page_it_came_from_is_kept_when_it_is_a_page_of_alerts(one):
    web, alert_id = one
    back = "/alerts?severity=medium&page=1"
    response = web.post(f"/alerts/{alert_id}/seen", {"view": "list", "back": back})
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/alerts?severity=medium&page=1")
    assert "notice=" in location


def test_from_the_detail_page_a_mark_returns_to_the_detail(one):
    web, alert_id = one
    response = web.post(f"/alerts/{alert_id}/dismiss", {"view": "detail"})
    assert response.headers["location"].startswith(f"/alerts/{alert_id}")


@pytest.mark.parametrize(
    "back",
    [
        "//evil.example/alerts",
        "https://evil.example/alerts",
        "http://127.0.0.1:8765/other",
        "/other",
        "/alerts/../sources",
        "javascript:alert(1)",
        "/alertsx",
        "/alerts?x=1\r\nSet-Cookie: a=b",
        "/alerts\\evil",
        "/alerts?" + "a" * 2000,
        "",
    ],
)
def test_the_return_address_is_never_an_open_redirect(one, back):
    web, alert_id = one
    response = web.post(f"/alerts/{alert_id}/seen", {"view": "list", "back": back})
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/alerts")
    assert "evil" not in location
    assert "\r" not in location
    assert "\n" not in location
    assert not location.startswith("//")


# ---- refusals ----


@pytest.mark.parametrize("verb", ["seen", "dismiss", "reopen"])
@pytest.mark.parametrize("bad", ["alt_" + "0" * 32, "nope", "' OR 1=1 --"])
def test_an_unknown_alert_is_gone_for_htmx_and_a_notice_for_a_form(web: Web, verb, bad):
    got = web.post_htmx(f"/alerts/{bad}/{verb}", {"view": "list"})
    assert_fragment(got, 404, refresh=True, text="no longer exists")
    plain = web.post(f"/alerts/{bad}/{verb}", {"view": "list"})
    assert plain.status_code == 303
    assert plain.headers["location"].startswith("/alerts?notice=")


def test_a_token_that_does_not_belong_is_refused_and_changes_nothing(one):
    web, alert_id = one
    response = web.client.post(
        f"/alerts/{alert_id}/seen",
        content=urlencode({"csrf_token": "x" * 64, "view": "list"}),
        headers={**HTMX, **FORM},
    )
    assert_fragment(response, 403, refresh=True, text="Reload the page")
    assert trail(web, alert_id) == ["created"]
    forgot = web.client.post(f"/alerts/{alert_id}/seen", data={"view": "list"})
    assert forgot.status_code == 403
    assert trail(web, alert_id) == ["created"]


@pytest.mark.parametrize("verb", ["seen", "dismiss", "reopen"])
def test_every_mark_refuses_a_request_without_a_valid_token(one, verb):
    web, alert_id = one
    for headers in (HTMX, {}):
        response = web.client.post(
            f"/alerts/{alert_id}/{verb}",
            content=urlencode({"view": "list"}),
            headers={**headers, **FORM},
        )
        assert response.status_code == 403
    assert trail(web, alert_id) == ["created"]


def test_the_wrong_content_type_is_refused(one):
    web, alert_id = one
    response = web.client.post(
        f"/alerts/{alert_id}/seen",
        content=b'{"view": "list"}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 415
    assert trail(web, alert_id) == ["created"]


def test_an_action_cannot_be_asked_for_with_a_plain_get(one):
    web, alert_id = one
    assert web.client.get(f"/alerts/{alert_id}/seen").status_code in (404, 405)
    assert trail(web, alert_id) == ["created"]


@pytest.mark.parametrize("view", ["", "row", "LIST", "list ", "<script>", "x" * 500])
def test_an_unknown_view_is_a_bad_request(one, view):
    web, alert_id = one
    assert web.post(f"/alerts/{alert_id}/seen", {"view": view}).status_code == 400
    assert trail(web, alert_id) == ["created"]


def test_a_missing_view_means_the_list(one):
    web, alert_id = one
    assert web.post(f"/alerts/{alert_id}/seen", {}).status_code == 303


@pytest.mark.parametrize(
    "reason", ["", "  ", "bell\x07", "r" * 501, "hidden" + chr(0x202E) + "text"]
)
def test_a_reason_that_is_not_plain_text_is_refused_and_the_row_keeps_its_state(one, reason):
    web, alert_id = one
    response = web.post_htmx(f"/alerts/{alert_id}/seen", {"view": "list", "reason": reason})
    assert response.status_code == 422
    assert buttons(row_html(response.text, alert_id)) == ["Mark seen", "Dismiss"]
    assert trail(web, alert_id) == ["created"]


def test_a_busy_database_leaves_the_row_as_it_was_and_says_to_try_again(one):
    web, alert_id = one
    client = impatient(web)
    token = token_of(client)
    with write_lock(web):
        response = client.post(
            f"/alerts/{alert_id}/seen",
            content=urlencode({"csrf_token": token, "view": "list"}),
            headers={**HTMX, **FORM},
        )
    assert response.status_code == 503
    assert "busy or unavailable" in visible_text(response.text)
    assert "SQLITE" not in response.text
    assert buttons(row_html(response.text, alert_id)) == ["Mark seen", "Dismiss"]
    assert trail(web, alert_id) == ["created"]
    again = client.post(
        f"/alerts/{alert_id}/seen",
        content=urlencode({"csrf_token": token, "view": "list"}),
        headers={**HTMX, **FORM},
    )
    assert again.status_code == 200
    with closing(web.services()) as services:
        assert services.alerts.detail(alert_id).alert.status.value == "seen"
