"""Deleting a source (UI-5): a confirmation first, alerts and audit events survive."""

import json
from contextlib import closing

from social_tracker.shared.errors import NotFoundError
from tests.entrypoints.web.conftest import Web, row_of, visible_text

CONFIRM = "/sources/src_one/delete"


def exists(web: Web, source_id: str = "src_one") -> bool:
    with closing(web.services()) as services:
        try:
            services.sources.get(source_id)
        except NotFoundError:
            return False
    return True


def test_the_list_links_to_the_edit_and_delete_pages_of_every_source(web: Web):
    web.seed()
    row = row_of(web.client.get("/").text, "src_one")
    assert 'href="/sources/src_one/edit"' in row
    assert 'href="/sources/src_one/delete"' in row


def test_the_confirmation_page_says_what_is_kept_and_deletes_nothing_by_itself(web: Web):
    web.seed(name="City News")
    response = web.client.get(CONFIRM)
    assert response.status_code == 200
    text = visible_text(response.text)
    assert "City News" in text
    assert "alerts and derived data are kept" in text
    assert 'action="/sources/src_one/delete"' in response.text
    assert 'name="csrf_token"' in response.text
    assert 'name="reason" value="via web UI"' in response.text
    assert exists(web)
    assert web.events("source_deleted") == []


def test_cancelling_leaves_the_source_alone(web: Web):
    web.seed()
    page = web.client.get(CONFIRM).text
    assert '<a href="/">Cancel</a>' in page
    assert web.client.get("/").status_code == 200
    assert exists(web)


def test_confirming_deletes_the_source_and_audits_who_and_why(web: Web):
    web.seed()
    response = web.post(CONFIRM, {"reason": "retired feed"})
    assert response.status_code == 303
    assert response.headers["location"] == "/?notice=deleted"
    assert not exists(web)
    assert "No sources yet" in visible_text(web.client.get("/").text)
    (event,) = web.events("source_deleted")
    detail = json.loads(event.detail)
    assert (event.source_id, detail["actor"], detail["reason"]) == (
        "src_one",
        "web:local",
        "retired feed",
    )


def test_the_audit_trail_of_a_deleted_source_survives(web: Web):
    web.seed()
    web.post("/sources/src_one/enable")
    web.post(CONFIRM)
    kinds = [e.kind for e in web.events() if e.source_id == "src_one"]
    assert kinds == ["source_added", "source_enabled", "source_deleted"]


def test_deleting_twice_is_a_not_found_notice_and_one_event(web: Web):
    web.seed()
    token = web.csrf()
    assert web.post(CONFIRM, token=token).headers["location"] == "/?notice=deleted"
    second = web.post(CONFIRM, token=token)
    assert second.status_code == 303
    assert second.headers["location"] == "/?notice=not_found"
    assert len(web.events("source_deleted")) == 1


def test_a_source_deleted_in_another_tab_leaves_a_stale_confirmation_harmless(web: Web):
    web.seed()
    assert web.client.get(CONFIRM).status_code == 200
    with closing(web.services()) as services:
        services.sources.delete("src_one", actor="other-tab", reason="gone elsewhere")
    response = web.post(CONFIRM)
    assert response.headers["location"] == "/?notice=not_found"
    assert "no longer exists" in visible_text(web.client.get(response.headers["location"]).text)


def test_the_confirmation_of_an_unknown_or_malformed_id_is_a_not_found_page(web: Web):
    for path in ("/sources/src_gone/delete", "/sources/UPPER/delete", "/sources/x/delete"):
        response = web.client.get(path)
        assert response.status_code == 404, path
        assert "does not exist" in visible_text(response.text)


def test_a_blank_reason_keeps_the_source_and_shows_the_confirmation_again(web: Web):
    web.seed()
    response = web.post(CONFIRM, {"reason": "   "})
    assert response.status_code == 422
    assert "reason" in visible_text(response.text).lower()
    assert 'action="/sources/src_one/delete"' in response.text
    assert exists(web)


def test_markup_in_the_name_is_inert_on_the_confirmation_page(web: Web):
    web.seed(name="<script>alert(1)</script>")
    page = web.client.get(CONFIRM).text
    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_the_analysis_flag_and_state_go_with_the_source_and_a_new_source_starts_clean(web: Web):
    web.seed()
    web.post("/sources/src_one/enable")
    web.post("/sources/src_one/analysis", {"on": "1"})
    web.post(CONFIRM)
    web.seed()  # the same id again
    with closing(web.services()) as services:
        (only,) = services.control.overview().sources
    assert (only.state.value, only.analysis) == ("disabled", False)


def test_deleting_a_malformed_id_is_a_not_found_notice_and_never_a_validation_page(web: Web):
    for bad in ("UPPER", "x", "a" * 200):
        response = web.post(f"/sources/{bad}/delete")
        assert response.status_code == 303, bad
        assert response.headers["location"] == "/?notice=not_found"
    assert web.events("source_deleted") == []
