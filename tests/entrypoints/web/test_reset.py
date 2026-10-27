"""The Reset button of a failed source: CSRF-protected like every change, audited, idempotent."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from social_tracker.bootstrap import DB_PATH_ENV
from social_tracker.control.domain import (
    MAX_CONSECUTIVE_FAILURES,
    CollectionState,
    FetchStatus,
    SourceState,
)
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from tests.entrypoints.web.conftest import Web, row_of, visible_text

FETCHED = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
RESET_FORM = 'action="/sources/src_one/reset"'


def make_failed(web: Web, source_id: str = "src_one", **fields: str) -> None:
    web.seed(source_id, **fields)
    web.post(f"/sources/{source_id}/enable")
    with Database(Path(web.env[DB_PATH_ENV])) as db:
        repo = SqliteControlRepository(db)
        repo.save_source_state(
            SourceState(source_id, CollectionState.FAILED, MAX_CONSECUTIVE_FAILURES, None)
        )
        repo.save_fetch_status(FetchStatus(source_id, FETCHED, "http 503"))


def stored(web: Web, source_id: str = "src_one") -> tuple[CollectionState, str | None]:
    with Database(Path(web.env[DB_PATH_ENV])) as db:
        repo = SqliteControlRepository(db)
        state = repo.get_source_state(source_id)
        status = repo.get_fetch_status(source_id)
    return state.state, status.last_error if status else None


# ---- the button ----


def test_only_a_failed_source_offers_a_reset(web: Web):
    web.seed("src_one", name="One")
    web.seed("src_three", name="Three")
    web.post("/sources/src_one/enable")
    make_failed(web, "src_two", name="Two")
    page = web.client.get("/").text
    assert RESET_FORM not in row_of(page, "src_one")  # idle
    assert "/reset" not in row_of(page, "src_three")  # disabled
    assert 'action="/sources/src_two/reset"' in row_of(page, "src_two")  # failed


def test_the_button_carries_the_csrf_token_and_a_screen_reader_label(web: Web):
    make_failed(web)
    row = row_of(web.client.get("/").text, "src_one")
    form = re.search(r'<form[^>]*action="/sources/src_one/reset".*?</form>', row, re.DOTALL)
    assert form
    assert 'name="csrf_token"' in form.group()
    assert 'aria-label="Reset ' in form.group()
    assert 'hx-post="/sources/src_one/reset"' in form.group()


# ---- the action ----


def test_a_reset_puts_the_source_back_to_idle_clears_the_error_and_audits(web: Web):
    make_failed(web)
    response = web.post("/sources/src_one/reset", {"reason": "fixed the feed"})
    assert response.status_code == 303
    assert response.headers["location"] == "/?notice=reset"
    assert stored(web) == (CollectionState.IDLE, None)
    (event,) = web.events("collection_reset")
    detail = json.loads(event.detail)
    assert (detail["actor"], detail["reason"]) == ("web:local", "fixed the feed")
    page = web.client.get("/?notice=reset").text
    assert "Source reset" in visible_text(page)
    assert RESET_FORM not in row_of(page, "src_one")
    assert "http 503" not in row_of(page, "src_one")


def test_the_default_reason_is_the_one_of_the_other_row_buttons(web: Web):
    make_failed(web)
    web.post("/sources/src_one/reset")
    (event,) = web.events("collection_reset")
    assert json.loads(event.detail)["reason"] == "via web UI"


def test_a_double_submit_resets_once_and_the_second_is_a_no_op(web: Web):
    make_failed(web)
    token = web.csrf()
    first = web.post("/sources/src_one/reset", token=token)
    second = web.post("/sources/src_one/reset", token=token)
    assert (first.status_code, second.status_code) == (303, 303)
    assert len(web.events("collection_reset")) == 1
    assert stored(web)[0] is CollectionState.IDLE


def test_resetting_a_source_that_is_not_failed_changes_nothing(web: Web):
    web.seed()
    web.post("/sources/src_one/enable")
    response = web.post("/sources/src_one/reset")
    assert response.status_code == 303
    assert web.events("collection_reset") == []
    assert stored(web)[0] is CollectionState.IDLE


def test_htmx_gets_the_row_as_the_server_holds_it_and_a_status_line(web: Web):
    make_failed(web)
    response = web.post_htmx("/sources/src_one/reset")
    assert response.status_code == 200
    assert 'id="row-src_one"' in response.text
    assert RESET_FORM not in response.text
    assert "reset" in visible_text(response.text).lower()
    again = web.post_htmx("/sources/src_one/reset")
    assert again.status_code == 200
    assert "not failed" in visible_text(again.text)
    assert len(web.events("collection_reset")) == 1


def test_an_unknown_or_malformed_source_is_gone_not_a_traceback(web: Web):
    plain = web.post("/sources/src_missing/reset")
    assert plain.status_code == 303
    assert plain.headers["location"] == "/?notice=not_found"
    assert web.post_htmx("/sources/src_missing/reset").status_code == 404
    assert web.post("/sources/Bad%20Id/reset").status_code in (303, 404)
    assert web.events("collection_reset") == []


def test_a_reset_without_a_valid_token_is_refused_and_changes_nothing(web: Web):
    make_failed(web)
    assert web.client.post("/sources/src_one/reset", data={"x": "1"}).status_code == 403
    assert web.post("/sources/src_one/reset", token="0" * 64).status_code == 403
    assert web.post("/sources/src_one/reset", token="").status_code == 403
    assert stored(web) == (CollectionState.FAILED, "http 503")
    assert web.events("collection_reset") == []


def test_a_blank_reason_is_rejected(web: Web):
    make_failed(web)
    assert web.post("/sources/src_one/reset", {"reason": "   "}).status_code in (400, 422)
    assert stored(web)[0] is CollectionState.FAILED
