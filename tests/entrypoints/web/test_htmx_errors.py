"""Every error answer to an htmx request is a small fragment, never a document (UI-6).

htmx swaps whatever comes back into the row, so a full error page would be pasted inside the
table. The fragment carries the message for the live region (out of band) and tells htmx not to
swap the row (`HX-Reswap: none`); a refused token also asks for a reload (`HX-Refresh`).
"""

import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV, Container, build_container
from social_tracker.entrypoints.web.app import create_app
from social_tracker.entrypoints.web.formdata import MAX_BODY_BYTES
from social_tracker.entrypoints.web.security import CSRF_COOKIE_NAME, CsrfTokens
from social_tracker.shared.errors import StorageError
from tests.entrypoints.web.conftest import BASE, Web, visible_text

SECRET = b"s" * 32
COOKIE = "A" * 43
HTMX = {"HX-Request": "true"}
FORM = {"Content-Type": "application/x-www-form-urlencoded"}


def token_for(cookie: str = COOKIE, secret: bytes = SECRET) -> str:
    return CsrfTokens(secret).token_for(cookie)


def client_for(factory: Callable[[], Container], secret: bytes = SECRET) -> TestClient:
    app = create_app(factory, host="127.0.0.1", secret=secret)
    client = TestClient(app, base_url=BASE, raise_server_exceptions=False)
    client.cookies.set(CSRF_COOKIE_NAME, COOKIE)
    return client


def assert_fragment(
    response: Response, status: int, *, refresh: bool, text: str | None = None
) -> None:
    assert response.status_code == status
    body = response.text
    assert "<!doctype" not in body.lower()
    assert "<html" not in body.lower()
    assert 'hx-swap-oob="innerHTML:#live-region"' in body
    assert response.headers["hx-reswap"] == "none"
    if refresh:
        assert response.headers["hx-refresh"] == "true"
    else:
        assert "hx-refresh" not in response.headers
    if text:
        assert text in visible_text(body)
    assert response.headers["x-frame-options"] == "DENY"  # still behind the guard


@pytest.fixture
def client(web: Web) -> TestClient:
    return client_for(web.services)


def post(client: TestClient, path: str, data: dict[str, str]) -> Response:
    # htmx always sends the content type, also for an empty form
    return client.post(path, content=urlencode(data), headers={**HTMX, **FORM})


# ---- refused before any route code ----


def test_a_token_issued_before_a_restart_is_refused_with_a_fragment_and_a_reload(web: Web):
    web.seed()
    old_token = token_for(secret=b"o" * 32)  # what the page served by the previous process held
    client = client_for(web.services)  # the restarted process has a new secret
    response = post(client, "/sources/src_one/enable", {"csrf_token": old_token})
    assert_fragment(response, 403, refresh=True, text="Reload the page")
    assert web.events("source_enabled") == []


@pytest.mark.parametrize(
    "form",
    [{}, {"csrf_token": ""}, {"csrf_token": "0" * 64}, {"csrf_token": "x"}],
    ids=["missing", "empty", "wrong", "short"],
)
def test_a_missing_or_wrong_token_is_a_403_fragment_that_reloads(client: TestClient, form):
    response = post(client, "/sources/src_one/enable", form)
    assert_fragment(response, 403, refresh=True)


def test_a_token_derived_from_another_cookie_is_refused_the_same_way(client: TestClient):
    response = post(client, "/sources/src_one/enable", {"csrf_token": token_for("B" * 43)})
    assert_fragment(response, 403, refresh=True)


def test_a_duplicated_token_field_uses_the_first_value_and_a_bad_first_one_is_refused(
    client: TestClient,
):
    body = f"csrf_token=bad&csrf_token={token_for()}"
    response = client.post("/sources/src_one/enable", content=body, headers={**HTMX, **FORM})
    assert_fragment(response, 403, refresh=True)


def test_a_malformed_body_is_a_400_fragment_without_a_reload(client: TestClient):
    response = client.post(
        "/sources/src_one/enable", content=b"\xff\xfe=1", headers={**HTMX, **FORM}
    )
    assert_fragment(response, 400, refresh=False)


def test_an_unknown_switch_value_is_a_400_fragment(client: TestClient):
    response = post(client, "/sources/src_one/analysis", {"csrf_token": token_for(), "on": "maybe"})
    assert_fragment(response, 400, refresh=False)


def test_an_oversized_declared_body_is_a_413_fragment(client: TestClient):
    response = client.post(
        "/sources/src_one/enable",
        content=b"x" * (MAX_BODY_BYTES + 1),
        headers={**HTMX, **FORM},
    )
    assert_fragment(response, 413, refresh=False, text="too large")


def test_an_oversized_streamed_body_is_a_413_fragment(client: TestClient):
    def chunks() -> Iterator[bytes]:  # no Content-Length: the limit is enforced while reading
        for _ in range(MAX_BODY_BYTES // 1024 + 2):
            yield b"a=" + b"x" * 1022

    response = client.post("/sources/src_one/enable", content=chunks(), headers={**HTMX, **FORM})
    assert_fragment(response, 413, refresh=False)


def test_an_unsupported_content_type_is_a_415_fragment(client: TestClient):
    response = client.post(
        "/sources/src_one/enable",
        content=b"{}",
        headers={**HTMX, "Content-Type": "application/json"},
    )
    assert_fragment(response, 415, refresh=False)


def test_an_unknown_route_and_a_wrong_method_are_fragments(client: TestClient):
    assert_fragment(post(client, "/nope", {}), 404, refresh=False)
    wrong = client.get("/analysis", headers=HTMX)
    assert_fragment(wrong, 405, refresh=False)


# ---- failures inside a route ----


@contextmanager
def write_lock(web: Web) -> Iterator[None]:
    holder = sqlite3.connect(web.env[DB_PATH_ENV], isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        yield
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def test_a_storage_failure_outside_a_switch_is_a_503_fragment(web: Web):
    env = {**web.env, BUSY_TIMEOUT_ENV: "50"}
    client = client_for(lambda: build_container(env))
    with write_lock(web):
        response = post(client, "/analysis", {"csrf_token": token_for(), "on": "1"})
    assert_fragment(response, 503, refresh=False, text="database is busy")
    assert "SQLITE" not in response.text


def test_a_store_that_cannot_be_opened_during_a_switch_is_a_503_fragment(web: Web):
    def unavailable() -> Container:
        raise StorageError("database error: SQLITE_CANTOPEN")

    client = client_for(unavailable)
    response = post(client, "/sources/src_one/enable", {"csrf_token": token_for()})
    assert_fragment(response, 503, refresh=False, text="database is busy")


def test_an_unexpected_failure_is_a_500_fragment_with_no_detail(web: Web, caplog):
    def broken() -> Container:
        raise RuntimeError("secret-" + "detail")

    client = client_for(broken)
    with caplog.at_level(logging.ERROR, logger="social_tracker.web"):
        response = post(client, "/analysis", {"csrf_token": token_for(), "on": "1"})
    assert_fragment(response, 500, refresh=False, text="Something went wrong")
    assert "secret-" not in response.text + caplog.text


def test_a_form_rejection_sent_by_htmx_is_a_fragment_too(web: Web):
    client = client_for(web.services)
    bad = {"csrf_token": token_for(), "id": "x", "name": "", "credibility": "9", "reason": "r"}
    response = post(client, "/sources/new/news_rss", bad)
    assert_fragment(response, 422, refresh=False)


def test_the_same_failures_without_htmx_stay_full_pages(client: TestClient):
    stale = client.post("/sources/src_one/enable", data={"csrf_token": "0" * 64})
    assert stale.status_code == 403
    assert stale.text.lower().startswith("<!doctype html>")
    assert "hx-reswap" not in stale.headers
    assert "hx-refresh" not in stale.headers
    assert "Reload the page" in visible_text(stale.text)


def test_a_foreign_host_over_htmx_is_plain_text_that_htmx_never_swaps(client: TestClient):
    response = client.post(
        "/sources/src_one/enable",
        content=f"csrf_token={token_for()}",
        headers={**HTMX, **FORM, "host": "evil.example"},
    )
    assert response.status_code == 400
    assert response.text == "Invalid host header"
    assert response.headers["hx-reswap"] == "none"
