"""Locked or failing storage, concurrent writers and unexpected errors: readable, never raw."""

import logging
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV, build_container
from social_tracker.entrypoints.web.app import create_app
from tests.entrypoints.web.conftest import BASE, TOKEN_FIELD, Web, visible_text
from tests.entrypoints.web.test_forms_add import BASE_FORM, VALID


@contextmanager
def write_lock(web: Web) -> Iterator[None]:
    """Another process holding the database's write lock."""
    holder = sqlite3.connect(web.env[DB_PATH_ENV], isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        yield
    finally:
        holder.execute("ROLLBACK")
        holder.close()


def impatient(web: Web) -> TestClient:
    env = {**web.env, BUSY_TIMEOUT_ENV: "50"}
    app = create_app(lambda: build_container(env), host="127.0.0.1")
    return TestClient(app, base_url=BASE)


def token_of(client: TestClient) -> str:
    found = TOKEN_FIELD.search(client.get("/").text)
    assert found
    return found.group(1)


def test_a_locked_database_gives_a_friendly_page_for_a_form_and_changes_nothing(web: Web):
    web.seed()
    client = impatient(web)
    token = token_of(client)
    with write_lock(web):
        response = client.post(
            "/sources/new/news_rss",
            data={
                **BASE_FORM,
                **VALID["news_rss"],
                "id": "second_one",
                "name": "Second",
                "csrf_token": token,
            },
        )
    assert response.status_code == 503
    text = visible_text(response.text)
    assert "database is busy or unavailable" in text
    assert "SQLITE" not in response.text
    assert "Traceback" not in response.text
    assert len(web.events("source_added")) == 1


def test_a_locked_database_during_a_switch_redirects_with_the_busy_notice(web: Web):
    web.seed()
    client = impatient(web)
    token = token_of(client)
    with write_lock(web):
        response = client.post(
            "/sources/src_one/enable", data={"csrf_token": token}, follow_redirects=False
        )
        assert response.headers["location"] == "/?notice=busy"
        page = client.get("/?notice=busy")  # readers never wait for the writer
        assert page.status_code == 200
        assert "database is busy" in visible_text(page.text)
    assert web.events("source_enabled") == []


def test_the_same_request_succeeds_once_the_lock_is_gone(web: Web):
    web.seed()
    client = impatient(web)
    token = token_of(client)
    with write_lock(web):
        client.post("/sources/src_one/enable", data={"csrf_token": token})
    done = client.post(
        "/sources/src_one/enable", data={"csrf_token": token}, follow_redirects=False
    )
    assert done.headers["location"] == "/?notice=enabled"


def test_concurrent_adds_of_the_same_name_let_exactly_one_win(web: Web):
    def add(n: int) -> int:
        client = TestClient(web.client.app, base_url=BASE)
        token = token_of(client)
        form = {**BASE_FORM, **VALID["news_rss"], "id": f"src_{n:02d}", "csrf_token": token}
        return client.post("/sources/new/news_rss", data=form, follow_redirects=False).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = sorted(pool.map(add, range(6)))
    assert codes == [303, 409, 409, 409, 409, 409]
    with closing(web.services()) as services:
        assert len(services.sources.list()) == 1


def test_concurrent_adds_of_different_sources_all_land(web: Web):
    def add(n: int) -> int:
        client = TestClient(web.client.app, base_url=BASE)
        form = {
            **BASE_FORM,
            **VALID["news_rss"],
            "id": f"src_{n:02d}",
            "name": f"Source {n}",
            "csrf_token": token_of(client),
        }
        return client.post("/sources/new/news_rss", data=form, follow_redirects=False).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(add, range(6)))
    assert codes == [303] * 6
    with closing(web.services()) as services:
        assert len(services.sources.list()) == 6


def test_an_unexpected_failure_in_a_post_is_a_plain_500_and_the_log_has_only_the_type(
    web: Web, caplog
):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 1:  # the page renders, then the write blows up
            raise RuntimeError("secret-" + "detail-in-message")
        return build_container(web.env)

    client = TestClient(
        create_app(flaky, host="127.0.0.1"), base_url=BASE, raise_server_exceptions=False
    )
    token = token_of(client)
    with caplog.at_level(logging.ERROR, logger="social_tracker.web"):
        response = client.post("/analysis", data={"csrf_token": token, "on": "1"})
    assert response.status_code == 500
    assert "Something went wrong" in response.text
    assert "secret-" not in response.text
    assert "secret-" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_a_corrupt_database_is_a_friendly_page_not_a_traceback(tmp_path: Path):
    bad = tmp_path / "bad.db"
    bad.write_text("this is not a database" * 50, encoding="utf-8")
    env = {DB_PATH_ENV: str(bad)}
    client = TestClient(
        create_app(lambda: build_container(env), host="127.0.0.1"),
        base_url=BASE,
        raise_server_exceptions=False,
    )
    response = client.get("/")
    assert response.status_code == 503
    assert "Traceback" not in response.text
    assert bad.read_text(encoding="utf-8").startswith("this is not a database")  # untouched
