"""Containers per request and their release: no query per row, and nothing left open (R6, R7)."""

import logging
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from social_tracker.bootstrap import Container, build_container
from social_tracker.entrypoints.web.app import create_app
from social_tracker.entrypoints.web.security import CSRF_COOKIE_NAME, CsrfTokens
from social_tracker.shared.errors import StorageError
from tests.actors import WHO
from tests.entrypoints.web.conftest import BASE, Web
from tests.entrypoints.web.test_forms_add import BASE_FORM, VALID

SECRET = b"r" * 32
COOKIE = "R" * 43
TOKEN = CsrfTokens(SECRET).token_for(COOKIE)
HTMX = {"HX-Request": "true"}


class Proxy:
    """Delegates everything to `inner`, except the attributes in `overrides`."""

    def __init__(self, inner: Any, **overrides: Any) -> None:
        self._inner, self._overrides = inner, overrides

    def __getattr__(self, name: str) -> Any:
        return self._overrides.get(name, getattr(self._inner, name))


@dataclass
class Counting:
    """A container factory that records every container opened, closed and SQL statement run."""

    env: dict[str, str]
    wrap: Callable[[Container], Any] = lambda container: container
    opened: int = 0
    closed: int = 0
    statements: list[str] = field(default_factory=list)

    def __call__(self) -> Any:
        real = build_container(self.env)
        self.opened += 1
        real._db._conn.set_trace_callback(self.statements.append)  # type: ignore[union-attr]
        counter = self

        def close() -> None:
            counter.closed += 1
            real.close()

        return Proxy(self.wrap(real), close=close)

    def client(self) -> TestClient:
        app = create_app(self, host="127.0.0.1", secret=SECRET)
        client = TestClient(app, base_url=BASE, raise_server_exceptions=False)
        client.cookies.set(CSRF_COOKIE_NAME, COOKIE)
        return client

    def reset(self) -> None:
        self.opened = self.closed = 0
        self.statements.clear()


def add_failing(web: Web, numbers: range) -> None:
    """Enabled sources that failed once: in backoff, so each row has a failure count and retry."""
    with closing(web.services()) as services:
        for n in numbers:
            source_id = f"src_{n:02d}"
            services.sources.add(
                {
                    "type": "news_rss",
                    "id": source_id,
                    "name": f"Source {n}",
                    "credibility": 3,
                    "feed_url": "https://example.org/rss.xml",
                },
                **WHO,
            )
            services.sources.enable(source_id, **WHO)
            services.control.start_collection(source_id, **WHO)
            services.control.fail_collection(source_id, **WHO)


def shapes(statements: list[str]) -> list[str]:
    return [sql.split()[0] for sql in statements]


def toggle(client: TestClient, source_id: str, action: str):
    return client.post(
        f"/sources/{source_id}/{action}",
        content=f"csrf_token={TOKEN}",
        headers={**HTMX, "Content-Type": "application/x-www-form-urlencoded"},
    )


# ---- no query per row ----


def test_the_list_opens_one_container_and_runs_the_same_statements_for_1_and_30_failing(
    web: Web,
):
    counting = Counting(web.env)
    client = counting.client()
    add_failing(web, range(1))
    counting.reset()
    assert client.get("/").text.count("<tr id=") == 1
    few = (counting.opened, shapes(counting.statements))
    add_failing(web, range(1, 30))
    counting.reset()
    page = client.get("/").text
    many = (counting.opened, shapes(counting.statements))
    assert page.count("<tr id=") == 30
    assert page.count("1 failure, retry after") == 30  # the values are real, not skipped
    assert few[0] == many[0] == 1
    assert few[1]  # the trace really recorded something
    assert many[1] == few[1]


def test_an_htmx_toggle_opens_one_container_and_the_same_statements_for_1_and_30_failing(
    web: Web,
):
    counting = Counting(web.env)
    client = counting.client()
    add_failing(web, range(1))
    counting.reset()
    assert toggle(client, "src_00", "disable").status_code == 200
    few = (counting.opened, shapes(counting.statements))
    add_failing(web, range(1, 30))
    with closing(web.services()) as services:
        services.sources.enable("src_00", **WHO)
    counting.reset()
    response = toggle(client, "src_00", "disable")
    many = (counting.opened, shapes(counting.statements))
    assert response.status_code == 200
    assert few[0] == many[0] == 1
    assert "BEGIN" in few[1]  # the write ran in the same trace as the read
    assert many[1] == few[1]


# ---- every container is closed ----


def form(**fields: str) -> dict[str, Any]:
    body = "&".join(f"{k}={v}" for k, v in {"csrf_token": TOKEN, **fields}.items())
    return {"content": body, "headers": {"Content-Type": "application/x-www-form-urlencoded"}}


def storage_failure(container: Container) -> Any:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise StorageError("database error: SQLITE_BUSY")

    return Proxy(container, sources=Proxy(container.sources, enable=boom, delete=boom, add=boom))


def crash(container: Container) -> Any:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("unexpected")

    return Proxy(container, control=Proxy(container.control, overview=boom))


SCENARIOS: list[tuple[str, Callable[[Container], Any], str, str, dict[str, Any]]] = [
    ("list", lambda c: c, "GET", "/", {}),
    ("htmx enable", lambda c: c, "POST", "/sources/src_one/enable", {**form(), "_h": 1}),
    ("htmx not found", lambda c: c, "POST", "/sources/src_gone/enable", {**form(), "_h": 1}),
    ("plain not found", lambda c: c, "POST", "/sources/src_gone/enable", form()),
    (
        "storage error, htmx",
        storage_failure,
        "POST",
        "/sources/src_one/enable",
        {**form(), "_h": 1},
    ),
    ("storage error, plain", storage_failure, "POST", "/sources/src_one/enable", form()),
    ("500 in a read", crash, "GET", "/", {}),
    ("500 in a toggle, htmx", crash, "POST", "/sources/src_one/enable", {**form(), "_h": 1}),
    ("edit form", lambda c: c, "GET", "/sources/src_one/edit", {}),
    ("edit form, not found", lambda c: c, "GET", "/sources/src_gone/edit", {}),
    ("delete", lambda c: c, "POST", "/sources/src_one/delete", form()),
    ("delete, storage error", storage_failure, "POST", "/sources/src_one/delete", form()),
    ("delete, not found", lambda c: c, "POST", "/sources/src_gone/delete", form()),
    ("analysis switch", lambda c: c, "POST", "/analysis", form(on="1")),
    (
        "add",
        lambda c: c,
        "POST",
        "/sources/new/news_rss",
        form(**{**BASE_FORM, **VALID["news_rss"], "id": "src_new", "name": "New"}),
    ),
    (
        "add, storage error",
        storage_failure,
        "POST",
        "/sources/new/news_rss",
        form(**{**BASE_FORM, **VALID["news_rss"], "id": "src_new", "name": "New"}),
    ),
    ("add, rejected", lambda c: c, "POST", "/sources/new/news_rss", form(id="x")),
]


@pytest.mark.parametrize(
    ("wrap", "method", "path", "kwargs"),
    [s[1:] for s in SCENARIOS],
    ids=[s[0] for s in SCENARIOS],
)
def test_every_container_a_request_opens_is_closed(web: Web, wrap, method, path, kwargs, caplog):
    web.seed()
    counting = Counting(web.env, wrap)
    client = counting.client()
    kwargs = dict(kwargs)
    headers = dict(kwargs.pop("headers", {}))
    if kwargs.pop("_h", None):
        headers.update(HTMX)
    with caplog.at_level(logging.ERROR, logger="social_tracker.web"):
        response = client.request(method, path, headers=headers, **kwargs)
    assert response.status_code != 0
    assert counting.opened >= 1
    assert counting.closed == counting.opened
