"""Fixtures for the real-browser checks: a live server on a free port and a headless Chromium.

These tests are opt-in (`uv run pytest -m browser`). Browsers are not downloaded by the test run:
install one once, outside the system drive if you like:

    set PLAYWRIGHT_BROWSERS_PATH=D:\\social_tracker_models\\playwright
    uv run playwright install chromium

and give the same variable to the test run. Without Playwright or a browser the tests are skipped.
"""

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV, TIMEZONE_ENV, build_container
from social_tracker.control.domain import ControlEvent
from social_tracker.entrypoints.web.app import create_app
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from tests.actors import WHO
from tests.sources.factories import payload

sync_api = pytest.importorskip("playwright.sync_api")

HOST = "127.0.0.1"


def free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


class _Thread(threading.Thread):
    def __init__(self, server: uvicorn.Server) -> None:
        super().__init__(daemon=True)
        self.server = server

    def run(self) -> None:
        self.server.run()


@dataclass
class Live:
    """The UI served by uvicorn from a thread. `restart` is a new process in the user's eyes: the
    port and the database stay, the CSRF secret (and so every token already issued) changes."""

    env: dict[str, str]
    port: int
    _thread: _Thread | None = field(default=None, init=False)
    _generation: int = field(default=0, init=False)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def start(self) -> None:
        self._generation += 1
        secret = bytes([self._generation]) * 32
        app = create_app(lambda: build_container(self.env), host=HOST, secret=secret)
        config = uvicorn.Config(app, host=HOST, port=self.port, log_level="error")
        server = uvicorn.Server(config)
        self._thread = _Thread(server)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            assert self._thread.is_alive(), "the server thread died while starting"
            assert time.monotonic() < deadline, "the server did not start"
            time.sleep(0.02)

    def stop(self) -> None:
        if self._thread is not None:
            self._thread.server.should_exit = True
            self._thread.join(timeout=10)
            self._thread = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def seed(self, source_id: str = "src_one", **extra: Any) -> None:
        with closing(build_container(self.env)) as services:
            services.sources.add(payload("news_rss", id=source_id, **extra), **WHO)

    def enable(self, source_id: str) -> None:
        with closing(build_container(self.env)) as services:
            services.sources.enable(source_id, **WHO)

    def events(self, kind: str) -> list[ControlEvent]:
        with Database(Path(self.env[DB_PATH_ENV])) as db:
            found = SqliteControlRepository(db).list_control_events()
        return [e for e in found if e.kind == kind]


@pytest.fixture
def live(tmp_path: Path) -> Iterator[Live]:
    env = {
        DB_PATH_ENV: str(tmp_path / "browser.db"),
        BUSY_TIMEOUT_ENV: "150",
        TIMEZONE_ENV: "UTC",
    }
    server = Live(env, free_port())
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="session")
def browser() -> Iterator[Any]:
    with sync_api.sync_playwright() as playwright:
        try:
            launched = playwright.chromium.launch(headless=True)
        except sync_api.Error as exc:
            pytest.skip(f"no Chromium available ({str(exc).splitlines()[0]})")
        try:
            yield launched
        finally:
            launched.close()


@pytest.fixture
def page(browser: Any) -> Iterator[Any]:
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(8000)
    problems: list[str] = []
    page.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.append(str(e)))
    page.problems = problems  # type: ignore[attr-defined]
    try:
        yield page
    finally:
        context.close()
