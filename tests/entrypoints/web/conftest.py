"""Fixtures to drive the real web UI (FastAPI TestClient) against a throwaway SQLite database."""

import html
import re
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV, Container, build_container
from social_tracker.control.domain import ControlEvent
from social_tracker.entrypoints.web.app import create_app
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from tests.actors import WHO
from tests.sources.factories import payload

BASE = "http://127.0.0.1:8765"
TOKEN_FIELD = re.compile(r'name="csrf_token" value="([^"]+)"')


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        self._skip += tag in ("script", "style")

    def handle_endtag(self, tag: str) -> None:
        self._skip -= tag in ("script", "style")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def visible_text(page: str) -> str:
    """The text a person reads: tags and scripts dropped, entities decoded, spaces collapsed."""
    parser = _Text()
    parser.feed(page)
    return re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts))).strip()


def row_of(page: str, source_id: str) -> str:
    found = re.search(rf'<tr id="row-{source_id}".*?</tr>', page, re.DOTALL)
    assert found, f"no row for {source_id}"
    return found.group()


@dataclass
class Web:
    client: TestClient
    env: dict[str, str]
    services: Callable[[], Container]

    def csrf(self) -> str:
        """A valid token for the client's current session (the client keeps the cookie)."""
        found = TOKEN_FIELD.search(self.client.get("/").text)
        assert found
        return found.group(1)

    def post(
        self, path: str, data: Mapping[str, str] | None = None, *, token: str | None = None
    ) -> Any:
        body = {"csrf_token": self.csrf() if token is None else token, **(data or {})}
        return self.client.post(path, data=body, follow_redirects=False)

    def post_htmx(self, path: str, data: Mapping[str, str] | None = None) -> Any:
        body = {"csrf_token": self.csrf(), **(data or {})}
        return self.client.post(path, data=body, headers={"HX-Request": "true"})

    def seed(self, source_id: str = "src_one", type_: str = "news_rss", **extra: Any) -> None:
        with closing(self.services()) as services:
            services.sources.add(payload(type_, id=source_id, **extra), **WHO)

    def events(self, kind: str | None = None) -> list[ControlEvent]:
        with Database(Path(self.env[DB_PATH_ENV])) as db:
            found = SqliteControlRepository(db).list_control_events()
        return [e for e in found if kind is None or e.kind == kind]


@pytest.fixture
def web(tmp_path: Path) -> Web:
    env = {DB_PATH_ENV: str(tmp_path / "web.db"), BUSY_TIMEOUT_ENV: "2000"}

    def services() -> Container:
        return build_container(env)

    app = create_app(services, host="127.0.0.1")
    return Web(TestClient(app, base_url=BASE), env, services)
