"""A client that trickles a request body cannot hold a worker open for ever (S-S3)."""

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import pytest

from social_tracker.bootstrap import build_container
from social_tracker.entrypoints.web import formdata
from social_tracker.entrypoints.web.app import create_app
from social_tracker.entrypoints.web.security import CSRF_COOKIE_NAME, CsrfTokens
from tests.entrypoints.web.conftest import Web, visible_text

SECRET = b"t" * 32
COOKIE = "T" * 43
TOKEN = CsrfTokens(SECRET).token_for(COOKIE)
BODY = f"csrf_token={TOKEN}&on=1".encode()

Step = tuple[float, bytes]  # (seconds to wait before this chunk, the chunk)


async def call(
    app: Any, steps: Sequence[Step], *, hang_after: bool, headers: Sequence[tuple[str, str]] = ()
) -> tuple[int, dict[str, str], bytes]:
    """Drive the ASGI app by hand: the body arrives in `steps`, then either ends or never ends."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/analysis",
        "raw_path": b"/analysis",
        "query_string": b"",
        "root_path": "",
        "server": ("127.0.0.1", 8765),
        "client": ("127.0.0.1", 50000),
        "headers": [
            (b"host", b"127.0.0.1:8765"),
            (b"content-type", b"application/x-www-form-urlencoded"),
            (b"cookie", f"{CSRF_COOKIE_NAME}={COOKIE}".encode()),
            *((k.encode(), v.encode()) for k, v in headers),
        ],
    }
    queue = list(steps)
    status = 0
    sent_headers: dict[str, str] = {}
    chunks: list[bytes] = []

    async def receive() -> dict[str, Any]:
        if queue:
            delay, chunk = queue.pop(0)
            await asyncio.sleep(delay)
            return {"type": "http.request", "body": chunk, "more_body": bool(queue) or hang_after}
        if hang_after:
            await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
            sent_headers.update({k.decode(): v.decode() for k, v in message["headers"]})
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return status, sent_headers, b"".join(chunks)


@pytest.fixture
def app(web: Web, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(formdata, "BODY_TIMEOUT_S", 0.4)
    return create_app(lambda: build_container(web.env), host="127.0.0.1", secret=SECRET)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_a_body_that_stops_arriving_is_a_408_page_after_the_deadline(app, web: Web):
    started = time.monotonic()
    status, headers, body = run(call(app, [(0, BODY[:5])], hang_after=True))
    assert status == 408
    assert 0.3 <= time.monotonic() - started < 5
    assert "took too long" in visible_text(body.decode())
    assert headers["x-frame-options"] == "DENY"
    assert web.events("global_analysis_changed") == []


def test_a_trickle_is_cut_by_the_total_deadline_not_by_the_gap_between_chunks(app, web: Web):
    steps: list[Step] = [(0.1, b"a")] * 30  # each gap is short; the whole takes 3 s
    started = time.monotonic()
    status, _, _ = run(call(app, steps, hang_after=True))
    assert status == 408
    assert time.monotonic() - started < 2
    assert web.events("global_analysis_changed") == []


def test_a_slow_but_complete_body_inside_the_deadline_is_accepted(app, web: Web):
    steps: list[Step] = [(0.05, BODY[:10]), (0.05, BODY[10:20]), (0.05, BODY[20:])]
    status, headers, _ = run(call(app, steps, hang_after=False))
    assert status == 303
    assert headers["location"] == "/?notice=analysis_on"
    assert len(web.events("global_analysis_changed")) == 1


def test_the_timeout_answers_an_htmx_request_with_a_fragment(app):
    status, headers, body = run(
        call(app, [(0, BODY[:5])], hang_after=True, headers=[("hx-request", "true")])
    )
    assert status == 408
    assert headers["hx-reswap"] == "none"
    assert "<!doctype" not in body.decode().lower()
    assert 'hx-swap-oob="innerHTML:#live-region"' in body.decode()


def test_a_declared_length_check_still_comes_before_any_waiting(app):
    steps: list[Step] = []
    status, _, _ = run(
        call(
            app,
            steps,
            hang_after=True,
            headers=[("content-length", str(formdata.MAX_BODY_BYTES + 1))],
        )
    )
    assert status == 413


def test_the_deadline_is_ten_seconds():
    assert formdata.BODY_TIMEOUT_S == 10
