"""A stand-in for Ollama: a stdlib HTTP server on 127.0.0.1, in a thread, with scripted answers.

It is NOT Ollama and never starts or reaches it. It exists so that a few tests can cross a real
socket (redirects, refused connections, trickled bodies, proxy variables).
"""

import json
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def json(self) -> Any:
        return json.loads(self.body)


@dataclass
class Script:
    """What the server does for the next requests: a list of callables taking the handler."""

    steps: list[Callable[[BaseHTTPRequestHandler], None]] = field(default_factory=list)
    default: Callable[[BaseHTTPRequestHandler], None] | None = None


def reply(status: int, body: bytes | dict | str = b"", headers: dict[str, str] | None = None):
    payload = (
        json.dumps(body).encode()
        if isinstance(body, dict)
        else (body.encode() if isinstance(body, str) else body)
    )

    def step(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(status)
        for key, value in (headers or {}).items():
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    return step


def redirect(location: str):
    return reply(302, b"", {"Location": location})


def trickle(total: int, pause_s: float):
    """Promise `total` bytes and send them one by one, slowly (slow-loris)."""

    def step(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Length", str(total))
        handler.end_headers()
        try:
            for _ in range(total):
                handler.wfile.write(b"x")
                handler.wfile.flush()
                time.sleep(pause_s)
        except OSError:
            pass

    return step


def silence(seconds: float):
    def step(handler: BaseHTTPRequestHandler) -> None:
        time.sleep(seconds)

    return step


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self.script = Script()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append(
                    Recorded(self.command, self.path, dict(self.headers.items()), body)
                )
                step = outer.script.steps.pop(0) if outer.script.steps else outer.script.default
                if step is None:
                    reply(404, {"error": "nothing scripted"})(self)
                else:
                    step(self)

            do_GET = do_POST = _handle  # noqa: N815

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.02), daemon=True
        )

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "FakeOllama":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def closed_port() -> int:
    """A loopback port nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
