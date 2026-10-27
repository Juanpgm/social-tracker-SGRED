"""A wall-clock deadline enforced INSIDE the I/O of an HTTP client (shared by the two clients).

A per-read timeout restarts on every byte, so a peer that sends one byte at a time (a slow-loris
on the response HEADERS as much as on the body) would hold the caller for as long as it likes.
These wrappers give every connect, TLS handshake, read and write at most the budget that is left
of ONE deadline, whatever the peer does. `remaining()` is called before every operation; when it
is not positive the operation fails with the timeout error of its kind, which `httpx2` maps to
its own `TimeoutException`.

Used by the feed fetcher (`collector.http_fetcher`) and by the Ollama adapter (`llm.ollama`). It
opens no connection of its own: it wraps the ones its transport makes.
"""

import ssl
from collections.abc import Callable, Iterable
from typing import Any

import httpcore2
import httpx2


class DeadlineStream(httpcore2.NetworkStream):
    """A connection whose every operation gets at most the budget that is left."""

    def __init__(self, inner: httpcore2.NetworkStream, remaining: Callable[[], float]) -> None:
        self._inner = inner
        self._remaining = remaining

    def _budget(self, timeout: float | None, expired: type[Exception]) -> float:
        left = self._remaining()
        if left <= 0:
            raise expired("deadline reached")
        return left if timeout is None else min(timeout, left)

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._inner.read(max_bytes, self._budget(timeout, httpcore2.ReadTimeout))

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._inner.write(buffer, self._budget(timeout, httpcore2.WriteTimeout))

    def close(self) -> None:
        self._inner.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore2.NetworkStream:
        budget = self._budget(timeout, httpcore2.ConnectTimeout)
        wrapped = self._inner.start_tls(ssl_context, server_hostname, budget)
        return DeadlineStream(wrapped, self._remaining)

    def get_extra_info(self, info: str) -> Any:
        return self._inner.get_extra_info(info)


class DeadlineBackend(httpcore2.NetworkBackend):
    def __init__(self, inner: httpcore2.NetworkBackend, remaining: Callable[[], float]) -> None:
        self._inner = inner
        self._remaining = remaining

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore2.NetworkStream:
        left = self._remaining()
        if left <= 0:
            raise httpcore2.ConnectTimeout("deadline reached")
        budget = left if timeout is None else min(timeout, left)
        stream = self._inner.connect_tcp(host, port, budget, local_address, socket_options)
        return DeadlineStream(stream, self._remaining)


class DeadlineTransport(httpx2.HTTPTransport):
    """The default HTTP transport with its connections wrapped in the deadline of one request."""

    def __init__(self, ssl_context: ssl.SSLContext, remaining: Callable[[], float]) -> None:
        super().__init__(verify=ssl_context, trust_env=False)
        self._pool.close()  # replaced below by a pool over the deadline-aware backend
        self._pool = httpcore2.ConnectionPool(
            ssl_context=ssl_context,
            network_backend=DeadlineBackend(httpcore2.SyncBackend(), remaining),
            http2=False,
            retries=0,
        )
