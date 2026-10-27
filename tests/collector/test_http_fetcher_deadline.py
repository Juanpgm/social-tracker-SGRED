"""The fetch deadline is enforced INSIDE the I/O: no peer can stretch a fetch past it.

These tests use real sockets and real time on loopback (scaled down: a deadline of one second).
A peer that sends one byte at a time resets every per-read timeout, so only a wall-clock budget
spread over every read, write, connect and name lookup keeps the fetch inside `deadline_s`.
"""

import socket
import threading
import time
from datetime import UTC, datetime

import httpcore2
import httpx2
import pytest

from social_tracker.collector.domain import FetchError, FetchErrorKind, FetchRequest
from social_tracker.collector.http_fetcher import FetchLimits, HttpFetcher, _DeadlineBackend
from social_tracker.shared.clock import FixedClock
from tests.collector.netserver import Server, read_request, trickle
from tests.collector.test_http_fetcher import FakeResolver, build, failure, ok, reply

NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
DEADLINE = 1.0
SLACK = 0.5  # what a loaded machine may add to the deadline
OVERLONG = 8.0  # how long a hostile peer would keep going if nothing stopped it
HEAD = b"HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\n"


def fetcher_for(*, timeout_s: float = 5.0, deadline_s: float = DEADLINE, **options) -> HttpFetcher:
    return HttpFetcher(
        FixedClock(NOW),
        limits=FetchLimits(timeout_s=timeout_s, deadline_s=deadline_s),
        private_hosts=["127.0.0.1"],
        allowed_ports=None,
        **options,
    )


def timed_failure(fetcher: HttpFetcher, url: str) -> tuple[FetchError, float]:
    started = time.monotonic()
    error = failure(lambda: fetcher.fetch(FetchRequest(url)))
    return error, time.monotonic() - started


def assert_ended_at_the_deadline(error: FetchError, elapsed: float) -> None:
    assert error.kind is FetchErrorKind.TIMEOUT
    assert elapsed < DEADLINE + SLACK
    assert elapsed > DEADLINE - 0.3  # it ran into the deadline, not into something else


# ---- a peer that never finishes the headers ----


def test_a_header_trickle_ends_at_the_deadline_not_when_the_peer_stops():
    def script(conn, stop):
        read_request(conn, stop)
        trickle(conn, stop, HEAD + b"X-Pad: " + b"a" * 400, every_s=0.2)

    with Server(script) as server:
        error, elapsed = timed_failure(fetcher_for(), server.url())
    assert_ended_at_the_deadline(error, elapsed)


def test_a_silent_peer_ends_at_the_deadline_even_with_a_long_per_read_timeout():
    def script(conn, stop):
        read_request(conn, stop)
        stop.wait(OVERLONG)

    with Server(script) as server:
        error, elapsed = timed_failure(fetcher_for(timeout_s=30.0), server.url())
    assert_ended_at_the_deadline(error, elapsed)


def test_a_peer_that_never_answers_a_tls_handshake_ends_at_the_deadline():
    def script(conn, stop):
        stop.wait(OVERLONG)  # accepts the connection and says nothing

    with Server(script) as server:
        error, elapsed = timed_failure(fetcher_for(timeout_s=30.0), server.url(scheme="https"))
    assert_ended_at_the_deadline(error, elapsed)


def test_a_per_read_timeout_shorter_than_the_deadline_still_applies():
    def script(conn, stop):
        read_request(conn, stop)
        stop.wait(OVERLONG)

    with Server(script) as server:
        error, elapsed = timed_failure(fetcher_for(timeout_s=0.3, deadline_s=5.0), server.url())
    assert error.kind is FetchErrorKind.TIMEOUT
    assert 0.2 < elapsed < 0.3 + SLACK


# ---- a peer that never finishes the body ----


def test_a_body_trickle_ends_at_the_deadline():
    def script(conn, stop):
        read_request(conn, stop)
        conn.sendall(HEAD + b"Content-Length: 100000\r\nConnection: close\r\n\r\n")
        trickle(conn, stop, b"<rss>" + b"x" * 400, every_s=0.2)

    with Server(script) as server:
        error, elapsed = timed_failure(fetcher_for(), server.url())
    assert_ended_at_the_deadline(error, elapsed)


def test_a_slow_body_that_finishes_inside_the_deadline_is_accepted():
    body = b"<rss>ok</rss>"

    def script(conn, stop):
        read_request(conn, stop)
        conn.sendall(HEAD + b"Content-Length: %d\r\nConnection: close\r\n\r\n" % len(body))
        trickle(conn, stop, body, every_s=0.03)

    with Server(script) as server:
        response = fetcher_for(deadline_s=5.0).fetch(FetchRequest(server.url()))
    assert response.body == body


# ---- a redirect chain spends one budget ----


def test_a_chain_of_slow_redirects_shares_one_deadline():
    def script(conn, stop):
        read_request(conn, stop)
        stop.wait(0.9)
        conn.sendall(b"HTTP/1.1 302 Found\r\nLocation: /next\r\nContent-Length: 0\r\n")
        conn.sendall(b"Connection: close\r\n\r\n")

    with Server(script) as server:
        error, elapsed = timed_failure(fetcher_for(timeout_s=5.0), server.url())
    assert error.kind is FetchErrorKind.TIMEOUT
    assert elapsed < DEADLINE + SLACK  # the second hop was cut short, not given a fresh budget
    assert server.connections <= 2


# ---- headers are bounded in size ----


def test_endless_headers_are_refused_by_size_not_by_time():
    def script(conn, stop):
        read_request(conn, stop)
        conn.sendall(HEAD)
        for _ in range(400):
            conn.sendall(b"X-Pad: " + b"a" * 1000 + b"\r\n")

    with Server(script) as server:
        error, elapsed = timed_failure(fetcher_for(), server.url())
    assert error.kind is FetchErrorKind.PROTOCOL
    assert elapsed < DEADLINE


# ---- name resolution runs under the deadline too ----


class SlowResolver:
    def __init__(self, delay_s: float, answer: list[str] | None = None) -> None:
        self.release = threading.Event()
        self.delay_s = delay_s
        self.answer = answer or ["93.184.216.34"]
        self.calls = 0

    def __call__(self, host: str, port: int) -> list[str]:
        self.calls += 1
        self.release.wait(self.delay_s)
        return self.answer


def test_a_name_lookup_that_hangs_is_cut_at_the_deadline():
    resolver = SlowResolver(OVERLONG)
    fetcher = fetcher_for(resolver=resolver)
    try:
        error, elapsed = timed_failure(fetcher, "http://slow.example.org/feed")
    finally:
        resolver.release.set()
    assert_ended_at_the_deadline(error, elapsed)
    assert resolver.calls == 1


def test_a_name_lookup_that_answers_inside_the_deadline_is_used():
    resolver = SlowResolver(0.2)
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.host)
        return reply(200, content=b"<rss/>")

    fetcher = HttpFetcher(
        FixedClock(NOW),
        limits=FetchLimits(timeout_s=5.0, deadline_s=DEADLINE),
        resolver=resolver,
        transport=httpx2.MockTransport(handler),
    )
    assert fetcher.fetch(FetchRequest("https://slow.example.org/feed")).body == b"<rss/>"
    assert seen == ["93.184.216.34"]


def test_a_resolver_failure_is_passed_on_unchanged():
    def broken(host: str, port: int) -> list[str]:
        raise FetchError(FetchErrorKind.DNS)

    fetcher = fetcher_for(resolver=broken)
    error = failure(lambda: fetcher.fetch(FetchRequest("http://x.example.org/")))
    assert error.kind is FetchErrorKind.DNS


def test_an_unexpected_resolver_exception_is_not_swallowed():
    def broken(host: str, port: int) -> list[str]:
        raise RuntimeError("boom")

    fetcher = fetcher_for(resolver=broken)
    with pytest.raises(RuntimeError):
        fetcher.fetch(FetchRequest("http://x.example.org/"))


def test_a_budget_already_spent_before_the_lookup_never_starts_one():
    resolver = FakeResolver()
    reads: list[int] = []

    def monotonic() -> float:
        reads.append(1)
        return 0.0 if len(reads) == 1 else 99.0

    fetcher, _, _ = build(ok(), resolver, limits=FetchLimits(deadline_s=5.0), monotonic=monotonic)
    error = failure(lambda: fetcher.fetch(FetchRequest("https://feeds.example.org/x")))
    assert error.kind is FetchErrorKind.TIMEOUT
    assert resolver.calls == []


def test_the_resolver_thread_is_a_daemon_so_a_hung_lookup_cannot_block_exit():
    resolver = SlowResolver(OVERLONG)
    fetcher = fetcher_for(resolver=resolver, deadline_s=0.3)
    try:
        failure(lambda: fetcher.fetch(FetchRequest("http://slow.example.org/feed")))
        lookups = [t for t in threading.enumerate() if t.name == "social-tracker-dns"]
        assert lookups
        assert all(t.daemon for t in lookups)
    finally:
        resolver.release.set()


def test_hung_lookups_cannot_pile_up_threads_without_bound(monkeypatch):
    monkeypatch.setattr("social_tracker.collector.http_fetcher.MAX_PENDING_LOOKUPS", 2)
    resolver = SlowResolver(OVERLONG)
    fetcher = fetcher_for(resolver=resolver, deadline_s=0.2)
    try:
        for _ in range(2):
            error = failure(lambda: fetcher.fetch(FetchRequest("http://slow.example.org/f")))
            assert error.kind is FetchErrorKind.TIMEOUT
        started = time.monotonic()
        error = failure(lambda: fetcher.fetch(FetchRequest("http://slow.example.org/f")))
        assert error.kind is FetchErrorKind.DNS  # refused at once: two lookups are still hanging
        assert time.monotonic() - started < 0.1
        assert resolver.calls == 2
        resolver.release.set()
        time.sleep(0.3)  # the finished lookups give their slots back
        resolver.delay_s = 0.0
        resolver.answer = ["10.0.0.1"]  # answered at once, and refused: nothing goes on the wire
        error = failure(lambda: fetcher.fetch(FetchRequest("http://slow.example.org/f")))
        assert error.kind is FetchErrorKind.BLOCKED_ADDRESS
    finally:
        resolver.release.set()


# ---- the deadline is checked before each address attempt ----


def test_no_further_address_is_tried_once_the_deadline_has_passed():
    clock = {"now": 0.0}
    attempts: list[str] = []

    def handler(request):
        attempts.append(request.url.host)
        clock["now"] += 10.0
        raise httpx2.ConnectError("refused")

    resolver = FakeResolver({"feeds.example.org": ["93.184.216.34", "8.8.4.4", "1.1.1.1"]})
    fetcher, _, _ = build(
        handler, resolver, limits=FetchLimits(deadline_s=5.0), monotonic=lambda: clock["now"]
    )
    error = failure(lambda: fetcher.fetch(FetchRequest("https://feeds.example.org/x")))
    assert error.kind is FetchErrorKind.TIMEOUT
    assert attempts == ["93.184.216.34"]


def test_all_addresses_are_tried_while_the_budget_lasts():
    attempts: list[str] = []

    def handler(request):
        attempts.append(request.url.host)
        raise httpx2.ConnectError("refused")

    resolver = FakeResolver({"feeds.example.org": ["93.184.216.34", "8.8.4.4"]})
    fetcher, _, _ = build(handler, resolver, monotonic=lambda: 0.0)
    error = failure(lambda: fetcher.fetch(FetchRequest("https://feeds.example.org/x")))
    assert error.kind is FetchErrorKind.CONNECT
    assert attempts == ["93.184.216.34", "8.8.4.4"]


def test_a_closed_port_fails_inside_the_deadline():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    error, elapsed = timed_failure(fetcher_for(), f"http://127.0.0.1:{port}/feed")
    # Windows keeps retrying a refused connection for about two seconds, so there the deadline
    # (not the refusal) ends it: both are failures well inside the budget.
    assert error.kind in (FetchErrorKind.CONNECT, FetchErrorKind.TIMEOUT)
    assert elapsed < DEADLINE + SLACK


# ---- the network backend spends the remaining budget ----


class RecordingStream(httpcore2.NetworkStream):
    def __init__(self) -> None:
        self.calls: list[tuple[str, float | None]] = []

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.calls.append(("read", timeout))
        return b"x"

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.calls.append(("write", timeout))

    def close(self) -> None:
        self.calls.append(("close", None))

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self.calls.append(("tls", timeout))
        return self


class RecordingBackend(httpcore2.NetworkBackend):
    def __init__(self, stream: RecordingStream | None = None) -> None:
        self.stream = stream or RecordingStream()
        self.connects: list[float | None] = []

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.connects.append(timeout)
        return self.stream


def backend_at(now: dict[str, float], deadline: float = 10.0):
    inner = RecordingBackend()
    return _DeadlineBackend(inner, lambda: deadline - now["t"]), inner


def test_every_operation_gets_the_smaller_of_its_timeout_and_the_remaining_budget():
    now = {"t": 0.0}
    backend, inner = backend_at(now)
    stream = backend.connect_tcp("1.2.3.4", 80, timeout=30.0)
    now["t"] = 7.5
    stream.read(10, timeout=30.0)
    stream.write(b"x", timeout=1.0)
    stream.read(10, timeout=None)
    stream.start_tls(None, "h", timeout=30.0)
    assert inner.connects == [10.0]
    assert inner.stream.calls == [("read", 2.5), ("write", 1.0), ("read", 2.5), ("tls", 2.5)]


@pytest.mark.parametrize("spent", [10.0, 10.5, 1e9])
def test_an_exhausted_budget_fails_without_touching_the_socket(spent):
    now = {"t": 0.0}
    backend, inner = backend_at(now)
    stream = backend.connect_tcp("1.2.3.4", 80, timeout=30.0)
    now["t"] = spent
    with pytest.raises(httpcore2.ReadTimeout):
        stream.read(10, timeout=30.0)
    with pytest.raises(httpcore2.WriteTimeout):
        stream.write(b"x", timeout=30.0)
    with pytest.raises(httpcore2.ConnectTimeout):
        stream.start_tls(None, "h", timeout=30.0)
    with pytest.raises(httpcore2.ConnectTimeout):
        backend.connect_tcp("1.2.3.4", 80, timeout=30.0)
    assert inner.stream.calls == []
    assert inner.connects == [10.0]  # only the first connect, made while the budget lasted


def test_closing_and_asking_about_the_socket_pass_through_after_the_deadline():
    now = {"t": 0.0}
    backend, inner = backend_at(now)
    stream = backend.connect_tcp("1.2.3.4", 80, timeout=1.0)
    now["t"] = 99.0
    stream.close()
    assert inner.stream.calls == [("close", None)]
    assert stream.get_extra_info("nothing") is None


def test_a_socket_error_is_still_a_read_error_not_a_timeout():
    class Broken(RecordingStream):
        def read(self, max_bytes, timeout=None):
            raise httpcore2.ReadError("reset")

    backend = _DeadlineBackend(RecordingBackend(Broken()), lambda: 5.0)
    with pytest.raises(httpcore2.ReadError):
        backend.connect_tcp("1.2.3.4", 80, timeout=1.0).read(1, 1.0)
