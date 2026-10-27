"""The httpx2 fetcher, offline: a mock transport replaces the network and a fake resolver DNS.

Every outgoing request goes to the validated IP address (the connection cannot be redirected by
a second, different DNS answer), carries the original host name in `Host`/SNI, and is judged
again at every redirect hop.
"""

import gzip
import logging
import socket
import zlib
from collections.abc import Callable, Iterator
from datetime import UTC, datetime

import httpx2
import pytest

from social_tracker.collector.domain import (
    MAX_RETRY_AFTER_S,
    FetchError,
    FetchErrorKind,
    FetchRequest,
    FetchResponse,
)
from social_tracker.collector.http_fetcher import FetchLimits, HttpFetcher, system_resolver
from social_tracker.collector.ports import Fetcher
from social_tracker.shared.clock import FixedClock

NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
PUBLIC = "93.184.216.34"
OTHER_PUBLIC = "8.8.4.4"
URL = "https://feeds.example.org/rss.xml"
Handler = Callable[[httpx2.Request], httpx2.Response]


class _Body(httpx2.SyncByteStream):
    """A response body that is still unread when the fetcher gets it, like a real connection."""

    def __init__(self, content: bytes | Iterator[bytes]) -> None:
        self._content = content

    def __iter__(self) -> Iterator[bytes]:
        if isinstance(self._content, bytes):
            yield self._content
        else:
            yield from self._content


def reply(
    status: int = 200,
    *,
    content: bytes | Iterator[bytes] = b"",
    headers: dict[str, str] | None = None,
) -> httpx2.Response:
    return httpx2.Response(status, headers=headers, stream=_Body(content))


class FakeResolver:
    def __init__(self, table: dict[str, list[str]] | None = None) -> None:
        self.table = {"feeds.example.org": [PUBLIC], **(table or {})}
        self.calls: list[tuple[str, int]] = []
        self.sequence: dict[str, list[list[str]]] = {}

    def __call__(self, host: str, port: int) -> list[str]:
        self.calls.append((host, port))
        if host in self.sequence and self.sequence[host]:
            return self.sequence[host].pop(0)
        if host not in self.table:
            raise FetchError(FetchErrorKind.DNS)
        return list(self.table[host])


class Net:
    """A scripted network: records the requests it receives."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.handler(request)


def build(
    handler: Handler,
    resolver: FakeResolver | None = None,
    *,
    limits: FetchLimits | None = None,
    private_hosts: frozenset[str] = frozenset(),
    monotonic: Callable[[], float] | None = None,
    **options: object,
) -> tuple[HttpFetcher, Net, FakeResolver]:
    net = Net(handler)
    resolver = resolver or FakeResolver()
    kwargs = {"monotonic": monotonic, **options} if monotonic else dict(options)
    fetcher = HttpFetcher(
        FixedClock(NOW),
        limits=limits or FetchLimits(),
        resolver=resolver,
        transport=httpx2.MockTransport(net),
        private_hosts=private_hosts,
        **kwargs,
    )
    return fetcher, net, resolver


def ok(body: bytes = b"<rss/>", **headers: str) -> Handler:
    return lambda request: reply(200, content=body, headers=headers)


def failure(call: Callable[[], object]) -> FetchError:
    with pytest.raises(FetchError) as caught:
        call()
    return caught.value


def get(fetcher: HttpFetcher, url: str = URL, **validators: str) -> FetchResponse:
    return fetcher.fetch(FetchRequest(url, **validators))


def test_it_satisfies_the_fetcher_port():
    fetcher, _, _ = build(ok())
    assert isinstance(fetcher, Fetcher)


# ---- a plain fetch ----


def test_a_successful_fetch_returns_the_body_and_the_validators():
    fetcher, _, _ = build(
        ok(b"<rss>x</rss>", etag='"abc"', **{"last-modified": "Sun, 01 Mar 2026"})
    )
    response = get(fetcher)
    assert response == FetchResponse(
        b"<rss>x</rss>", etag='"abc"', last_modified="Sun, 01 Mar 2026", not_modified=False
    )


def test_the_request_goes_to_the_validated_address_with_the_original_host():
    fetcher, net, resolver = build(ok())
    get(fetcher)
    (request,) = net.requests
    assert resolver.calls == [("feeds.example.org", 443)]
    assert request.url.host == PUBLIC
    assert request.url.scheme == "https"
    assert request.headers["host"] == "feeds.example.org"
    assert request.extensions["sni_hostname"] == "feeds.example.org"
    assert request.url.path == "/rss.xml"


def test_plain_http_has_no_sni_and_keeps_a_non_default_port_in_the_host_header():
    fetcher, net, resolver = build(ok())
    get(fetcher, "http://feeds.example.org:8080/a/b?x=1")
    (request,) = net.requests
    assert resolver.calls == [("feeds.example.org", 8080)]
    assert request.url.port == 8080
    assert request.headers["host"] == "feeds.example.org:8080"
    assert "sni_hostname" not in request.extensions
    assert request.url.query == b"x=1"


def test_an_ipv6_literal_is_contacted_directly():
    fetcher, net, resolver = build(ok())
    get(fetcher, "http://[2606:2800:220:1::1]/x")
    assert resolver.calls == []
    assert net.requests[0].url.host == "2606:2800:220:1::1"


def test_the_request_carries_no_credentials_and_identifies_itself():
    fetcher, net, _ = build(ok())
    get(fetcher)
    headers = net.requests[0].headers
    assert headers["user-agent"].startswith("social-tracker/")
    assert headers["accept-encoding"] == "gzip"
    for forbidden in ("authorization", "cookie", "proxy-authorization", "referer"):
        assert forbidden not in headers


def test_the_client_ignores_environment_proxies_and_never_follows_redirects_itself(monkeypatch):
    seen: dict[str, object] = {}
    real_client = httpx2.Client

    def spy(**kwargs):
        seen.update(kwargs)
        return real_client(**kwargs)

    monkeypatch.setattr("social_tracker.collector.http_fetcher.httpx2.Client", spy)
    fetcher, _, _ = build(ok())
    get(fetcher)
    assert seen["trust_env"] is False
    assert seen["follow_redirects"] is False


def test_an_empty_body_is_returned_as_is_for_the_parser_to_judge():
    fetcher, _, _ = build(ok(b""))
    assert get(fetcher).body == b""


# ---- conditional requests ----


def test_validators_are_sent_as_conditional_headers():
    fetcher, net, _ = build(ok())
    get(fetcher, etag='"v1"', last_modified="Sun, 01 Mar 2026 10:00:00 GMT")
    headers = net.requests[0].headers
    assert headers["if-none-match"] == '"v1"'
    assert headers["if-modified-since"] == "Sun, 01 Mar 2026 10:00:00 GMT"


def test_without_validators_the_request_is_unconditional():
    fetcher, net, _ = build(ok())
    get(fetcher)
    assert "if-none-match" not in net.requests[0].headers
    assert "if-modified-since" not in net.requests[0].headers


def test_not_modified_is_reported_without_a_body():
    fetcher, _, _ = build(lambda r: reply(304, headers={"etag": '"v1"'}))
    response = get(fetcher, etag='"v1"')
    assert response.not_modified is True
    assert response.body == b""
    assert response.etag == '"v1"'


@pytest.mark.parametrize(
    ("header", "value"),
    [("etag", "e" * 513), ("etag", "bad\x7fvalue"), ("last-modified", "m" * 101), ("etag", "   ")],
)
def test_oversize_or_unprintable_validators_are_dropped(header, value):
    fetcher, _, _ = build(lambda r: reply(200, content=b"x", headers={header: value}))
    response = get(fetcher)
    assert response.etag is None
    assert response.last_modified is None


# ---- http failures ----


@pytest.mark.parametrize("status", [400, 404, 410, 500, 502, 503])
def test_error_statuses_are_fetch_errors_with_a_fixed_message(status):
    fetcher, _, _ = build(lambda r: reply(status, content=b"<html>secret page</html>"))
    error = failure(lambda: get(fetcher))
    assert (error.kind, error.status) == (FetchErrorKind.HTTP, status)
    assert error.message == f"http {status}"
    assert error.degraded is False
    assert "secret" not in error.message


@pytest.mark.parametrize("status", [401, 403, 429])
def test_access_and_rate_limit_statuses_are_degraded(status):
    fetcher, _, _ = build(lambda r: reply(status))
    error = failure(lambda: get(fetcher))
    assert error.degraded is True
    assert error.status == status


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("120", 120.0),
        ("0", 0.0),
        ("Sun, 01 Mar 2026 12:10:00 GMT", 600.0),
        ("Sat, 28 Feb 2026 12:00:00 GMT", 0.0),
        ("99999999999999999999", float(MAX_RETRY_AFTER_S)),
        ("-30", None),
        ("soon", None),
        ("1.5", None),
        ("", None),
    ],
)
@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_is_parsed_from_seconds_or_a_date_and_garbage_is_ignored(
    status, header, expected
):
    fetcher, _, _ = build(lambda r: reply(status, headers={"retry-after": header}))
    assert failure(lambda: get(fetcher)).retry_after_s == expected


def test_a_missing_retry_after_header_is_none():
    fetcher, _, _ = build(lambda r: reply(429))
    assert failure(lambda: get(fetcher)).retry_after_s is None


# ---- transport failures ----


@pytest.mark.parametrize(
    ("raised", "kind"),
    [
        (httpx2.ReadTimeout("slow"), FetchErrorKind.TIMEOUT),
        (httpx2.ConnectTimeout("slow"), FetchErrorKind.TIMEOUT),
        (httpx2.PoolTimeout("busy"), FetchErrorKind.TIMEOUT),
        (httpx2.ConnectError("refused"), FetchErrorKind.CONNECT),
        (httpx2.RemoteProtocolError("garbage"), FetchErrorKind.PROTOCOL),
        (httpx2.ReadError("reset"), FetchErrorKind.PROTOCOL),
        (httpx2.WriteError("reset"), FetchErrorKind.PROTOCOL),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, Exception) else v.value,
)
def test_transport_failures_become_typed_fetch_errors(raised, kind):
    def boom(request):
        raise raised

    fetcher, _, _ = build(boom)
    error = failure(lambda: get(fetcher))
    assert error.kind is kind
    assert "refused" not in error.message
    assert "slow" not in error.message


def test_the_next_address_is_tried_when_a_connection_is_refused():
    resolver = FakeResolver({"feeds.example.org": [PUBLIC, OTHER_PUBLIC]})

    def handler(request):
        if request.url.host == PUBLIC:
            raise httpx2.ConnectError("refused")
        return reply(200, content=b"<rss/>")

    fetcher, net, _ = build(handler, resolver)
    assert get(fetcher).body == b"<rss/>"
    assert [r.url.host for r in net.requests] == [PUBLIC, OTHER_PUBLIC]


def test_every_address_failing_is_one_connection_error():
    resolver = FakeResolver({"feeds.example.org": [PUBLIC, OTHER_PUBLIC]})

    def handler(request):
        raise httpx2.ConnectError("refused")

    fetcher, net, _ = build(handler, resolver)
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.CONNECT
    assert len(net.requests) == 2


def test_a_name_that_does_not_resolve_is_a_dns_failure_without_any_request():
    fetcher, net, _ = build(ok())
    error = failure(lambda: get(fetcher, "https://unknown.example.org/x"))
    assert error.kind is FetchErrorKind.DNS
    assert net.requests == []


def test_the_system_resolver_reports_lookup_errors_and_empty_answers_as_dns_failures(monkeypatch):
    def refuse(*args, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert failure(lambda: system_resolver("nowhere.example", 443)).kind is FetchErrorKind.DNS
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    assert failure(lambda: system_resolver("nowhere.example", 443)).kind is FetchErrorKind.DNS


def test_the_system_resolver_returns_the_addresses_in_order_without_repeats(monkeypatch):
    answer = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC, 443)),
        (socket.AF_INET, socket.SOCK_DGRAM, 17, "", (PUBLIC, 443)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:2800:220:1::1", 443, 0, 0)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: answer)
    assert system_resolver("example.org", 443) == [PUBLIC, "2606:2800:220:1::1"]


# ---- ssrf ----


@pytest.mark.parametrize(
    "answer",
    [
        ["127.0.0.1"],
        ["10.0.0.5"],
        ["169.254.169.254"],
        ["::1"],
        ["192.168.1.1"],
        [PUBLIC, "10.0.0.5"],
        ["::ffff:127.0.0.1"],
        ["100.64.0.9"],
        ["0.0.0.0"],
    ],
)
def test_a_host_that_resolves_to_an_internal_address_is_never_contacted(answer):
    resolver = FakeResolver({"feeds.example.org": answer})
    fetcher, net, _ = build(ok(), resolver)
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.BLOCKED_ADDRESS
    assert net.requests == []


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost.invalid/",
        "http://[::1]/",
        "http://10.1.2.3:8080/x",
        "http://169.254.169.254/latest/meta-data/",
        "https://192.168.0.1/",
    ],
)
def test_internal_literal_addresses_and_names_are_refused(url):
    resolver = FakeResolver({"localhost.invalid": ["127.0.0.1"]})
    fetcher, net, _ = build(ok(), resolver)
    assert failure(lambda: get(fetcher, url)).kind is FetchErrorKind.BLOCKED_ADDRESS
    assert net.requests == []


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org/x",
        "file:///etc/passwd",
        "http://2130706433/",
        "https://" + "user:pw" + "@feeds.example.org/",
        "gopher://x/",
        "",
    ],
)
def test_urls_that_are_not_plain_http_never_reach_the_network(url):
    fetcher, net, resolver = build(ok())
    assert failure(lambda: get(fetcher, url)).kind is FetchErrorKind.BAD_URL
    assert net.requests == []
    assert resolver.calls == []


def test_the_private_host_allowlist_lets_a_named_internal_host_through_and_logs_it(caplog):
    resolver = FakeResolver({"intranet.corp": ["10.20.30.40"]})
    fetcher, net, _ = build(ok(), resolver, private_hosts=frozenset({"intranet.corp"}))
    with caplog.at_level(logging.WARNING, logger="social_tracker.collector"):
        assert get(fetcher, "http://intranet.corp/feed").body == b"<rss/>"
    assert net.requests[0].url.host == "10.20.30.40"
    assert any(
        "intranet.corp" in r.getMessage() and "private" in r.getMessage() for r in caplog.records
    )


def test_the_allowlist_is_exact_and_off_by_default():
    resolver = FakeResolver({"intranet.corp": ["10.20.30.40"], "other.corp": ["10.20.30.41"]})
    allowed = frozenset({"intranet.corp"})
    fetcher, _, _ = build(ok(), resolver, private_hosts=allowed)
    assert failure(lambda: get(fetcher, "http://other.corp/feed")).kind is (
        FetchErrorKind.BLOCKED_ADDRESS
    )
    assert failure(lambda: get(fetcher, "http://sub.intranet.corp/feed")).kind is FetchErrorKind.DNS
    default, _, _ = build(ok(), resolver)
    assert failure(lambda: get(default, "http://intranet.corp/feed")).kind is (
        FetchErrorKind.BLOCKED_ADDRESS
    )


def test_an_allowlisted_ip_literal_is_contactable():
    fetcher, net, _ = build(ok(), private_hosts=frozenset({"10.20.30.40"}))
    assert get(fetcher, "http://10.20.30.40/feed").body == b"<rss/>"
    assert net.requests[0].url.host == "10.20.30.40"


# ---- redirects ----


def redirect(location: str, status: int = 302) -> httpx2.Response:
    return reply(status, headers={"location": location})


def test_a_redirect_is_followed_and_the_new_host_is_validated_and_pinned_again():
    resolver = FakeResolver({"cdn.example.net": [OTHER_PUBLIC]})

    def handler(request):
        if request.headers["host"] == "feeds.example.org":
            return redirect("https://cdn.example.net/moved.xml", 301)
        return reply(200, content=b"<rss>moved</rss>")

    fetcher, net, _ = build(handler, resolver)
    assert get(fetcher).body == b"<rss>moved</rss>"
    assert [r.url.host for r in net.requests] == [PUBLIC, OTHER_PUBLIC]
    assert [c[0] for c in resolver.calls] == ["feeds.example.org", "cdn.example.net"]
    assert net.requests[1].url.path == "/moved.xml"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_every_redirect_status_is_followed(status):
    def handler(request):
        if request.url.path == "/rss.xml":
            return redirect("/new.xml", status)
        return reply(200, content=b"<rss/>")

    fetcher, net, _ = build(handler)
    assert get(fetcher).body == b"<rss/>"
    assert net.requests[1].url.path == "/new.xml"


@pytest.mark.parametrize(
    "location",
    [
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://10.0.0.1:9200/",
        "http://metadata.internal/computeMetadata/v1/",
    ],
)
def test_a_redirect_to_an_internal_address_is_blocked(location):
    resolver = FakeResolver({"metadata.internal": ["169.254.169.254"]})
    fetcher, net, _ = build(lambda r: redirect(location), resolver)
    # started from plain http so the address check, not the downgrade check, is what is exercised
    error = failure(lambda: get(fetcher, "http://feeds.example.org/rss.xml"))
    assert error.kind is FetchErrorKind.BLOCKED_ADDRESS
    assert len(net.requests) == 1  # only the first hop was ever sent


@pytest.mark.parametrize(
    "location",
    [
        "file:///etc/passwd",
        "ftp://example.org/x",
        "javascript:1",
        "http://2130706433/",
        "http://" + "user:pw" + "@example.org/",
    ],
)
def test_a_redirect_to_another_scheme_or_a_disguised_address_is_refused(location):
    fetcher, net, _ = build(lambda r: redirect(location))
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.BAD_URL
    assert len(net.requests) == 1


def test_dns_rebinding_between_hops_is_caught_because_every_hop_resolves_again():
    resolver = FakeResolver()
    resolver.sequence["feeds.example.org"] = [[PUBLIC], ["10.0.0.7"]]

    def handler(request):
        return redirect("https://feeds.example.org/second.xml")

    fetcher, net, _ = build(handler, resolver)
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.BLOCKED_ADDRESS
    assert len(net.requests) == 1


def test_the_answer_that_was_validated_is_the_one_connected_to():
    resolver = FakeResolver()
    resolver.sequence["feeds.example.org"] = [
        [PUBLIC],
        ["10.0.0.7"],
    ]  # a second lookup would differ
    fetcher, net, _ = build(ok(), resolver)
    get(fetcher)
    assert net.requests[0].url.host == PUBLIC
    assert len(resolver.calls) == 1


def test_a_redirect_loop_ends_in_too_many_redirects():
    fetcher, net, _ = build(lambda r: redirect(URL))
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_MANY_REDIRECTS
    assert len(net.requests) == FetchLimits().max_redirects + 1


def test_a_chain_just_within_the_limit_succeeds_and_one_more_hop_fails():
    limits = FetchLimits(max_redirects=3)

    def chain(hops: int) -> Handler:
        def handler(request):
            done = int(request.url.params.get("n", "0"))
            if done < hops:
                return redirect(f"/rss.xml?n={done + 1}")
            return reply(200, content=b"<rss/>")

        return handler

    fetcher, _, _ = build(chain(3), limits=limits)
    assert get(fetcher).body == b"<rss/>"
    fetcher, _, _ = build(chain(4), limits=limits)
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_MANY_REDIRECTS


@pytest.mark.parametrize("response", [reply(302), reply(302, headers={"location": ""})])
def test_a_redirect_without_a_usable_location_is_an_http_error(response):
    fetcher, _, _ = build(lambda r: response)
    error = failure(lambda: get(fetcher))
    assert (error.kind, error.status) == (FetchErrorKind.HTTP, 302)


def test_cookies_set_by_a_response_are_not_sent_on_the_next_hop():
    def handler(request):
        if request.url.path == "/rss.xml":
            return reply(302, headers={"location": "/b.xml", "set-cookie": "sid=1; Path=/"})
        return reply(200, content=b"<rss/>")

    fetcher, net, _ = build(handler)
    get(fetcher)
    assert "cookie" not in net.requests[1].headers


def test_a_redirect_status_that_is_not_a_redirect_is_an_error_not_followed():
    fetcher, net, _ = build(lambda r: reply(300, headers={"location": "/x"}))
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.HTTP
    assert len(net.requests) == 1


# ---- size ----

LIMITS = FetchLimits(max_bytes=1000)


def chunks(total: int, size: int = 100) -> Iterator[bytes]:
    sent = 0
    while sent < total:
        piece = min(size, total - sent)
        sent += piece
        yield b"x" * piece


def test_a_body_exactly_at_the_limit_is_accepted_and_one_byte_more_is_not():
    fetcher, _, _ = build(ok(b"x" * 1000), limits=LIMITS)
    assert len(get(fetcher).body) == 1000
    fetcher, _, _ = build(ok(b"x" * 1001), limits=LIMITS)
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_LARGE


def test_a_declared_content_length_over_the_limit_is_refused_before_reading():
    read = []

    def body() -> Iterator[bytes]:
        read.append(1)
        yield b"x"

    fetcher, _, _ = build(
        lambda r: reply(200, content=body(), headers={"content-length": "5000"}),
        limits=LIMITS,
    )
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_LARGE
    assert read == []


def test_a_stream_without_a_length_is_cut_off_when_it_grows_past_the_limit():
    produced = []

    def endless() -> Iterator[bytes]:
        for _ in range(10_000):
            produced.append(1)
            yield b"x" * 100

    fetcher, _, _ = build(lambda r: reply(200, content=endless()), limits=LIMITS)
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_LARGE
    assert len(produced) < 50


def test_a_garbage_content_length_header_is_ignored_and_the_cap_still_applies():
    fetcher, _, _ = build(
        lambda r: reply(200, content=b"x" * 1500, headers={"content-length": "abc"}),
        limits=LIMITS,
    )
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_LARGE
    fetcher, _, _ = build(
        lambda r: reply(200, content=b"<rss/>", headers={"content-length": "abc"}),
        limits=LIMITS,
    )
    assert get(fetcher).body == b"<rss/>"


# ---- compression ----


def gz(payload: bytes) -> httpx2.Response:
    return reply(200, content=gzip.compress(payload), headers={"content-encoding": "gzip"})


def test_a_gzip_body_is_decompressed():
    fetcher, _, _ = build(lambda r: gz(b"<rss>compressed</rss>"))
    assert get(fetcher).body == b"<rss>compressed</rss>"


def test_a_gzip_bomb_is_stopped_at_the_limit_without_inflating_it():
    bomb = gzip.compress(b"\0" * 50_000_000)
    assert len(bomb) < 100_000  # the compressed form alone is within the cap
    fetcher, _, _ = build(
        lambda r: reply(200, content=bomb, headers={"content-encoding": "gzip"}),
        limits=FetchLimits(max_bytes=100_000),
    )
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_LARGE


def test_a_gzip_body_inflating_to_exactly_the_limit_is_accepted():
    fetcher, _, _ = build(lambda r: gz(b"y" * 1000), limits=LIMITS)
    assert len(get(fetcher).body) == 1000
    fetcher, _, _ = build(lambda r: gz(b"y" * 1001), limits=LIMITS)
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_LARGE


@pytest.mark.parametrize(
    "content",
    [
        b"this is not gzip",
        gzip.compress(b"<rss>truncated gzip</rss>" * 50)[:-12],
        b"\x1f\x8b" + b"\x00" * 20,
    ],
    ids=["not-gzip", "truncated-gzip", "corrupt-header"],
)
def test_a_broken_gzip_body_is_a_decode_error(content):
    fetcher, _, _ = build(
        lambda r: reply(200, content=content, headers={"content-encoding": "gzip"})
    )
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.DECODE


@pytest.mark.parametrize(
    "encoding", ["br", "zstd", "deflate", "gzip, gzip", "compress", "x-unknown"]
)
def test_an_encoding_that_was_not_requested_is_refused(encoding):
    fetcher, _, _ = build(
        lambda r: reply(200, content=zlib.compress(b"x"), headers={"content-encoding": encoding})
    )
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.DECODE


@pytest.mark.parametrize("encoding", ["identity", "", "GZIP"])
def test_identity_and_case_insensitive_gzip_labels_are_understood(encoding):
    body = gzip.compress(b"<rss/>") if encoding == "GZIP" else b"<rss/>"
    fetcher, _, _ = build(
        lambda r: reply(200, content=body, headers={"content-encoding": encoding})
    )
    assert get(fetcher).body == b"<rss/>"


# ---- time ----


def test_a_response_that_drips_past_the_overall_deadline_is_a_timeout():
    ticks = iter(range(0, 1000))

    def drip() -> Iterator[bytes]:
        for _ in range(100):
            yield b"x"

    fetcher, _, _ = build(
        lambda r: reply(200, content=drip()),
        limits=FetchLimits(deadline_s=5.0),
        monotonic=lambda: float(next(ticks)),
    )
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TIMEOUT


def test_a_chain_of_redirects_cannot_outlast_the_deadline():
    clock = {"now": 0.0}

    def handler(request):
        clock["now"] += 4.0
        return redirect("/again")

    fetcher, net, _ = build(
        handler, limits=FetchLimits(deadline_s=10.0), monotonic=lambda: clock["now"]
    )
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TIMEOUT
    assert len(net.requests) <= 3


# ---- configuration ----


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_s": 0},
        {"timeout_s": -1},
        {"deadline_s": 0},
        {"max_bytes": 0},
        {"max_redirects": -1},
        {"max_bytes": True},
        {"timeout_s": float("nan")},
        {"deadline_s": float("inf")},
    ],
)
def test_invalid_limits_are_refused(kwargs):
    with pytest.raises(ValueError, match="must be"):
        FetchLimits(**kwargs)


def test_zero_redirects_means_a_redirect_is_an_error():
    fetcher, _, _ = build(lambda r: redirect("/x"), limits=FetchLimits(max_redirects=0))
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_MANY_REDIRECTS
