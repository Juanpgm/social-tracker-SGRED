"""Which ports, redirects and validators the fetcher accepts (offline, mock transport)."""

import logging

import pytest

from social_tracker.collector.domain import FetchErrorKind
from social_tracker.collector.http_fetcher import DEFAULT_ALLOWED_PORTS, HttpFetcher
from tests.collector.test_http_fetcher import (
    URL,
    FakeResolver,
    build,
    failure,
    get,
    ok,
    redirect,
    reply,
)

OTHER = "https://cdn.example.net/feed.xml"


def other_host_resolver() -> FakeResolver:
    return FakeResolver({"cdn.example.net": ["8.8.4.4"], "feeds.example.org": ["93.184.216.34"]})


# ---- destination ports ----


def test_the_default_ports_are_the_web_ones():
    assert frozenset({80, 443, 8080, 8443}) == DEFAULT_ALLOWED_PORTS


@pytest.mark.parametrize("port", [80, 443, 8080, 8443])
def test_the_default_ports_are_allowed(port):
    fetcher, net, _ = build(ok())
    get(fetcher, f"http://feeds.example.org:{port}/x")
    assert len(net.requests) == 1


@pytest.mark.parametrize("port", [1, 21, 22, 25, 79, 81, 3306, 5432, 6379, 8000, 8444, 9200, 65535])
def test_other_ports_are_refused_before_anything_is_resolved_or_sent(port):
    resolver = FakeResolver()
    fetcher, net, _ = build(ok(), resolver)
    error = failure(lambda: get(fetcher, f"http://feeds.example.org:{port}/x"))
    assert error.kind is FetchErrorKind.BLOCKED_ADDRESS
    assert net.requests == []
    assert resolver.calls == []


def test_a_redirect_to_a_port_that_is_not_allowed_is_refused():
    fetcher, net, _ = build(lambda r: redirect("https://feeds.example.org:6379/"))
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.BLOCKED_ADDRESS
    assert len(net.requests) == 1


def test_the_allowed_ports_can_be_replaced_by_the_operator():
    fetcher, net, _ = build(ok(), allowed_ports=frozenset({8000}))
    get(fetcher, "http://feeds.example.org:8000/x")
    assert len(net.requests) == 1
    assert failure(lambda: get(fetcher, URL)).kind is FetchErrorKind.BLOCKED_ADDRESS


def test_no_port_list_allows_every_port():
    fetcher, net, _ = build(ok(), allowed_ports=None)
    get(fetcher, "http://feeds.example.org:22/x")
    assert len(net.requests) == 1


def test_the_default_fetcher_uses_the_default_ports():
    from social_tracker.shared.clock import SystemClock

    assert HttpFetcher(SystemClock())._allowed_ports == DEFAULT_ALLOWED_PORTS


# ---- scheme downgrade ----


def test_a_redirect_from_https_to_http_is_refused_and_not_followed(caplog):
    fetcher, net, _ = build(lambda r: redirect("http://feeds.example.org/plain.xml"))
    with caplog.at_level(logging.WARNING, logger="social_tracker.collector"):
        error = failure(lambda: get(fetcher))
    assert error.kind is FetchErrorKind.INSECURE_REDIRECT
    assert len(net.requests) == 1
    assert "downgrade" in caplog.text
    assert "feeds.example.org" not in caplog.text
    assert "plain.xml" not in caplog.text


def test_a_downgrade_later_in_the_chain_is_refused_too():
    def handler(request):
        if request.url.path == "/rss.xml":
            return redirect("/step2")
        return redirect("http://feeds.example.org/plain.xml")

    fetcher, net, _ = build(handler)
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.INSECURE_REDIRECT
    assert len(net.requests) == 2


def test_a_redirect_from_http_to_https_is_followed():
    def handler(request):
        if request.url.scheme == "http":
            return redirect("https://feeds.example.org/secure.xml")
        return reply(200, content=b"<rss>secure</rss>")

    fetcher, net, _ = build(handler)
    assert get(fetcher, "http://feeds.example.org/rss.xml").body == b"<rss>secure</rss>"
    assert len(net.requests) == 2


def test_http_to_http_stays_allowed():
    def handler(request):
        if request.url.path == "/rss.xml":
            return redirect("http://feeds.example.org/b.xml")
        return reply(200, content=b"<rss/>")

    fetcher, _, _ = build(handler)
    assert get(fetcher, "http://feeds.example.org/rss.xml").body == b"<rss/>"


# ---- validators go only to the original authority ----

VALIDATORS = {"etag": '"v1"', "last_modified": "Sun, 01 Mar 2026 10:00:00 GMT"}


def test_validators_are_not_sent_to_another_host_after_a_redirect():
    def handler(request):
        if request.headers["host"] == "feeds.example.org":
            return redirect(OTHER)
        return reply(200, content=b"<rss/>")

    fetcher, net, _ = build(handler, other_host_resolver())
    get(fetcher, **VALIDATORS)
    first, second = net.requests
    assert first.headers["if-none-match"] == '"v1"'
    assert "if-none-match" not in second.headers
    assert "if-modified-since" not in second.headers


def test_validators_are_not_sent_to_another_port_of_the_same_host():
    def handler(request):
        if request.headers["host"] == "feeds.example.org":
            return redirect("https://feeds.example.org:8443/rss.xml")
        return reply(200, content=b"<rss/>")

    fetcher, net, _ = build(handler)
    get(fetcher, **VALIDATORS)
    assert "if-none-match" not in net.requests[1].headers


def test_validators_travel_on_with_a_redirect_inside_the_same_authority():
    def handler(request):
        if request.url.path == "/rss.xml":
            return redirect("/moved.xml")
        return reply(200, content=b"<rss/>")

    fetcher, net, _ = build(handler)
    get(fetcher, **VALIDATORS)
    assert net.requests[1].headers["if-none-match"] == '"v1"'
    assert net.requests[1].headers["if-modified-since"] == VALIDATORS["last_modified"]


def test_after_leaving_the_authority_and_coming_back_validators_return_only_to_the_original():
    def handler(request):
        host, path = request.headers["host"], request.url.path
        if host == "feeds.example.org" and path == "/rss.xml":
            return redirect(OTHER)
        if host == "cdn.example.net":
            return redirect("https://feeds.example.org/back.xml")
        return reply(200, content=b"<rss/>")

    fetcher, net, _ = build(handler, other_host_resolver())
    get(fetcher, **VALIDATORS)
    assert "if-none-match" in net.requests[0].headers
    assert "if-none-match" not in net.requests[1].headers
    assert "if-none-match" in net.requests[2].headers
