"""The optional host allowlist and https-only mode of the fetcher (used by the catalog refresher).

Both are off by default, so the feed collector is unchanged. When set, they are judged at the first
hop and again at every redirect hop, before anything is resolved or sent.
"""

import pytest

from social_tracker.collector.domain import FetchErrorKind
from tests.collector.test_http_fetcher import (
    FakeResolver,
    build,
    failure,
    get,
    ok,
    redirect,
)

USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
EMSC = "https://www.seismicportal.eu/fdsnws/event/1/query?format=json"
HOSTS = frozenset({"earthquake.usgs.gov", "www.seismicportal.eu"})


def catalog_resolver() -> FakeResolver:
    return FakeResolver(
        {
            "earthquake.usgs.gov": ["8.8.4.4"],
            "www.seismicportal.eu": ["93.184.216.34"],
            "evil.example.net": ["8.8.8.8"],
        }
    )


def fetcher_for(handler=None, **options):
    options.setdefault("allowed_hosts", HOSTS)
    options.setdefault("https_only", True)
    return build(handler or ok(b"{}"), catalog_resolver(), **options)


@pytest.mark.parametrize("url", [USGS, EMSC])
def test_a_listed_host_is_fetched(url):
    fetcher, net, _ = fetcher_for()
    assert get(fetcher, url).body == b"{}"
    assert len(net.requests) == 1


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.net/x",
        "https://earthquake.usgs.gov.evil.example.net/x",  # a listed name as a prefix
        "https://evilearthquake.usgs.gov/x",
        "https://sub.earthquake.usgs.gov/x",  # a subdomain is not the listed host
        "https://usgs.gov/x",
        "https://earthquake.usgs.gov./x",  # the trailing-dot spelling is not the listed host
        "https://www.seismicportal.eu@evil.example.net/x",  # credentials are refused anyway
    ],
)
def test_an_unlisted_host_is_refused_before_resolving_or_sending(url):
    fetcher, net, resolver = fetcher_for()
    error = failure(lambda: get(fetcher, url))
    assert error.kind in (FetchErrorKind.HOST_NOT_ALLOWED, FetchErrorKind.BAD_URL)
    assert net.requests == []
    assert resolver.calls == []


def test_the_host_comparison_ignores_case():
    fetcher, net, _ = fetcher_for()
    get(fetcher, "https://EARTHQUAKE.USGS.GOV/fdsnws/event/1/query")
    assert len(net.requests) == 1


def test_a_redirect_to_an_unlisted_host_is_refused_after_the_first_request():
    fetcher, net, resolver = fetcher_for(lambda r: redirect("https://evil.example.net/steal"))
    error = failure(lambda: get(fetcher, USGS))
    assert error.kind is FetchErrorKind.HOST_NOT_ALLOWED
    assert len(net.requests) == 1
    assert ("evil.example.net", 443) not in resolver.calls


def test_a_redirect_between_listed_hosts_is_followed():
    def handler(request):
        if request.headers["host"] == "earthquake.usgs.gov":
            return redirect(EMSC)
        return ok(b'{"ok": true}')(request)

    fetcher, net, _ = fetcher_for(handler)
    assert get(fetcher, USGS).body == b'{"ok": true}'
    assert len(net.requests) == 2


def test_a_downgrade_to_plain_http_on_a_listed_host_is_refused():
    fetcher, net, _ = fetcher_for(
        lambda r: redirect("http://earthquake.usgs.gov/fdsnws/event/1/query"),
        allowed_ports=None,
    )
    assert failure(lambda: get(fetcher, USGS)).kind is FetchErrorKind.INSECURE_REDIRECT
    assert len(net.requests) == 1


def test_https_only_refuses_a_plain_http_first_hop():
    fetcher, net, resolver = fetcher_for()
    error = failure(lambda: get(fetcher, "http://earthquake.usgs.gov/fdsnws/event/1/query"))
    assert error.kind is FetchErrorKind.BAD_URL
    assert net.requests == []
    assert resolver.calls == []


def test_plain_http_is_allowed_when_https_only_is_off():
    fetcher, net, _ = fetcher_for(https_only=False)
    get(fetcher, "http://earthquake.usgs.gov/fdsnws/event/1/query")
    assert len(net.requests) == 1


def test_an_empty_allowlist_allows_no_host_while_none_allows_all():
    fetcher, net, _ = fetcher_for(allowed_hosts=frozenset())
    assert failure(lambda: get(fetcher, USGS)).kind is FetchErrorKind.HOST_NOT_ALLOWED
    assert net.requests == []
    open_fetcher, open_net, _ = fetcher_for(allowed_hosts=None)
    get(open_fetcher, "https://evil.example.net/x")
    assert len(open_net.requests) == 1


def test_the_failure_message_names_neither_the_host_nor_the_url():
    fetcher, _, _ = fetcher_for()
    error = failure(lambda: get(fetcher, "https://evil.example.net/secret?token=abc"))
    assert error.message == "host not allowed"
    assert "evil" not in error.message
    assert "secret" not in error.message


def test_the_listed_hosts_are_still_checked_against_the_address_policy():
    resolver = FakeResolver({"earthquake.usgs.gov": ["10.0.0.5"]})
    fetcher, net, _ = build(ok(b"{}"), resolver, allowed_hosts=HOSTS, https_only=True)
    assert failure(lambda: get(fetcher, USGS)).kind is FetchErrorKind.BLOCKED_ADDRESS
    assert net.requests == []
