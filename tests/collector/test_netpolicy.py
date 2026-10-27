"""Address policy: which URLs and IP addresses the collector may contact."""

import ipaddress
from datetime import UTC, datetime

import pytest

from social_tracker.collector.domain import (
    MAX_RETRY_AFTER_S,
    FetchError,
    FetchErrorKind,
    parse_retry_after,
)
from social_tracker.collector.netpolicy import (
    is_blocked_address,
    parse_fetch_url,
    redirect_target,
    usable_addresses,
)

NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


def kind_of(call) -> FetchErrorKind:
    with pytest.raises(FetchError) as caught:
        call()
    return caught.value.kind


# ---- urls ----


@pytest.mark.parametrize(
    ("url", "scheme", "host", "port", "target", "authority"),
    [
        ("http://example.org/rss.xml", "http", "example.org", 80, "/rss.xml", "example.org"),
        ("https://example.org/rss", "https", "example.org", 443, "/rss", "example.org"),
        (
            "https://Example.ORG:8443/a/b?x=1&y=2",
            "https",
            "example.org",
            8443,
            "/a/b?x=1&y=2",
            "example.org:8443",
        ),
        ("http://example.org", "http", "example.org", 80, "/", "example.org"),
        ("http://example.org?q=1", "http", "example.org", 80, "/?q=1", "example.org"),
        ("https://example.org/a#frag", "https", "example.org", 443, "/a", "example.org"),
        ("http://example.org:80/x", "http", "example.org", 80, "/x", "example.org"),
        ("https://93.184.216.34/x", "https", "93.184.216.34", 443, "/x", "93.184.216.34"),
        (
            "http://[2606:2800:220:1::1]:8080/x",
            "http",
            "2606:2800:220:1::1",
            8080,
            "/x",
            "[2606:2800:220:1::1]:8080",
        ),
        (
            "https://bücher.example/feed",
            "https",
            "xn--bcher-kva.example",
            443,
            "/feed",
            "xn--bcher-kva.example",
        ),
        ("https://example.org./feed", "https", "example.org.", 443, "/feed", "example.org."),
    ],
)
def test_a_valid_address_is_split_into_what_the_connection_needs(
    url, scheme, host, port, target, authority
):
    parsed = parse_fetch_url(url)
    assert (parsed.scheme, parsed.host, parsed.port) == (scheme, host, port)
    assert (parsed.target, parsed.authority) == (target, authority)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org/x",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "gopher://example.org",
        "data:text/plain,hi",
        "//example.org/x",
        "example.org/x",
        "http://",
        "http:///path",
        "https://" + "user:pass" + "@example.org/x",
        "https://user@example.org/x",
        "http://example.org:0/x",
        "http://example.org:99999/x",
        "http://example.org:abc/x",
        "http://exa mple.org/x",
        "http://example.org/x\ny",
        "http://example.org/\x00",
        "http://[::1/x",
        "http://[fe80::1%25eth0]/x",
        "http://example.org/" + "a" * 2100,
        "http://2130706433/",
        "http://0x7f.0.0.1/",
        "http://127.1/",
        "http://017700000001/",
        "http://0177.0.0.1/",
        "http://-bad.example/",
        "",
        "   ",
    ],
)
def test_an_address_that_is_not_a_plain_http_url_is_refused(url):
    assert kind_of(lambda: parse_fetch_url(url)) is FetchErrorKind.BAD_URL


@pytest.mark.parametrize("bad", [None, 5, b"http://example.org", ["http://example.org"]])
def test_a_non_string_address_is_refused(bad):
    assert kind_of(lambda: parse_fetch_url(bad)) is FetchErrorKind.BAD_URL


def test_a_host_name_may_be_253_characters_long_but_not_254():
    label = "a" * 63
    longest = ".".join([label, label, label, "a" * 61])
    assert len(longest) == 253
    assert parse_fetch_url(f"http://{longest}/").host == longest
    with pytest.raises(FetchError) as caught:
        parse_fetch_url(f"http://{longest}a/")
    assert caught.value.kind is FetchErrorKind.BAD_URL


def test_host_names_that_only_look_numeric_in_hex_letters_are_fine():
    assert parse_fetch_url("http://decade.bee.example/").host == "decade.bee.example"


# ---- addresses ----

BLOCKED = [
    "127.0.0.1",
    "127.255.255.254",
    "10.0.0.1",
    "10.255.255.255",
    "172.16.0.1",
    "172.31.255.255",
    "192.168.0.1",
    "192.168.255.255",
    "169.254.169.254",
    "169.254.0.1",
    "0.0.0.0",
    "0.1.2.3",
    "100.64.0.1",
    "100.127.255.255",
    "192.0.0.1",
    "192.0.2.1",
    "198.18.0.1",
    "198.51.100.1",
    "203.0.113.1",
    "224.0.0.1",
    "239.255.255.255",
    "240.0.0.1",
    "255.255.255.255",
    "::1",
    "::",
    "fe80::1",
    "febf::1",
    "fc00::1",
    "fd12:3456::1",
    "ff02::1",
    "2001:db8::1",
    "::ffff:127.0.0.1",
    "::ffff:10.0.0.1",
    "::ffff:169.254.169.254",
    "::ffff:192.168.1.1",
    "64:ff9b::7f00:1",
    "64:ff9b::a9fe:a9fe",
    "2002:7f00:1::1",
    "2002:a9fe:a9fe::1",
    # IPv4-compatible (::/96) and IPv4-translated (::ffff:0:0/96) forms: judged by the IPv4 inside
    "::7f00:1",
    "::127.0.0.1",
    "::a00:1",
    "::a9fe:a9fe",
    "::2",
    "::ffff:0:7f00:1",
    "::ffff:0:a00:1",
    "::ffff:0:0",
    "fec0::1",
    "feff:ffff::1",
]
ALLOWED = [
    "93.184.216.34",
    "8.8.8.8",
    "1.1.1.1",
    "172.15.255.255",
    "172.32.0.1",
    "11.0.0.1",
    "100.63.255.255",
    "100.128.0.1",
    "2606:2800:220:1:248:1893:25c8:1946",
    "2a00:1450:4001:81b::200e",
    "::ffff:8.8.8.8",
    "64:ff9b::808:808",
    "2002:5db8:d822::1",
    "::808:808",
    "::ffff:0:808:808",
    "fe00::1",  # just outside fec0::/10 (and not link-local either)
]


@pytest.mark.parametrize("address", BLOCKED)
def test_internal_and_special_purpose_addresses_are_blocked(address):
    assert is_blocked_address(ipaddress.ip_address(address)) is True


@pytest.mark.parametrize("address", ALLOWED)
def test_public_addresses_are_allowed(address):
    assert is_blocked_address(ipaddress.ip_address(address)) is False


def test_usable_addresses_keep_the_order_and_drop_repeats():
    assert usable_addresses(["8.8.8.8", "1.1.1.1", "8.8.8.8"], allow_private=False) == [
        "8.8.8.8",
        "1.1.1.1",
    ]


@pytest.mark.parametrize(
    "answer",
    [
        ["127.0.0.1"],
        ["10.1.2.3"],
        ["8.8.8.8", "169.254.169.254"],
        ["::1"],
        ["8.8.8.8", "::ffff:10.0.0.1"],
    ],
)
def test_one_blocked_address_in_the_answer_blocks_the_whole_host(answer):
    assert kind_of(lambda: usable_addresses(answer, allow_private=False)) is (
        FetchErrorKind.BLOCKED_ADDRESS
    )


def test_the_allowlist_override_lets_private_answers_through():
    assert usable_addresses(["10.1.2.3", "192.168.0.9"], allow_private=True) == [
        "10.1.2.3",
        "192.168.0.9",
    ]


@pytest.mark.parametrize("answer", [[], ["not-an-ip"], [""], ["999.1.1.1"], ["8.8.8.8", "bogus"]])
def test_an_empty_or_malformed_answer_is_a_dns_failure(answer):
    assert kind_of(lambda: usable_addresses(answer, allow_private=False)) is FetchErrorKind.DNS


def test_a_link_local_zone_suffix_in_an_answer_is_refused():
    assert kind_of(lambda: usable_addresses(["fe80::1%eth0"], allow_private=False)) is (
        FetchErrorKind.DNS
    )


# ---- redirects ----


@pytest.mark.parametrize(
    ("base", "location", "expected"),
    [
        ("https://example.org/a/feed", "/other", "https://example.org/other"),
        ("https://example.org/a/feed", "next.xml", "https://example.org/a/next.xml"),
        ("https://example.org/a", "https://other.example/x", "https://other.example/x"),
        ("https://example.org/a", "//cdn.example/x", "https://cdn.example/x"),
        ("http://example.org/a", "https://example.org/a#frag", "https://example.org/a"),
        ("https://example.org/a?x=1", "?y=2", "https://example.org/a?y=2"),
    ],
)
def test_redirect_locations_resolve_against_the_current_address(base, location, expected):
    assert redirect_target(base, location) == expected


@pytest.mark.parametrize("location", ["", "   ", "x" * 3000, "/a\nb", "/a\x00"])
def test_a_missing_or_malformed_location_is_refused(location):
    assert kind_of(lambda: redirect_target("https://example.org/a", location)) is (
        FetchErrorKind.BAD_URL
    )


# ---- retry-after ----


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("120", 120.0),
        ("0", 0.0),
        ("  30  ", 30.0),
        ("Sun, 01 Mar 2026 12:05:00 GMT", 300.0),
        ("Sun, 01 Mar 2026 12:00:00 GMT", 0.0),
        ("Sun, 01 Mar 2026 11:00:00 GMT", 0.0),  # already past
        ("Sun, 01 Mar 2026 12:00:30 +0000", 30.0),
        ("Sun, 01 Mar 2026 12:05:00 -0000", 300.0),  # "-0000" is a date without a zone
        ("Sun, 01 Mar 2026 12:05:00", 300.0),
        ("99999999999999999999", float(MAX_RETRY_AFTER_S)),
        ("604801", float(MAX_RETRY_AFTER_S)),
        ("604800", 604800.0),
        ("Fri, 01 Jan 2100 00:00:00 GMT", float(MAX_RETRY_AFTER_S)),
    ],
)
def test_retry_after_in_seconds_or_http_date(value, seconds):
    assert parse_retry_after(value, NOW) == seconds


@pytest.mark.parametrize(
    "value",
    [
        "-5",
        "-0",
        "1.5",
        "1e3",
        "abc",
        "",
        "   ",
        "12 seconds",
        "0x10",
        "Sun, 99 Foo 2026",
        "NaN",
        "١٢",
        "1" + "0" * 5000 + "x",
    ],
)
def test_garbage_or_negative_retry_after_is_ignored(value):
    assert parse_retry_after(value, NOW) is None


@pytest.mark.parametrize("value", [None, 5, b"120"])
def test_non_text_retry_after_is_ignored(value):
    assert parse_retry_after(value, NOW) is None


# ---- errors ----


@pytest.mark.parametrize(
    ("status", "degraded"),
    [(401, True), (403, True), (429, True), (404, False), (500, False), (503, False), (400, False)],
)
def test_access_and_rate_limit_statuses_mean_degraded(status, degraded):
    error = FetchError.http(status)
    assert error.degraded is degraded
    assert error.message.startswith(f"http {status}")


def test_error_messages_are_fixed_short_phrases():
    messages = {
        kind: FetchError(kind).message for kind in FetchErrorKind if kind is not FetchErrorKind.HTTP
    }
    assert all(0 < len(m) < 60 for m in messages.values())
    assert len(set(messages.values())) == len(messages)
