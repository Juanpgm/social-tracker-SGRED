"""Strict parsing of the `Host` header and of the addresses the server accepts (S-W2, S-S1)."""

import pytest
from fastapi.testclient import TestClient

from social_tracker.bootstrap import build_container
from social_tracker.entrypoints.web.app import create_app
from social_tracker.entrypoints.web.security import (
    LOOPBACK_NAMES,
    allowed_hosts,
    host_allowed,
    host_name,
)
from social_tracker.shared.errors import ValidationError
from tests.entrypoints.web.conftest import Web


@pytest.mark.parametrize(
    ("header", "name"),
    [
        ("localhost", "localhost"),
        ("LocalHost", "localhost"),
        ("localhost:8765", "localhost"),
        ("localhost:0", "localhost"),
        ("localhost:65535", "localhost"),
        ("localhost.", "localhost"),
        ("LOCALHOST.:80", "localhost"),
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.1:8765", "127.0.0.1"),
        ("[::1]", "::1"),
        ("[::1]:8765", "::1"),
        ("[0:0:0:0:0:0:0:1]:80", "::1"),
        ("[0000:0000::0001]", "::1"),
        ("[FE80::1]:9", "fe80::1"),
        ("radio-1.example.org", "radio-1.example.org"),
    ],
)
def test_a_well_formed_host_header_gives_its_canonical_name(header: str, name: str):
    assert host_name(header) == name


@pytest.mark.parametrize(
    "header",
    [
        "",
        " ",
        ".",
        ".:80",
        " localhost",
        "localhost ",
        "localhost:8765 ",
        "localhost\n",
        "\tlocalhost",
        "local host",
        "localhost:80@evil.com",
        "evil.com@localhost",
        "user:pw@localhost",
        "@localhost",
        "localhost@",
        ":80",
        ":",
        "localhost:",
        "localhost:port",
        "localhost:+80",
        "localhost:-1",
        "localhost:123456",
        "localhost:65536",
        "localhost:80:80",
        "::1",
        "::1:8765",
        "[::1",
        "::1]",
        "[]",
        "[]:80",
        "[::1]x",
        "[::1]:",
        "[::1]:abc",
        "[localhost]",
        "[127.0.0.1]",
        "[::g]",
        "[1:2:3:4:5:6:7:8:9]",
        "localhost/",
        "localhost/path",
        "localhost?x",
        "localhost#x",
        "local\x00host",
        "locálhost",
        "ｌｏｃａｌｈｏｓｔ",
        "localhost:٨٠",  # Arabic-Indic digits are not digits here
        "Kali",  # the Kelvin sign lower-cases to an ASCII "k"
        "a_b",
        "a b",
    ],
)
def test_a_malformed_or_ambiguous_host_header_has_no_name(header: str):
    assert host_name(header) is None
    assert host_allowed(header, LOOPBACK_NAMES | {"kali", "fe80::1"}) is False


def test_a_missing_host_header_is_not_allowed():
    assert host_allowed(None, LOOPBACK_NAMES) is False


def test_the_bound_name_is_compared_after_normalising_both_sides():
    hosts = allowed_hosts("FE80:0:0:0:0:0:0:1")
    assert "fe80::1" in hosts
    assert host_allowed("[fe80::1]:8765", hosts)
    assert host_allowed("[FE80:0::1]", hosts)
    assert not host_allowed("[fe80::2]", hosts)
    assert not host_allowed("fe80::1", hosts)  # unbracketed is not a valid Host header


@pytest.mark.parametrize("bound", ["127.0.0.1", "localhost", "::1", "[::1]", "LOCALHOST."])
def test_a_loopback_bind_adds_nothing_beyond_the_loopback_names(bound: str):
    assert allowed_hosts(bound) == LOOPBACK_NAMES


@pytest.mark.parametrize("bound", ["fe80::1", "[fe80::1]", "FE80::1"])
def test_an_unbracketed_or_bracketed_ipv6_bind_is_accepted_in_its_canonical_form(bound: str):
    assert allowed_hosts(bound) == LOOPBACK_NAMES | {"fe80::1"}


@pytest.mark.parametrize("bound", ["0.0.0.0", "::", "[::]", "0:0:0:0:0:0:0:0"])
def test_a_wildcard_bind_names_no_host_of_its_own(bound: str):
    hosts = allowed_hosts(bound)
    assert hosts == LOOPBACK_NAMES
    assert not host_allowed("0.0.0.0:8765", hosts)  # the "0.0.0.0 day" Host value
    assert not host_allowed("[::]:8765", hosts)


def test_a_named_bind_and_the_extra_names_are_lower_cased_and_stripped_of_a_trailing_dot():
    hosts = allowed_hosts("Radio.Local", ("Other.Example.", "192.168.1.5"))
    assert hosts == LOOPBACK_NAMES | {"radio.local", "other.example", "192.168.1.5"}


@pytest.mark.parametrize(
    "bad",
    [
        "",
        " ",
        "a b",
        "host:80",
        "user@host",
        "http://host",
        "host/x",
        "0.0.0.0",
        "::",
        "[::1",
        "ho_st",
        "hóst",
    ],
)
def test_an_invalid_extra_allowed_host_is_refused_when_the_app_is_built(bad: str):
    with pytest.raises(ValidationError):
        allowed_hosts("127.0.0.1", (bad,))


@pytest.mark.parametrize(
    "bad", ["", "a b", "user@host", "host:80", "ho_st", "[::1", "::1]", "[]", "[[::1]]"]
)
def test_an_invalid_bound_host_is_refused(bad: str):
    with pytest.raises(ValidationError):
        allowed_hosts(bad)


# ---- through the app ----


@pytest.mark.parametrize(
    "host",
    [
        "localhost:80@evil.com",
        "localhost ",
        " localhost",
        ":80",
        "localhost:",
        "[::1",
        "::1",
        "localhost:99999999",
        "127.0.0.1\t",
    ],
)
def test_the_app_refuses_malformed_host_headers(web: Web, host: str):
    for response in (
        web.client.get("/", headers={"host": host}),
        web.client.post("/analysis", data={"on": "1"}, headers={"host": host}),
    ):
        assert response.status_code == 400, host
        assert response.text == "Invalid host header"


@pytest.mark.parametrize("host", ["localhost.", "LOCALHOST.:8765", "[0:0:0:0:0:0:0:1]:8765"])
def test_the_app_accepts_equivalent_spellings_of_the_loopback_names(web: Web, host: str):
    assert web.client.get("/", headers={"host": host}).status_code == 200


def serve(tmp_path, bound: str, extra=()) -> TestClient:
    env = {"SOCIAL_TRACKER_DB_PATH": str(tmp_path / "h.db")}
    app = create_app(lambda: build_container(env), host=bound, extra_hosts=tuple(extra))
    return TestClient(app, base_url="http://127.0.0.1:8765")


@pytest.mark.parametrize(
    ("bound", "good", "bad"),
    [
        ("fe80::1", "[fe80::1]:8765", "[fe80::2]:8765"),
        ("::", "localhost", "[::]:8765"),
        ("0.0.0.0", "127.0.0.1", "0.0.0.0:8765"),
        ("192.168.1.20", "192.168.1.20:8765", "192.168.1.21"),
    ],
)
def test_the_app_serves_the_bound_name_and_nothing_else(tmp_path, bound, good, bad):
    client = serve(tmp_path, bound)
    assert client.get("/", headers={"host": good}).status_code == 200
    assert client.get("/", headers={"host": bad}).status_code == 400


def test_extra_allowed_hosts_are_served_and_others_are_not(tmp_path):
    client = serve(tmp_path, "0.0.0.0", ("Radio.Local", "10.0.0.7"))
    assert client.get("/", headers={"host": "radio.local:8765"}).status_code == 200
    assert client.get("/", headers={"host": "10.0.0.7"}).status_code == 200
    assert client.get("/", headers={"host": "10.0.0.8"}).status_code == 400
