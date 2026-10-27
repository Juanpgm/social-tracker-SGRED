"""Composition of the worker: settings from the environment, injected fetcher, one database."""

from pathlib import Path

import pytest

from social_tracker.bootstrap import (
    ALLOW_PRIVATE_HOSTS_ENV,
    ALLOWED_PORTS_ENV,
    DB_PATH_ENV,
    RETENTION_DAYS_ENV,
    build_container,
    build_worker,
    parse_allowed_ports,
    parse_private_hosts,
    parse_retention_days,
)
from social_tracker.collector.http_fetcher import DEFAULT_ALLOWED_PORTS, HttpFetcher
from social_tracker.items.domain import DEFAULT_RETENTION_DAYS
from social_tracker.shared.errors import ValidationError
from tests.actors import WHO
from tests.collector.harness import FEED, FakeFetcher
from tests.sources.factories import payload


def env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {DB_PATH_ENV: str(tmp_path / "w.db"), **extra}


# ---- retention days ----


@pytest.mark.parametrize(
    ("raw", "days"),
    [
        ("", DEFAULT_RETENTION_DAYS),
        ("   ", DEFAULT_RETENTION_DAYS),
        ("30", 30),
        ("7", 7),
        ("0", 0),
        ("3650", 3650),
        (" 14 ", 14),
    ],
)
def test_retention_days_come_from_the_environment_with_a_default(raw, days):
    assert parse_retention_days(raw) == days


@pytest.mark.parametrize("raw", ["-1", "3651", "1.5", "abc", "1e3", "0x10", "٣٠", "30 days", "+5"])
def test_invalid_retention_days_are_refused_with_the_variable_name(raw):
    with pytest.raises(ValidationError) as caught:
        parse_retention_days(raw)
    assert RETENTION_DAYS_ENV in caught.value.message


def test_the_container_exposes_an_item_service_with_the_configured_retention(tmp_path):
    container = build_container(env(tmp_path, **{RETENTION_DAYS_ENV: "9"}))
    try:
        assert container.items.retention_days == 9
        assert container.items.list_items() == []
    finally:
        container.close()


def test_an_invalid_retention_stops_the_start_before_the_database_is_opened(tmp_path):
    with pytest.raises(ValidationError):
        build_container(env(tmp_path, **{RETENTION_DAYS_ENV: "-3"}))
    assert not (tmp_path / "w.db").exists()


# ---- private hosts ----


@pytest.mark.parametrize(
    ("raw", "hosts"),
    [
        ("", frozenset()),
        ("   ", frozenset()),
        ("intranet.corp", frozenset({"intranet.corp"})),
        ("A.Corp, b.corp ,10.0.0.5", frozenset({"a.corp", "b.corp", "10.0.0.5"})),
        (",,a.corp,,", frozenset({"a.corp"})),
    ],
)
def test_private_hosts_are_a_comma_separated_list_of_names(raw, hosts):
    assert parse_private_hosts(raw) == hosts


@pytest.mark.parametrize(
    "raw", ["a b", "a.corp/path", "http://a.corp", "a.corp:80", "x\x00y", "*", "a" * 300, "u@host"]
)
def test_a_malformed_private_host_is_refused(raw):
    with pytest.raises(ValidationError) as caught:
        parse_private_hosts(raw)
    assert ALLOW_PRIVATE_HOSTS_ENV in caught.value.message


# ---- destination ports ----


@pytest.mark.parametrize(
    ("raw", "ports"),
    [
        ("", DEFAULT_ALLOWED_PORTS),
        ("  ", DEFAULT_ALLOWED_PORTS),
        (",,", DEFAULT_ALLOWED_PORTS),
        ("80", frozenset({80})),
        ("80, 443 ,8000", frozenset({80, 443, 8000})),
        ("1,65535", frozenset({1, 65535})),
        ("443,443", frozenset({443})),
    ],
)
def test_allowed_ports_are_a_comma_separated_list_with_the_web_ports_as_default(raw, ports):
    assert parse_allowed_ports(raw) == ports


@pytest.mark.parametrize(
    "raw", ["0", "65536", "-1", "http", "80;443", "8o80", "80 443", "1.5", "٨٠", "+80", "1" * 9]
)
def test_a_malformed_port_list_is_refused_with_the_variable_name(raw):
    with pytest.raises(ValidationError) as caught:
        parse_allowed_ports(raw)
    assert ALLOWED_PORTS_ENV in caught.value.message


def test_the_configured_ports_reach_the_fetcher(tmp_path):
    built = build_worker(env(tmp_path, **{ALLOWED_PORTS_ENV: "8000, 9000"}))
    try:
        assert built.fetcher._allowed_ports == frozenset({8000, 9000})
    finally:
        built.close()


def test_an_invalid_port_list_stops_the_start_before_the_database_is_opened(tmp_path):
    with pytest.raises(ValidationError):
        build_worker(env(tmp_path, **{ALLOWED_PORTS_ENV: "99999"}))
    assert not (tmp_path / "w.db").exists()


# ---- the worker ----


def test_the_worker_uses_the_real_http_fetcher_unless_one_is_injected(tmp_path):
    built = build_worker(env(tmp_path))
    try:
        assert isinstance(built.fetcher, HttpFetcher)
    finally:
        built.close()


def test_an_injected_fetcher_drives_a_whole_collection_through_the_real_wiring(tmp_path):
    fetcher = FakeFetcher(FEED)
    built = build_worker(env(tmp_path), fetcher=fetcher)
    try:
        built.sources.add(payload(id="src_news", enabled=True), **WHO)
        built.worker.start()
        built.control.set_global_analysis(True, **WHO)
        summary = built.worker.run_pass()
        assert [r.new for r in summary.results] == [2]
        assert built.items.list_items(source_id="src_news")[0].source_name == "Some Source"
    finally:
        built.close()


def test_closing_the_worker_container_releases_the_database(tmp_path):
    built = build_worker(env(tmp_path), fetcher=FakeFetcher())
    built.close()
    built.close()  # safe to repeat
    reopened = build_container(env(tmp_path))
    reopened.close()
