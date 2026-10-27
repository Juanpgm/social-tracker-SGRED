"""Composition of the worker: settings from the environment, injected fetcher, one database."""

from pathlib import Path

import pytest

from social_tracker.bootstrap import (
    ALLOW_PRIVATE_HOSTS_ENV,
    ALLOWED_PORTS_ENV,
    DB_PATH_ENV,
    MAX_BYTES_PER_FETCH_ENV,
    MAX_BYTES_PER_SOURCE_ENV,
    MAX_ITEMS_PER_SOURCE_ENV,
    RETENTION_DAYS_ENV,
    build_container,
    build_worker,
    parse_allowed_ports,
    parse_ingest_limits,
    parse_private_hosts,
    parse_retention_days,
)
from social_tracker.collector.domain import IngestLimits
from social_tracker.collector.http_fetcher import DEFAULT_ALLOWED_PORTS, FetchLimits, HttpFetcher
from social_tracker.items.domain import DEFAULT_RETENTION_DAYS
from social_tracker.persistence.worker_lock import WorkerLockHeldError
from social_tracker.shared.errors import ValidationError
from tests.actors import WHO
from tests.collector.harness import FEED, FakeFetcher, dated_now
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


# ---- ingest limits ----


def test_ingest_limits_default_when_the_variables_are_blank_or_absent():
    assert parse_ingest_limits({}) == IngestLimits()
    blank = {
        MAX_ITEMS_PER_SOURCE_ENV: " ",
        MAX_BYTES_PER_SOURCE_ENV: "",
        MAX_BYTES_PER_FETCH_ENV: "",
    }
    assert parse_ingest_limits(blank) == IngestLimits()


def test_ingest_limits_come_from_the_environment():
    limits = parse_ingest_limits(
        {
            MAX_ITEMS_PER_SOURCE_ENV: "500",
            MAX_BYTES_PER_SOURCE_ENV: " 1048576 ",
            MAX_BYTES_PER_FETCH_ENV: "65536",
        }
    )
    assert limits == IngestLimits(500, 1048576, 65536)


@pytest.mark.parametrize(
    "name", [MAX_ITEMS_PER_SOURCE_ENV, MAX_BYTES_PER_SOURCE_ENV, MAX_BYTES_PER_FETCH_ENV]
)
@pytest.mark.parametrize(
    "raw", ["0", "-1", "1.5", "abc", "1e6", "0x10", "٣٠", "+5", "9" * 16, "10 MB"]
)
def test_a_malformed_ingest_limit_is_refused_with_the_variable_name(name, raw):
    with pytest.raises(ValidationError) as caught:
        parse_ingest_limits({name: raw})
    assert name in caught.value.message


def test_the_configured_limits_and_the_retention_reach_the_collector(tmp_path):
    settings = {
        MAX_ITEMS_PER_SOURCE_ENV: "7",
        RETENTION_DAYS_ENV: "9",
    }
    built = build_worker(env(tmp_path, **settings))
    try:
        collector = built.worker._collector
        assert collector._limits.max_items_per_source == 7
        assert collector._retention_days == 9
    finally:
        built.close()


def test_an_invalid_limit_stops_the_start_before_the_database_is_opened(tmp_path):
    with pytest.raises(ValidationError):
        build_worker(env(tmp_path, **{MAX_BYTES_PER_FETCH_ENV: "0"}))
    assert not (tmp_path / "w.db").exists()


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
    "raw",
    ["0", "65536", "-1", "http", "80;443", "8o80", "80 443", "1.5", "٨٠", "+80", "1" * 9]
    + ["9" * 5000],
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


# ---- one worker per database ----


def test_a_second_worker_on_the_same_database_is_refused_while_the_first_runs(tmp_path):
    first = build_worker(env(tmp_path))
    try:
        with pytest.raises(WorkerLockHeldError) as caught:
            build_worker(env(tmp_path))
        assert "another worker" in caught.value.message
        assert first.container.control.global_analysis_enabled() is False  # the first one is fine
    finally:
        first.close()


def test_the_lock_is_released_when_the_worker_container_is_closed(tmp_path):
    build_worker(env(tmp_path)).close()
    again = build_worker(env(tmp_path))
    again.close()
    again.close()  # closing twice is harmless


def test_the_lock_is_released_when_wiring_the_worker_fails_after_it_was_taken(
    tmp_path, monkeypatch
):
    def broken(*args, **kwargs):
        raise RuntimeError("wiring bug")

    monkeypatch.setattr("social_tracker.bootstrap.Worker", broken)
    with pytest.raises(RuntimeError):
        build_worker(env(tmp_path))
    monkeypatch.undo()
    build_worker(env(tmp_path)).close()  # neither the lock nor the database was left behind


def test_a_worker_and_a_plain_container_can_share_the_database(tmp_path):
    worker = build_worker(env(tmp_path))
    try:
        other = build_container(env(tmp_path))
        other.close()
    finally:
        worker.close()


def test_the_claim_timeout_covers_the_fetch_deadline_with_a_margin(tmp_path):
    built = build_worker(env(tmp_path))
    try:
        timeout = built.worker._collector.claim_timeout_s
        assert timeout >= FetchLimits().deadline_s + 60
    finally:
        built.close()


# ---- the worker ----


def test_the_worker_uses_the_real_http_fetcher_unless_one_is_injected(tmp_path):
    built = build_worker(env(tmp_path))
    try:
        assert isinstance(built.fetcher, HttpFetcher)
    finally:
        built.close()


def test_an_injected_fetcher_drives_a_whole_collection_through_the_real_wiring(tmp_path):
    fetcher = FakeFetcher(dated_now(FEED))  # the real clock is in play
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
