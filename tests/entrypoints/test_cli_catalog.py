"""`catalog refresh` and `catalog status`: the operator's view of the official-catalog cache."""

from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.bootstrap import build_catalog
from social_tracker.catalog.domain import CatalogEvent, CatalogSource
from social_tracker.collector.domain import FetchError, FetchErrorKind
from social_tracker.entrypoints import cli as cli_module
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.db import Database
from tests.catalog.support import ScriptedFetcher, emsc_payload, usgs_payload
from tests.entrypoints.conftest import WHY, Cli


@pytest.fixture
def fetcher(monkeypatch: pytest.MonkeyPatch) -> ScriptedFetcher:
    scripted = ScriptedFetcher()
    scripted.on("usgs", usgs_payload({"id": "us1", "time": datetime.now(UTC) - timedelta(hours=1)}))
    scripted.on("emsc", emsc_payload({"id": "em1", "time": datetime.now(UTC) - timedelta(hours=1)}))
    monkeypatch.setattr(cli_module, "build_catalog", lambda: build_catalog(fetcher=scripted))
    return scripted


def analysis_on(cli: Cli) -> None:
    cli.ok("analysis", "on", "--reason", WHY)


def test_the_group_lists_its_commands(cli: Cli):
    result = cli("catalog", "--help")
    assert result.exit_code == 0
    assert "refresh" in result.output
    assert "status" in result.output


# ---- status ----


def test_the_status_of_an_empty_cache_says_so(cli: Cli):
    out = cli.ok("catalog", "status").output
    assert "available: no" in out
    assert "usgs: 0 event(s); last good refresh never" in out
    assert "emsc: 0 event(s); last good refresh never" in out
    assert "newest event: none" in out
    assert "STALE" in out


def test_the_status_shows_counts_times_failures_and_staleness(cli: Cli):
    now = datetime.now(UTC)
    with Database(cli.db_path) as db:
        repo = SqliteCatalogRepository(db)
        event = CatalogEvent(
            CatalogSource.USGS, "us1", now - timedelta(hours=2), 3.5, -76.5, 5.0, "x"
        )
        repo.upsert_events([event], now)
        repo.record_success(CatalogSource.USGS, now - timedelta(minutes=5))
        repo.record_success(CatalogSource.EMSC, now - timedelta(days=1))
        repo.record_failure(CatalogSource.EMSC, now, now + timedelta(minutes=2), "http 503")
    out = cli.ok("catalog", "status").output
    lines = {line.split(":")[0]: line for line in out.splitlines()}
    assert "available: yes" in out
    assert "complete: no" in out
    assert "usgs: 1 event(s)" in lines["usgs"]
    assert "STALE" not in lines["usgs"]
    assert "failures 0" in lines["usgs"]
    assert "STALE" in lines["emsc"]
    assert "failures 1" in lines["emsc"]
    assert "http 503" in lines["emsc"]


def test_the_status_neutralises_control_characters_in_what_it_prints(cli: Cli):
    now = datetime.now(UTC)
    with Database(cli.db_path) as db:
        SqliteCatalogRepository(db).record_failure(
            CatalogSource.USGS, now, now, "bad \x1b[31mred\x07 text"
        )
    out = cli.ok("catalog", "status").output
    assert "\x1b" not in out
    assert "\x07" not in out


# ---- refresh ----


def test_refresh_does_nothing_while_global_analysis_is_off(cli: Cli, fetcher: ScriptedFetcher):
    result = cli("catalog", "refresh")
    assert result.exit_code == 3
    assert "global analysis is off" in result.stderr
    assert fetcher.calls == []
    assert "usgs: 0 event(s)" in cli.ok("catalog", "status").output


def test_refresh_dry_run_also_stays_off_the_network_while_analysis_is_off(
    cli: Cli, fetcher: ScriptedFetcher
):
    assert cli("catalog", "refresh", "--dry-run").exit_code == 3
    assert fetcher.calls == []


def test_refresh_stores_both_catalogs_and_reports_them(cli: Cli, fetcher: ScriptedFetcher):
    analysis_on(cli)
    result = cli.ok("catalog", "refresh")
    assert "usgs: refreshed (1 new, 0 updated, 0 unchanged, 0 discarded)" in result.output
    assert "emsc: refreshed (1 new, 0 updated, 0 unchanged, 0 discarded)" in result.output
    status = cli.ok("catalog", "status").output
    assert "available: yes" in status
    assert "complete: yes" in status
    assert "usgs: 1 event(s)" in status


def test_a_second_manual_refresh_fetches_again_and_finds_nothing_new(
    cli: Cli, fetcher: ScriptedFetcher
):
    analysis_on(cli)
    cli.ok("catalog", "refresh")
    out = cli.ok("catalog", "refresh").output
    assert "usgs: refreshed (0 new, 0 updated, 1 unchanged, 0 discarded)" in out


def test_dry_run_reports_what_would_change_and_stores_nothing(cli: Cli, fetcher: ScriptedFetcher):
    analysis_on(cli)
    out = cli.ok("catalog", "refresh", "--dry-run").output
    assert "usgs: dry run (1 new, 0 unchanged, 0 discarded)" in out
    assert len(fetcher.calls) == 4
    assert "usgs: 0 event(s)" in cli.ok("catalog", "status").output


def test_a_failing_catalog_exits_1_and_the_other_is_still_refreshed(
    cli: Cli, fetcher: ScriptedFetcher
):
    analysis_on(cli)
    fetcher.on("usgs", FetchError(FetchErrorKind.TIMEOUT))
    result = cli("catalog", "refresh")
    assert result.exit_code == 1
    assert "usgs: failed (timeout)" in result.output
    assert "emsc: refreshed (1 new" in result.output
    status = cli.ok("catalog", "status").output
    assert "usgs: 0 event(s)" in status
    assert "failures 1" in status
    assert "emsc: 1 event(s)" in status


@pytest.mark.parametrize(
    "outcome",
    [
        FetchError.http(429, retry_after_s=60.0),
        FetchError.http(503),
        FetchError(FetchErrorKind.DNS),
        FetchError(FetchErrorKind.HOST_NOT_ALLOWED),
        RuntimeError("boom with a secret"),
        b"not json at all",
    ],
    ids=["429", "503", "dns", "redirect", "crash", "garbage"],
)
def test_no_failure_of_a_catalog_is_ever_a_traceback(cli: Cli, fetcher: ScriptedFetcher, outcome):
    analysis_on(cli)
    fetcher.on("emsc", outcome)
    result = cli("catalog", "refresh")
    assert result.exit_code == 1
    assert "Traceback" not in result.output + result.stderr
    assert "secret" not in result.output + result.stderr
    assert "emsc: failed" in result.output


def test_a_catalog_being_refreshed_elsewhere_is_reported_and_not_an_error(
    cli: Cli, fetcher: ScriptedFetcher
):
    analysis_on(cli)
    now = datetime.now(UTC)
    with Database(cli.db_path) as db:
        SqliteCatalogRepository(db).try_claim(CatalogSource.USGS, now, now + timedelta(minutes=2))
    result = cli("catalog", "refresh")
    assert result.exit_code == 0
    assert "usgs: skipped, another refresh is running" in result.output
    assert fetcher.calls_to("usgs") == []
    assert "emsc: refreshed" in result.output


def test_an_empty_answer_is_a_quiet_day_and_exits_0(cli: Cli, fetcher: ScriptedFetcher):
    analysis_on(cli)
    fetcher.on("emsc", b"")
    result = cli("catalog", "refresh")
    assert result.exit_code == 0
    assert "emsc: refreshed (0 new, 0 updated, 0 unchanged, 0 discarded)" in result.output


def test_events_a_catalog_stopped_listing_are_reported_as_removed(
    cli: Cli, fetcher: ScriptedFetcher
):
    analysis_on(cli)
    cli.ok("catalog", "refresh")
    fetcher.on("usgs", usgs_payload({"id": "us2", "time": datetime.now(UTC) - timedelta(hours=1)}))
    out = cli.ok("catalog", "refresh").output
    assert "usgs: refreshed (1 new, 0 updated, 0 unchanged, 0 discarded, 1 removed)" in out
    assert "removed" not in out.splitlines()[1]


def test_the_status_flags_a_catalog_whose_last_answer_was_cut_short(cli: Cli):
    now = datetime.now(UTC)
    with Database(cli.db_path) as db:
        repo = SqliteCatalogRepository(db)
        repo.record_success(CatalogSource.USGS, now, truncated=True)
        repo.record_success(CatalogSource.EMSC, now)
    lines = cli.ok("catalog", "status").output.splitlines()
    assert "complete: no" in lines[1]
    assert "[CUT SHORT]" in lines[3]
    assert "[CUT SHORT]" not in lines[4]
