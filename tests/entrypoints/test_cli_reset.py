"""`sources reset ID`: put a failed source back to work, audited, safe to repeat."""

from datetime import UTC, datetime

import pytest

from social_tracker.control.domain import (
    MAX_CONSECUTIVE_FAILURES,
    CollectionState,
    FetchStatus,
    SourceState,
)
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from tests.entrypoints.conftest import WHY, Cli, table

FETCHED = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def make_failed(cli: Cli, source_id: str = "src_news") -> None:
    cli.add(source_id)
    cli.ok("sources", "enable", source_id, "--reason", WHY)
    with Database(cli.db_path) as db:
        repo = SqliteControlRepository(db)
        repo.save_source_state(
            SourceState(source_id, CollectionState.FAILED, MAX_CONSECUTIVE_FAILURES, None)
        )
        repo.save_fetch_status(FetchStatus(source_id, FETCHED, "http 503"))


def state_of(cli: Cli, source_id: str = "src_news") -> str:
    rows = {row[0]: row for row in table(cli.ok("sources", "list").stdout)[1:]}
    return rows[source_id][3]


def test_a_failed_source_is_reset_and_the_reset_is_audited(cli: Cli):
    make_failed(cli)
    result = cli.ok("sources", "reset", "src_news", "--reason", "feed fixed", "--actor", "ana")
    assert result.stdout.strip() == "reset src_news"
    assert state_of(cli) == "idle"
    (event,) = cli.details("collection_reset")
    assert (event["actor"], event["reason"]) == ("ana", "feed fixed")
    assert (event["from"], event["to"]) == ("failed", "idle")


def test_the_reset_clears_the_stale_error_in_the_listing_data(cli: Cli):
    make_failed(cli)
    cli.ok("sources", "reset", "src_news", "--reason", WHY)
    with Database(cli.db_path) as db:
        status = SqliteControlRepository(db).get_fetch_status("src_news")
    assert (status.last_error, status.last_fetch_at) == (None, FETCHED)


def test_resetting_again_says_there_is_nothing_to_do_and_writes_no_event(cli: Cli):
    make_failed(cli)
    cli.ok("sources", "reset", "src_news", "--reason", WHY)
    again = cli.ok("sources", "reset", "src_news", "--reason", WHY)
    assert "not failed" in again.stdout
    assert len(cli.details("collection_reset")) == 1


def test_a_healthy_source_is_not_touched(cli: Cli):
    cli.add()
    cli.ok("sources", "enable", "src_news", "--reason", WHY)
    result = cli.ok("sources", "reset", "src_news", "--reason", WHY)
    assert "not failed" in result.stdout
    assert cli.details("collection_reset") == []
    assert state_of(cli) == "idle"


def test_an_unknown_source_is_refused_with_exit_1(cli: Cli):
    cli.ok("status")
    result = cli("sources", "reset", "src_missing", "--reason", WHY)
    assert result.exit_code == 1
    assert result.stderr.startswith("error: ")
    assert result.stdout == ""


@pytest.mark.parametrize(
    "args", [["src_news"], ["src_news", "--reason", "  "], ["Bad Id", "-r", "x"]]
)
def test_a_missing_reason_or_a_bad_id_is_a_clean_error(cli: Cli, args):
    make_failed(cli)
    result = cli("sources", "reset", *args)
    assert result.exit_code in (1, 2)
    assert "Traceback" not in result.stderr
    assert cli.details("collection_reset") == []


def test_reset_is_listed_in_the_help(cli: Cli):
    assert "reset" in cli.ok("sources", "--help").stdout
