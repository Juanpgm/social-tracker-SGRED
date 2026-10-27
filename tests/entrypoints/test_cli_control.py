"""`social-tracker analysis ...`, `status` and the behaviour shared by every command."""

import sqlite3

import pytest
from typer.testing import CliRunner

from social_tracker.bootstrap import DB_PATH_ENV, build_container
from social_tracker.control.service import (
    EVENT_GLOBAL_ANALYSIS,
    EVENT_SOURCE_ANALYSIS,
    ControlService,
)
from social_tracker.shared.errors import (
    ConflictError,
    DomainError,
    InvalidTransition,
    NotFoundError,
    StorageError,
    ValidationError,
)
from tests.entrypoints.conftest import WHY, Cli, table
from tests.entrypoints.test_cli_sources import assert_clean_failure

# ---- analysis ----


def test_the_global_switch_turns_on_and_off_and_audits_actor_and_reason(cli: Cli):
    on = cli.ok("analysis", "on", "--reason", WHY, "--actor", "alice")
    assert (on.stdout, on.stderr) == ("global analysis is now on\n", "")
    off = cli.ok("analysis", "off", "-r", "incident over")
    assert off.stdout == "global analysis is now off\n"
    first, second = cli.details(EVENT_GLOBAL_ANALYSIS)
    assert (first["actor"], first["reason"], first["to"]) == ("alice", WHY, "active")
    assert (second["actor"], second["reason"], second["to"]) == (
        "tester",
        "incident over",
        "paused",
    )


def test_a_redundant_global_switch_is_reported_and_not_audited(cli: Cli):
    assert cli.ok("analysis", "off", "--reason", WHY).stdout == "global analysis is already off\n"
    cli.ok("analysis", "on", "--reason", WHY)
    assert cli.ok("analysis", "on", "--reason", WHY).stdout == "global analysis is already on\n"
    assert len(cli.details(EVENT_GLOBAL_ANALYSIS)) == 1


def test_the_per_source_switch_is_independent_of_the_global_one(cli: Cli):
    cli.add()
    on = cli.ok("analysis", "on", "--source", "src_news", "--reason", WHY)
    assert on.stdout == "analysis for src_news is now on\n"
    assert cli.ok("analysis", "on", "--source", "src_news", "-r", WHY).stdout == (
        "analysis for src_news is already on\n"
    )
    assert cli.details(EVENT_GLOBAL_ANALYSIS) == []
    (detail,) = cli.details(EVENT_SOURCE_ANALYSIS)
    assert (detail["actor"], detail["reason"], detail["to"]) == ("tester", WHY, "on")
    off = cli.ok("analysis", "off", "--source", "src_news", "-r", WHY)
    assert off.stdout == "analysis for src_news is now off\n"
    assert table(cli.ok("sources", "list").stdout)[1][4] == "off"


def test_analysis_can_be_flagged_on_a_disabled_source_but_never_runs_there(cli: Cli):
    cli.add()
    cli.ok("analysis", "on", "--source", "src_news", "--reason", WHY)
    cli.ok("analysis", "on", "--reason", WHY)
    row = table(cli.ok("sources", "list").stdout)[1]
    assert row[2:5] == ["no", "disabled", "on"]
    assert "effective analysis: 0 source(s)" in cli.ok("status").stdout


@pytest.mark.parametrize("command", ["on", "off"])
def test_the_per_source_switch_on_an_unknown_or_malformed_source_fails_cleanly(
    cli: Cli, command: str
):
    missing = cli("analysis", command, "--source", "src_zzz", "--reason", WHY)
    assert "source not found" in assert_clean_failure(missing)
    malformed = cli("analysis", command, "--source", "NOT OK", "--reason", WHY)
    assert "not a valid slug" in assert_clean_failure(malformed)
    assert cli.events() == []


@pytest.mark.parametrize("command", ["on", "off"])
def test_the_switches_need_a_reason_and_refuse_a_blank_one(cli: Cli, command: str):
    missing = cli("analysis", command)
    assert missing.exit_code == 2
    assert "--reason" in missing.stderr
    blank = cli("analysis", command, "--reason", "  ")
    assert "reason" in assert_clean_failure(blank)
    bad_actor = cli("analysis", command, "--reason", WHY, "--actor", "")
    assert "actor" in assert_clean_failure(bad_actor)
    assert cli.events() == []


# ---- status ----


def test_status_of_a_fresh_system(cli: Cli):
    result = cli.ok("status")
    assert result.stdout == (
        "global analysis: off\n"
        "alert threshold: 0.5\n"
        "sources: 0 total, 0 enabled\n"
        "collection states: none\n"
        "effective analysis: 0 source(s)\n"
    )
    assert result.stderr == ""


def test_status_summarises_switches_states_and_effective_analysis(cli: Cli):
    cli.add("src_a", "Alpha")
    cli.add("src_b", "Beta")
    cli.add("src_c", "Gamma")
    cli.ok("sources", "enable", "src_a", "--reason", WHY)
    cli.ok("sources", "enable", "src_b", "--reason", WHY)
    cli.ok("analysis", "on", "--source", "src_a", "--reason", WHY)
    cli.ok("analysis", "on", "--source", "src_c", "--reason", WHY)  # disabled: never effective
    assert "global analysis: off\n" in cli.ok("status").stdout  # paused: nothing runs
    assert "effective analysis: 0 source(s)\n" in cli.ok("status").stdout
    cli.ok("analysis", "on", "--reason", WHY)
    assert cli.ok("status").stdout == (
        "global analysis: on\n"
        "alert threshold: 0.5\n"
        "sources: 3 total, 2 enabled\n"
        "collection states: disabled 1, idle 2\n"
        "effective analysis: 1 source(s)\n"
    )


def test_status_does_not_write_anything(cli: Cli):
    cli.add()
    before = len(cli.events())
    cli.ok("status")
    assert len(cli.events()) == before


def test_status_of_a_corrupt_stored_flag_is_a_clean_error(cli: Cli):
    cli.add()
    with sqlite3.connect(cli.db_path) as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES ('source_analysis.src_news', 'maybe')"
        )
    assert "stored" in assert_clean_failure(cli("status"))


# ---- behaviour shared by every command ----


@pytest.mark.parametrize(
    "error", [ValidationError, ConflictError, NotFoundError, InvalidTransition, StorageError]
)
def test_every_kind_of_domain_error_becomes_a_clean_message_and_exit_code_one(
    cli: Cli, monkeypatch: pytest.MonkeyPatch, error: type[DomainError]
):
    def refuse(*args, **kwargs):
        raise error("refused for a reason")

    monkeypatch.setattr(ControlService, "set_global_analysis", refuse)
    result = cli("analysis", "on", "--reason", WHY)
    assert assert_clean_failure(result) == "error: refused for a reason\n"


def test_an_unknown_command_and_missing_arguments_are_usage_errors(cli: Cli):
    assert cli("frobnicate").exit_code == 2
    assert cli("sources", "enable").exit_code == 2
    assert cli("sources", "add", "src_news", "--reason", WHY).exit_code == 2
    assert not cli.db_path.exists()


def test_help_lists_the_command_groups_and_never_opens_the_database(cli: Cli):
    top = " ".join(cli.ok("--help").stdout.split())
    for word in ("sources", "analysis", "status"):
        assert word in top
    listing = " ".join(cli.ok("sources", "--help").stdout.split())
    for word in ("add", "list", "edit", "enable", "disable", "delete", "import", "export"):
        assert word in listing
    assert not cli.db_path.exists()


def test_output_that_the_terminal_cannot_encode_is_escaped_instead_of_crashing(cli: Cli):
    cli.runner = CliRunner(charset="cp1252")  # a Windows console or pipe: no CJK
    name = "Ciudad " + chr(0x57CE)
    cli.ok(
        "sources", "add", "src_news", "--name", name,
        "--type", "news_rss", "--set", "feed_url=https://example.org/rss.xml", "--reason", WHY,
    )  # fmt: skip
    listing = cli.ok("sources", "list")
    assert "Ciudad " + chr(92) + "u57ce" in listing.stdout
    assert listing.stderr == ""


def test_status_shows_a_stored_threshold_as_a_plain_decimal(cli: Cli):
    cli.ok("status")  # creates the database
    container = build_container({DB_PATH_ENV: str(cli.db_path)})
    try:
        container.control.set_alert_min_confidence(0.00001, actor="tester", reason=WHY)
    finally:
        container.close()
    assert "alert threshold: 0.00001\n" in cli.ok("status").stdout


@pytest.mark.parametrize("group", [[], ["sources"], ["analysis"]])
def test_a_command_group_without_a_subcommand_prints_its_help(cli: Cli, group: list[str]):
    result = cli(*group)
    assert "Usage:" in result.stdout
    assert "Commands" in result.stdout
    assert not cli.db_path.exists()
