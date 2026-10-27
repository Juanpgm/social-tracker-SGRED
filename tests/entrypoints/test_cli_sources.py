"""`social-tracker sources ...`: add, list, edit, enable, disable, delete."""

import sqlite3

import pytest

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV
from social_tracker.entrypoints.inputs import ACTOR_ENV
from social_tracker.sources.service import (
    SOURCE_ADDED,
    SOURCE_DELETED,
    SOURCE_DISABLED,
    SOURCE_EDITED,
    SOURCE_ENABLED,
)
from tests.entrypoints.conftest import NEWS, WHY, Cli, table

PASTED = "ghp_" + "0123456789abcdef"  # built by concatenation: the hygiene scan flags the literal


def assert_clean_failure(result, *, code: int = 1) -> str:
    """A failed command: non-zero exit, a one-line `error:` on stderr, nothing on stdout."""
    assert result.exit_code == code
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.stderr
    return result.stderr


# ---- add ----


def test_add_creates_a_disabled_source_and_audits_actor_and_reason(cli: Cli):
    result = cli.ok("sources", "add", "src_news", "--name", "City News", *NEWS, "--reason", WHY)
    assert result.stdout == "added src_news (news_rss)\n"
    assert result.stderr == ""
    (detail,) = cli.details(SOURCE_ADDED)
    assert (detail["actor"], detail["reason"], detail["type"]) == ("tester", WHY, "news_rss")
    assert table(cli.ok("sources", "list").stdout)[1] == [
        "src_news", "news_rss", "no", "disabled", "off", "City News",
    ]  # fmt: skip


def test_add_takes_every_type_specific_field_from_set(cli: Cli):
    cli.ok(
        "sources", "add", "ig_main", "--type", "instagram", "--name", "IG", "--reason", WHY,
        "--set", "handle=some.handle", "--set", "secret_ref=env:IG_MAIN",
        "--set", "include_stories=false", "--set", "poll_interval_s=900",
        "--set", "credibility=5",
    )  # fmt: skip
    exported = cli.ok("sources", "export").stdout
    assert "secret_ref: env:IG_MAIN" in exported
    assert "include_stories: false" in exported
    assert "credibility: 5" in exported


def test_the_actor_option_overrides_the_environment_and_the_environment_the_default(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
):
    cli.ok("sources", "add", "one_src", "--name", "One", *NEWS, "--reason", WHY, "--actor", "alice")
    monkeypatch.setattr("getpass.getuser", lambda: "osuser")
    cli.ok(
        "sources", "add", "two_src", "--name", "Two", *NEWS, "--reason", WHY, env={ACTOR_ENV: None}
    )
    cli.ok("sources", "add", "three_src", "--name", "Three", *NEWS, "--reason", WHY)
    assert [d["actor"] for d in cli.details(SOURCE_ADDED)] == ["alice", "cli:osuser", "tester"]


def test_a_missing_reason_is_a_usage_error_and_creates_nothing(cli: Cli):
    result = cli("sources", "add", "src_news", "--name", "City News", *NEWS)
    assert result.exit_code == 2
    assert "--reason" in result.stderr
    assert not cli.db_path.exists()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--reason", ""),
        ("--reason", "   "),
        ("--reason", "line one\nline two"),
        ("--reason", "bell\x07"),
        ("--reason", "x" * 501),
        ("--actor", ""),
        ("--actor", "  "),
        ("--actor", "tab\there"),
        ("--actor", "x" * 101),
    ],
)
def test_a_blank_oversized_or_control_character_actor_or_reason_is_refused(
    cli: Cli, flag: str, value: str
):
    args = ["sources", "add", "src_news", "--name", "City News", *NEWS]
    args += ["--reason", WHY, "--actor", "alice"]
    args[args.index(flag) + 1] = value
    stderr = assert_clean_failure(cli(*args))
    assert flag.removeprefix("--") in stderr
    assert cli.ok("sources", "list").stdout == "No sources registered.\n"
    assert cli.events() == []


def test_the_boundary_lengths_of_actor_and_reason_are_accepted(cli: Cli):
    cli.ok(
        "sources", "add", "src_news", "--name", "City News", *NEWS,
        "--reason", "r" * 500, "--actor", "a" * 100,
    )  # fmt: skip
    (detail,) = cli.details(SOURCE_ADDED)
    assert (len(detail["reason"]), len(detail["actor"])) == (500, 100)


def test_a_duplicate_id_or_a_duplicate_name_in_another_case_is_a_conflict(cli: Cli):
    cli.add("src_news", "City News")
    again = cli("sources", "add", "src_news", "--name", "Other", *NEWS, "--reason", WHY)
    assert "id already exists" in assert_clean_failure(again)
    shouted = cli("sources", "add", "src_other", "--name", " CITY news ", *NEWS, "--reason", WHY)
    assert "name already exists" in assert_clean_failure(shouted)
    assert len(table(cli.ok("sources", "list").stdout)) == 2  # header + the first source only


def test_an_unknown_type_a_missing_field_and_a_bad_id_are_validation_errors(cli: Cli):
    unknown = cli("sources", "add", "src_news", "--type", "tiktok", "--name", "N", "--reason", WHY)
    assert "unknown source type" in assert_clean_failure(unknown)
    missing = cli(
        "sources", "add", "src_news", "--type", "news_rss", "--name", "N", "--reason", WHY
    )
    assert "feed_url" in assert_clean_failure(missing)
    bad_id = cli("sources", "add", "Bad Id!", "--name", "N", *NEWS, "--reason", WHY)
    assert "invalid slug" in assert_clean_failure(bad_id)


@pytest.mark.parametrize("field", ["id", "type", "name"])
def test_fields_with_a_dedicated_option_cannot_be_given_through_set(cli: Cli, field: str):
    result = cli(
        "sources", "add", "src_news", "--name", "N", *NEWS, "--set", f"{field}=other",
        "--reason", WHY,
    )  # fmt: skip
    assert "dedicated option" in assert_clean_failure(result)


def test_a_pasted_secret_is_rejected_and_never_echoed(cli: Cli):
    as_ref = cli(
        "sources", "add", "ig_main", "--type", "instagram", "--name", "IG", "--reason", WHY,
        "--set", "handle=h", "--set", f"secret_ref={PASTED}",
    )  # fmt: skip
    as_field = cli(
        "sources", "add", "src_news", "--name", "N", *NEWS, "--set", f"token={PASTED}",
        "--reason", WHY,
    )  # fmt: skip
    as_url = cli(
        "sources", "add", "src_rss", "--type", "news_rss", "--name", "R", "--reason", WHY,
        "--set", f"feed_url=https://example.org/rss?token={PASTED}",
    )  # fmt: skip
    for result in (as_ref, as_field, as_url):
        assert_clean_failure(result)
        assert PASTED not in result.stdout + result.stderr
    assert cli.events() == []


def test_a_malformed_set_pair_is_refused_without_echoing_it(cli: Cli):
    result = cli(
        "sources", "add", "src_news", "--name", "N", *NEWS, "--set", PASTED, "--reason", WHY
    )
    assert "KEY=VALUE" in assert_clean_failure(result)
    assert PASTED not in result.stderr


# ---- list ----


def test_list_of_an_empty_registry_says_so(cli: Cli):
    result = cli.ok("sources", "list")
    assert (result.stdout, result.stderr) == ("No sources registered.\n", "")


def test_list_shows_one_row_per_source_sorted_by_id_with_the_name_last(cli: Cli):
    cli.add("src_b", "Beta News")
    cli.add("src_a", "Alpha  with   spaces")
    cli.ok("sources", "enable", "src_b", "--reason", WHY)
    cli.ok("analysis", "on", "--source", "src_b", "--reason", WHY)
    assert table(cli.ok("sources", "list").stdout) == [
        ["ID", "TYPE", "ENABLED", "STATE", "ANALYSIS", "NAME"],
        ["src_a", "news_rss", "no", "disabled", "off", "Alpha  with   spaces"],
        ["src_b", "news_rss", "yes", "idle", "on", "Beta News"],
    ]


def test_list_columns_stay_aligned_for_long_ids_and_types(cli: Cli):
    cli.add("a_very_long_source_identifier_indeed", "Long")
    cli.ok("sources", "add", "ig_main", "--type", "instagram", "--name", "I", "--reason", WHY,
           "--set", "handle=h", "--set", "secret_ref=env:IG")  # fmt: skip
    lines = cli.ok("sources", "list").stdout.splitlines()
    columns = {line.index(line.split()[1]) for line in lines}
    assert len(columns) == 1  # the TYPE column starts at the same offset in every row


def test_list_of_a_corrupt_database_row_is_a_clean_error(cli: Cli):
    cli.add()
    with sqlite3.connect(cli.db_path) as conn:
        conn.execute("UPDATE sources SET name = char(7)")
    assert "stored source row is invalid" in assert_clean_failure(cli("sources", "list"))


# ---- edit ----


def test_edit_changes_fields_and_audits_what_changed(cli: Cli):
    cli.add()
    result = cli.ok(
        "sources", "edit", "src_news", "--set", "credibility=5",
        "--set", 'keywords=["flood","fire"]', "--reason", WHY,
    )  # fmt: skip
    assert result.stdout == "updated src_news\n"
    (detail,) = cli.details(SOURCE_EDITED)
    assert detail["changed"] == ["credibility", "keywords"]
    assert (detail["actor"], detail["reason"]) == ("tester", WHY)
    exported = cli.ok("sources", "export").stdout
    assert "credibility: 5" in exported
    assert "- flood" in exported


def test_edit_with_identical_values_is_a_reported_noop_without_an_event(cli: Cli):
    cli.add()
    cli.ok("sources", "edit", "src_news", "--set", "credibility=4", "--reason", WHY)
    again = cli.ok("sources", "edit", "src_news", "--set", "credibility=4", "--reason", WHY)
    assert again.stdout == "no changes for src_news\n"
    assert len(cli.details(SOURCE_EDITED)) == 1


def test_edit_can_clear_an_optional_field_with_null(cli: Cli):
    cli.add()
    cli.ok("sources", "edit", "src_news", "--set", 'keywords=["a"]', "--reason", WHY)
    cli.ok("sources", "edit", "src_news", "--set", "keywords=null", "--reason", WHY)
    assert "keywords" not in cli.ok("sources", "export").stdout


def test_edit_without_any_set_is_refused(cli: Cli):
    cli.add()
    result = cli("sources", "edit", "src_news", "--reason", WHY)
    assert "nothing to change" in assert_clean_failure(result)


@pytest.mark.parametrize("field", ["id=other_id", "type=instagram", "enabled=true", "created_at=1"])
def test_immutable_fields_cannot_be_edited(cli: Cli, field: str):
    cli.add()
    result = cli("sources", "edit", "src_news", "--set", field, "--reason", WHY)
    assert "cannot be edited" in assert_clean_failure(result)
    assert cli.details(SOURCE_EDITED) == []


def test_edit_of_an_unknown_source_or_a_taken_name_or_a_bad_value_fails_cleanly(cli: Cli):
    cli.add("src_a", "Alpha")
    cli.add("src_b", "Beta")
    unknown = cli("sources", "edit", "src_zzz", "--set", "credibility=2", "--reason", WHY)
    assert "source not found" in assert_clean_failure(unknown)
    taken = cli("sources", "edit", "src_b", "--set", 'name="alpha"', "--reason", WHY)
    assert "name already exists" in assert_clean_failure(taken)
    bad = cli("sources", "edit", "src_b", "--set", "credibility=9", "--reason", WHY)
    assert "credibility" in assert_clean_failure(bad)
    unknown_key = cli("sources", "edit", "src_b", "--set", f"{PASTED}=1", "--reason", WHY)
    assert PASTED not in assert_clean_failure(unknown_key)


# ---- enable / disable / delete ----


def test_enable_and_disable_toggle_the_source_and_audit_each_change_once(cli: Cli):
    cli.add()
    assert cli.ok("sources", "enable", "src_news", "--reason", WHY).stdout == "enabled src_news\n"
    row = table(cli.ok("sources", "list").stdout)[1]
    assert row[2:4] == ["yes", "idle"]
    repeat = cli.ok("sources", "enable", "src_news", "--reason", WHY)
    assert repeat.stdout == "src_news is already enabled\n"
    assert cli.ok("sources", "disable", "src_news", "--reason", WHY).stdout == "disabled src_news\n"
    row = table(cli.ok("sources", "list").stdout)[1]
    assert row[2:4] == ["no", "disabled"]
    again = cli.ok("sources", "disable", "src_news", "--reason", WHY)
    assert again.stdout == "src_news is already disabled\n"
    assert [len(cli.details(k)) for k in (SOURCE_ENABLED, SOURCE_DISABLED)] == [1, 1]
    assert cli.details(SOURCE_ENABLED)[0]["actor"] == "tester"


@pytest.mark.parametrize("command", ["enable", "disable", "delete"])
def test_commands_on_an_unknown_or_malformed_source_id_fail_cleanly(cli: Cli, command: str):
    missing = cli("sources", command, "src_zzz", "--reason", WHY)
    assert "source not found" in assert_clean_failure(missing)
    malformed = cli("sources", command, "../etc", "--reason", WHY)
    assert "not a valid slug" in assert_clean_failure(malformed)


@pytest.mark.parametrize("command", ["enable", "disable", "delete"])
def test_commands_need_a_reason(cli: Cli, command: str):
    cli.add()
    result = cli("sources", command, "src_news")
    assert result.exit_code == 2
    assert "--reason" in result.stderr


def test_delete_removes_the_source_but_keeps_the_audit_trail(cli: Cli):
    cli.add()
    cli.ok("sources", "enable", "src_news", "--reason", WHY)
    result = cli.ok("sources", "delete", "src_news", "--reason", "retired", "--actor", "bob")
    assert result.stdout == "deleted src_news\n"
    assert cli.ok("sources", "list").stdout == "No sources registered.\n"
    (detail,) = cli.details(SOURCE_DELETED)
    assert (detail["actor"], detail["reason"], detail["state"]) == ("bob", "retired", "idle")
    assert len(cli.details(SOURCE_ADDED)) == 1  # earlier events survive the delete
    second = cli("sources", "delete", "src_news", "--reason", WHY)
    assert "source not found" in assert_clean_failure(second)


# ---- environment and storage failures ----


def test_an_unusable_database_location_is_a_clean_error(cli: Cli, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    result = cli("sources", "list", env={DB_PATH_ENV: str(blocker / "x.db")})
    assert "database location is not usable" in assert_clean_failure(result)
    assert "blocker" not in result.stderr


def test_a_locked_database_fails_with_a_clean_error_and_works_again_once_released(cli: Cli):
    cli.add()
    holder = sqlite3.connect(cli.db_path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        result = cli(
            "sources", "enable", "src_news", "--reason", WHY, env={BUSY_TIMEOUT_ENV: "100"}
        )
        assert "SQLITE_BUSY" in assert_clean_failure(result)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    cli.ok("sources", "enable", "src_news", "--reason", WHY)


def test_an_invalid_busy_timeout_is_reported_before_anything_is_touched(cli: Cli):
    result = cli("sources", "list", env={BUSY_TIMEOUT_ENV: "soon"})
    assert BUSY_TIMEOUT_ENV in assert_clean_failure(result)
    assert not cli.db_path.exists()
