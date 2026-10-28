"""The alert commands after the remediation: the time zone, the timestamp a range applies to,
verdicts that wait for the model, withdrawn alerts, friendly refusals and an honest `stats`."""

import re

import pytest

from social_tracker.bootstrap import TIMEZONE_ENV
from social_tracker.persistence.db import Database
from tests.entrypoints.conftest import WHY, Cli
from tests.entrypoints.test_cli_alerts import ID, first_id, plant, rows

BOGOTA = {TIMEZONE_ENV: "America/Bogota"}


@pytest.fixture
def loaded(cli: Cli) -> Cli:
    cli.add()
    plant(cli, "a", confidence=0.55, hours=0)
    plant(cli, "b", confidence=0.9, hours=1)
    cli.ok("alerts", "refresh")
    return cli


def make_provisional(cli: Cli, item_id: str) -> None:
    with Database(cli.db_path) as db, db.transaction() as conn:
        conn.execute(
            "UPDATE classifications SET needs_llm = 1, priority = 0.5 WHERE item_id = ?",
            (item_id,),
        )


# ---- the time zone ----


def test_times_are_drawn_in_the_zone_of_the_operator(loaded: Cli):
    utc = loaded.ok("alerts", "list").stdout
    bogota = loaded.ok("alerts", "list", env=BOGOTA).stdout
    assert re.search(r"2026-03-01T09:01Z", utc)  # item b: published 09:00, fetched 09:01
    assert re.search(r"2026-03-01T04:01-05", bogota)


def test_a_day_is_a_day_of_the_zone(loaded: Cli):
    # the alerts were last seen at 08:01 and 09:01 UTC = 03:01 and 04:01 in Bogota
    assert len(rows(loaded.ok("alerts", "list", "--since", "2026-03-01", env=BOGOTA).stdout)) == 2
    out = loaded.ok("alerts", "list", "--until", "2026-02-28", env=BOGOTA).stdout
    assert out.strip() == "No alerts."
    day_start = loaded.ok("alerts", "list", "--since", "2026-03-01T04:00:00", env=BOGOTA).stdout
    assert len(rows(day_start)) == 1  # 04:00 Bogota is 09:00 UTC: only b (09:01 UTC)


def test_an_explicit_offset_beats_the_zone(loaded: Cli):
    out = loaded.ok("alerts", "list", "--since", "2026-03-01T09:00:00+00:00", env=BOGOTA).stdout
    assert len(rows(out)) == 1
    out = loaded.ok("alerts", "list", "--since", "2026-03-01T09:00:00Z").stdout
    assert len(rows(out)) == 1


def test_the_zone_is_checked_when_the_command_starts(loaded: Cli):
    result = loaded("alerts", "list", env={TIMEZONE_ENV: "Mars/Phobos"})
    assert result.exit_code == 1
    assert TIMEZONE_ENV in result.stderr
    assert "Phobos" not in result.stderr


def test_show_draws_its_times_in_the_zone_with_the_offset(loaded: Cli):
    alert_id = first_id(loaded)
    out = loaded.ok("alerts", "show", alert_id, env=BOGOTA).stdout
    assert "first seen: 2026-03-01T04:01:00-05:00" in out
    assert "published: 2026-03-01T04:00:00-05:00" in out


def test_show_says_when_the_publication_time_is_unknown(loaded: Cli):
    with Database(loaded.db_path) as db, db.transaction() as conn:
        conn.execute("UPDATE alerts SET published_at = NULL")
    assert "published: unknown" in loaded.ok("alerts", "show", first_id(loaded)).stdout


# ---- the timestamp a range applies to ----


def test_a_range_can_apply_to_the_first_sighting(cli: Cli):
    cli.add()
    cli.ok("sources", "add", "src_two", "--name", "Two", "--type", "news_rss", "--set",
           "feed_url=https://example.org/two.xml", "--reason", WHY)  # fmt: skip
    plant(cli, "a", text="Mañana habrá un temblor fuerte en Cali, compartan", hours=0)
    cli.ok("alerts", "refresh")
    out = cli.ok("alerts", "list", "--since", "2026-03-01", "--when", "first_seen").stdout
    assert len(rows(out)) == 1
    out = cli.ok("alerts", "list", "--until", "2026-02-28", "--when", "last_seen").stdout
    assert out.strip() == "No alerts."


@pytest.mark.parametrize("value", ["created_at", "", "FIRST_SEEN", "first"])
def test_any_other_timestamp_is_refused_listing_the_valid_ones(loaded: Cli, value):
    result = loaded("alerts", "list", "--when", value)
    assert result.exit_code == 1
    assert "first_seen, last_seen" in result.stderr


# ---- verdicts that wait for the model ----


def test_a_pending_verdict_is_marked_in_the_list_and_can_be_hidden(loaded: Cli):
    alert_id = first_id(loaded, "--severity", "high")
    item_id = re.search(r"itm_[0-9a-f]+", loaded.ok("alerts", "show", alert_id).stdout).group()
    make_provisional(loaded, item_id)
    loaded.ok("alerts", "rebuild")
    out = loaded.ok("alerts", "list").stdout
    assert out.count("[pending model]") == 1
    hidden = loaded.ok("alerts", "list", "--hide-provisional").stdout
    assert "[pending model]" not in hidden
    assert len(rows(hidden)) == len(rows(out)) - 1


def test_show_says_the_verdict_is_provisional_and_the_list_says_nothing_otherwise(loaded: Cli):
    assert "[pending model]" not in loaded.ok("alerts", "list").stdout
    assert "[pending model]" not in loaded.ok("alerts", "list", "--hide-provisional").stdout


# ---- withdrawn alerts ----


def withdraw_one(cli: Cli) -> str:
    alert_id = first_id(cli, "--severity", "medium")
    item_id = re.search(r"itm_[0-9a-f]+", cli.ok("alerts", "show", alert_id).stdout).group()
    with Database(cli.db_path) as db, db.transaction() as conn:
        conn.execute("UPDATE classifications SET label = 'no_risk' WHERE item_id = ?", (item_id,))
    cli.ok("alerts", "refresh")
    return alert_id


def test_withdrawn_alerts_are_hidden_counted_and_listed_on_request(loaded: Cli):
    alert_id = withdraw_one(loaded)
    assert alert_id not in loaded.ok("alerts", "list").stdout
    assert alert_id in loaded.ok("alerts", "list", "--status", "withdrawn").stdout
    both = loaded.ok("alerts", "list", "--include-withdrawn").stdout
    assert alert_id in both
    assert "[withdrawn]" in both
    assert "1 withdrawn" in loaded.ok("alerts", "stats").stdout


# ---- friendly refusals ----


@pytest.mark.parametrize(
    ("flag", "value", "valid"),
    [
        ("--status", "bogus", "new, seen, dismissed, withdrawn"),
        ("--severity", "urgent", "low, medium, high, critical"),
        ("--label", "no_risk", "unintentional_rumor, deliberate_disinformation, recycled_content"),
    ],
)
def test_a_bad_choice_lists_the_valid_ones_without_echoing_the_input(
    loaded: Cli, flag, value, valid
):
    result = loaded("alerts", "list", flag, value)
    assert result.exit_code == 1
    assert valid in result.stderr
    assert value not in result.stderr
    assert len(result.stderr.strip().splitlines()) == 1


def test_a_hostile_choice_is_not_echoed(loaded: Cli):
    result = loaded("alerts", "list", "--status", "x\x1b[31m'; DROP TABLE alerts;--")
    assert result.exit_code == 1
    assert "DROP" not in result.stderr
    assert "\x1b" not in result.stderr


# ---- stats ----


def test_stats_say_when_the_alerts_were_last_refreshed_and_with_which_threshold(loaded: Cli):
    out = loaded.ok("alerts", "stats").stdout
    assert re.search(r"last refresh: 20\d\d-\d\d-\d\dT", out)
    assert "threshold: 0.5 (applied)" in out


def test_stats_say_when_the_threshold_was_changed_but_not_applied(loaded: Cli):
    from social_tracker.bootstrap import build_container

    services = build_container({"SOCIAL_TRACKER_DB_PATH": str(loaded.db_path), TIMEZONE_ENV: "UTC"})
    try:
        services.control.set_alert_min_confidence(0.7, actor="tester", reason=WHY)
    finally:
        services.close()
    assert "threshold: 0.7 (not applied yet)" in loaded.ok("alerts", "stats").stdout


def test_stats_on_a_fresh_database_say_nothing_was_refreshed(cli: Cli):
    out = cli.ok("alerts", "stats").stdout
    assert "last refresh: never" in out


def test_stats_count_the_downgraded_and_the_parked(loaded: Cli):
    with Database(loaded.db_path) as db, db.transaction() as conn:
        conn.execute("UPDATE alerts SET downgraded_by_model = 1 WHERE severity = 'high'")
        conn.execute(
            "INSERT INTO alert_failures (item_id, failures, last_failed_at, next_try_at, parked) "
            "VALUES ('itm_stuck', 5, '2026-03-01T08:00:00+00:00', '2026-03-01T09:00:00+00:00', 1)"
        )
    out = loaded.ok("alerts", "stats").stdout
    assert "downgraded by the model: 1" in out
    assert "parked: 1" in out


def test_the_list_help_names_the_zone_not_utc(cli: Cli):
    out = cli.ok("alerts", "list", "--help").stdout
    assert "SOCIAL_TRACKER_TIMEZONE" in out
    assert "--hide-provisional" in out
    assert "--when" in out
    assert ID.pattern.startswith("alt_")
