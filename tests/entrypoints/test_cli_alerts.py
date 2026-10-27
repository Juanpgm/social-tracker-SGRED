"""`social-tracker alerts ...`: list, show, mark, rebuild and the threshold from the shell."""

import re
from datetime import timedelta

import pytest

from social_tracker.classification.domain import Classification, Label, Signal, Stage, Topic
from social_tracker.persistence.classification_repo import SqliteClassificationRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from tests.entrypoints.conftest import WHY, Cli
from tests.items.factories import NOW, make_item

RUMOR, DELIBERATE, RECYCLED = (
    Label.UNINTENTIONAL_RUMOR,
    Label.DELIBERATE_DISINFORMATION,
    Label.RECYCLED_CONTENT,
)
ID = re.compile(r"alt_[0-9a-f]{32}")


def plant(
    cli: Cli,
    name: str,
    *,
    label: Label = RUMOR,
    confidence: float = 0.6,
    hours: int = 0,
    text: str | None = None,
    title: str = "",
    catalog: str = "unavailable",
    topics: tuple[Topic, ...] = (Topic.EARTHQUAKE,),
    signals: tuple[Signal, ...] | None = None,
):
    """An item with a stored verdict, straight into the database."""
    item = make_item(
        name,
        title=title,
        text=text or f"Texto {name}: mañana habrá un temblor fuerte en Cali",
        published_at=NOW + timedelta(hours=hours),
    )
    verdict = Classification(
        item_id=item.id,
        label=label,
        confidence=confidence,
        stage=Stage.RULES,
        rationale="prediction_claim",
        classified_at=NOW,
        classifier_version="cascade-2/topics-0",
        signals=(Signal("prediction_claim", RUMOR, 0.45),) if signals is None else signals,
        topics=topics,
        catalog_status=catalog,
    )
    with Database(cli.db_path) as db:
        SqliteItemRepository(db).add_new([item])
        SqliteClassificationRepository(db).add(verdict)
    return item


@pytest.fixture
def loaded(cli: Cli) -> Cli:
    """A source and three alerts of different severity and label."""
    cli.add()
    plant(cli, "a", confidence=0.55, hours=0)
    plant(cli, "b", label=DELIBERATE, confidence=0.9, hours=1)
    plant(cli, "c", label=RECYCLED, confidence=0.75, hours=2, topics=(Topic.SECRETARIAT,))
    cli.ok("alerts", "refresh")
    return cli


def rows(output: str) -> list[list[str]]:
    return [line.split() for line in output.splitlines()[1:] if line.startswith("alt_")]


def first_id(cli: Cli, *filters: str) -> str:
    found = ID.search(cli.ok("alerts", "list", *filters).stdout)
    assert found
    return found.group()


# ---- refresh and rebuild ----


def test_refresh_on_an_empty_database_does_nothing_and_succeeds(cli: Cli):
    result = cli.ok("alerts", "refresh")
    assert result.stdout.strip() == (
        "0 created, 0 merged, 0 updated, 0 withdrawn, 0 brought back, 0 unchanged, 0 skipped, "
        "0 error(s)"
    )


def test_refresh_raises_the_alerts_and_a_second_run_finds_nothing_new(cli: Cli):
    cli.add()
    plant(cli, "a", confidence=0.55)
    plant(cli, "b", confidence=0.3)
    first = cli.ok("alerts", "refresh").stdout
    assert first.startswith("1 created, 0 merged")
    again = cli.ok("alerts", "refresh").stdout
    assert again.startswith("0 created, 0 merged, 0 updated")


def test_rebuild_is_idempotent_and_follows_the_threshold(loaded: Cli):
    assert loaded.ok("alerts", "rebuild").stdout.startswith("0 created, 0 merged, 0 updated")
    loaded.ok("alerts", "threshold", "set", "0.8", "--reason", WHY)
    assert loaded.ok("alerts", "refresh").stdout.startswith("0 created, 0 merged, 0 updated")
    assert "2 withdrawn" in loaded.ok("alerts", "rebuild").stdout
    assert len(rows(loaded.ok("alerts", "list").stdout)) == 1
    assert "0 withdrawn" in loaded.ok("alerts", "rebuild").stdout


# ---- list ----


def test_an_empty_list_says_so(cli: Cli):
    result = cli.ok("alerts", "list")
    assert result.stdout.strip() == "No alerts."


def test_the_list_is_newest_first_with_what_a_person_needs_to_decide(loaded: Cli):
    out = loaded.ok("alerts", "list").stdout
    header = out.splitlines()[0].split()
    assert header[:7] == ["ID", "SEVERITY", "LABEL", "STATUS", "CONF", "OCC", "LAST_SEEN"]
    body = rows(out)
    assert [r[2] for r in body] == [
        "recycled_content",
        "deliberate_disinformation",
        "unintentional_rumor",
    ]
    assert [r[1] for r in body] == ["high", "critical", "medium"]
    assert {r[3] for r in body} == {"new"}
    assert [r[4] for r in body] == ["0.75", "0.90", "0.55"]


@pytest.mark.parametrize(
    ("flags", "labels"),
    [
        (["--severity", "critical"], ["deliberate_disinformation"]),
        (["--label", "recycled_content"], ["recycled_content"]),
        (
            ["--source", "src_news"],
            ["recycled_content", "deliberate_disinformation", "unintentional_rumor"],
        ),
        (["--source", "src_other"], []),
        (
            ["--status", "new"],
            ["recycled_content", "deliberate_disinformation", "unintentional_rumor"],
        ),
        (["--status", "seen"], []),
        (["--tag", "secretariat"], ["recycled_content"]),
        (["--tag", "prediction", "--severity", "medium"], ["unintentional_rumor"]),
        (["--limit", "1"], ["recycled_content"]),
        (["--limit", "1", "--offset", "1"], ["deliberate_disinformation"]),
        (["--limit", "5", "--offset", "3"], []),
    ],
)
def test_the_filters_and_the_page(loaded: Cli, flags, labels):
    out = loaded.ok("alerts", "list", *flags).stdout
    if labels:
        assert [r[2] for r in rows(out)] == labels
    else:
        assert out.strip() == "No alerts."


def test_the_date_range_takes_days_and_moments(loaded: Cli):
    day = NOW.date().isoformat()
    assert len(rows(loaded.ok("alerts", "list", "--since", day, "--until", day).stdout)) == 3
    far = (NOW + timedelta(days=3)).date().isoformat()
    assert loaded.ok("alerts", "list", "--since", far).stdout.strip() == "No alerts."
    moment = (NOW + timedelta(hours=1, minutes=2)).isoformat()
    assert [r[2] for r in rows(loaded.ok("alerts", "list", "--since", moment).stdout)] == [
        "recycled_content"
    ]


@pytest.mark.parametrize(
    "flags",
    [
        ["--severity", "urgent"],
        ["--label", "no_risk"],
        ["--status", "archived"],
        ["--source", "' OR 1=1 --"],
        ["--tag", "Robert'); DROP TABLE alerts;--"],
        ["--since", "yesterday"],
        ["--until", "2026-02-30"],
        ["--since", "2026-03-02", "--until", "2026-03-01"],
        ["--severity", ""],
    ],
)
def test_a_malformed_filter_is_refused_with_one_error_line_and_exit_1(loaded: Cli, flags):
    result = loaded("alerts", "list", *flags)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith("error: ")
    assert len(result.stderr.strip().splitlines()) == 1
    assert "DROP" not in result.stderr


@pytest.mark.parametrize(
    "flags", [["--limit", "0"], ["--limit", "201"], ["--offset", "-1"], ["--limit", "x"]]
)
def test_a_page_out_of_range_is_a_usage_error(loaded: Cli, flags):
    assert loaded("alerts", "list", *flags).exit_code == 2


def test_sql_in_a_filter_is_never_run(loaded: Cli):
    loaded("alerts", "list", "--tag", "x'; DROP TABLE alerts; --")
    assert len(rows(loaded.ok("alerts", "list").stdout)) == 3


# ---- show ----


def test_show_prints_the_alert_its_evidence_its_items_and_its_trail(loaded: Cli):
    alert_id = first_id(loaded, "--label", "deliberate_disinformation")
    out = loaded.ok("alerts", "show", alert_id).stdout
    assert f"alert: {alert_id}" in out
    for line in (
        "severity: critical",
        "label: deliberate_disinformation",
        "confidence: 0.900",
        "status: new",
        "occurrences: 1",
        "source: City News (src_news)",
        "rationale: prediction_claim",
        "deciding stage: rules",
        "catalog: unavailable",
        "signals: prediction_claim (0.45)",
        "tags: earthquake, prediction",
    ):
        assert line in out, line
    assert "Texto b" in out
    assert "created" in out


def test_show_says_when_a_source_was_removed(loaded: Cli):
    alert_id = first_id(loaded)
    loaded.ok("sources", "delete", "src_news", "--reason", WHY)
    out = loaded.ok("alerts", "show", alert_id).stdout
    assert "source: City News (src_news) [source removed]" in out
    assert "[source removed]" in loaded.ok("alerts", "list").stdout


@pytest.mark.parametrize("bad", ["alt_" + "0" * 32, "x", "' OR 1=1 --", "A" * 200])
def test_an_unknown_alert_is_refused(loaded: Cli, bad):
    result = loaded("alerts", "show", bad)
    assert result.exit_code == 1
    assert result.stderr.strip() == "error: alert not found"


def test_hostile_text_cannot_move_the_terminal(cli: Cli):
    cli.add()
    plant(cli, "x", text="Mañana temblor " + chr(0x202E) + chr(0x1B) + "[31m" + " ya", title="T")
    cli.ok("alerts", "refresh")
    alert_id = first_id(cli)
    out = cli.ok("alerts", "show", alert_id).stdout + cli.ok("alerts", "list").stdout
    assert chr(0x1B) not in out
    assert chr(0x202E) not in out


# ---- marking ----


def mark(cli: Cli, verb: str, alert_id: str, *extra: str):
    return cli("alerts", verb, alert_id, "--reason", WHY, "--actor", "ana", *extra)


def test_marking_walks_the_legal_moves_and_audits_each(loaded: Cli):
    alert_id = first_id(loaded)
    assert mark(loaded, "seen", alert_id).stdout.strip() == f"{alert_id}: new -> seen"
    assert mark(loaded, "dismiss", alert_id).stdout.strip() == f"{alert_id}: seen -> dismissed"
    assert mark(loaded, "reopen", alert_id).stdout.strip() == f"{alert_id}: dismissed -> seen"
    shown = loaded.ok("alerts", "show", alert_id).stdout
    assert shown.count("status_changed") == 3
    assert "ana" in shown
    assert WHY in shown


def test_marking_again_changes_nothing_and_succeeds(loaded: Cli):
    alert_id = first_id(loaded)
    mark(loaded, "seen", alert_id)
    again = mark(loaded, "seen", alert_id)
    assert again.exit_code == 0
    assert again.stdout.strip() == f"{alert_id}: already seen, nothing changed"
    assert loaded.ok("alerts", "show", alert_id).stdout.count("status_changed") == 1


def test_an_illegal_move_is_refused_with_exit_1(loaded: Cli):
    alert_id = first_id(loaded)
    result = mark(loaded, "reopen", alert_id)  # nobody has looked at it yet
    assert (result.exit_code, result.stdout) == (1, "")
    assert result.stderr.strip() == "error: only a dismissed alert can be reopened"


def test_marking_an_unknown_alert_is_refused(loaded: Cli):
    result = mark(loaded, "dismiss", "alt_" + "f" * 32)
    assert (result.exit_code, result.stderr.strip()) == (1, "error: alert not found")


def test_marking_needs_a_reason(loaded: Cli):
    assert loaded("alerts", "seen", first_id(loaded)).exit_code == 2


def test_marking_refuses_a_blank_reason_or_actor(loaded: Cli):
    alert_id = first_id(loaded)
    assert loaded("alerts", "seen", alert_id, "--reason", "  ").exit_code == 1
    assert loaded("alerts", "seen", alert_id, "--reason", WHY, "--actor", "").exit_code == 1


def test_the_actor_comes_from_the_environment_by_default(loaded: Cli):
    alert_id = first_id(loaded)
    loaded.ok("alerts", "seen", alert_id, "--reason", WHY)
    assert "tester" in loaded.ok("alerts", "show", alert_id).stdout


# ---- stats ----


def test_stats_count_by_status_and_severity(loaded: Cli):
    mark(loaded, "seen", first_id(loaded))
    out = loaded.ok("alerts", "stats").stdout
    assert "alerts: 3 (2 new, 0 withdrawn)" in out
    assert "seen 1" in out
    assert "critical 1" in out


def test_stats_on_an_empty_database(cli: Cli):
    assert cli.ok("alerts", "stats").stdout.startswith("alerts: 0 (0 new, 0 withdrawn)")


# ---- the threshold ----


def test_the_threshold_starts_at_half_and_is_set_with_a_reason(cli: Cli):
    assert cli.ok("alerts", "threshold", "show").stdout.strip() == "alert threshold: 0.5"
    result = cli.ok("alerts", "threshold", "set", "0.65", "--reason", WHY, "--actor", "ana")
    assert result.stdout.strip() == "alert threshold set to 0.65 (was 0.5)"
    assert cli.ok("alerts", "threshold", "show").stdout.strip() == "alert threshold: 0.65"
    (event,) = cli.details("alert_threshold_changed")
    assert (event["actor"], event["reason"], event["from"], event["to"]) == (
        "ana",
        WHY,
        "0.5",
        "0.65",
    )
    assert "alert threshold: 0.65" in cli.ok("status").stdout


def test_setting_the_same_threshold_again_is_a_no_op(cli: Cli):
    cli.ok("alerts", "threshold", "set", "0.7", "--reason", WHY)
    again = cli.ok("alerts", "threshold", "set", "0.7", "--reason", WHY)
    assert again.stdout.strip() == "alert threshold unchanged: 0.7"
    assert len(cli.details("alert_threshold_changed")) == 1


@pytest.mark.parametrize("value", ["0", "0.0", "1", "1.0", "0.5", "0.01", "0.999"])
def test_the_threshold_accepts_zero_up_to_one(cli: Cli, value):
    assert cli("alerts", "threshold", "set", value, "--reason", WHY).exit_code == 0


@pytest.mark.parametrize("value", ["1.01", "2", "abc", "nan", "inf", "1e-1", "0,5", "50%", "", " "])
def test_a_threshold_out_of_range_or_not_a_number_is_refused(cli: Cli, value):
    result = cli("alerts", "threshold", "set", value, "--reason", WHY)
    assert result.exit_code == 1
    assert result.stderr.startswith("error: ")
    assert cli.ok("alerts", "threshold", "show").stdout.strip() == "alert threshold: 0.5"


def test_a_negative_threshold_is_refused(cli: Cli):
    result = cli("alerts", "threshold", "set", "--reason", WHY, "--", "-0.1")
    assert result.exit_code == 1
    assert cli.details("alert_threshold_changed") == []


def test_the_threshold_needs_a_reason(cli: Cli):
    assert cli("alerts", "threshold", "set", "0.7").exit_code == 2


def test_a_new_threshold_decides_what_the_next_refresh_raises(cli: Cli):
    cli.add()
    plant(cli, "a", confidence=0.55)
    plant(cli, "b", confidence=0.75, hours=1)
    cli.ok("alerts", "threshold", "set", "0.7", "--reason", WHY)
    cli.ok("alerts", "refresh")
    assert [r[4] for r in rows(cli.ok("alerts", "list").stdout)] == ["0.75"]


# ---- the shell ----


def test_an_unexpected_failure_is_a_bug_report_not_a_traceback_with_values(
    loaded: Cli, monkeypatch: pytest.MonkeyPatch
):
    from social_tracker.alerts.service import AlertService

    def boom(self, *args, **kwargs):
        raise RuntimeError("".join(["secret", "-value-in-message"]))

    monkeypatch.setattr(AlertService, "list", boom)
    result = loaded("alerts", "list")
    assert result.exit_code == 70
    assert "this is a bug" in result.stderr
    assert "secret-value-in-message" not in result.stderr


def test_the_alert_commands_never_publish_share_or_notify(cli: Cli):
    from social_tracker.entrypoints.cli import app

    group = next(g for g in app.registered_groups if g.name == "alerts")
    names = {c.name or c.callback.__name__ for c in group.typer_instance.registered_commands}
    names |= {g.name for g in group.typer_instance.registered_groups}
    assert names >= {"list", "show", "seen", "dismiss", "reopen", "rebuild", "refresh", "stats"}
    for word in (
        "publish",
        "post",
        "share",
        "send",
        "notify",
        "email",
        "webhook",
        "draft",
        "approve",
    ):
        assert not any(word in name for name in names), word
    help_text = cli.ok("alerts", "--help").stdout.lower()
    for word in ("publish", "webhook", "email", "approve", "draft"):
        assert word not in help_text.replace("no external notifications", ""), word


def test_the_alerts_group_without_a_command_shows_its_help(cli: Cli):
    result = cli("alerts")
    shown = result.stdout + result.stderr
    assert "list" in shown
    assert "threshold" in shown
