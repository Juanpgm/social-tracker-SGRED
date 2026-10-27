"""`social-tracker classify ...` and `topics ...`: the cascade and its vocabulary from the shell."""

from datetime import timedelta
from pathlib import Path

import pytest

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV
from social_tracker.classification.yaml_io import default_topic_config, load_topic_config
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from tests.entrypoints.conftest import WHY, Cli
from tests.items.factories import NOW, make_item

PREDICTION = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
REPORT = "El SGC reportó un sismo de magnitud 4.2 en Cali"
OFF_TOPIC = "El equipo ganó el partido de ayer en el estadio de la ciudad"


def seed(cli: Cli, *items) -> None:
    with Database(cli.db_path) as db:
        SqliteItemRepository(db).add_new(items)


def sample() -> list:
    return [
        make_item("a", title="Alerta", text=PREDICTION, published_at=NOW),
        make_item("b", title="Sismo", text=REPORT, published_at=NOW + timedelta(hours=1)),
        make_item("c", title="Fútbol", text=OFF_TOPIC, published_at=NOW + timedelta(hours=2)),
    ]


@pytest.fixture
def ready(cli: Cli) -> Cli:
    """A source with everything switched on, and three items waiting."""
    cli.add()
    cli.ok("sources", "enable", "src_news", "--reason", WHY)
    cli.ok("analysis", "on", "--source", "src_news", "--reason", WHY)
    cli.ok("analysis", "on", "--reason", WHY)
    seed(cli, *sample())
    return cli


def write(tmp_path: Path, text: str, name: str = "topics.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---- classify run ----


def test_run_explains_why_nothing_is_classified_while_nothing_is_eligible(cli: Cli):
    cli.add()
    seed(cli, *sample())
    result = cli("classify", "run")
    assert result.exit_code == 3  # nothing could be done: distinct from success and from refusal
    assert "no source is eligible" in result.stderr
    assert "global analysis" in result.stderr
    assert result.stdout == ""


def test_run_classifies_the_pending_items_and_reports_the_counts(ready: Cli):
    result = ready.ok("classify", "run")
    assert result.stdout.strip() == (
        "classified 2 item(s), 1 out of scope, 0 skipped, 0 error(s); 1 queued for the LLM"
    )
    again = ready.ok("classify", "run")
    assert again.stdout.strip() == (
        "classified 0 item(s), 0 out of scope, 0 skipped, 0 error(s); 0 queued for the LLM"
    )


def test_run_honours_the_limit(ready: Cli):
    first = ready.ok("classify", "run", "--limit", "1")
    assert first.stdout.startswith("classified 1 item(s)")
    rest = ready.ok("classify", "run", "--limit", "5")
    assert rest.stdout.startswith("classified 1 item(s), 1 out of scope")


@pytest.mark.parametrize("limit", ["0", "-1", "1001", "abc", "1.5"])
def test_run_refuses_a_bad_limit_as_a_usage_error(ready: Cli, limit: str):
    result = ready("classify", "run", "--limit", limit)
    assert result.exit_code == 2
    assert result.stdout == ""


def test_run_leaves_items_of_a_source_with_analysis_off_alone(cli: Cli):
    cli.add()
    cli.ok("sources", "enable", "src_news", "--reason", WHY)
    cli.ok("analysis", "on", "--reason", WHY)  # the source's own switch stays off
    seed(cli, *sample())
    result = cli("classify", "run")
    assert (result.exit_code, "no source is eligible" in result.stderr) == (3, True)
    assert "No classifications yet." in cli.ok("classify", "stats").stdout


def test_run_exits_with_status_one_when_a_storage_error_stopped_it(ready: Cli):
    with Database(ready.db_path).transaction():  # another process holds the write lock
        from_outside = ready("classify", "run", env={BUSY_TIMEOUT_ENV: "50"})
    assert from_outside.exit_code == 1
    assert "1 error(s)" in from_outside.stdout
    assert "classified 0 item(s)" in from_outside.stdout
    assert "a storage error stopped the run" in from_outside.stderr
    assert ready.ok("classify", "run").stdout.startswith("classified 2 item(s)")  # it resumes


# ---- classify show ----


def item_id(name: str) -> str:
    return next(i.id for i in sample() if i.external_id == name)


def test_show_prints_the_verdict_of_an_item(ready: Cli):
    ready.ok("classify", "run")
    out = ready.ok("classify", "show", item_id("a")).stdout
    assert f"item: {item_id('a')}" in out
    assert "label: unintentional_rumor" in out
    assert "confidence: 0.560" in out
    assert "stage: rules" in out
    assert "topics: earthquake" in out
    assert "prediction_claim (0.45)" in out
    assert "flags: needs_llm" in out
    assert "catalog: unavailable" in out
    assert "source credibility used: 3" in out
    assert "classifier version: cascade-2/topics-0" in out
    assert "rationale: unintentional_rumor candidate from rules" in out


def test_show_lists_every_flag_that_is_set(ready: Cli):
    seed(ready, make_item("z", title="Raro", text="sismo xkqzv wpfhj zzzq", published_at=NOW))
    ready.ok("classify", "run")
    out = ready.ok("classify", "show", make_item("z").id).stdout
    assert "label: unclassified" in out
    assert "flags: needs_llm, low_confidence" in out
    assert "signals: none" in out


def test_show_prints_an_out_of_scope_verdict_with_its_reason(ready: Cli):
    ready.ok("classify", "run")
    out = ready.ok("classify", "show", item_id("c")).stdout
    assert "label: out_of_scope" in out
    assert "reason: no_topic_match" in out
    assert "flags: none" in out


def test_show_refuses_an_unknown_or_invalid_id(ready: Cli):
    ready.ok("classify", "run")
    unknown = ready("classify", "show", "itm_unknown")
    assert (unknown.exit_code, unknown.stdout) == (1, "")
    assert "error: this item has no classification" in unknown.stderr
    bad = ready("classify", "show", "BAD ID")
    assert (bad.exit_code, bad.stdout) == (1, "")


def test_run_exits_zero_when_a_source_is_eligible_but_nothing_is_pending(ready: Cli):
    ready.ok("classify", "run")
    again = ready("classify", "run")
    assert again.exit_code == 0  # eligible sources exist: "nothing to do" is not a failure
    assert again.stdout.startswith("classified 0 item(s)")
    assert again.stderr == ""


class _Container:
    def __init__(self, report):
        self.classifier = self
        self._report = report

    def classify_pending(self, limit):
        return self._report

    def close(self):
        pass


def run_with_report(monkeypatch, cli: Cli, **fields):
    from social_tracker.classification.service import ClassifyReport
    from social_tracker.entrypoints import cli as module

    report = ClassifyReport(eligible_sources=1, **fields)
    monkeypatch.setattr(module, "build_container", lambda: _Container(report))
    return cli("classify", "run")


def test_a_storage_error_and_item_errors_are_told_apart(monkeypatch, cli: Cli):
    storage = run_with_report(monkeypatch, cli, storage_errors=1)
    assert storage.exit_code == 1
    assert "a storage error stopped the run" in storage.stderr
    assert "1 error(s)" in storage.stdout

    items = run_with_report(monkeypatch, cli, errors=2, classified=3)
    assert items.exit_code == 0  # the run completed; the items are retried
    assert "a storage error" not in items.stderr
    assert "2 item(s) could not be classified" in items.stderr
    assert "2 error(s)" in items.stdout


def test_parked_items_and_a_reached_candidate_limit_are_reported(monkeypatch, cli: Cli):
    result = run_with_report(monkeypatch, cli, failed=1, candidate_cap_hits=4, classified=1)
    assert "1 parked as classification_failed" in result.stdout
    assert "candidate limit was reached 4 time(s)" in result.stdout


def test_a_clean_report_mentions_neither(monkeypatch, cli: Cli):
    result = run_with_report(monkeypatch, cli, classified=2)
    assert "parked" not in result.stdout
    assert "candidate limit" not in result.stdout
    assert result.stderr == ""


# ---- classify reclassify ----


def test_reclassify_sends_out_of_scope_items_back_and_the_next_run_uses_the_new_vocabulary(
    ready: Cli, tmp_path: Path
):
    ready.ok("classify", "run")
    old = ready.ok("classify", "show", item_id("c")).stdout
    assert "label: out_of_scope" in old
    path = write(tmp_path, "version: 1\nearthquake: [sismo]\nother_disaster: [estadio]\n")
    ready.ok("topics", "set-file", str(path), "--reason", WHY)
    result = ready.ok("classify", "reclassify", "--status", "out_of_scope")
    assert result.stdout.startswith("1 item(s) sent back to the queue")
    assert "No classifications" not in ready.ok("classify", "stats").stdout
    ready.ok("classify", "run")
    shown = ready.ok("classify", "show", item_id("c")).stdout
    assert "topics: other_disaster" in shown  # the stadium is now vocabulary
    assert "classifier version: cascade-2/topics-1" in shown


def test_reclassify_honours_the_limit_and_says_when_there_is_nothing_to_redo(ready: Cli):
    seed(ready, make_item("o2", title="Fútbol", text=OFF_TOPIC + " otra vez", published_at=NOW))
    ready.ok("classify", "run")
    assert ready.ok(
        "classify", "reclassify", "--status", "out_of_scope", "--limit", "1"
    ).stdout.startswith("1 item(s)")
    assert ready.ok(
        "classify", "reclassify", "--status", "classification_failed"
    ).stdout.startswith("0 item(s)")


@pytest.mark.parametrize("status", ["classified", "new", "bogus", "", "OUT_OF_SCOPE"])
def test_reclassify_refuses_any_other_status_as_a_usage_error(ready: Cli, status: str):
    result = ready("classify", "reclassify", "--status", status)
    assert result.exit_code == 2
    assert result.stdout == ""


@pytest.mark.parametrize("limit", ["0", "-1", "1001", "abc"])
def test_reclassify_refuses_a_bad_limit(ready: Cli, limit: str):
    assert (
        ready("classify", "reclassify", "--status", "out_of_scope", "--limit", limit).exit_code == 2
    )


def test_reclassify_needs_a_status(ready: Cli):
    assert ready("classify", "reclassify").exit_code == 2


# ---- classify stats ----


def test_stats_summarize_the_stored_verdicts(ready: Cli):
    ready.ok("classify", "run")
    out = ready.ok("classify", "stats").stdout
    assert "classifications: 3" in out
    assert "no_risk 1" in out
    assert "out_of_scope 1" in out
    assert "unintentional_rumor 1" in out
    assert "no_topic_match 1" in out
    assert "queued for the LLM: 1" in out
    assert "low confidence: 0" in out
    assert "duplicates: 0" in out
    assert "waiting for their original's judgment: 0" in out
    assert "near copies: 0" in out
    assert "parked as classification_failed: 0" in out


def test_stats_of_an_empty_store(cli: Cli):
    assert cli.ok("classify", "stats").stdout.strip() == "No classifications yet."


# ---- topics ----


def test_topics_show_prints_the_configuration_in_force_as_loadable_yaml(cli: Cli):
    out = cli.ok("topics", "show").stdout
    assert out.startswith("# topics: defaults shipped with the application (version 0)")
    assert "verify before relying" in out
    assert load_topic_config(out) == default_topic_config()


def test_topics_set_file_installs_a_new_vocabulary_and_audits_it(cli: Cli, tmp_path: Path):
    path = write(tmp_path, "version: 1\nother_disaster: [granizo]\nplaces: [Cali]\n")
    assert cli.ok("topics", "set-file", str(path), "--reason", WHY).stdout.strip() == (
        "topics updated: version 1"
    )
    again = cli.ok("topics", "set-file", str(path), "--reason", WHY)
    assert again.stdout.strip() == "topics unchanged: version 1"
    [event] = cli.details("topic_config_changed")
    assert (event["actor"], event["reason"], event["from"], event["to"]) == ("tester", WHY, 0, 1)
    shown = cli.ok("topics", "show").stdout
    assert shown.startswith("# topics: stored version 1")
    assert "granizo" in shown


def test_the_new_vocabulary_applies_to_the_next_classification(ready: Cli, tmp_path: Path):
    seed(
        ready, make_item("g", title="Granizo", text="El granizo cayó sobre Cali", published_at=NOW)
    )
    ready.ok("classify", "run")
    assert "label: out_of_scope" in ready.ok("classify", "show", make_item("g").id).stdout
    path = write(tmp_path, "version: 1\nother_disaster: [granizo]\nplaces: [Cali]\n")
    ready.ok("topics", "set-file", str(path), "--reason", WHY)
    seed(ready, make_item("h", title="Granizo", text="Otro granizo sobre Cali", published_at=NOW))
    ready.ok("classify", "run")
    shown = ready.ok("classify", "show", make_item("h").id).stdout
    assert "topics: other_disaster" in shown
    assert "classifier version: cascade-2/topics-1" in shown


@pytest.mark.parametrize(
    "text",
    [
        "",
        "version: 2\nearthquake: [sismo]\n",
        "version: 1\nunknown: 1\n",
        "version: 1\nearthquake: [sismo]\nearthquake: [temblor]\n",
        "version: 1\nearthquake: &a [sismo]\nplaces: *a\n",
        "version: 1\nearthquake: [sismo, Sismo]\n",
        "[",
    ],
)
def test_an_invalid_file_is_refused_and_changes_nothing(cli: Cli, tmp_path: Path, text: str):
    result = cli("topics", "set-file", str(write(tmp_path, text)), "--reason", WHY)
    assert (result.exit_code, result.stdout) == (1, "")
    assert result.stderr.startswith("error: ")
    assert cli.details("topic_config_changed") == []
    assert cli.ok("topics", "show").stdout.startswith("# topics: defaults")


def test_an_oversize_file_is_refused_without_being_parsed(cli: Cli, tmp_path: Path):
    path = write(tmp_path, "version: 1\nearthquake: [sismo]\n" + "#" * 70_000)
    result = cli("topics", "set-file", str(path), "--reason", WHY)
    assert result.exit_code == 1
    assert "larger than 65536 bytes" in result.stderr


def test_a_file_of_exactly_the_limit_is_read(cli: Cli, tmp_path: Path):
    head = "version: 1\nearthquake: [sismo]\n# "
    path = tmp_path / "exact.yaml"
    path.write_bytes((head + "x" * (65536 - len(head) - 1) + "\n").encode())
    assert path.stat().st_size == 65536
    assert "version 1" in cli.ok("topics", "set-file", str(path), "--reason", WHY).stdout


def test_a_missing_file_or_a_directory_is_refused_without_naming_a_path(cli: Cli, tmp_path: Path):
    for target in (tmp_path / "nope.yaml", tmp_path):
        result = cli("topics", "set-file", str(target), "--reason", WHY)
        assert result.exit_code == 1
        assert str(tmp_path) not in result.stderr
        assert result.stdout == ""


def test_a_file_that_is_not_utf8_is_refused(cli: Cli, tmp_path: Path):
    path = tmp_path / "latin1.yaml"
    path.write_bytes("version: 1\nearthquake: [sísmo]\n".encode("latin-1"))
    result = cli("topics", "set-file", str(path), "--reason", WHY)
    assert result.exit_code == 1
    assert "not valid UTF-8" in result.stderr


def test_topics_reset_returns_to_the_defaults_once(cli: Cli, tmp_path: Path):
    cli.ok(
        "topics", "set-file", str(write(tmp_path, "version: 1\nearthquake: [sismo]\n")), "-r", WHY
    )
    assert cli.ok("topics", "reset", "--reason", WHY).stdout.strip() == (
        "topics reset to the defaults: version 2"
    )
    assert cli.ok("topics", "reset", "--reason", WHY).stdout.strip() == (
        "topics already use the defaults"
    )
    assert load_topic_config(cli.ok("topics", "show").stdout) == default_topic_config()


@pytest.mark.parametrize("command", [["set-file", "x.yaml"], ["reset"]])
def test_changing_topics_needs_a_reason(cli: Cli, command: list[str]):
    result = cli("topics", *command)
    assert result.exit_code == 2
    assert cli.details("topic_config_changed") == []


def test_a_blank_reason_is_refused_by_the_service(cli: Cli, tmp_path: Path):
    path = write(tmp_path, "version: 1\nearthquake: [sismo]\n")
    result = cli("topics", "set-file", str(path), "--reason", "  ")
    assert result.exit_code == 1


def test_the_commands_are_listed_in_the_help(cli: Cli):
    top = cli.ok("--help").stdout
    assert "classify" in top
    assert "topics" in top
    assert "set-file" in cli.ok("topics", "--help").stdout
    assert "stats" in cli.ok("classify", "--help").stdout
