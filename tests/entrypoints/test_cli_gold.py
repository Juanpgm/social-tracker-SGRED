"""`social-tracker gold ...` and `classify calibrate`: labeling texts by hand and measuring the
cascade with them. Synthetic texts only; the real judge is never reached."""

import re
from pathlib import Path

import pytest

from social_tracker import bootstrap
from social_tracker.classification.domain import Label
from social_tracker.entrypoints import cli as cli_module
from social_tracker.llm.domain import JudgeError, JudgeErrorKind
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from tests.entrypoints.conftest import WHY, Cli
from tests.items.factories import NOW, make_item
from tests.llm.harness import ScriptedJudge
from tests.llm.support import response

PREDICTION = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
REPORT = "Un sismo de magnitud 4.5 se registró anoche en Jamundí sin reportes de daños ni heridos"
OFF_TOPIC = "El equipo ganó el partido de ayer en el estadio de la ciudad"
SEED = Path(__file__).parents[1] / "data" / "gold_examples.yaml"
ID = re.compile(r"gld_[0-9a-f]{16}")


def text_file(tmp_path: Path, text: str, name: str = "t.txt") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def add(cli: Cli, tmp_path: Path, text: str, label: str, *extra: str, name: str = "t.txt"):
    path = text_file(tmp_path, text, name)
    return cli("gold", "add", "--text-file", str(path), "--label", label, *extra)


@pytest.fixture
def labeled(cli: Cli, tmp_path: Path) -> Cli:
    for n, (text, label) in enumerate(
        [(PREDICTION, "unintentional_rumor"), (REPORT, "no_risk"), (OFF_TOPIC, "out_of_scope")]
    ):
        assert add(cli, tmp_path, text, label, name=f"t{n}.txt").exit_code == 0
    return cli


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> ScriptedJudge:
    scripted = ScriptedJudge()
    scripted.default = response(label=Label.NO_RISK, confidence=0.9)
    monkeypatch.setattr(
        cli_module, "build_llm", lambda stop=None: bootstrap.build_llm(judge=scripted, stop=stop)
    )
    return scripted


# ---- add ----


def test_adding_a_text_prints_its_id_and_label(cli: Cli, tmp_path: Path):
    result = add(cli, tmp_path, PREDICTION, "unintentional_rumor", "--note", "a chain")
    assert result.exit_code == 0
    assert re.fullmatch(r"added gld_[0-9a-f]{16} \(unintentional_rumor\)\n", result.stdout)


def test_the_same_text_again_is_not_added_twice(cli: Cli, tmp_path: Path):
    add(cli, tmp_path, PREDICTION, "no_risk")
    again = add(cli, tmp_path, PREDICTION, "no_risk")
    assert again.exit_code == 0
    assert "already in the gold set" in again.stdout
    assert cli.ok("gold", "list").stdout.count("gld_") == 1


def test_another_label_for_a_text_that_is_there_is_refused(cli: Cli, tmp_path: Path):
    add(cli, tmp_path, PREDICTION, "no_risk")
    result = add(cli, tmp_path, PREDICTION, "unintentional_rumor")
    assert (result.exit_code, result.stdout) == (1, "")
    assert "another label" in result.stderr


@pytest.mark.parametrize("label", ["rumor", "NO_RISK", "", "no_risk ", "drop table"])
def test_an_unknown_label_is_refused(cli: Cli, tmp_path: Path, label):
    result = add(cli, tmp_path, PREDICTION, label)
    assert result.exit_code == 1
    assert result.stderr.startswith("error: ")
    assert cli.ok("gold", "list").stdout.strip() == "No gold examples yet."


@pytest.mark.parametrize("credibility", ["0", "6", "-1", "x", "3.5"])
def test_a_credibility_out_of_range_is_a_usage_error(cli: Cli, tmp_path: Path, credibility):
    result = add(cli, tmp_path, PREDICTION, "no_risk", "--credibility", credibility)
    assert result.exit_code == 2


def test_the_options_that_shape_an_example_are_kept(cli: Cli, tmp_path: Path):
    add(
        cli, tmp_path, PREDICTION, "no_risk", "--note", "n1", "--credibility", "5",
        "--url", "https://example.org/a", "--actor", "ana",
    )  # fmt: skip
    (row,) = (line for line in cli.ok("gold", "list").stdout.splitlines() if "gld_" in line)
    assert "no_risk" in row
    assert "n1" in row
    exported = cli.ok("gold", "export").stdout
    assert "credibility: 5" in exported
    assert "url: https://example.org/a" in exported


def test_a_note_that_is_not_plain_text_is_refused(cli: Cli, tmp_path: Path):
    assert add(cli, tmp_path, PREDICTION, "no_risk", "--note", "x" * 501).exit_code == 1


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"   \n\t", id="blank"),
        pytest.param(b"\xff\xfe\x00bad", id="binary"),
        pytest.param(b"x" * 200_000, id="huge"),
    ],
)
def test_a_text_file_that_is_empty_binary_or_huge_is_refused(cli: Cli, tmp_path: Path, content):
    path = tmp_path / "bad.txt"
    path.write_bytes(content)
    result = cli("gold", "add", "--text-file", str(path), "--label", "no_risk")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert str(tmp_path) not in result.stderr  # no path in a refusal


def test_a_missing_file_or_a_directory_is_refused(cli: Cli, tmp_path: Path):
    for target in (tmp_path / "missing.txt", tmp_path):
        result = cli("gold", "add", "--text-file", str(target), "--label", "no_risk")
        assert result.exit_code == 1
        assert result.stderr.startswith("error: ")


def test_control_characters_in_a_file_are_cleaned_not_stored(cli: Cli, tmp_path: Path):
    add(cli, tmp_path, "Mañana\x1b[31m habrá\x07 un temblor", "no_risk")
    shown = cli.ok("gold", "list").stdout + cli.ok("gold", "export").stdout
    assert "\x1b" not in shown
    assert "\x07" not in shown


def test_a_text_comes_from_a_file_or_from_a_stored_item_not_both_and_not_neither(
    cli: Cli, tmp_path: Path
):
    path = text_file(tmp_path, PREDICTION)
    both = cli("gold", "add", "--text-file", str(path), "--item", "itm_abc", "--label", "no_risk")
    neither = cli("gold", "add", "--label", "no_risk")
    assert both.exit_code == 2
    assert neither.exit_code == 2


def test_an_item_that_was_collected_can_be_labeled_by_its_id(cli: Cli):
    cli.add()
    item = make_item("a", title="Alerta", text=PREDICTION, published_at=NOW)
    with Database(cli.db_path) as db:
        SqliteItemRepository(db).add_new([item])
    result = cli("gold", "add", "--item", item.id, "--label", "unintentional_rumor")
    assert result.exit_code == 0, result.stderr
    assert "Mañana habrá un temblor fuerte" in cli.ok("gold", "export").stdout
    assert "Alerta" in cli.ok("gold", "export").stdout


@pytest.mark.parametrize("item_id", ["itm_" + "0" * 32, "x", "' OR 1=1 --"])
def test_an_unknown_item_is_refused(cli: Cli, item_id):
    result = cli("gold", "add", "--item", item_id, "--label", "no_risk")
    assert result.exit_code == 1
    assert "not found" in result.stderr


# ---- label (interactive) ----


def test_label_asks_for_the_label_and_the_note(cli: Cli, tmp_path: Path):
    path = text_file(tmp_path, PREDICTION)
    result = cli.runner.invoke(
        cli_module.app,
        ["gold", "label", "--text-file", str(path)],
        input="unintentional_rumor\na chain\n",
        env=cli_env(cli),
    )
    assert result.exit_code == 0, result.output
    assert "Label" in result.stdout
    assert "added gld_" in result.stdout
    assert "a chain" in cli.ok("gold", "list").stdout


def test_label_asks_again_for_a_label_that_is_not_one(cli: Cli, tmp_path: Path):
    path = text_file(tmp_path, PREDICTION)
    result = cli.runner.invoke(
        cli_module.app,
        ["gold", "label", "--text-file", str(path)],
        input="nonsense\nno_risk\n\n",
        env=cli_env(cli),
    )
    assert result.exit_code == 0, result.output
    assert "not one of" in result.stdout or "Error" in result.stdout
    assert "added gld_" in result.stdout


def test_label_without_anything_to_answer_with_gives_up_without_storing(cli: Cli, tmp_path: Path):
    path = text_file(tmp_path, PREDICTION)
    result = cli.runner.invoke(
        cli_module.app, ["gold", "label", "--text-file", str(path)], input="", env=cli_env(cli)
    )
    assert result.exit_code != 0
    assert cli.ok("gold", "list").stdout.strip() == "No gold examples yet."


def test_label_with_the_label_already_given_asks_nothing(cli: Cli, tmp_path: Path):
    path = text_file(tmp_path, PREDICTION)
    result = cli("gold", "label", "--text-file", str(path), "--label", "no_risk", "--note", "n")
    assert result.exit_code == 0
    assert "added gld_" in result.stdout


def cli_env(cli: Cli) -> dict[str, str]:
    from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV
    from social_tracker.entrypoints.inputs import ACTOR_ENV

    return {DB_PATH_ENV: str(cli.db_path), ACTOR_ENV: "tester", BUSY_TIMEOUT_ENV: "2000"}


# ---- list, show, remove ----


def test_list_shows_the_examples_in_the_order_added(labeled: Cli):
    out = labeled.ok("gold", "list").stdout
    labels = [line.split()[1] for line in out.splitlines()[1:] if line.startswith("gld_")]
    assert labels == ["unintentional_rumor", "no_risk", "out_of_scope"]
    assert out.splitlines()[0].split()[:3] == ["ID", "LABEL", "CRED"]


def test_list_filters_and_pages(labeled: Cli):
    assert labeled.ok("gold", "list", "--label", "no_risk").stdout.count("gld_") == 1
    assert labeled.ok("gold", "list", "--limit", "2").stdout.count("gld_") == 2
    assert labeled.ok("gold", "list", "--limit", "2", "--offset", "2").stdout.count("gld_") == 1
    assert labeled("gold", "list", "--label", "bogus").exit_code == 1
    assert labeled("gold", "list", "--limit", "0").exit_code == 2


def test_a_long_text_is_shortened_in_the_list_and_whole_in_show(cli: Cli, tmp_path: Path):
    add(cli, tmp_path, "Mañana habrá un temblor " + "muy " * 100 + "fuerte", "no_risk")
    listed = cli.ok("gold", "list").stdout
    assert "..." in listed
    gold_id = ID.search(listed).group()
    shown = cli.ok("gold", "show", gold_id).stdout
    assert shown.count("muy") == 100
    assert "label: no_risk" in shown


def test_show_and_remove_refuse_an_unknown_id(cli: Cli):
    for command in ("show", "remove"):
        result = cli("gold", command, "gld_" + "0" * 16)
        assert (result.exit_code, result.stderr.strip()) == (1, "error: gold example not found")


def test_remove_forgets_one_example(labeled: Cli):
    gold_id = ID.search(labeled.ok("gold", "list").stdout).group()
    assert labeled.ok("gold", "remove", gold_id).stdout.strip() == f"removed {gold_id}"
    assert labeled.ok("gold", "list").stdout.count("gld_") == 2


# ---- export ----


def test_export_prints_yaml_in_the_shape_of_the_seed(labeled: Cli):
    out = labeled.ok("gold", "export").stdout
    assert out.startswith("# Gold set exported by social-tracker")
    assert out.count("- id: gld_") == 3
    assert "label: out_of_scope" in out


def test_export_to_a_file_refuses_to_overwrite_unless_forced(labeled: Cli, tmp_path: Path):
    target = tmp_path / "gold.yaml"
    first = labeled.ok("gold", "export", "-o", str(target))
    assert "exported 3 example(s)" in first.stdout
    assert target.read_text(encoding="utf-8").count("- id: gld_") == 3
    again = labeled("gold", "export", "-o", str(target))
    assert again.exit_code == 1
    assert "--force" in again.stderr
    forced = labeled("gold", "export", "-o", str(target), "--force", "--label", "no_risk")
    assert forced.exit_code == 0
    assert target.read_text(encoding="utf-8").count("- id: gld_") == 1


def test_export_of_an_empty_set_is_a_valid_empty_file(cli: Cli):
    assert "examples: []" in cli.ok("gold", "export").stdout


def test_the_export_feeds_the_calibration(labeled: Cli, tmp_path: Path):
    target = tmp_path / "gold.yaml"
    labeled.ok("gold", "export", "-o", str(target))
    out = labeled.ok("classify", "calibrate", "--file", str(target)).stdout
    assert "3 example(s)" in out


# ---- calibrate ----


def test_calibrating_the_seed_prints_the_whole_report(cli: Cli):
    out = cli.ok("classify", "calibrate", "--file", str(SEED)).stdout
    assert re.search(r"gold set: \d+ example\(s\) from the file", out)
    assert "accuracy: 100.0%" in out
    assert re.search(r"escalation: \d+ of \d+ \(\d+\.\d%\) would go to the language model", out)
    for needle in (
        "per label:",
        "confusion matrix (rows: expected, columns: predicted):",
        "alert threshold sweep (current 0.50):",
        "suggested alert_min_confidence:",
        "unintentional_rumor",
        "out_of_scope",
        "report only: nothing was changed",
        "alerts threshold set",
    ):
        assert needle in out, needle
    assert "without an expected label were left out" in out


def test_the_sweep_has_one_line_per_threshold_and_marks_the_current_one(cli: Cli):
    out = cli.ok("classify", "calibrate", "--file", str(SEED)).stdout
    sweep = out.split("alert threshold sweep")[1].split("suggested")[0]
    thresholds = re.findall(r"^\s+(0\.\d\d)\s", sweep, re.MULTILINE)
    assert thresholds[0] == "0.30"
    assert thresholds[-1] == "0.95"
    assert len(thresholds) == 14
    assert re.search(r"^\s+0\.50 \*", sweep, re.MULTILINE)


def test_the_report_is_the_same_every_time(cli: Cli):
    first = cli.ok("classify", "calibrate", "--file", str(SEED)).stdout
    assert cli.ok("classify", "calibrate", "--file", str(SEED)).stdout == first


def test_the_current_threshold_shown_is_the_one_in_force(cli: Cli):
    cli.ok("alerts", "threshold", "set", "0.72", "--reason", WHY)
    out = cli.ok("classify", "calibrate", "--file", str(SEED)).stdout
    assert "alert threshold sweep (current 0.72)" in out
    assert re.search(r"^\s+0\.72 \*", out, re.MULTILINE)


def test_calibrating_never_changes_the_threshold_or_anything_stored(cli: Cli):
    cli.ok("alerts", "threshold", "set", "0.6", "--reason", WHY)
    before = len(cli.events())
    cli.ok("classify", "calibrate", "--file", str(SEED))
    assert cli.ok("alerts", "threshold", "show").stdout.strip() == "alert threshold: 0.6"
    assert len(cli.events()) == before


def test_the_local_set_is_calibrated_when_no_file_is_given(labeled: Cli):
    out = labeled.ok("classify", "calibrate").stdout
    assert "3 example(s) from the local gold set" in out


def test_an_empty_local_set_says_how_to_start_and_exits_3(cli: Cli):
    result = cli("classify", "calibrate")
    assert result.exit_code == 3
    assert "gold add" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "content",
    [
        "",
        "just text",
        "examples: 5",
        "!!python/object/apply:os.system ['x']",
        "examples: [{id: a}]",
    ],
)
def test_a_bad_gold_file_is_refused_without_echoing_it(cli: Cli, tmp_path: Path, content):
    path = text_file(tmp_path, content, "bad.yaml")
    result = cli("classify", "calibrate", "--file", str(path))
    assert result.exit_code == 1
    assert result.stderr.startswith("error: ")
    assert "os.system" not in result.stderr


def test_a_missing_gold_file_is_refused(cli: Cli, tmp_path: Path):
    result = cli("classify", "calibrate", "--file", str(tmp_path / "nope.yaml"))
    assert result.exit_code == 1


def test_a_file_without_any_labeled_example_exits_3(cli: Cli, tmp_path: Path):
    path = text_file(tmp_path, "examples: [{id: a, text: hola, expect: {in_scope: true}}]")
    result = cli("classify", "calibrate", "--file", str(path))
    assert result.exit_code == 3


# ---- the model is only ever asked when asked for ----


def test_without_the_flag_the_model_is_never_built_or_asked(cli: Cli, monkeypatch, judge):
    built = []
    monkeypatch.setattr(cli_module, "build_llm", lambda stop=None: built.append(1) or None)
    cli.ok("classify", "calibrate", "--file", str(SEED))
    assert built == []
    assert judge.calls == 0


def test_with_the_flag_the_escalated_texts_go_to_the_local_model(cli: Cli, judge):
    out = cli.ok("classify", "calibrate", "--file", str(SEED), "--with-llm").stdout
    assert judge.calls > 0
    match = re.search(r"the model answered (\d+) of (\d+)", out)
    assert match
    assert int(match.group(1)) == judge.calls == int(match.group(2))
    assert "local model" in out


def test_the_model_answers_move_the_report_but_not_the_stored_data(cli: Cli, judge):
    plain = cli.ok("classify", "calibrate", "--file", str(SEED)).stdout
    judged = cli.ok("classify", "calibrate", "--file", str(SEED), "--with-llm").stdout
    assert plain != judged
    assert "report only: nothing was changed" in judged


def test_a_model_server_that_is_down_is_reported_and_the_cascade_report_stands(cli: Cli, judge):
    judge.identity = JudgeError(JudgeErrorKind.UNREACHABLE, "down")
    result = cli("classify", "calibrate", "--file", str(SEED), "--with-llm")
    assert result.exit_code == 0
    assert "accuracy: 100.0%" in result.stdout
    assert "3 failed" in result.stdout
    assert judge.calls == 0  # the text never went anywhere: the server was not identified
    assert "language model" in result.stderr


def test_with_a_busy_model_the_run_still_reports(cli: Cli, judge, monkeypatch):
    from social_tracker.llm.service import EscalationService

    monkeypatch.setattr(EscalationService, "judge_text", lambda self, *a, **k: None)
    result = cli("classify", "calibrate", "--file", str(SEED), "--with-llm")
    assert result.exit_code == 0
    assert "busy" in result.stdout
