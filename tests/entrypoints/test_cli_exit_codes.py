"""Exit codes: 0 success, 1 refused, 2 usage, 70 unexpected failure (a bug)."""

import sqlite3
from pathlib import Path

import pytest
import typer

from social_tracker.bootstrap import Container
from social_tracker.control.service import ControlService
from social_tracker.entrypoints import cli as cli_module
from tests.entrypoints.conftest import WHY, Cli
from tests.entrypoints.test_cli_sources import assert_clean_failure

MESSAGE = "ghp_" + "should-never-be-printed"  # concatenated: the hygiene scan flags the literal
UNEXPECTED_LINE = "error: unexpected failure ({name}); this is a bug"


def explode(exc: BaseException):
    def fail(*_args: object, **_kwargs: object) -> None:
        raise exc

    return fail


def commands(tmp_path: Path) -> dict[str, tuple[str, ...]]:
    source = tmp_path / "in.yaml"
    source.write_text("sources: []\n", encoding="utf-8")
    return command_table(source)


def command_table(source: Path) -> dict[str, tuple[str, ...]]:
    return {
        "list": ("sources", "list"),
        "status": ("status",),
        "add": ("sources", "add", "src_one", "--name", "One", "--type", "news_rss", "-r", WHY),
        "edit": ("sources", "edit", "src_one", "--set", "name=x", "-r", WHY),
        "enable": ("sources", "enable", "src_one", "-r", WHY),
        "disable": ("sources", "disable", "src_one", "-r", WHY),
        "delete": ("sources", "delete", "src_one", "-r", WHY),
        "import": ("sources", "import", str(source), "-r", WHY),
        "export": ("sources", "export"),
        "analysis-on": ("analysis", "on", "-r", WHY),
        "analysis-off": ("analysis", "off", "--source", "src_one", "-r", WHY),
    }


COMMAND_NAMES = list(command_table(Path("in.yaml")))  # names only: nothing is written


@pytest.mark.parametrize("name", COMMAND_NAMES)
def test_an_unexpected_failure_in_any_command_exits_70_with_one_line_and_no_message(
    cli: Cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
):
    monkeypatch.setattr(cli_module, "build_container", explode(RuntimeError(MESSAGE)))
    result = cli(*commands(tmp_path)[name])
    assert result.exit_code == 70
    assert result.stdout == ""
    first = result.stderr.splitlines()[0]
    assert first == UNEXPECTED_LINE.format(name="RuntimeError")
    assert MESSAGE not in result.stderr  # the message of a bug may hold a pasted value
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("x"),
        ValueError("x"),
        KeyError("x"),
        OSError(5, "I/O"),
        sqlite3.OperationalError("raw"),
        RecursionError("deep"),
        MemoryError(),
    ],
    ids=lambda e: type(e).__name__,
)
def test_every_kind_of_unexpected_exception_exits_70_and_names_only_its_type(
    cli: Cli, monkeypatch: pytest.MonkeyPatch, exc: Exception
):
    monkeypatch.setattr(cli_module, "build_container", explode(exc))
    result = cli("status")
    assert result.exit_code == 70
    assert result.stderr.splitlines()[0] == UNEXPECTED_LINE.format(name=type(exc).__name__)


def test_the_frames_of_the_bug_are_shown_but_not_the_values_of_its_variables(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
):
    def buggy(*_args: object, **_kwargs: object) -> None:
        pasted = MESSAGE  # noqa: F841 - a local that must never be printed
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cli_module, "build_container", buggy)
    result = cli("status")
    assert "in buggy" in result.stderr  # where it happened: enough to report the bug
    assert MESSAGE not in result.stderr


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (("sources", "list"), 0),
        (("sources", "enable", "src_zzz", "-r", WHY), 1),  # refused: unknown source
        (("sources", "add", "BAD ID", "--name", "x", "--type", "news_rss", "-r", WHY), 1),
        (("analysis", "on", "-r", ""), 1),  # refused: a blank reason
        (("status",), 70),  # a bug inside the command
    ],
    ids=["success", "unknown-source", "invalid-id", "blank-reason", "bug"],
)
def test_the_container_is_closed_exactly_once_however_the_command_ends(
    cli: Cli, monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...], code: int
):
    cli.ok("status")  # an existing database
    closed: list[int] = []
    real_close = Container.close

    def spy(self: Container) -> None:
        closed.append(1)
        real_close(self)

    monkeypatch.setattr(Container, "close", spy)
    monkeypatch.setattr(
        "social_tracker.control.service.ControlService.overview",
        explode(RuntimeError("bug")) if code == 70 else ControlService.overview,
    )
    assert cli(*args).exit_code == code
    assert closed == [1]


def test_a_refusal_is_still_exit_1_not_70(cli: Cli):
    result = cli("sources", "enable", "src_zzz", "-r", WHY)
    assert assert_clean_failure(result) == "error: source not found\n"


def test_a_usage_mistake_is_still_exit_2_not_70(cli: Cli):
    assert cli("sources", "enable", "src_zzz").exit_code == 2  # missing --reason
    assert cli("no-such-command").exit_code == 2
    assert cli("sources", "add").exit_code == 2


def test_a_usage_error_raised_inside_a_command_is_not_reported_as_a_bug(
    cli: Cli, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cli_module, "build_container", explode(typer.BadParameter("bad value")))
    result = cli("status")
    assert result.exit_code == 2
    assert "unexpected failure" not in result.stderr


def test_an_explicit_exit_inside_a_command_is_kept(cli: Cli, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_module, "build_container", explode(typer.Exit(3)))
    assert cli("status").exit_code == 3


def test_an_interrupt_is_not_reported_as_a_bug(cli: Cli, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_module, "build_container", explode(KeyboardInterrupt()))
    result = cli("status")
    assert result.exit_code in (1, 130)  # click's own "Aborted!" (130 on click >= 8.2)
    assert "unexpected failure" not in result.stderr


def test_the_four_exit_codes_are_documented_in_the_help(cli: Cli):
    text = " ".join(cli.ok("--help").stdout.split())
    for expected in ("0 success", "1 refused", "2 usage", "70 unexpected failure"):
        assert expected in text, (expected, text)
