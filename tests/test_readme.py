"""The README examples must stay true: its YAML loads, its variables and commands exist, every
documented command runs and every documented flag is a real option of the real Click command."""

import itertools
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from social_tracker.bootstrap import (
    ALLOW_PRIVATE_HOSTS_ENV,
    ALLOWED_PORTS_ENV,
    BUSY_TIMEOUT_ENV,
    DB_PATH_ENV,
    DEFAULT_DB_PATH,
    MAX_BYTES_PER_FETCH_ENV,
    MAX_BYTES_PER_SOURCE_ENV,
    MAX_ITEMS_PER_SOURCE_ENV,
    RETENTION_DAYS_ENV,
)
from social_tracker.entrypoints.cli import app
from social_tracker.entrypoints.inputs import ACTOR_ENV
from social_tracker.sources.yaml_io import MAX_FILE_BYTES, load_sources

README = Path(__file__).resolve().parent.parent / "README.md"
FENCE = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
SUBCOMMAND = re.compile(r"(?<![\w-])(?:(sources|analysis|items) ([a-z]+)|(status|serve|run))\b")
FILE_SUFFIXES = {"-wal", "-shm"}  # SQLite sidecar files, written like flags in the prose
CONTINUATION = "\\\n"
EXECUTABLE = "uv run social-tracker"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def blocks(text: str, language: str) -> list[str]:
    return [body for lang, body in FENCE.findall(text) if lang == language]


def test_the_source_file_example_is_a_valid_import_file_with_every_source_type(readme: str):
    (example,) = blocks(readme, "yaml")
    loaded = load_sources(example, default_timestamp=datetime(2026, 1, 1, tzinfo=UTC))
    assert sorted(s.type for s in loaded) == [
        "instagram",
        "live_stream",
        "news_rss",
        "recorded_audio",
    ]


def test_every_environment_variable_and_default_is_documented(readme: str):
    for name in (
        DB_PATH_ENV,
        BUSY_TIMEOUT_ENV,
        ACTOR_ENV,
        RETENTION_DAYS_ENV,
        ALLOW_PRIVATE_HOSTS_ENV,
        ALLOWED_PORTS_ENV,
        MAX_ITEMS_PER_SOURCE_ENV,
        MAX_BYTES_PER_SOURCE_ENV,
        MAX_BYTES_PER_FETCH_ENV,
    ):
        assert f"`{name}`" in readme
    assert DEFAULT_DB_PATH in readme
    assert f"{MAX_FILE_BYTES:,}" in readme


def test_the_exit_codes_and_the_working_directory_warning_are_documented(readme: str):
    for code in ("`0`", "`1`", "`2`", "`70`"):
        assert f"| {code} |" in readme
    assert "depends on the working directory" in readme
    assert "database: ..." in readme


# ---- the documented commands, checked against the real Click command tree ----


def command_tree() -> Any:
    return get_command(app)


def is_group(command: Any) -> bool:
    return hasattr(command, "commands")


def leaf_commands(group: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    found: dict[tuple[str, ...], Any] = {}
    for name, command in group.commands.items():
        if is_group(command):
            found.update(leaf_commands(command, (*prefix, name)))
        else:
            found[(*prefix, name)] = command
    return found


def option_names(command: Any) -> set[str]:
    names = {"--help"}
    for param in command.params:
        names.update(param.opts, param.secondary_opts)
    return names


def resolve(args: list[str]) -> tuple[Any, list[str]]:
    """Follow group names as far as `args` go: the command reached and what is left over."""
    command = command_tree()
    rest = list(args)
    while is_group(command) and rest and rest[0] in command.commands:
        command = command.commands[rest.pop(0)]
    return command, rest


def runs_of(block: str) -> list[list[str]]:
    """The `uv run social-tracker ...` commands of a bash block, continuation lines joined."""
    lines = block.replace(CONTINUATION, " ").splitlines()
    return [shlex.split(line)[3:] for line in lines if line.startswith(EXECUTABLE)]


def all_runs(readme: str) -> list[list[str]]:
    return [args for block in blocks(readme, "bash") for args in runs_of(block)]


def test_every_command_and_flag_in_the_readme_examples_exists(readme: str):
    runs = all_runs(readme)
    assert len(runs) >= 12
    for args in runs:
        command, rest = resolve(args)
        known = option_names(command)
        for token in rest:
            if token.startswith("-"):
                assert token.split("=")[0] in known, (args, token, sorted(known))
            else:
                assert not is_group(command), (args, token)  # an unknown subcommand


def test_the_flag_check_rejects_a_misspelt_flag_and_command():
    command, rest = resolve(["sources", "import", "f.yaml", "--dryrun"])
    assert "--dryrun" in rest
    assert "--dryrun" not in option_names(command)
    command, rest = resolve(["sources", "nope"])
    assert is_group(command)
    assert rest == ["nope"]


BASH_BLOCKS = len(blocks(README.read_text(encoding="utf-8"), "bash"))


@pytest.mark.parametrize("index", range(BASH_BLOCKS))
def test_each_bash_example_runs_to_success_on_a_fresh_database(
    readme: str, tmp_path: Path, index: int
):
    runs = runs_of(blocks(readme, "bash")[index])
    if not runs:
        pytest.skip("no social-tracker command in this block (install or development steps)")
    environ = {
        DB_PATH_ENV: str(tmp_path / "readme.db"),
        ACTOR_ENV: "readme",
        BUSY_TIMEOUT_ENV: "2000",
    }
    runner = CliRunner()
    for args in runs:
        result = runner.invoke(app, args, env=environ)
        assert result.exit_code == 0, (args, result.stdout, result.stderr)


def command_cells(readme: str) -> list[str]:
    """The first column of the `Commands` table."""
    lines = readme.split("| Command | What it does |", 1)[1].splitlines()[2:]  # skip the divider
    rows = list(itertools.takewhile(lambda line: line.startswith("|"), lines))
    assert rows
    return [row.split("|")[1] for row in rows]


def test_the_commands_table_matches_the_real_commands_and_their_flags(readme: str):
    real = leaf_commands(command_tree())
    documented: set[tuple[str, ...]] = set()
    for cell in command_cells(readme):
        text = cell.replace("`", "")
        paths = [tuple(part for part in m.groups() if part) for m in SUBCOMMAND.finditer(text)]
        assert paths, cell
        for path in paths:
            assert path in real, (cell, path)
            documented.add(path)
            for flag in FLAG.findall(text):
                assert flag in option_names(real[path]), (cell, path, flag)
    assert documented == set(real)  # nothing undocumented, nothing invented


def test_every_flag_named_in_code_anywhere_in_the_readme_is_a_real_option(readme: str):
    every: set[str] = set()
    for command in leaf_commands(command_tree()).values():
        every |= option_names(command)
    prose = FENCE.sub("", readme)
    named = {flag for code in re.findall(r"`([^`]*)`", prose) for flag in FLAG.findall(code)}
    named -= FILE_SUFFIXES
    assert {"--reason", "--dry-run", "--output", "--force", "--set", "--source"} <= named
    assert named <= every, sorted(named - every)


def test_the_web_ui_section_documents_the_real_defaults(readme: str):
    from social_tracker.entrypoints.cli import DEFAULT_HOST, DEFAULT_PORT
    from social_tracker.entrypoints.web.formdata import MAX_BODY_BYTES
    from social_tracker.entrypoints.web.forms import ACTOR, DEFAULT_REASON

    section = readme.split("## Web UI", 1)[1].split("\n## ", 1)[0]
    assert f"http://{DEFAULT_HOST}:{DEFAULT_PORT}" in section
    assert f"`{ACTOR}`" in section
    assert f"`{DEFAULT_REASON}`" in section
    assert f"{MAX_BODY_BYTES // 1024} KiB" in section
    assert "--allow-non-loopback" in section
