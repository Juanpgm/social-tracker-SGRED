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
    NEAR_DUPLICATE_THRESHOLD_ENV,
    RECYCLE_DAYS_ENV,
    RETENTION_DAYS_ENV,
)
from social_tracker.collector.domain import MAX_RETRY_AFTER_S, IngestLimits
from social_tracker.collector.feed import MAX_ENTRIES
from social_tracker.collector.http_fetcher import (
    DEFAULT_ALLOWED_PORTS,
    MAX_PENDING_LOOKUPS,
    FetchLimits,
)
from social_tracker.collector.normalize import CONTENT_DROPPED_CHARS
from social_tracker.entrypoints.cli import app
from social_tracker.entrypoints.inputs import ACTOR_ENV
from social_tracker.items.domain import DEFAULT_RETENTION_DAYS, MAX_TOMBSTONES_PER_SOURCE
from social_tracker.sources.yaml_io import MAX_FILE_BYTES, load_sources

README = Path(__file__).resolve().parent.parent / "README.md"
FENCE = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
SUBCOMMAND = re.compile(
    r"(?<![\w-])(?:(sources|analysis|items|classify|topics) ([a-z][a-z-]*)|(status|serve|run))\b"
)
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
        RECYCLE_DAYS_ENV,
        NEAR_DUPLICATE_THRESHOLD_ENV,
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


# ---- the claims about the collector must be the ones the code enforces ----


def collecting(readme: str) -> str:
    """The 'Collecting content' section as one line of single-spaced text."""
    section = readme.split("## Collecting content", 1)[1].split("## Classifying content", 1)[0]
    return " ".join(section.split())


@pytest.mark.parametrize(
    "stale",
    [
        "nothing here fetches",
        "nothing here fetches or analyses",
        "are not part of this slice",
        "workers are not part",
        "does not fetch yet",
        "nothing fetches yet",
        "no worker exists",
        "until the collection worker exists",
    ],
)
def test_the_readme_no_longer_claims_that_nothing_is_fetched(readme: str, stale: str):
    assert stale not in " ".join(readme.lower().split())


def test_the_introduction_says_that_a_collection_worker_exists(readme: str):
    intro = readme.split("## Install", 1)[0]
    assert "collection worker" in intro
    assert "`run`" in intro


def test_the_quickstart_explains_why_run_comes_before_analysis_on(readme: str):
    quickstart = " ".join(readme.split("## Quickstart", 1)[1].split("### Commands", 1)[0].split())
    assert "starting a worker switches global analysis off" in quickstart.lower()
    assert "`run`" in quickstart
    assert "--once --no-reset-analysis" in quickstart
    assert quickstart.index("social-tracker run") < quickstart.index(
        'social-tracker analysis on -r "start the system"', quickstart.index("social-tracker run")
    )


def test_the_fetch_bounds_in_the_readme_are_the_enforced_ones(readme: str):
    section = collecting(readme)
    limits = FetchLimits()
    ports = sorted(DEFAULT_ALLOWED_PORTS)
    listed = ", ".join(map(str, ports[:-1])) + f" and {ports[-1]}"
    assert f"{limits.timeout_s:g} s per network operation and {limits.deadline_s:g} s overall" in (
        section
    )
    assert f"limited to {limits.max_bytes // 1024**2} MiB" in section
    assert f"Redirects (at most {limits.max_redirects})" in section
    assert f"only on the ports {listed}" in section
    assert "100 KiB of headers" in section
    assert f"at most {MAX_PENDING_LOOKUPS} may hang at once" in section
    assert "wall-clock deadline enforced inside" in section


def test_the_limits_on_what_a_feed_can_cost_are_the_enforced_ones(readme: str):
    section, defaults = collecting(readme), IngestLimits()
    assert f"at most {defaults.max_bytes_per_fetch // 1024**2} MiB of items" in section
    assert f"at most {defaults.max_items_per_source:,} items" in section
    assert f"{defaults.max_bytes_per_source // 1024**2} MiB held per source" in section
    assert f"`{defaults.max_items_per_source}`" in readme  # the variable table
    assert f"`{defaults.max_bytes_per_source}`" in readme
    assert f"`{defaults.max_bytes_per_fetch}`" in readme
    assert f"{MAX_ENTRIES} are read" in section


def test_the_retry_retention_and_text_numbers_are_the_enforced_ones(readme: str):
    section = collecting(readme)
    assert f"up to {MAX_RETRY_AFTER_S // 86400} days" in section
    assert f"{MAX_TOMBSTONES_PER_SOURCE:,} per source" in section
    assert f"{CONTENT_DROPPED_CHARS:,} characters or more" in section
    assert f"({DEFAULT_RETENTION_DAYS} by default)" in section
    assert "`SOCIAL_TRACKER_RETENTION_DAYS` | How many days collected items are kept" in readme


def test_the_readme_tells_every_view_of_item_text_to_escape_it(readme: str):
    section = collecting(readme)
    assert "MUST escape all four" in section
    assert "Titles, texts, authors and `raw_payload`" in section


def test_the_stop_and_single_worker_behaviour_is_documented(readme: str):
    section = collecting(readme)
    assert "130" in section
    assert "143" in section
    assert ".worker-lock" in section
    assert "another worker is already running" in section
    assert "sources reset ID" in section


def classifying(readme: str) -> str:
    """The 'Classifying content' section as one line of single-spaced text."""
    section = readme.split("## Classifying content", 1)[1].split("## Audit log", 1)[0]
    return " ".join(section.split())


def test_the_signal_table_lists_every_signal_with_its_label_and_default_weight(readme: str):
    from social_tracker.classification.domain import SIGNALS

    section = readme.split("## Classifying content", 1)[1].split("## Audit log", 1)[0]
    rows = {
        cells[1].strip(" `"): [c.strip() for c in cells]
        for cells in (line.split("|") for line in section.splitlines() if line.startswith("| `"))
    }
    assert set(rows) == set(SIGNALS)
    for name, spec in SIGNALS.items():
        assert rows[name][2] == spec.label.value, name
        assert rows[name][3] == f"{spec.weight:.2f}", name


def test_the_classification_numbers_in_the_readme_are_the_enforced_ones(readme: str):
    from social_tracker.classification.domain import ClassifierSettings
    from social_tracker.classification.rules import PRIOR_STEP
    from social_tracker.classification.service import DEFAULT_BATCH
    from social_tracker.classification.topics import (
        MAX_TERM_CHARS,
        MAX_TERM_WORDS,
        MAX_TERMS_PER_GROUP,
    )
    from social_tracker.classification.yaml_io import MAX_CONFIG_BYTES

    s, text = ClassifierSettings(), classifying(readme)
    for fragment in (
        f"[{s.escalate_low}, {s.escalate_high})",
        f"under {s.inconclusive_below}",
        f"at least {s.conflict_min}",
        f"closer than {s.conflict_margin}",
        f"{s.max_text_chars:,} characters",
        f"{DEFAULT_BATCH} items at a time",
        f"{PRIOR_STEP} per step",
        f"at most {MAX_CONFIG_BYTES // 1024} KiB",
        f"over {MAX_TERMS_PER_GROUP} terms",
        f"over {MAX_TERM_CHARS} characters or {MAX_TERM_WORDS} words",
        f"{s.recycle_days} days exactly",
        f"({s.near_duplicate_threshold})",
        "0.8 is settled, 0.79 is not; 0.4 is queued, 0.39 is not",
    ):
        assert fragment in text, fragment


def test_the_placeholders_are_declared_unverified_and_the_catalog_is_not_a_source(readme: str):
    text = classifying(readme)
    assert "verify before relying" in text
    assert "official-domain.example" in text
    assert "never a data source" in text
    assert "never published or sent" in text
