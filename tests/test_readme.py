"""The README examples must stay true: its YAML loads, its variables and commands exist."""

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.main import get_command

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV, DEFAULT_DB_PATH
from social_tracker.entrypoints.cli import app
from social_tracker.entrypoints.inputs import ACTOR_ENV
from social_tracker.sources.yaml_io import MAX_FILE_BYTES, load_sources

README = Path(__file__).resolve().parent.parent / "README.md"
FENCE = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)


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
    for name in (DB_PATH_ENV, BUSY_TIMEOUT_ENV, ACTOR_ENV):
        assert f"`{name}`" in readme
    assert DEFAULT_DB_PATH in readme
    assert f"{MAX_FILE_BYTES:,}" in readme


def test_every_command_the_readme_runs_exists(readme: str):
    root = get_command(app)
    runs = re.findall(
        r"^uv run social-tracker (\w+)(?: (\w+))?", "\n".join(blocks(readme, "bash")), re.M
    )
    assert runs
    for group, sub in runs:
        command = root.commands[group]
        if sub and hasattr(command, "commands"):
            assert sub in command.commands, (group, sub)
