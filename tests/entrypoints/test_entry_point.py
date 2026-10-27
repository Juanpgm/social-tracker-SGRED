"""The `social-tracker` console script is declared and really starts the CLI."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import typer

from social_tracker.bootstrap import DB_PATH_ENV
from social_tracker.entrypoints import cli


def test_the_console_script_points_at_the_typer_app(repo_root: Path):
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {"social-tracker": "social_tracker.entrypoints.cli:app"}
    assert isinstance(cli.app, typer.Typer)


def test_the_app_runs_as_a_real_process_with_separate_streams_and_exit_codes(tmp_path: Path):
    env = {"SOCIAL_TRACKER_ACTOR": "proc", DB_PATH_ENV: str(tmp_path / "proc.db")}
    run = [sys.executable, "-c", "from social_tracker.entrypoints.cli import app; app()"]

    def invoke(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*run, *args], capture_output=True, text=True, env=_environ(env), check=False
        )

    listing = invoke("sources", "list")
    assert (listing.returncode, listing.stdout, listing.stderr) == (
        0,
        "No sources registered.\n",
        "",
    )
    refused = invoke("sources", "enable", "src_zzz", "--reason", "why")
    assert (refused.returncode, refused.stdout) == (1, "")
    assert refused.stderr == "error: source not found\n"
    usage = invoke("sources", "enable", "src_zzz")
    assert usage.returncode == 2


def _environ(extra: dict[str, str]) -> dict[str, str]:
    return {**os.environ, **extra}
