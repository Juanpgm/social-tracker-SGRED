"""Fixtures to drive the real CLI against a throwaway SQLite database."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
from typer.testing import CliRunner

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV
from social_tracker.control.domain import ControlEvent
from social_tracker.entrypoints.cli import app
from social_tracker.entrypoints.inputs import ACTOR_ENV
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database

NEWS = ("--type", "news_rss", "--set", "feed_url=https://example.org/rss.xml")
WHY = "scheduled maintenance"


@dataclass
class Cli:
    db_path: Path
    runner: CliRunner

    def __call__(self, *args: str, env: Mapping[str, str | None] | None = None) -> Result:
        environ: dict[str, str | None] = {
            DB_PATH_ENV: str(self.db_path),
            ACTOR_ENV: "tester",
            BUSY_TIMEOUT_ENV: "2000",
        }
        environ.update(env or {})
        result = self.runner.invoke(app, list(args), env=environ)
        # An escaped exception would be a traceback on the operator's terminal: fail loudly.
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception
        return result

    def ok(self, *args: str, env: Mapping[str, str | None] | None = None) -> Result:
        result = self(*args, env=env)
        assert result.exit_code == 0, result.stderr
        return result

    def add(self, source_id: str = "src_news", name: str = "City News") -> Result:
        return self.ok("sources", "add", source_id, "--name", name, *NEWS, "--reason", WHY)

    def events(self) -> list[ControlEvent]:
        with Database(self.db_path) as db:
            return SqliteControlRepository(db).list_control_events()

    def details(self, kind: str) -> list[dict[str, Any]]:
        return [json.loads(e.detail) for e in self.events() if e.kind == kind]


@pytest.fixture
def cli(tmp_path: Path) -> Cli:
    return Cli(tmp_path / "cli.db", CliRunner())


def table(output: str) -> list[list[str]]:
    """The listing as rows of cells: the first five columns split on whitespace, the name kept
    whole as the sixth."""
    return [line.split(None, 5) for line in output.splitlines() if line.strip()]
