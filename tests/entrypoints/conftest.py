"""Fixtures to drive the real CLI against a throwaway SQLite database."""

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
from typer.testing import CliRunner

from social_tracker import bootstrap
from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV, TIMEZONE_ENV
from social_tracker.control.domain import ControlEvent
from social_tracker.entrypoints import cli as cli_module
from social_tracker.entrypoints.cli import app
from social_tracker.entrypoints.inputs import ACTOR_ENV
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from tests.items.factories import NOW, make_item
from tests.llm.harness import ScriptedJudge


def scrubbed_env(**extra: str) -> dict[str, str]:
    """The ambient environment without any `SOCIAL_TRACKER_*` variable, plus `extra`: a developer's
    own settings (a real database path, a short busy timeout) must never reach a test process."""
    ambient = {k: v for k, v in os.environ.items() if not k.startswith("SOCIAL_TRACKER_")}
    return {**ambient, **extra}


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
            TIMEZONE_ENV: "UTC",
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


# ---- the language-model stage: a scripted judge and a database with items waiting for it ----

PREDICTION = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
SECOND = "Dicen que mañana habrá un sismo muy fuerte en Cali, pásalo a todos"
REPORT = "El SGC reportó un sismo de magnitud 4.2 en Cali"


@pytest.fixture
def judge(monkeypatch: pytest.MonkeyPatch) -> ScriptedJudge:
    scripted = ScriptedJudge()
    monkeypatch.setattr(
        cli_module, "build_llm", lambda stop=None: bootstrap.build_llm(judge=scripted, stop=stop)
    )
    return scripted


def seed(cli: Cli, *items) -> None:
    with Database(cli.db_path) as db:
        SqliteItemRepository(db).add_new(items)


@pytest.fixture
def queued(cli: Cli) -> Cli:
    """A source with everything on and items classified, some of them waiting for the LLM."""
    cli.add()
    cli.ok("sources", "enable", "src_news", "--reason", WHY)
    cli.ok("analysis", "on", "--source", "src_news", "--reason", WHY)
    cli.ok("analysis", "on", "--reason", WHY)
    seed(
        cli,
        make_item("a", title="Alerta", text=PREDICTION, published_at=NOW),
        make_item("b", title="Alerta 2", text=SECOND, published_at=NOW + timedelta(hours=1)),
        make_item("c", title="Sismo", text=REPORT, published_at=NOW + timedelta(hours=2)),
    )
    cli.ok("classify", "run")
    return cli


def waiting(cli: Cli) -> int:
    match = re.search(r"queued for the LLM: (\d+)", cli.ok("classify", "stats").stdout)
    assert match is not None
    return int(match.group(1))
