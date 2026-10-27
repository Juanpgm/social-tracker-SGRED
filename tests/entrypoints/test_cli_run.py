"""`social-tracker run`: the worker from the command line, offline (the fetcher is a script)."""

import signal
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

from social_tracker.bootstrap import DB_PATH_ENV, build_container
from social_tracker.bootstrap import build_worker as real_build_worker
from social_tracker.collector.domain import FetchError
from social_tracker.control.domain import CollectionState
from tests.actors import WHO
from tests.collector.harness import FEED, FEED_URL, FakeFetcher
from tests.entrypoints.conftest import WHY, Cli


@pytest.fixture
def fetcher(monkeypatch) -> FakeFetcher:
    """Replaces the HTTP client: no test of `run` can reach the network."""
    script = FakeFetcher(FEED)

    def build(environ=None, **options):
        return real_build_worker(environ, fetcher=script, **options)

    monkeypatch.setattr("social_tracker.entrypoints.cli.build_worker", build)
    return script


def add_news(cli: Cli, source_id: str = "src_news") -> None:
    cli.add(source_id, name=f"News {source_id}")
    cli.ok("sources", "enable", source_id, "--reason", WHY)


def turn_analysis_on_soon(cli: Cli, delay_s: float = 0.15) -> threading.Thread:
    """What the operator does in a second terminal once the worker is up."""

    def operator() -> None:
        time.sleep(delay_s)
        container = build_container({DB_PATH_ENV: str(cli.db_path)})
        with closing(container):
            container.control.set_global_analysis(True, **WHO)

    thread = threading.Thread(target=operator)
    thread.start()
    return thread


# ---- once ----


def test_once_with_nothing_registered_finishes_cleanly(cli: Cli, fetcher):
    result = cli.ok("run", "--once")
    assert fetcher.requests == []
    assert "global analysis is off" in result.stdout


def test_starting_the_worker_switches_global_analysis_off_with_an_audited_event(cli: Cli, fetcher):
    cli.ok("analysis", "on", "--reason", WHY)
    cli.ok("run", "--once")
    assert cli.ok("status").stdout.count("global analysis: off") == 1
    last = cli.details("global_analysis_changed")[-1]
    assert (last["actor"], last["reason"], last["from"], last["to"]) == (
        "system:worker",
        "worker startup",
        "active",
        "paused",
    )


def test_a_once_pass_right_after_the_reset_explains_why_nothing_was_collected(cli: Cli, fetcher):
    add_news(cli)
    cli.ok("analysis", "on", "--reason", WHY)
    result = cli.ok("run", "--once")
    assert fetcher.requests == []
    assert "nothing was collected" in result.stdout
    assert "analysis on" in result.stdout


def test_other_commands_never_reset_global_analysis(cli: Cli):
    cli.ok("analysis", "on", "--reason", WHY)
    for command in (["status"], ["sources", "list"], ["items", "list"], ["items", "search", "x"]):
        cli.ok(*command)
    assert "global analysis: on" in cli.ok("status").stdout
    assert [d["actor"] for d in cli.details("global_analysis_changed")] == ["tester"]


@pytest.mark.parametrize("tick", ["0", "-1", "abc", "nan", "3601"])
def test_an_invalid_tick_is_a_usage_error(cli: Cli, fetcher, tick):
    assert cli("run", "--once", "--interval-tick", tick).exit_code == 2


def test_a_valid_tick_is_accepted(cli: Cli, fetcher):
    assert cli("run", "--once", "--interval-tick", "0.01").exit_code == 0
    assert cli("run", "--once", "--interval-tick", "3600").exit_code == 0


def test_a_broken_database_is_a_clean_refusal(cli: Cli, fetcher, tmp_path: Path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    result = cli("run", "--once", env={DB_PATH_ENV: str(blocker / "x.db")})
    assert result.exit_code == 1
    assert result.stderr.startswith("error: ")
    assert "Traceback" not in result.stderr


# ---- running and stopping ----


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_a_stop_signal_during_a_collection_lets_the_unit_finish_and_exits_cleanly(
    cli: Cli, fetcher, signum
):
    add_news(cli)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

    def stop_during_fetch(request):
        signal.raise_signal(signum)
        return FEED

    fetcher.then(stop_during_fetch)
    operator = turn_analysis_on_soon(cli)
    started = time.monotonic()
    result = cli("run", "--interval-tick", "0.02")
    operator.join(timeout=10)
    assert result.exit_code == 0, result.stderr
    assert time.monotonic() - started < 25
    assert "worker stopped" in result.stdout
    assert len(fetcher.requests) == 1
    with closing(build_container({DB_PATH_ENV: str(cli.db_path)})) as services:
        assert len(services.items.list_items()) == 2
        overview = services.control.overview()
    assert overview.sources[0].state is CollectionState.IDLE
    assert overview.sources[0].last_fetch_at is not None
    assert {s: signal.getsignal(s) for s in before} == before  # handlers restored


def test_the_worker_collects_what_is_due_once_the_operator_turns_analysis_on(cli: Cli, fetcher):
    add_news(cli)

    def finish_after_first_pass(request):
        threading.Timer(0.05, signal.raise_signal, args=(signal.SIGTERM,)).start()
        return FEED

    fetcher.then(finish_after_first_pass)
    operator = turn_analysis_on_soon(cli)
    result = cli("run", "--interval-tick", "0.02")
    operator.join(timeout=10)
    assert result.exit_code == 0, result.stderr
    with closing(build_container({DB_PATH_ENV: str(cli.db_path)})) as services:
        assert len(services.items.list_items()) == 2


def test_logs_go_to_stderr_and_never_contain_urls_or_item_text(cli: Cli, fetcher):
    add_news(cli)

    def stop_soon(request):
        threading.Timer(0.05, signal.raise_signal, args=(signal.SIGTERM,)).start()
        return FetchError.http(503)

    fetcher.then(stop_soon)
    operator = turn_analysis_on_soon(cli)
    result = cli("run", "--interval-tick", "0.02")
    operator.join(timeout=10)
    assert result.exit_code == 0, result.stderr
    combined = result.stdout + result.stderr
    assert FEED_URL not in combined
    assert "news.example.org" not in combined
    assert "http 503" in result.stderr


def test_the_run_command_is_listed_in_the_help(cli: Cli):
    out = cli.ok("--help").stdout
    assert "run" in out
    assert "items" in out
