"""`social-tracker run`: the worker from the command line, offline (the fetcher is a script)."""

import signal
import time
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest

from social_tracker.bootstrap import DB_PATH_ENV, build_container
from social_tracker.bootstrap import build_worker as real_build_worker
from social_tracker.collector.domain import FetchError
from social_tracker.control.domain import CollectionState
from tests.actors import WHO
from tests.collector.harness import FEED, FEED_URL, FakeFetcher, dated_now
from tests.entrypoints.conftest import WHY, Cli


class Operator:
    """What a person does in a second terminal while the worker runs, made deterministic: the
    queued actions run when the worker first goes idle (its sleeper), which is after its startup
    reset and a first pass, never before. A real sleep would race the reset."""

    def __init__(self, cli: Cli) -> None:
        self._cli = cli
        self._actions: list[Callable[[], object]] = []

    def turn_analysis_on(self) -> None:
        def act() -> None:
            container = build_container({DB_PATH_ENV: str(self._cli.db_path)})
            with closing(container):
                container.control.set_global_analysis(True, **WHO)

        self._actions.append(act)

    def sleeper(self, seconds: float) -> None:
        if self._actions:
            self._actions.pop(0)()
        else:
            time.sleep(0.001)

    @property
    def pending(self) -> int:
        return len(self._actions)


@pytest.fixture
def operator(cli: Cli) -> Operator:
    return Operator(cli)


@pytest.fixture
def fetcher(monkeypatch, operator: Operator) -> FakeFetcher:
    """Replaces the HTTP client: no test of `run` can reach the network."""
    script = FakeFetcher(dated_now(FEED))  # the real clock is in play: dates must be recent

    def build(environ=None, **options):
        return real_build_worker(environ, fetcher=script, sleeper=operator.sleeper, **options)

    monkeypatch.setattr("social_tracker.entrypoints.cli.build_worker", build)
    return script


def add_news(cli: Cli, source_id: str = "src_news") -> None:
    cli.add(source_id, name=f"News {source_id}")
    cli.ok("sources", "enable", source_id, "--reason", WHY)


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


def test_a_plain_once_pass_cannot_collect_and_says_how_to_run_a_diagnostic_pass(cli: Cli, fetcher):
    add_news(cli)
    cli.ok("analysis", "on", "--reason", WHY)
    result = cli.ok("run", "--once")
    assert fetcher.requests == []
    assert "nothing was collected" in result.stdout
    assert "--once --no-reset-analysis" in result.stdout
    assert "analysis on" in result.stdout


# ---- once, without the startup reset ----


def test_once_without_the_reset_collects_what_is_due_while_analysis_is_on(cli: Cli, fetcher):
    add_news(cli)
    cli.ok("analysis", "on", "--reason", WHY)
    result = cli.ok("run", "--once", "--no-reset-analysis")
    assert len(fetcher.requests) == 1
    assert "1 collected (2 new item(s))" in result.stdout
    with closing(build_container({DB_PATH_ENV: str(cli.db_path)})) as services:
        assert len(services.items.list_items()) == 2


def test_a_pass_classifies_what_it_collected_and_says_so(cli: Cli, fetcher):
    add_news(cli)
    cli.ok("analysis", "on", "--reason", WHY)
    cli.ok("analysis", "on", "--source", "src_news", "--reason", WHY)
    result = cli.ok("run", "--once", "--no-reset-analysis")
    assert "classification: classified 1 item(s), 1 out of scope, 0 skipped, 0 error(s)" in (
        result.stdout
    )
    assert "No classifications yet." not in cli.ok("classify", "stats").stdout


def test_a_pass_also_says_what_the_alert_step_did(cli: Cli, fetcher):
    add_news(cli)
    cli.ok("analysis", "on", "--reason", WHY)
    cli.ok("analysis", "on", "--source", "src_news", "--reason", WHY)
    result = cli.ok("run", "--once", "--no-reset-analysis")
    assert "alerts: 0 created, 0 merged, 0 updated, 0 withdrawn, 0 brought back" in result.stdout


def test_once_without_the_reset_leaves_analysis_on_and_writes_no_startup_event(cli: Cli, fetcher):
    add_news(cli)
    cli.ok("analysis", "on", "--reason", WHY)
    cli.ok("run", "--once", "--no-reset-analysis")
    assert "global analysis: on" in cli.ok("status").stdout
    assert [d["actor"] for d in cli.details("global_analysis_changed")] == ["tester"]
    assert "worker startup" not in [d["reason"] for d in cli.details("global_analysis_changed")]


def test_once_without_the_reset_and_with_analysis_off_collects_nothing_and_says_so(
    cli: Cli, fetcher
):
    add_news(cli)
    result = cli.ok("run", "--once", "--no-reset-analysis")
    assert fetcher.requests == []
    assert "global analysis is off: nothing was collected" in result.stdout
    assert "analysis on" in result.stdout
    assert "--no-reset-analysis" not in result.stdout  # it was already given: no pointless hint
    assert cli.details("global_analysis_changed") == []


def test_once_without_the_reset_still_recovers_a_collection_a_crash_left_running(cli: Cli, fetcher):
    add_news(cli)
    cli.ok("analysis", "on", "--reason", WHY)
    with closing(build_container({DB_PATH_ENV: str(cli.db_path)})) as services:
        services.control.start_collection("src_news", actor="system:worker", reason="crashed")
    cli.ok("run", "--once", "--no-reset-analysis")
    assert "backoff 1" in cli.ok("status").stdout


@pytest.mark.parametrize("extra", [[], ["--interval-tick", "1"]])
def test_the_no_reset_flag_without_once_is_a_usage_error_and_starts_nothing(
    cli: Cli, fetcher, extra
):
    result = cli("run", "--no-reset-analysis", *extra)
    assert result.exit_code == 2
    assert "--once" in result.stderr
    assert not cli.db_path.exists()  # nothing was opened, nothing was reset
    assert fetcher.requests == []


def test_the_no_reset_flag_is_listed_in_the_help_for_diagnostics(cli: Cli):
    out = cli.ok("run", "--help").stdout
    assert "--no-reset-analysis" in out


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
    cli: Cli, fetcher, operator: Operator, signum
):
    add_news(cli)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}

    def stop_during_fetch(request):
        signal.raise_signal(signum)
        return dated_now(FEED)

    fetcher.then(stop_during_fetch)
    operator.turn_analysis_on()
    started = time.monotonic()
    result = cli("run", "--interval-tick", "0.02")
    assert operator.pending == 0
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


@pytest.mark.parametrize(
    ("signum", "code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)], ids=["SIGINT", "SIGTERM"]
)
def test_a_second_stop_signal_exits_at_once_and_the_next_start_recovers_the_abandoned_unit(
    cli: Cli, fetcher, operator: Operator, signum, code
):
    add_news(cli)

    def impatient_operator(request):
        signal.raise_signal(signum)
        signal.raise_signal(signum)
        raise AssertionError("the fetch must have been abandoned by the second signal")

    fetcher.then(impatient_operator)
    operator.turn_analysis_on()
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    result = cli("run", "--interval-tick", "0.02")
    assert result.exit_code == code
    assert "worker stopped" not in result.stdout
    assert {s: signal.getsignal(s) for s in before} == before
    # the database was closed properly and the abandoned unit is still marked as collecting
    assert "collecting 1" in cli.ok("status").stdout
    with closing(build_container({DB_PATH_ENV: str(cli.db_path)})) as services:
        assert services.items.list_items() == []
    cli.ok("run", "--once")
    assert "backoff 1" in cli.ok("status").stdout


def test_the_worker_collects_what_is_due_once_the_operator_turns_analysis_on(
    cli: Cli, fetcher, operator: Operator
):
    add_news(cli)

    def finish_after_first_pass(request):
        signal.raise_signal(signal.SIGTERM)
        return dated_now(FEED)

    fetcher.then(finish_after_first_pass)
    operator.turn_analysis_on()
    result = cli("run", "--interval-tick", "0.02")
    assert operator.pending == 0
    assert result.exit_code == 0, result.stderr
    with closing(build_container({DB_PATH_ENV: str(cli.db_path)})) as services:
        assert len(services.items.list_items()) == 2


def test_logs_go_to_stderr_and_never_contain_urls_or_item_text(
    cli: Cli, fetcher, operator: Operator
):
    add_news(cli)

    def stop_soon(request):
        signal.raise_signal(signal.SIGTERM)
        return FetchError.http(503)

    fetcher.then(stop_soon)
    operator.turn_analysis_on()
    result = cli("run", "--interval-tick", "0.02")
    assert operator.pending == 0
    assert result.exit_code == 0, result.stderr
    combined = result.stdout + result.stderr
    assert FEED_URL not in combined
    assert "news.example.org" not in combined
    assert "http 503" in result.stderr


def test_the_run_command_is_listed_in_the_help(cli: Cli):
    out = cli.ok("--help").stdout
    assert "run" in out
    assert "items" in out
