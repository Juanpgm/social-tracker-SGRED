"""`social-tracker run` takes the single-worker lock: a second worker is refused at once, and a
worker that is killed leaves nothing behind that blocks the next one (real processes)."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from social_tracker.bootstrap import DB_PATH_ENV
from social_tracker.persistence.worker_lock import WorkerLock
from tests.entrypoints.conftest import Cli, scrubbed_env

ENTRY = "from social_tracker.entrypoints.cli import app; app()"


def run_cli(db: Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", ENTRY, *args],
        env=scrubbed_env(**{DB_PATH_ENV: str(db)}),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def start_daemon(db: Path) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", ENTRY, "run", "--interval-tick", "0.05"],
        env=scrubbed_env(**{DB_PATH_ENV: str(db)}),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline()  # printed once the worker is built, i.e. after the lock
    assert "worker running" in line, process.stderr.read() if process.stderr else line
    return process


@pytest.mark.parametrize(
    "args", [["run", "--once"], ["run"], ["run", "--once", "--no-reset-analysis"]]
)
def test_a_worker_is_refused_while_the_lock_is_held(cli: Cli, args):
    cli.ok("analysis", "on", "--reason", "x")
    holder = WorkerLock(cli.db_path)
    holder.acquire()
    try:
        result = cli(*args)
    finally:
        holder.release()
    assert result.exit_code == 1
    assert result.stderr.startswith("error: another worker is already running")
    assert "Traceback" not in result.stderr
    assert "global analysis: on" in cli.ok("status").stdout  # it did not even reset the switch
    assert [d["actor"] for d in cli.details("global_analysis_changed")] == ["tester"]


def test_the_lock_does_not_get_in_the_way_of_other_commands(cli: Cli):
    holder = WorkerLock(cli.db_path)
    cli.ok("status")  # creates the database
    holder.acquire()
    try:
        cli.ok("status")
        cli.add()
        cli.ok("sources", "list")
        cli.ok("items", "list")
    finally:
        holder.release()


def test_a_second_run_is_refused_while_a_real_daemon_runs_and_works_after_it_is_killed(
    tmp_path: Path,
):
    db = tmp_path / "w.db"
    daemon = start_daemon(db)
    try:
        for args in (["run", "--once"], ["run", "--once", "--no-reset-analysis"]):
            refused = run_cli(db, *args)
            assert refused.returncode == 1, refused.stderr
            assert "another worker is already running" in refused.stderr
        daemon.kill()  # a crash: nothing gets to clean up
        daemon.wait(timeout=10)
        deadline = time.monotonic() + 15
        while True:
            outcome = run_cli(db, "run", "--once")
            if outcome.returncode == 0:
                break
            assert time.monotonic() < deadline, outcome.stderr
            time.sleep(0.1)
        assert "global analysis is off" in outcome.stdout
    finally:
        if daemon.poll() is None:
            daemon.kill()
        daemon.wait(timeout=10)
        for pipe in (daemon.stdout, daemon.stderr):
            if pipe is not None:
                pipe.close()


def test_a_leftover_lock_file_from_a_crashed_worker_blocks_nobody(tmp_path: Path):
    db = tmp_path / "w.db"
    assert run_cli(db, "run", "--once").returncode == 0
    leftover = db.with_name(db.name + ".worker-lock")
    assert leftover.exists()  # the file stays, the lock does not
    assert run_cli(db, "run", "--once").returncode == 0
