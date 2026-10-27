"""Several CLI processes on the same database at once: writers serialise, readers never wait,
nothing corrupts the data.

Every process imports the application first and then waits for a start gate (a file), so the
group really starts its work at the same moment instead of one process finishing before the next
one has even been spawned.
"""

import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV
from social_tracker.sources.service import SOURCE_ADDED
from social_tracker.sources.yaml_io import load_sources
from tests.entrypoints.conftest import Cli, scrubbed_env, table

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
GATE_ENV = "ST_TEST_START_GATE"
GATED_RUN = (
    "import os, time\n"
    "from social_tracker.entrypoints.cli import app\n"
    f"gate = os.environ[{GATE_ENV!r}]\n"
    "while not os.path.exists(gate):\n"
    "    time.sleep(0.002)\n"
    "app()\n"
)
RESULT_TIMEOUT_S = 120


class Group:
    """Processes spawned behind one start gate."""

    def __init__(self, db_path: Path, gate: Path, busy_timeout_ms: int = 20_000) -> None:
        self.db_path, self.gate, self.busy_timeout_ms = db_path, gate, busy_timeout_ms
        self.processes: list[subprocess.Popen[str]] = []

    def spawn(self, *args: str) -> None:
        env = scrubbed_env(
            **{
                DB_PATH_ENV: str(self.db_path),
                BUSY_TIMEOUT_ENV: str(self.busy_timeout_ms),
                "SOCIAL_TRACKER_ACTOR": "proc",
                GATE_ENV: str(self.gate),
            }
        )
        self.processes.append(
            subprocess.Popen(
                [sys.executable, "-c", GATED_RUN, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        )

    def go(self) -> list[tuple[int, str, str]]:
        """Open the gate for everyone, then collect the results."""
        self.gate.write_text("go", encoding="utf-8")
        return finish(self.processes, timeout=RESULT_TIMEOUT_S)


def finish(
    processes: list[subprocess.Popen[str]], timeout: float = RESULT_TIMEOUT_S
) -> list[tuple[int, str, str]]:
    """Collect every process. On a timeout kill ALL of them (and reap them) before re-raising, so
    a hung child never outlives the test."""
    deadline = time.monotonic() + timeout
    results = []
    try:
        for process in processes:
            out, err = process.communicate(timeout=max(deadline - time.monotonic(), 0.01))
            results.append((process.returncode, out, err))
    except subprocess.TimeoutExpired:
        for process in processes:
            process.kill()
        for process in processes:
            process.communicate()
        raise
    return results


def add_args(source_id: str, name: str) -> tuple[str, ...]:
    return (
        *("sources", "add", source_id, "--name", name, "--type", "news_rss"),
        *("--set", "feed_url=https://example.org/rss.xml", "--reason", "race"),
    )


@pytest.fixture
def group(cli: Cli, tmp_path: Path) -> Group:
    return Group(cli.db_path, tmp_path / "start.gate")


def sources_file(prefix: str, count: int) -> str:
    """An import file of `count` news sources whose ids AND names are unique to `prefix`."""
    entries = [
        f"  - id: {prefix}_{n:02d}\n    type: news_rss\n    name: {prefix} feed {n:02d}\n"
        f"    feed_url: https://example.org/{prefix}/{n}.xml\n"
        for n in range(count)
    ]
    return "sources:\n" + "".join(entries)


def listed_ids(cli: Cli) -> list[str]:
    return [row[0] for row in table(cli.ok("sources", "list").stdout)[1:]]


# ---- the harness itself ----


def test_a_timeout_kills_every_child_and_is_reraised(tmp_path: Path):
    sleepers = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        finish(sleepers, timeout=0.5)
    assert all(process.poll() is not None for process in sleepers)


def test_the_gate_holds_every_process_until_it_is_opened(group: Group):
    group.spawn("--help")
    group.spawn("--help")
    time.sleep(1.0)  # long enough for the interpreters to start and import the app
    assert [process.poll() for process in group.processes] == [None, None]
    results = group.go()
    assert [code for code, _, _ in results] == [0, 0]


# ---- writers ----


def test_writers_adding_different_sources_at_once_all_succeed(cli: Cli, group: Group):
    cli.ok("status")  # create and migrate the database once, as a deployed one would be
    ids = [f"src_{n}" for n in range(5)]
    for source_id in ids:
        group.spawn(*add_args(source_id, f"Name {source_id}"))
    results = group.go()
    assert [code for code, _, _ in results] == [0] * 5, results
    assert listed_ids(cli) == ids
    assert len(cli.details(SOURCE_ADDED)) == 5


def test_writers_racing_for_the_same_id_produce_exactly_one_winner(cli: Cli, group: Group):
    cli.ok("status")
    for n in range(4):
        group.spawn(*add_args("src_race", f"Name {n}"))
    results = group.go()
    codes = sorted(code for code, _, _ in results)
    assert codes == [0, 1, 1, 1], results
    for code, out, err in results:
        if code == 1:
            assert out == ""
            assert err == "error: a source with this id already exists\n"
    assert len(cli.details(SOURCE_ADDED)) == 1


@pytest.mark.parametrize("round_number", range(3))
def test_racing_first_use_of_a_brand_new_database_still_works(
    cli: Cli, group: Group, round_number: int
):
    # no warm-up: the processes race, at the same instant, to create the file, switch it to WAL
    # and migrate it
    assert not cli.db_path.exists()
    for n in range(6):
        group.spawn(*add_args(f"new_{n}", f"New {n}"))
    results = group.go()
    assert [code for code, _, _ in results] == [0] * 6, results
    assert all(err == "" for _, _, err in results), results
    assert len(table(cli.ok("sources", "list").stdout)) == 7


def test_racing_first_use_by_readers_and_writers_still_works(cli: Cli, group: Group):
    assert not cli.db_path.exists()
    for n in range(3):
        group.spawn(*add_args(f"mix_{n}", f"Mix {n}"))
    group.spawn("sources", "list")
    group.spawn("status")
    results = group.go()
    assert [code for code, _, _ in results] == [0] * 5, results


# ---- readers never wait for a writer ----


@pytest.fixture
def writer_holding_the_lock(cli: Cli):
    cli.ok("status")  # an existing database
    holder = sqlite3.connect(cli.db_path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    yield holder
    holder.execute("ROLLBACK")
    holder.close()


@pytest.mark.parametrize("command", [("sources", "list"), ("status"), ("sources", "export")])
def test_readers_succeed_at_once_while_another_connection_holds_the_write_lock(
    cli: Cli, writer_holding_the_lock: sqlite3.Connection, command: tuple[str, ...] | str
):
    args = (command,) if isinstance(command, str) else command
    started = time.monotonic()
    result = cli.ok(*args, env={BUSY_TIMEOUT_ENV: "20000"})
    assert time.monotonic() - started < 10  # nobody waited for the 20 s busy timeout
    assert result.stderr == ""


def test_readers_in_separate_processes_succeed_while_the_write_lock_is_held(
    cli: Cli, group: Group, writer_holding_the_lock: sqlite3.Connection
):
    group.spawn("sources", "list")
    group.spawn("status")
    started = time.monotonic()
    results = group.go()
    assert time.monotonic() - started < 15
    assert [code for code, _, _ in results] == [0, 0], results


def test_a_writer_behind_a_held_lock_gives_up_after_the_busy_timeout_with_a_clean_error(
    cli: Cli, writer_holding_the_lock: sqlite3.Connection
):
    result = cli(
        *("sources", "add", "src_late", "--name", "Late", *("--type", "news_rss")),
        *("--set", "feed_url=https://example.org/rss.xml", "--reason", "late"),
        env={BUSY_TIMEOUT_ENV: "200"},
    )
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "error: database error: SQLITE_BUSY\n"


# ---- concurrent import and export ----


def test_imports_and_exports_at_once_all_succeed_and_every_export_is_a_valid_snapshot(
    cli: Cli, group: Group, tmp_path: Path
):
    cli.ok("status")
    imports = []
    for n in range(3):
        path = tmp_path / f"in{n}.yaml"
        path.write_text(sources_file(f"imp{n}", 4), encoding="utf-8")
        imports.append(path)
        group.spawn("sources", "import", str(path), "--reason", "bulk")
    for _ in range(3):
        group.spawn("sources", "export")
    results = group.go()
    assert [code for code, _, _ in results] == [0] * 6, results
    assert all(err == "" for _, _, err in results), results
    all_ids = {f"imp{n}_{m:02d}" for n in range(3) for m in range(4)}
    for _, out, _ in results[3:]:
        exported = {s.id for s in load_sources(out, default_timestamp=_NOW)}
        assert exported <= all_ids
        # a snapshot never shows half of one import: each file's four sources arrive together
        for n in range(3):
            present = {i for i in exported if i.startswith(f"imp{n}_")}
            assert len(present) in (0, 4), (n, sorted(exported))
    assert set(listed_ids(cli)) == all_ids


def test_the_same_import_run_by_several_processes_adds_each_source_once(
    cli: Cli, group: Group, tmp_path: Path
):
    cli.ok("status")
    path = tmp_path / "same.yaml"
    path.write_text(sources_file("same", 5), encoding="utf-8")
    for _ in range(4):
        group.spawn("sources", "import", str(path), "--reason", "same file")
    results = group.go()
    assert [code for code, _, _ in results] == [0] * 4, results
    assert sum(out.startswith("imported: 5 added") for _, out, _ in results) == 1
    assert (
        sum(out.startswith("imported: 0 added, 0 updated, 5 unchanged") for _, out, _ in results)
        == 3
    )
    assert len(cli.details(SOURCE_ADDED)) == 5


def test_exports_forced_onto_the_same_file_never_leave_it_partial(
    cli: Cli, group: Group, tmp_path: Path
):
    cli.ok("sources", "import", str(_write(tmp_path, sources_file("big", 40))), "-r", "seed")
    target = tmp_path / "shared-out.yaml"
    for _ in range(5):
        group.spawn("sources", "export", "--output", str(target), "--force")
    results = group.go()
    assert [code for code, _, _ in results] == [0] * 5, results
    assert len(load_sources(target.read_text(encoding="utf-8"), default_timestamp=_NOW)) == 40
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".export-")) == []


def test_exports_without_force_onto_the_same_new_file_have_exactly_one_winner(
    cli: Cli, group: Group, tmp_path: Path
):
    cli.ok("sources", "import", str(_write(tmp_path, sources_file("one", 3))), "-r", "seed")
    target = tmp_path / "new-out.yaml"
    for _ in range(5):
        group.spawn("sources", "export", "--output", str(target))
    results = group.go()
    assert sorted(code for code, _, _ in results) == [0, 1, 1, 1, 1], results
    for code, out, err in results:
        if code == 1:
            assert out == ""
            assert err == "error: output file already exists; use --force to overwrite it\n"
    assert len(load_sources(target.read_text(encoding="utf-8"), default_timestamp=_NOW)) == 3
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".export-")) == []


def _write(directory: Path, content: str) -> Path:
    path = directory / "seed.yaml"
    path.write_text(content, encoding="utf-8")
    return path
