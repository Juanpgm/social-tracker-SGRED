"""Several CLI processes on the same database at once: writers serialise, none corrupts the data."""

import os
import subprocess
import sys
from pathlib import Path

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV
from social_tracker.sources.service import SOURCE_ADDED
from tests.entrypoints.conftest import Cli, table

RUN = [sys.executable, "-c", "from social_tracker.entrypoints.cli import app; app()"]


def spawn(db_path: Path, *args: str) -> subprocess.Popen[str]:
    env = {
        **os.environ,
        DB_PATH_ENV: str(db_path),
        BUSY_TIMEOUT_ENV: "20000",
        "SOCIAL_TRACKER_ACTOR": "proc",
    }
    return subprocess.Popen(
        [*RUN, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def add_args(source_id: str, name: str) -> tuple[str, ...]:
    return (
        *("sources", "add", source_id, "--name", name, "--type", "news_rss"),
        *("--set", "feed_url=https://example.org/rss.xml", "--reason", "race"),
    )


def finish(processes: list[subprocess.Popen[str]]) -> list[tuple[int, str, str]]:
    results = []
    for process in processes:
        out, err = process.communicate(timeout=120)
        results.append((process.returncode, out, err))
    return results


def test_writers_adding_different_sources_at_once_all_succeed(cli: Cli):
    cli.ok("status")  # create and migrate the database once, as a deployed one would be
    ids = [f"src_{n}" for n in range(5)]
    results = finish([spawn(cli.db_path, *add_args(i, f"Name {i}")) for i in ids])
    assert [code for code, _, _ in results] == [0] * 5, results
    assert [row[0] for row in table(cli.ok("sources", "list").stdout)[1:]] == ids
    assert len(cli.details(SOURCE_ADDED)) == 5


def test_writers_racing_for_the_same_id_produce_exactly_one_winner(cli: Cli):
    cli.ok("status")
    results = finish([spawn(cli.db_path, *add_args("src_race", f"Name {n}")) for n in range(4)])
    codes = sorted(code for code, _, _ in results)
    assert codes == [0, 1, 1, 1], results
    for code, out, err in results:
        if code == 1:
            assert out == ""
            assert err == "error: a source with this id already exists\n"
    assert len(cli.details(SOURCE_ADDED)) == 1


def test_racing_first_use_of_a_brand_new_database_still_works(cli: Cli):
    # no warm-up: the processes race to create the file, switch it to WAL and migrate it
    results = finish([spawn(cli.db_path, *add_args(f"new_{n}", f"New {n}")) for n in range(4)])
    assert [code for code, _, _ in results] == [0] * 4, results
    assert len(table(cli.ok("sources", "list").stdout)) == 5
