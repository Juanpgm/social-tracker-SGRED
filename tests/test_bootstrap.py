"""The composition root: environment in, wired services out."""

import sqlite3
import time
from pathlib import Path

import pytest

from social_tracker.bootstrap import (
    BUSY_TIMEOUT_ENV,
    DB_PATH_ENV,
    DEFAULT_DB_PATH,
    build_container,
)
from social_tracker.shared.errors import StorageError, ValidationError
from tests.actors import WHO
from tests.sources.factories import payload


def test_the_default_database_lives_under_data_and_its_directory_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    container = build_container({})
    try:
        assert DEFAULT_DB_PATH == "data/social_tracker.db"
        assert (tmp_path / "data" / "social_tracker.db").is_file()
    finally:
        container.close()


def test_the_environment_variable_overrides_the_path_and_missing_parents_are_created(
    tmp_path: Path,
):
    target = tmp_path / "a" / "b" / "my.db"
    container = build_container({DB_PATH_ENV: str(target)})
    container.close()
    assert target.is_file()
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_path_variable_means_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blank: str
):
    monkeypatch.chdir(tmp_path)
    build_container({DB_PATH_ENV: blank}).close()
    assert (tmp_path / "data" / "social_tracker.db").is_file()


def test_the_services_are_wired_to_one_shared_database_and_survive_a_restart(tmp_path: Path):
    env = {DB_PATH_ENV: str(tmp_path / "wired.db")}
    first = build_container(env)
    try:
        first.sources.add(payload(), **WHO)
        first.sources.enable("src_one", **WHO)
        first.control.set_global_analysis(True, **WHO)
        # the registry and the control service share one transaction scope: both writes are
        # visible to the control service's read model
        assert first.control.overview().sources[0].state.value == "idle"
    finally:
        first.close()
    second = build_container(env)
    try:
        overview = second.control.overview()
        assert overview.global_analysis is True
        assert [o.source.id for o in overview.sources] == ["src_one"]
    finally:
        second.close()


def test_timestamps_come_from_the_system_clock(tmp_path: Path):
    container = build_container({DB_PATH_ENV: str(tmp_path / "clock.db")})
    try:
        before = time.time()
        source = container.sources.add(payload(), **WHO)
        assert before - 1 <= source.created_at.timestamp() <= time.time() + 1
    finally:
        container.close()


def test_close_releases_the_file_and_is_idempotent(tmp_path: Path):
    path = tmp_path / "close.db"
    container = build_container({DB_PATH_ENV: str(path)})
    container.close()
    container.close()
    path.unlink()  # a lingering handle would make this fail on Windows
    for suffix in ("-wal", "-shm"):
        assert not Path(str(path) + suffix).exists()
    with pytest.raises(StorageError):
        container.sources.list()


def test_a_parent_that_is_a_file_is_a_clean_storage_error(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.raises(StorageError) as caught:
        build_container({DB_PATH_ENV: str(blocker / "x.db")})
    assert "blocker" not in caught.value.message  # fixed message, the path is not echoed


def test_a_directory_in_place_of_the_database_file_is_a_clean_storage_error(tmp_path: Path):
    with pytest.raises(StorageError):
        build_container({DB_PATH_ENV: str(tmp_path)})


def test_a_file_that_is_not_a_database_is_refused_and_left_untouched(tmp_path: Path):
    path = tmp_path / "junk.db"
    path.write_bytes(b"this is not a sqlite database at all, just text" * 10)
    before = path.read_bytes()
    with pytest.raises(StorageError):
        build_container({DB_PATH_ENV: str(path)})
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "bad", ["abc", "0", "-5", "1.5", "1e3", "600001", "99999999999", "ghp_leak"]
)
def test_an_invalid_busy_timeout_is_a_validation_error(tmp_path: Path, bad: str):
    with pytest.raises(ValidationError) as caught:
        build_container({DB_PATH_ENV: str(tmp_path / "t.db"), BUSY_TIMEOUT_ENV: bad})
    assert BUSY_TIMEOUT_ENV in caught.value.message
    assert "ghp_leak" not in caught.value.message  # the offending value is never echoed
    assert not (tmp_path / "t.db").exists()  # validated before the database is touched


def test_a_busy_database_fails_after_the_configured_timeout_not_the_default(tmp_path: Path):
    path = tmp_path / "busy.db"
    build_container({DB_PATH_ENV: str(path)}).close()
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    container = build_container({DB_PATH_ENV: str(path), BUSY_TIMEOUT_ENV: "100"})
    try:
        started = time.monotonic()
        with pytest.raises(StorageError) as caught:
            container.sources.add(payload(), **WHO)
        assert time.monotonic() - started < 3  # the default timeout is 5 s
        assert "SQLITE_BUSY" in caught.value.message
    finally:
        container.close()
        holder.execute("ROLLBACK")
        holder.close()


@pytest.mark.parametrize("edge", ["1", "600000", " 250 "])
def test_the_busy_timeout_accepts_its_range_boundaries(tmp_path: Path, edge: str):
    build_container({DB_PATH_ENV: str(tmp_path / "e.db"), BUSY_TIMEOUT_ENV: edge}).close()


def test_a_blank_busy_timeout_means_the_default(tmp_path: Path):
    build_container({DB_PATH_ENV: str(tmp_path / "b.db"), BUSY_TIMEOUT_ENV: ""}).close()
