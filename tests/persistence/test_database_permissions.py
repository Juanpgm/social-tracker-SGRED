"""A new database is private to its owner: directory 0700 (only if we create it), file 0600.

POSIX modes cannot be observed on Windows (the stat mode there only reflects the read-only
flag), so the real-mode tests are skipped there and the intent is pinned by spying on the calls
that carry the modes, which runs everywhere.
"""

import os
import stat
from pathlib import Path

import pytest

from social_tracker.persistence.db import Database, load_migrations
from social_tracker.shared.errors import StorageError

POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason="POSIX permission bits are not enforced or reported on Windows"
)


def mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@POSIX_ONLY
def test_a_new_database_in_a_new_directory_is_private(tmp_path: Path):
    path = tmp_path / "data" / "app.db"
    with Database(path):
        assert mode_of(path.parent) == 0o700
        assert mode_of(path) == 0o600


@POSIX_ONLY
def test_the_wal_and_shm_files_inherit_the_private_mode(tmp_path: Path):
    path = tmp_path / "app.db"
    with Database(path) as db:
        with db.transaction() as conn:
            conn.execute("CREATE TABLE t (x)")
        sidecars = [Path(str(path) + suffix) for suffix in ("-wal", "-shm")]
        assert all(p.exists() for p in sidecars)
        assert [mode_of(p) for p in sidecars] == [0o600, 0o600]


@POSIX_ONLY
def test_an_existing_directory_and_database_keep_their_modes(tmp_path: Path):
    directory = tmp_path / "shared"
    directory.mkdir()
    directory.chmod(0o755)
    path = directory / "app.db"
    Database(path).close()
    path.chmod(0o644)
    Database(path).close()
    assert mode_of(directory) == 0o755
    assert mode_of(path) == 0o644


@POSIX_ONLY
def test_a_restrictive_umask_cannot_loosen_the_modes_but_a_loose_one_cannot_either(tmp_path: Path):
    old = os.umask(0o000)
    try:
        path = tmp_path / "loose" / "app.db"
        with Database(path):
            assert (mode_of(path.parent), mode_of(path)) == (0o700, 0o600)
    finally:
        os.umask(old)


def test_the_directory_is_created_with_mode_0700(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []
    real_mkdir = Path.mkdir

    def spy(self: Path, *args: object, **kwargs: object) -> None:
        calls.append({"path": self, **kwargs})
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", spy)
    Database(tmp_path / "data" / "app.db").close()
    assert calls == [{"path": tmp_path / "data", "parents": True, "exist_ok": True, "mode": 0o700}]


def test_the_file_is_created_exclusively_with_mode_0600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[Path, int, int]] = []
    real_open = os.open

    def spy(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        calls.append((Path(path), flags, mode))
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", spy)
    target = tmp_path / "app.db"
    Database(target).close()
    ((path, flags, mode),) = calls
    assert path == target
    assert mode == 0o600
    assert flags & os.O_CREAT
    assert flags & os.O_EXCL


def test_an_existing_database_is_never_recreated_or_changed(tmp_path: Path):
    target = tmp_path / "app.db"
    with Database(target) as db, db.transaction() as conn:
        conn.execute("CREATE TABLE keep (x)")
        conn.execute("INSERT INTO keep VALUES (42)")
    with Database(target) as db:
        assert [tuple(r) for r in db.query("SELECT x FROM keep")] == [(42,)]


def test_a_file_that_appears_between_the_check_and_the_create_is_used_as_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # another process won the race to create the file: our exclusive create must just step aside
    target = tmp_path / "app.db"
    real_open = os.open

    def racing(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if Path(path) == target:
            target.write_bytes(b"")
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", racing)
    with Database(target) as db:
        assert db.query("SELECT count(*) FROM schema_migrations")[0][0] == len(load_migrations())


def test_a_directory_in_place_of_the_file_is_still_a_clean_storage_error(tmp_path: Path):
    with pytest.raises(StorageError):
        Database(tmp_path)
