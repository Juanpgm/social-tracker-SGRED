"""One worker per database: an exclusive advisory lock file next to the database file.

The lock belongs to the OPERATING SYSTEM, not to a file's existence: it is gone when the holder
exits or crashes, so a stale lock file never blocks anyone (a hard kill is tested with a real
second process).
"""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from social_tracker.persistence.worker_lock import LOCK_SUFFIX, WorkerLock, WorkerLockHeldError
from social_tracker.shared.errors import DomainError

HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from social_tracker.persistence.worker_lock import WorkerLock

    lock = WorkerLock(Path(sys.argv[1]))
    lock.acquire()
    print("HELD", flush=True)
    time.sleep(float(sys.argv[2]))
    """
)


def start_holder(db: Path, seconds: float = 60.0) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(db), str(seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    assert line.strip() == "HELD", process.stderr.read() if process.stderr else line
    return process


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "w.db"


def test_the_lock_file_sits_next_to_the_database(db):
    lock = WorkerLock(db)
    assert lock.path == db.with_name(db.name + LOCK_SUFFIX)
    lock.acquire()
    try:
        assert lock.path.exists()
        assert lock.path.parent == db.parent
    finally:
        lock.release()


def test_a_second_holder_in_the_same_process_is_refused_with_a_clear_message(db):
    first = WorkerLock(db)
    first.acquire()
    try:
        with pytest.raises(WorkerLockHeldError) as caught:
            WorkerLock(db).acquire()
    finally:
        first.release()
    assert isinstance(caught.value, DomainError)
    assert "another worker" in caught.value.message
    assert str(db) not in caught.value.message  # error messages never carry paths


def test_the_lock_is_free_again_after_release(db):
    first = WorkerLock(db)
    first.acquire()
    first.release()
    second = WorkerLock(db)
    second.acquire()
    second.release()


def test_release_is_idempotent_and_safe_without_acquire(db):
    lock = WorkerLock(db)
    lock.release()
    lock.acquire()
    lock.release()
    lock.release()


def test_acquiring_twice_with_the_same_object_is_harmless(db):
    lock = WorkerLock(db)
    lock.acquire()
    try:
        lock.acquire()  # already ours
        with pytest.raises(WorkerLockHeldError):
            WorkerLock(db).acquire()
    finally:
        lock.release()
    WorkerLock(db).acquire()


def test_the_context_manager_releases_on_error(db):
    with pytest.raises(RuntimeError), WorkerLock(db):
        raise RuntimeError("boom")
    again = WorkerLock(db)
    again.acquire()
    again.release()


def test_different_databases_have_independent_locks(tmp_path):
    one, two = WorkerLock(tmp_path / "a.db"), WorkerLock(tmp_path / "b.db")
    one.acquire()
    try:
        two.acquire()
        two.release()
    finally:
        one.release()


def test_a_leftover_lock_file_does_not_block(db):
    lock = WorkerLock(db)
    lock.path.write_text("pid 1\n")  # what a crashed holder left behind
    lock.acquire()
    lock.release()


def test_a_missing_directory_is_a_clean_error_not_a_traceback(tmp_path):
    with pytest.raises(DomainError):
        WorkerLock(tmp_path / "nope" / "w.db").acquire()


# ---- two processes ----


def test_a_second_process_cannot_take_a_lock_another_process_holds(db):
    holder = start_holder(db)
    try:
        with pytest.raises(WorkerLockHeldError):
            WorkerLock(db).acquire()
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_the_lock_is_released_when_the_holder_is_killed_without_any_cleanup(db):
    holder = start_holder(db)
    holder.kill()  # a crash: no finally block, no atexit, nothing runs
    holder.wait(timeout=10)
    deadline = time.monotonic() + 10
    while True:
        lock = WorkerLock(db)
        try:
            lock.acquire()
            break
        except WorkerLockHeldError:
            assert time.monotonic() < deadline, "the dead holder's lock was never released"
            time.sleep(0.05)
    lock.release()


def test_the_lock_is_released_when_the_holder_exits_normally(db):
    holder = start_holder(db, seconds=0.1)
    holder.wait(timeout=10)
    lock = WorkerLock(db)
    lock.acquire()
    lock.release()
