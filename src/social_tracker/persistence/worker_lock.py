"""One worker per database: an exclusive advisory lock on a file next to the database file.

`social-tracker run` (the daemon and `--once` alike) takes the lock before it touches anything and
refuses to start, with a clear message, while another process holds it. Two workers on one
database would each treat the other's collection in flight as interrupted and fail it, and could
fetch the same source twice.

The lock is held by the operating system on an open file, not by the file's existence
(`msvcrt.locking` on Windows, `fcntl.flock` elsewhere), so it disappears when the holder exits OR
crashes or is killed: a leftover `.worker-lock` file never blocks a new worker. Because the
previous owner is certainly gone once the lock is ours, the worker may treat every collection it
finds still running as interrupted (see `CollectorService.recover_interrupted`).

The lock is advisory: it stops other workers, not other programs. Network file systems may not
honour it; keep the database on a local disk.
"""

import contextlib
import os
import sys
from pathlib import Path
from types import TracebackType

from social_tracker.shared.errors import DomainError, StorageError

LOCK_SUFFIX = ".worker-lock"
_PRIVATE_FILE_MODE = 0o600


class WorkerLockHeldError(DomainError):
    """Another worker process already runs on this database."""

    code = "worker_lock_held"


def _try_lock(fd: int) -> bool:
    """Take the lock without waiting; False if somebody else holds it."""
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


class WorkerLock:
    def __init__(self, db_path: Path) -> None:
        self.path = db_path.with_name(db_path.name + LOCK_SUFFIX)
        self._fd: int | None = None

    def acquire(self) -> None:
        """Take the lock or raise `WorkerLockHeldError`. Taking it again is a no-op."""
        if self._fd is not None:
            return
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, _PRIVATE_FILE_MODE)
        except OSError:
            raise StorageError("cannot open the worker lock file") from None
        try:
            if not _try_lock(fd):
                raise WorkerLockHeldError(
                    "another worker is already running on this database "
                    "(stop it first; only one worker per database may run)"
                )
            self._note_owner(fd)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd

    @staticmethod
    def _note_owner(fd: int) -> None:
        """Write the process id for a human reading the file; the lock does not depend on it."""
        with contextlib.suppress(OSError):
            os.lseek(fd, 1, os.SEEK_SET)  # byte 0 is the locked byte on Windows
            os.ftruncate(fd, 1)
            os.write(fd, f"pid {os.getpid()}\n".encode("ascii"))

    def release(self) -> None:
        """Let go of the lock. Safe to call more than once or without having acquired it."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            _unlock(fd)
        os.close(fd)

    def __enter__(self) -> "WorkerLock":
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
