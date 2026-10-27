"""SQLite database access: one guarded connection, transactions, error mapping, migrations.

Design:
- One connection per `Database`, shared by threads behind a re-entrant lock; every statement
  runs while holding it, so the connection is never used concurrently. Several `Database`
  instances (or processes) may open the same file: WAL plus `busy_timeout` serialise writers.
- Writes go through `transaction()` (BEGIN IMMEDIATE ... COMMIT/ROLLBACK). Reads use `query()`.
  A nested `transaction()` on the owning thread joins the outer one through a SAVEPOINT, so the
  application layer can make several repository calls atomic.
- Single-writer caveat (slices 2 and 3): `query()` and `transaction()` take the same re-entrant
  lock, so a long write transaction blocks the reads of every other thread in the process until
  it commits. WAL would allow concurrent readers, so a later slice should give reads their own
  connection. A lock timeout (busy) is still reported as the generic `StorageError`; mapping it
  to a distinct, retryable error is future work.
- Every `sqlite3.Error` is translated: unique violations become `ConflictError`, everything
  else `StorageError`. Messages are fixed strings plus SQLite's constant error name; they never
  carry SQL text, parameters, paths or file contents.
- A zero-byte file is a valid, empty SQLite database and is initialised normally. A file that
  is not a database is never modified: opening fails with `StorageError`.
- Permissions: a parent directory that this class creates gets mode 0700 (the leaf only; any
  intermediate directory follows the umask) and a database file it creates gets mode 0600, made
  with an exclusive create so an existing file keeps whatever mode its owner gave it. `-wal` and
  `-shm` inherit the mode of the main file. Windows has no POSIX modes: there the file simply
  inherits the ACL of its directory.
- `schema_migrations` belongs to the runner (created before any migration runs), not to 0001.
  It records `(version, applied_at, checksum)`; on every open the checksum of each applied
  migration is compared with the file (line endings normalised) and a mismatch is a
  `MigrationChecksumError`. There is no upgrade path from a table without `checksum`: the
  project had no deployed data when it was introduced, so such a database is rejected.
- Migrations run inside the runner's own transaction and must not manage transactions
  (BEGIN/COMMIT/END/ROLLBACK/SAVEPOINT/RELEASE are rejected up front; trigger bodies are fine).
  `PRAGMA foreign_keys` is a no-op inside a transaction, so a future table-rebuild migration
  must copy data inside the transaction and use `PRAGMA defer_foreign_keys = ON` instead.
"""

import hashlib
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ConflictError, DomainError, StorageError

DEFAULT_BUSY_TIMEOUT_MS = 5000
_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_CLOSE_CHECKPOINT_WAIT_MS = 100
_MIGRATION_FILE_RE = re.compile(r"([0-9]{4})_([a-z0-9_]+)\.sql")
_ERROR_NAME_RE = re.compile(r"[A-Z_]{1,40}")
_UNIQUE_ERRORS = frozenset({"SQLITE_CONSTRAINT_UNIQUE", "SQLITE_CONSTRAINT_PRIMARYKEY"})

# The final shape of the bookkeeping table. The project has no deployed data yet, so there is
# no upgrade path from earlier shapes: a table without `checksum` is rejected, not migrated.
_CREATE_MIGRATIONS_TABLE = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "version INTEGER NOT NULL PRIMARY KEY CHECK (version >= 1), "
    "applied_at TEXT NOT NULL, "
    "checksum TEXT NOT NULL)"
)


# ---- timestamps -----------------------------------------------------------------------


def format_utc(value: datetime) -> str:
    """ISO-8601 text in UTC (`...+00:00`), the only timestamp format the schema accepts."""
    return ensure_utc(value).isoformat()


def parse_utc(text: Any) -> datetime:
    """Inverse of `format_utc`; a malformed or naive value is a `StorageError`."""
    if not isinstance(text, str):
        raise StorageError("invalid stored timestamp")
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        raise StorageError("invalid stored timestamp") from None
    if value.tzinfo is None or value.utcoffset() is None:
        raise StorageError("invalid stored timestamp")
    return value.astimezone(UTC)


# ---- error mapping --------------------------------------------------------------------


def map_sqlite_error(exc: sqlite3.Error) -> DomainError:
    name = getattr(exc, "sqlite_errorname", None)
    if not isinstance(name, str) or not _ERROR_NAME_RE.fullmatch(name):
        name = type(exc).__name__
    if name in _UNIQUE_ERRORS:
        return ConflictError("unique constraint violated")
    return StorageError(f"database error: {name}")


# ---- migrations -----------------------------------------------------------------------


class MigrationChecksumError(StorageError):
    """An applied migration no longer matches the file it was applied from."""

    code = "migration_checksum_mismatch"


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        """SHA-256 of the SQL with line endings normalised, so CRLF checkouts do not drift."""
        normalised = self.sql.replace("\r\n", "\n").replace("\r", "\n")
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


_LEADING_NOISE_RE = re.compile(r"(?:\s+|--[^\n]*(?:\n|$)|/\*.*?(?:\*/|$))*", re.DOTALL)
_FIRST_WORD_RE = re.compile(r"[A-Za-z]+")
_TRANSACTION_KEYWORDS = frozenset({"BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE"})


def _statements(sql: str) -> Iterator[str]:
    """Split a script into statements the way SQLite does (trigger bodies stay in one piece)."""
    start = 0
    for index, char in enumerate(sql):
        if char == ";" and sqlite3.complete_statement(sql[start : index + 1]):
            yield sql[start : index + 1]
            start = index + 1
    if sql[start:].strip():
        yield sql[start:]


def _has_transaction_control(sql: str) -> bool:
    """True if any top-level statement is BEGIN/COMMIT/END/ROLLBACK/SAVEPOINT/RELEASE.

    Only the first keyword of each statement counts, so `BEGIN ... END` inside a CREATE TRIGGER
    and keywords inside strings, comments or quoted identifiers are not mistaken for it.
    """
    for statement in _statements(sql):
        body = statement[_LEADING_NOISE_RE.match(statement).end() :]
        word = _FIRST_WORD_RE.match(body)
        if word is not None and word.group().upper() in _TRANSACTION_KEYWORDS:
            return True
    return False


def validate_migrations(migrations: Sequence[Migration]) -> tuple[Migration, ...]:
    """Versions must be exactly 1..n in order: no gaps, no duplicates, nothing at zero."""
    ordered = tuple(migrations)
    if not ordered:
        raise StorageError("no migrations defined")
    for expected, migration in enumerate(ordered, start=1):
        if (
            not isinstance(migration.version, int)
            or isinstance(migration.version, bool)
            or migration.version != expected
        ):
            raise StorageError("migrations must be numbered 1..n without gaps or duplicates")
        if not isinstance(migration.sql, str) or not migration.sql.strip():
            raise StorageError("a migration must not be empty")
        if "\x00" in migration.sql:
            raise StorageError("a migration must not contain NUL characters")
        try:
            migration.sql.encode("utf-8")
        except UnicodeEncodeError:
            raise StorageError("a migration must be valid UTF-8 text") from None
        if _has_transaction_control(migration.sql):
            raise StorageError("a migration must not manage transactions itself")
    return ordered


def load_migrations(source: Traversable | None = None) -> tuple[Migration, ...]:
    """Discover `NNNN_name.sql` files (default: the packaged migrations), sorted by number."""
    directory = source if source is not None else files("social_tracker.persistence.migrations")
    found: list[Migration] = []
    for entry in directory.iterdir():
        if not entry.name.lower().endswith(".sql"):
            continue
        match = _MIGRATION_FILE_RE.fullmatch(entry.name)
        if match is None or not entry.is_file():
            raise StorageError("migration files must be named NNNN_name.sql")
        try:
            sql = entry.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise StorageError("a migration file could not be read as UTF-8") from None
        found.append(Migration(int(match.group(1)), match.group(2), sql))
    return validate_migrations(sorted(found, key=lambda m: m.version))


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_MIGRATIONS_TABLE)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(schema_migrations)")}
    if "checksum" not in columns:
        raise StorageError("schema_migrations has an unsupported shape: recreate the database")


def _applied_checksums(conn: sqlite3.Connection) -> dict[int, str]:
    rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    if not all(isinstance(v, int) and isinstance(c, str) for v, c in rows):
        raise StorageError("schema_migrations is corrupt")
    return dict(rows)


def _check_history(applied: dict[int, str], known: Sequence[Migration]) -> None:
    if set(applied) - {m.version for m in known}:
        raise StorageError("database was created by a newer version of the application")
    if set(applied) != set(range(1, len(applied) + 1)):
        raise StorageError("applied migrations are not a contiguous prefix of the known ones")
    for migration in known:
        recorded = applied.get(migration.version)
        if recorded is not None and recorded != migration.checksum:
            raise MigrationChecksumError("an applied migration was modified after it was applied")


def _rollback(conn: sqlite3.Connection) -> None:
    with suppress(sqlite3.Error):
        if conn.in_transaction:
            conn.execute("ROLLBACK")


def _apply_pending(conn: sqlite3.Connection, migrations: Sequence[Migration]) -> None:
    _ensure_migrations_table(conn)
    applied = _applied_checksums(conn)
    _check_history(applied, migrations)
    for migration in migrations:
        if migration.version in applied:
            continue
        try:
            # executescript would COMMIT a transaction opened beforehand, so BEGIN is part of it.
            # Migrations are screened for transaction control up front; the check below is the
            # backstop should one slip through.
            conn.executescript("BEGIN IMMEDIATE;\n" + migration.sql)
            if not conn.in_transaction:
                raise StorageError("a migration must not manage transactions itself")
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at, checksum) VALUES (?, ?, ?)",
                (migration.version, format_utc(datetime.now(UTC)), migration.checksum),
            )
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            _rollback(conn)
            if migration.version in _applied_checksums(conn):
                continue  # another process applied it while we waited for the write lock
            raise map_sqlite_error(exc) from None
        except BaseException:
            _rollback(conn)
            raise


def _enable_wal(conn: sqlite3.Connection, timeout_ms: int) -> Any:
    """Switch to WAL, retrying while another connection holds the file.

    SQLite does not call the busy handler for this pragma (it returns SQLITE_BUSY at once when
    it would have to wait for an exclusive lock), so concurrent first opens of a new file need
    an explicit retry bounded by the same timeout.
    """
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        try:
            return conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        except sqlite3.OperationalError as exc:
            busy = str(getattr(exc, "sqlite_errorname", "")).startswith("SQLITE_BUSY")
            if not busy or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _checked_path(path: object) -> Path:
    """Turn programmer errors (None, bytes, NUL characters) into a fixed-message StorageError."""
    if not isinstance(path, str | Path):
        raise StorageError("database path must be a str or a Path")
    if "\x00" in str(path):
        raise StorageError("database path must not contain NUL characters")
    return Path(path)


def _create_private_file(path: Path) -> None:
    """Create the database file with mode 0600 unless it already exists.

    The exclusive create means an existing file (ours from an earlier run, or one a concurrent
    process just made) is never touched, and SQLite then opens an empty file as a new database.
    The `-wal` and `-shm` files SQLite creates later copy the mode of the main file.
    """
    with suppress(FileExistsError):
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE_MODE))


# ---- database -------------------------------------------------------------------------


class Database:
    def __init__(
        self,
        path: Path | str,
        *,
        migrations: Sequence[Migration] | None = None,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        known = validate_migrations(migrations if migrations is not None else load_migrations())
        if (
            not isinstance(busy_timeout_ms, int)
            or isinstance(busy_timeout_ms, bool)
            or busy_timeout_ms <= 0
        ):
            raise StorageError("busy_timeout_ms must be a positive integer")
        self._path = _checked_path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._tx_depth = 0  # only touched while holding `_lock`, i.e. by the owning thread
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
            _create_private_file(self._path)
            conn = sqlite3.connect(self._path, isolation_level=None, check_same_thread=False)
        except (OSError, ValueError):
            raise StorageError("database location is not usable") from None
        except sqlite3.Error as exc:
            raise map_sqlite_error(exc) from None
        try:
            conn.row_factory = sqlite3.Row
            self._configure(conn, busy_timeout_ms)
            _apply_pending(conn, known)
        except sqlite3.Error as exc:
            conn.close()
            raise map_sqlite_error(exc) from None
        except BaseException:
            conn.close()
            raise
        self._conn = conn

    @staticmethod
    def _configure(conn: sqlite3.Connection, busy_timeout_ms: int) -> None:
        # busy_timeout first: switching journal mode already needs the file lock.
        conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        mode = _enable_wal(conn, busy_timeout_ms)
        if str(mode).lower() != "wal":
            raise StorageError("write-ahead logging is not available for this database")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

    @property
    def path(self) -> Path:
        return self._path

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageError("database is closed")
        return self._conn

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            conn = self._connection()
            try:
                return conn.execute(sql, params).fetchall()
            except sqlite3.Error as exc:
                raise map_sqlite_error(exc) from None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Exclusive write transaction. Rolls back on any exception, re-raising it.

        Joinable: while a thread holds a transaction, a nested `transaction()` on that thread
        runs inside a SAVEPOINT. The inner block releases it on success; on failure it rolls
        back to it, so the outer code can catch the error and keep its own writes. Rolling back
        the outer transaction undoes the inner work too. Other threads never join: they wait
        for the lock until the owner finishes.
        """
        with self._lock:
            conn = self._connection()
            if self._tx_depth == 0:
                yield from self._outermost(conn)
            else:
                yield from self._nested(conn)

    def _outermost(self, conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise map_sqlite_error(exc) from None
        self._tx_depth = 1
        try:
            try:
                yield conn
            except BaseException as exc:
                _rollback(conn)
                if isinstance(exc, sqlite3.Error):
                    raise map_sqlite_error(exc) from None
                raise
            if self._conn is None:
                raise StorageError("database was closed during the transaction")
            try:
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                _rollback(conn)
                raise map_sqlite_error(exc) from None
        finally:
            self._tx_depth = 0

    def _nested(self, conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        self._tx_depth += 1
        name = f"st_sp_{self._tx_depth}"
        try:
            try:
                conn.execute(f"SAVEPOINT {name}")
            except sqlite3.Error as exc:
                raise map_sqlite_error(exc) from None
            try:
                yield conn
            except BaseException as exc:
                # SQLite may already have rolled everything back (RAISE(ROLLBACK), disk errors);
                # then there is no savepoint left and the outer COMMIT reports the failure.
                with suppress(sqlite3.Error):
                    conn.execute(f"ROLLBACK TO {name}")
                    conn.execute(f"RELEASE {name}")
                if isinstance(exc, sqlite3.Error):
                    raise map_sqlite_error(exc) from None
                raise
            try:
                conn.execute(f"RELEASE {name}")
            except sqlite3.Error as exc:
                raise map_sqlite_error(exc) from None
        finally:
            self._tx_depth -= 1

    def close(self) -> None:
        """Release the file. Never raises.

        The WAL is checkpointed and truncated first (best effort) so a `-wal` file does not
        linger while other connections keep the database open. A process that stays up for
        days needs the same housekeeping while running: SQLite checkpoints automatically at
        ~1000 pages, but a long-lived reader can stall that, so 24/7 workers should
        `PRAGMA wal_checkpoint(TRUNCATE)` periodically (e.g. between collection cycles).
        """
        with self._lock:
            conn, self._conn = self._conn, None
            if conn is None:
                return
            try:
                with suppress(sqlite3.Error):
                    # Do not sit out the full busy timeout behind a reader: give up quickly.
                    conn.execute(f"PRAGMA busy_timeout={_CLOSE_CHECKPOINT_WAIT_MS}")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
