"""SQLite database access: one guarded connection, transactions, error mapping, migrations.

Design:
- One connection per `Database`, shared by threads behind a re-entrant lock; every statement
  runs while holding it, so the connection is never used concurrently. Several `Database`
  instances (or processes) may open the same file: WAL plus `busy_timeout` serialise writers.
- Writes go through `transaction()` (BEGIN IMMEDIATE ... COMMIT/ROLLBACK). Reads use `query()`.
- Every `sqlite3.Error` is translated: unique violations become `ConflictError`, everything
  else `StorageError`. Messages are fixed strings plus SQLite's constant error name; they never
  carry SQL text, parameters, paths or file contents.
- A zero-byte file is a valid, empty SQLite database and is initialised normally. A file that
  is not a database is never modified: opening fails with `StorageError`.
- `schema_migrations` belongs to the runner (created before any migration runs), not to 0001.
"""

import hashlib
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
    if conn.in_transaction:
        with suppress(sqlite3.Error):
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
            conn.executescript("BEGIN IMMEDIATE;\n" + migration.sql)
            if not conn.in_transaction:
                raise StorageError("a migration must not end the transaction itself")
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
        self._path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._path, isolation_level=None, check_same_thread=False)
        except OSError:
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
        """Exclusive write transaction. Rolls back on any exception, re-raising it."""
        with self._lock:
            conn = self._connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise map_sqlite_error(exc) from None
            try:
                yield conn
            except BaseException as exc:
                _rollback(conn)
                if isinstance(exc, sqlite3.Error):
                    raise map_sqlite_error(exc) from None
                raise
            try:
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                _rollback(conn)
                raise map_sqlite_error(exc) from None

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
