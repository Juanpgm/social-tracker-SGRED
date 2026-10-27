"""Input handling of the CLI: who is acting, `--set KEY=VALUE` pairs and bounded file I/O.

Every failure is a `DomainError` with a fixed message. Messages never echo option values, keys
or file contents (a mistyped `--set` may well be a pasted secret) and never include paths.
"""

import contextlib
import getpass
import json
import os
import stat
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_tracker.shared.errors import ConflictError, ValidationError
from social_tracker.sources.yaml_io import MAX_FILE_BYTES

ACTOR_ENV = "SOCIAL_TRACKER_ACTOR"
_UNKNOWN_USER = "unknown"
_IS_WINDOWS = os.name == "nt"
_REPLACE_ATTEMPTS = 40
_REPLACE_PAUSE_S = 0.025


def _os_user() -> str:
    try:
        return getpass.getuser()
    except (OSError, KeyError, ImportError):
        return _UNKNOWN_USER


def resolve_actor(explicit: str | None, environ: Mapping[str, str]) -> str:
    """`--actor` as given (even blank: the service rejects it), else `$SOCIAL_TRACKER_ACTOR`,
    else `cli:<os user>`. Only the service decides whether the result is a valid actor."""
    if explicit is not None:
        return explicit
    return environ.get(ACTOR_ENV, "").strip() or f"cli:{_os_user()}"


def _value(raw: str) -> Any:
    """JSON when it parses (numbers, booleans, null, lists), the plain text otherwise."""
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        return raw


def parse_fields(pairs: Sequence[str], *, reserved: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Turn repeated `--set KEY=VALUE` options into a field mapping."""
    fields: dict[str, Any] = {}
    for pair in pairs:
        key, equals, raw = pair.partition("=")
        if not equals or not key.strip():
            raise ValidationError("--set expects KEY=VALUE")
        if key in reserved:
            raise ValidationError(
                "this field has a dedicated option and cannot be given with --set"
            )
        if key in fields:
            raise ValidationError("a --set field was given more than once")
        fields[key] = _value(raw)
    return fields


def _read_failure(reason: str, what: str = "import file") -> ValidationError:
    return ValidationError(f"cannot read the {what}: {reason}")


def _write_failure(reason: str) -> ValidationError:
    return ValidationError(f"cannot write the output file: {reason}")


def read_import_text(
    path: Path, *, max_bytes: int = MAX_FILE_BYTES, what: str = "import file"
) -> str:
    """Read an import file as text, never more than `max_bytes` + 1 bytes of it (`what` names
    the file in the messages).

    Only regular files are read (checked with `fstat` on the opened descriptor, so there is no
    check-then-open race): a FIFO, device or directory is refused instead of blocking on it or
    reading from it forever. The file is opened non-blocking so a FIFO without a writer cannot
    hang the open itself. A symbolic link to a regular file is followed.

    The size limit is on bytes read from disk, applied BEFORE decoding or parsing, so an oversized
    or endless file cannot exhaust memory. A UTF-8 byte order mark is accepted and dropped.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise _read_failure(exc.strerror or "I/O error", what) from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _read_failure("not a regular file", what)
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise _read_failure(exc.strerror or "I/O error", what) from None
    finally:
        os.close(fd)
    if len(raw) > max_bytes:
        raise ValidationError(f"{what} is larger than {max_bytes} bytes")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValidationError(f"{what} is not valid UTF-8") from None


def _existing_mode(path: Path) -> int | None:
    try:
        return os.lstat(path).st_mode
    except FileNotFoundError:
        return None


def write_export(path: Path, text: str, *, force: bool) -> None:
    """Write `text` as UTF-8 bytes (no newline translation) so that the target is never left
    partial or empty.

    The bytes go to a private temporary file in the target's directory (created exclusively with
    mode 0600 where the platform has modes), are flushed to disk, and only then published under
    the target name: `os.replace` with `force`, a hard link without it (which fails if the name
    exists, so a race cannot clobber a file; a filesystem without hard links needs `--force`).
    On any failure, interrupts included, the temporary file is removed and an existing target is
    left as it was. A symbolic link or any other non-regular file as target is refused. The
    published file keeps the temporary file's owner-only mode, not the mode of a replaced file.
    """
    data = text.encode("utf-8")
    try:
        _publish(path, data, force=force)
    except OSError as exc:
        raise _write_failure(exc.strerror or "I/O error") from None


def _publish(path: Path, data: bytes, *, force: bool) -> None:
    mode = _existing_mode(path)
    if mode is not None:
        if stat.S_ISLNK(mode):
            raise _write_failure("refusing to write through a symbolic link")
        if not stat.S_ISREG(mode):
            raise _write_failure("not a regular file")
        if not force:
            raise _exists()
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".export-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if force:
            _replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise _exists() from None
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)


def _replace(source: str, target: Path) -> None:
    """`os.replace`, retried for about a second on Windows, where replacing a file that another
    process is replacing (or has open for a moment) fails with a transient `PermissionError`.
    Elsewhere a denied replace is a real permission problem and is raised at once."""
    for attempt in range(1, _REPLACE_ATTEMPTS + 1):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if not _IS_WINDOWS or attempt == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_PAUSE_S)


def _exists() -> ConflictError:
    return ConflictError("output file already exists; use --force to overwrite it")
