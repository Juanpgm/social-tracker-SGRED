"""Input handling of the CLI: who is acting, `--set KEY=VALUE` pairs and bounded file I/O.

Every failure is a `DomainError` with a fixed message. Messages never echo option values, keys
or file contents (a mistyped `--set` may well be a pasted secret) and never include paths.
"""

import getpass
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from social_tracker.shared.errors import ConflictError, ValidationError
from social_tracker.sources.yaml_io import MAX_FILE_BYTES

ACTOR_ENV = "SOCIAL_TRACKER_ACTOR"
_UNKNOWN_USER = "unknown"


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


def read_import_text(path: Path) -> str:
    """Read an import file as text, never more than `MAX_FILE_BYTES` + 1 bytes of it.

    The limit is on bytes read from disk, applied BEFORE decoding or parsing, so an oversized or
    endless file cannot exhaust memory. A UTF-8 byte order mark is accepted and dropped.
    """
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise ValidationError(
            f"cannot read the import file: {exc.strerror or 'I/O error'}"
        ) from None
    if len(raw) > MAX_FILE_BYTES:
        raise ValidationError(f"import file is larger than {MAX_FILE_BYTES} bytes")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValidationError("import file is not valid UTF-8") from None


def write_export(path: Path, text: str, *, force: bool) -> None:
    """Write `text` as UTF-8 bytes (no newline translation). An existing file is only replaced
    with `force`; without it the file is created exclusively, so a race cannot clobber one."""
    data = text.encode("utf-8")
    try:
        with path.open("wb" if force else "xb") as handle:
            handle.write(data)
    except FileExistsError:
        raise ConflictError("output file already exists; use --force to overwrite it") from None
    except OSError as exc:
        raise ValidationError(
            f"cannot write the output file: {exc.strerror or 'I/O error'}"
        ) from None
