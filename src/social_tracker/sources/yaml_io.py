"""YAML codec for source files: strict, bounded and free of secrets.

File format: a top-level mapping with one key, `sources`, holding a list of source mappings.
Credentials never appear: a source names its secret through `secret_ref`, and every other
unknown key (a pasted `token:`, `password:`...) is rejected by the source model.

Loading is hardened because the text is untrusted:
- only `yaml.SafeLoader` machinery is used; explicit tags (including `!!python/...`) are
  rejected before anything is constructed, as are anchors and aliases (billion-laughs) and
  merge keys, so the document can never expand beyond its own size;
- the text is capped in bytes, in parser events, in nesting depth and in entries;
- duplicate mapping keys and multiple documents are rejected;
- an invalid file is reported as ALL of its problems at once (`SourceFileError.problems`),
  with entry index and KNOWN field names only: no message ever echoes a key or a value.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import yaml

from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError
from social_tracker.sources.domain import Source, SourceBase, normalize_name, parse_source

MAX_FILE_BYTES = 1_000_000
MAX_SOURCES = 500
MAX_EVENTS = 100_000  # parser events: bounds work even for a file that fits in MAX_FILE_BYTES
MAX_DEPTH = 8  # mapping -> sources list -> entry -> keywords list is 4; the rest is headroom
_ROOT_KEY = "sources"
_CODE_INVALID = "validation_error"
_CODE_DUPLICATE = "duplicate"
_MAX_MESSAGE_PROBLEMS = 10

# Written first, in this order; timestamps last; every other field in between, in model order.
_HEAD_FIELDS = ("id", "type", "name", "mode", "credibility", "enabled", "secret_ref")
_TAIL_FIELDS = ("created_at", "updated_at")


@dataclass(frozen=True)
class SourceFileProblem:
    """One problem found in a source file. `index` is the entry position, None for the file."""

    index: int | None
    code: str
    fields: tuple[str, ...]
    message: str


class SourceFileError(ValidationError):
    """A source file was rejected; `problems` lists every one of them, in entry order."""

    def __init__(self, problems: Iterable[SourceFileProblem]) -> None:
        self.problems: tuple[SourceFileProblem, ...] = tuple(problems)
        shown = [
            f"entry {p.index}: {p.message}" if p.index is not None else p.message
            for p in self.problems[:_MAX_MESSAGE_PROBLEMS]
        ]
        hidden = len(self.problems) - len(shown)
        if hidden > 0:
            shown.append(f"and {hidden} more")
        fields = dict.fromkeys(f for p in self.problems for f in p.fields)
        super().__init__(f"invalid source file: {'; '.join(shown)}", tuple(fields))


def _file_problem(message: str, *fields: str) -> SourceFileProblem:
    return SourceFileProblem(None, _CODE_INVALID, fields, message)


class _Rejected(Exception):  # noqa: N818 - internal control flow, never leaves this module
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys.

    Merge keys (`<<`) are rejected too: there is no constructor for the merge tag, so building
    the key node below fails before any merge could happen.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                if key in seen:
                    raise _Rejected("duplicate mapping key")
                seen.add(key)
            except TypeError:
                continue  # unhashable key: the base class reports it
        return super().construct_mapping(node, deep)


class _PlainDumper(yaml.SafeDumper):
    """Never emits anchors or aliases, even when the same object appears twice."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _scan(text: str) -> None:
    """Walk the parser events (nothing is constructed) and refuse anything unsafe or unbounded."""
    depth = documents = 0
    for count, event in enumerate(yaml.parse(text, Loader=yaml.SafeLoader), start=1):
        if count > MAX_EVENTS:
            raise _Rejected("document is too complex")
        if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None) is not None:
            raise _Rejected("anchors and aliases are not allowed")
        if getattr(event, "tag", None) is not None:
            raise _Rejected("explicit YAML tags are not allowed")
        if isinstance(event, yaml.DocumentStartEvent):
            documents += 1
            if documents > 1:
                raise _Rejected("multiple YAML documents are not allowed")
        elif isinstance(event, yaml.MappingStartEvent | yaml.SequenceStartEvent):
            depth += 1
            if depth > MAX_DEPTH:
                raise _Rejected("nesting is too deep")
        elif isinstance(event, yaml.MappingEndEvent | yaml.SequenceEndEvent):
            depth -= 1


def _parse_document(text: str) -> Any:
    if not text.strip():
        return None  # blank (tabs included) is an empty document, not a syntax error
    try:
        _scan(text)
        return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except _Rejected as exc:
        raise SourceFileError([_file_problem(exc.message)]) from None
    except yaml.YAMLError:
        raise SourceFileError([_file_problem("malformed or unsupported YAML document")]) from None
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise SourceFileError([_file_problem("malformed or unsupported YAML document")]) from None


def _with_default_timestamps(entry: dict[Any, Any], default: datetime | None) -> dict[Any, Any]:
    filled = dict(entry)
    if "created_at" not in filled and default is not None:
        filled["created_at"] = default
    if "updated_at" not in filled and filled.get("created_at") is not None:
        filled["updated_at"] = filled["created_at"]
    return filled


def _check_root(document: Any) -> tuple[list[Any] | None, list[SourceFileProblem]]:
    if document is None:
        message = "document is empty: expected a mapping with a 'sources' list"
        return None, [_file_problem(message, _ROOT_KEY)]
    if not isinstance(document, dict):
        return None, [_file_problem("document must be a mapping with a 'sources' list", _ROOT_KEY)]
    if _ROOT_KEY not in document:
        return None, [_file_problem("missing 'sources' list", _ROOT_KEY)]
    entries = document[_ROOT_KEY]
    if not isinstance(entries, list):
        return None, [_file_problem("'sources' must be a list", _ROOT_KEY)]
    if len(entries) > MAX_SOURCES:
        return None, [_file_problem(f"too many entries: at most {MAX_SOURCES} are allowed")]
    extra = [_file_problem("unexpected top-level key: only 'sources' is allowed")]
    return entries, extra if len(document) > 1 else []


def load_sources(text: str, *, default_timestamp: datetime | None = None) -> list[Source]:
    """Parse a source file into sources, in file order, or raise `SourceFileError`.

    `default_timestamp` fills in a missing `created_at` (and `updated_at` follows `created_at`),
    so hand-written files need not carry timestamps.
    """
    if default_timestamp is not None:
        default_timestamp = ensure_utc(default_timestamp)
    if not isinstance(text, str):
        raise SourceFileError([_file_problem("source file must be text")])
    if len(text) > MAX_FILE_BYTES or len(text.encode("utf-8", "surrogatepass")) > MAX_FILE_BYTES:
        raise SourceFileError([_file_problem(f"file is larger than {MAX_FILE_BYTES} bytes")])
    entries, problems = _check_root(_parse_document(text))
    if entries is None:
        raise SourceFileError(problems)

    sources: list[tuple[int, Source]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(SourceFileProblem(index, _CODE_INVALID, (), "entry must be a mapping"))
            continue
        try:
            sources.append(
                (index, parse_source(_with_default_timestamps(entry, default_timestamp)))
            )
        except ValidationError as exc:
            problems.append(SourceFileProblem(index, _CODE_INVALID, exc.fields, exc.message))
    problems.extend(_duplicates(sources))
    if problems:
        raise SourceFileError(sorted(problems, key=lambda p: -1 if p.index is None else p.index))
    return [source for _, source in sources]


def _duplicates(sources: list[tuple[int, Source]]) -> list[SourceFileProblem]:
    by_id: dict[str, int] = {}
    by_name: dict[str, int] = {}
    found: list[SourceFileProblem] = []
    for index, source in sources:
        if source.id in by_id:
            message = f"duplicate id: same as entry {by_id[source.id]}"
            found.append(SourceFileProblem(index, _CODE_DUPLICATE, ("id",), message))
            continue
        by_id[source.id] = index
        name = normalize_name(source.name)
        if name in by_name:
            message = f"duplicate name: same as entry {by_name[name]}"
            found.append(SourceFileProblem(index, _CODE_DUPLICATE, ("name",), message))
            continue
        by_name[name] = index
    return found


def _entry_of(source: Source) -> dict[str, Any]:
    data = {k: v for k, v in source.model_dump().items() if v is not None}
    for key in _TAIL_FIELDS:
        data[key] = data[key].isoformat()
    middle = [k for k in data if k not in _HEAD_FIELDS and k not in _TAIL_FIELDS]
    return {k: data[k] for k in (*_HEAD_FIELDS, *middle, *_TAIL_FIELDS) if k in data}


def dump_sources(sources: Iterable[Source]) -> str:
    """Render sources as YAML: sorted by id, fixed key order, secret references only."""
    try:
        items = list(sources)
    except TypeError:
        raise ValidationError("sources must be an iterable of sources") from None
    if not all(isinstance(item, SourceBase) for item in items):
        raise ValidationError("every item must be a source")
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise ValidationError("source ids must be unique", ("id",))
    entries = [_entry_of(item) for item in sorted(items, key=lambda s: s.id)]
    return yaml.dump(
        {_ROOT_KEY: entries},
        Dumper=_PlainDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=MAX_FILE_BYTES,
    )
