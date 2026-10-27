"""Source entity: a pydantic v2 discriminated union on `type`. Pure, no I/O.

Credentials never live in a source: sources point at a secret by reference (`secret_ref`,
shaped `<scheme>:<name>`).
Every other unknown key is rejected by `extra="forbid"`; the model has no dict-typed
fields, so there is no nested place for a credential to hide and no key blacklist to evade.
"""

import re
import unicodedata
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)
from pydantic import BaseModel as _PydanticBase
from pydantic import ValidationError as _PydanticValidationError

from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError
from social_tracker.shared.ids import validate_slug
from social_tracker.shared.text import has_control_chars, has_unsafe_chars, is_blank_text

# Product rules, named so 1d/1e (CLI, docs) and tests reference one source of truth.
NEWS_RSS_MIN_POLL_S = 120
INSTAGRAM_MIN_POLL_S = 900
RECORDED_AUDIO_MIN_POLL_S = 1
CHUNK_SECONDS_MIN = 10
CHUNK_SECONDS_MAX = 60
KEYWORDS_MAX_ITEMS = 200
KEYWORD_MAX_LENGTH = 200
LANGUAGE_MAX_LENGTH = 35  # longest well-formed BCP-47 tag we care about
# Simple BCP-47-like tag: language subtag then hyphen-separated alphanumeric subtags.
LANGUAGE_PATTERN = r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$"
# secret_ref names a secret held elsewhere and must look like `<scheme>:<name>`: `env:NAME` (an
# environment variable), `keyring:NAME` (an OS keyring entry) or `file:NAME` (a file under the
# secrets directory). A pasted raw token (`ghp_...`, `sk-live-...`) has no scheme and is
# refused; one pasted AFTER a valid scheme is still bounded by the charset and length below.
REF_SCHEMES = ("env", "keyring", "file")
REF_NAME_MAX = 64
REF_NAME_PATTERN = rf"^({'|'.join(REF_SCHEMES)}):[A-Za-z0-9_./-]{{1,{REF_NAME_MAX}}}$"

MAX_URL_LENGTH = 2048
MAX_PATH_LENGTH = 4096
# Secret-looking URL parameter names. Long names match as substrings ("x-api-key",
# "access_token"); short ambiguous ones only as whole tokens so "author" and "design" stay valid.
# All patterns are linear: plain alternation or single bounded character classes.
_SECRET_NAME_RE = re.compile(
    r"password|passwd|token|secret|api[_-]?key|signature|session|cookie|access[_-]?key"
    r"|authorization"
)
_SECRET_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_SECRET_TOKENS = frozenset({"auth", "oauth", "sig", "pwd"})
_PERCENT_RUN_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})+")
_QUERY_SEPARATOR_RE = re.compile(r"[&;]")
_MAX_DECODE_PASSES = 3  # servers sometimes decode twice; "%2574oken" must not slip through
HTTP_SCHEMES = frozenset({"http", "https"})
STREAM_SCHEMES = frozenset({"http", "https", "rtmp", "icecast"})


# ---- helpers --------------------------------------------------------------------------


_NORMALIZE_MAX_PASSES = 5  # real text settles in 1-2 passes; the bound only guards a bug


def _normalize_once(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip()).casefold()


def normalize_name(name: str) -> str:
    """Canonical form used for case/whitespace-insensitive uniqueness.

    A single NFC + casefold pass is not idempotent (casefolding can produce sequences that NFC
    would reorder or recompose), which would make a stored `normalized_name` unfindable by
    itself. The pass is therefore repeated until it reaches a fixed point, so
    `normalize_name(normalize_name(x)) == normalize_name(x)` for every input.
    """
    current = name
    for _ in range(_NORMALIZE_MAX_PASSES):
        following = _normalize_once(current)
        if following == current:
            return current
        current = following
    raise ValidationError("name cannot be normalized to a stable form")


def _as_value_error[T](check: Callable[[Any], T]) -> Callable[[Any], T]:
    """Adapt a domain validator (raises `ValidationError`) to pydantic (expects `ValueError`)."""

    def wrapped(value: Any) -> T:
        try:
            return check(value)
        except ValidationError as exc:
            raise ValueError(exc.message) from None

    return wrapped


def _iso_or_datetime(value: Any) -> Any:
    """Accept datetimes and ISO-8601 strings only: no epochs, floats, bytes or numeric text."""
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("timestamp must be an ISO-8601 string or a datetime") from None
    if isinstance(value, datetime):
        return value
    raise ValueError("timestamp must be an ISO-8601 string or a datetime")


def _list_or_tuple(value: Any) -> Any:
    """Keywords may arrive as a list (JSON/YAML) or tuple; sets, dicts, strings and bytes no."""
    if isinstance(value, list | tuple):
        return tuple(value)
    raise ValueError("keywords must be a list")


def _name(value: str) -> str:
    if is_blank_text(value):
        raise ValueError("name must not be blank")
    if has_control_chars(value):
        raise ValueError("name must not contain control characters")
    if has_unsafe_chars(value):
        raise ValueError("name must not contain hidden characters")
    return value


def _keyword(value: str) -> str:
    if has_unsafe_chars(value):
        raise ValueError("keyword must not contain hidden characters")
    return value


def _percent_decode(text: str) -> str:
    def decode_run(match: re.Match[str]) -> str:
        raw = bytes.fromhex(match.group().replace("%", ""))
        return raw.decode("utf-8", errors="replace")

    for _ in range(_MAX_DECODE_PASSES):
        decoded = _PERCENT_RUN_RE.sub(decode_run, text)
        if decoded == text:
            break
        text = decoded
    return text


def _is_secret_name(raw_name: str) -> bool:
    name = unicodedata.normalize("NFKC", _percent_decode(raw_name)).casefold()
    if _SECRET_NAME_RE.search(name):
        return True
    return any(tok in _SECRET_TOKENS for tok in _SECRET_TOKEN_SPLIT_RE.split(name))


def _carries_secret_parameter(query: str, *, require_value: bool) -> bool:
    for pair in _QUERY_SEPARATOR_RE.split(query):
        name, sep, _ = pair.partition("=")
        if require_value and not sep:
            continue
        if _is_secret_name(name):
            return True
    return False


def _check_url(value: str, schemes: frozenset[str]) -> str:
    if len(value) > MAX_URL_LENGTH:
        raise ValueError(f"URL is longer than {MAX_URL_LENGTH} characters")
    if not value.strip():
        raise ValueError("URL must not be blank")
    if any(ch.isspace() for ch in value) or has_control_chars(value):
        raise ValueError("URL must not contain whitespace or control characters")
    scheme, sep, rest = value.partition("://")
    if not sep or scheme.lower() not in schemes:
        raise ValueError(f"URL scheme must be one of: {', '.join(sorted(schemes))}")
    authority = re.match(r"[^/?#]*", rest)
    authority_text = authority.group() if authority else ""
    if "@" in authority_text:
        raise ValueError("URL must not embed credentials (user:password@host)")
    if authority_text.startswith("["):
        host = authority_text.split("]", 1)[0].removeprefix("[")
    else:
        host = authority_text.split(":", 1)[0]
    if not host:
        raise ValueError("URL must have a host")
    tail = rest[len(authority_text) :]
    before_fragment, _, fragment = tail.partition("#")
    _, _, query = before_fragment.partition("?")
    if _carries_secret_parameter(query, require_value=False) or _carries_secret_parameter(
        fragment, require_value=True
    ):
        raise ValueError("URL must not carry credentials in query parameters or fragment")
    return value


def _http_url(value: str) -> str:
    return _check_url(value, HTTP_SCHEMES)


def _stream_url(value: str) -> str:
    return _check_url(value, STREAM_SCHEMES)


def _watch_dir(value: str) -> str:
    if len(value) > MAX_PATH_LENGTH:
        raise ValueError(f"path is longer than {MAX_PATH_LENGTH} characters")
    if not value.strip():
        raise ValueError("path must not be blank")
    if has_control_chars(value):
        raise ValueError("path must not contain control characters")
    return value


# ---- field types ----------------------------------------------------------------------

Slug = Annotated[
    str, StringConstraints(strict=True), AfterValidator(_as_value_error(validate_slug))
]
Name = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=100),
    AfterValidator(_as_value_error(_name)),
]
Credibility = Annotated[int, Field(strict=True, ge=1, le=5)]
Timestamp = Annotated[
    datetime, BeforeValidator(_iso_or_datetime), AfterValidator(_as_value_error(ensure_utc))
]
SecretRef = Annotated[str, StringConstraints(strict=True, pattern=REF_NAME_PATTERN)]
HttpUrl = Annotated[str, StringConstraints(strict=True), AfterValidator(_http_url)]
StreamUrl = Annotated[str, StringConstraints(strict=True), AfterValidator(_stream_url)]
WatchDir = Annotated[str, StringConstraints(strict=True), AfterValidator(_watch_dir)]
Keyword = Annotated[
    str,
    StringConstraints(
        strict=True, strip_whitespace=True, min_length=1, max_length=KEYWORD_MAX_LENGTH
    ),
    AfterValidator(_keyword),
]
InstagramHandle = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9._]{1,30}$")]
Language = Annotated[
    str, StringConstraints(strict=True, max_length=LANGUAGE_MAX_LENGTH, pattern=LANGUAGE_PATTERN)
]


# ---- entity ---------------------------------------------------------------------------


class SourceBase(_PydanticBase):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Slug
    name: Name
    credibility: Credibility = 3
    enabled: bool = Field(default=False, strict=True)
    secret_ref: SecretRef | None = None
    created_at: Timestamp
    updated_at: Timestamp

    @model_validator(mode="after")
    def _check_timestamps(self) -> "SourceBase":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        return self

    def with_changes(self, changes: Mapping[str, Any], /) -> "Source":
        """Return a NEW, fully re-validated source with `changes` applied.

        The changes are DATA, passed as one positional-only mapping, so no key can collide with
        a parameter name (a key called `self` used to crash a keyword-based version). Update
        paths must use this instead of `model_copy(update=...)`, which skips validation.
        """
        if not isinstance(changes, Mapping):
            raise ValidationError("changes must be a mapping of field names to values")
        return parse_source({**self.model_dump(), **changes})

    @property
    def normalized_name(self) -> str:
        return normalize_name(self.name)


class NewsRssSource(SourceBase):
    type: Literal["news_rss"]
    mode: Literal["recorded"] = "recorded"
    feed_url: HttpUrl
    poll_interval_s: int = Field(default=600, strict=True, ge=NEWS_RSS_MIN_POLL_S)
    keywords: Annotated[tuple[Keyword, ...], BeforeValidator(_list_or_tuple)] | None = Field(
        default=None, max_length=KEYWORDS_MAX_ITEMS
    )


class InstagramSource(SourceBase):
    type: Literal["instagram"]
    mode: Literal["recorded"] = "recorded"
    handle: InstagramHandle
    secret_ref: SecretRef  # required for Instagram
    include_stories: bool = Field(default=True, strict=True)
    include_posts: bool = Field(default=True, strict=True)
    poll_interval_s: int = Field(default=1800, strict=True, ge=INSTAGRAM_MIN_POLL_S)


class LiveStreamSource(SourceBase):
    type: Literal["live_stream"]
    mode: Literal["streaming"] = "streaming"
    stream_url: StreamUrl
    chunk_seconds: int = Field(default=30, strict=True, ge=CHUNK_SECONDS_MIN, le=CHUNK_SECONDS_MAX)
    language: Language = "es"


class RecordedAudioSource(SourceBase):
    type: Literal["recorded_audio"]
    mode: Literal["recorded"] = "recorded"
    feed_url: HttpUrl | None = None
    watch_dir: WatchDir | None = None
    poll_interval_s: int = Field(default=1800, strict=True, ge=RECORDED_AUDIO_MIN_POLL_S)

    @model_validator(mode="after")
    def _exactly_one_origin(self) -> "RecordedAudioSource":
        if (self.feed_url is None) == (self.watch_dir is None):
            raise ValueError("exactly one of feed_url or watch_dir is required")
        return self


Source = NewsRssSource | InstagramSource | LiveStreamSource | RecordedAudioSource
"""The union of concrete source types; use this in signatures instead of `SourceBase`."""

_SOURCE_SCHEMA = Annotated[Source, Field(discriminator="type")]
SOURCE_TYPES: tuple[str, ...] = ("news_rss", "instagram", "live_stream", "recorded_audio")

_ADAPTER: TypeAdapter[Source] = TypeAdapter(_SOURCE_SCHEMA)


_KNOWN_FIELDS = frozenset(
    name
    for model in (NewsRssSource, InstagramSource, LiveStreamSource, RecordedAudioSource)
    for name in model.model_fields
)
_UNKNOWN_FIELD_MESSAGE = "unknown field (not part of this source type)"


def _describe_errors(exc: _PydanticValidationError) -> tuple[str, tuple[str, ...]]:
    """Build a message and the KNOWN field names involved, without echoing user input.

    Unknown keys and unknown type tags are user-controlled (a mistyped key may well be a
    pasted secret), so they never reach the message or `fields`.
    """
    parts: dict[str, None] = {}
    fields: dict[str, None] = {}
    for err in exc.errors(include_url=False, include_input=False, include_context=False):
        loc = [str(p) for p in err["loc"] if p not in SOURCE_TYPES]
        kind = err["type"]
        if kind == "extra_forbidden":
            parts[f"<source>: {_UNKNOWN_FIELD_MESSAGE}"] = None
        elif kind == "union_tag_invalid":
            parts[f"type: unknown source type; expected one of {', '.join(SOURCE_TYPES)}"] = None
            fields["type"] = None
        else:
            parts[f"{'.'.join(loc) or '<source>'}: {err['msg']}"] = None
            if kind == "union_tag_not_found":
                fields["type"] = None
            elif loc and loc[0] in _KNOWN_FIELDS:
                fields[loc[0]] = None
    return "; ".join(parts), tuple(fields)


def parse_source(data: Any) -> Source:
    """Validate raw input into the right Source variant or raise `ValidationError`."""
    try:
        return _ADAPTER.validate_python(data)
    except _PydanticValidationError as exc:
        message, fields = _describe_errors(exc)
        raise ValidationError(message, fields) from None
