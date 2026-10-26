"""Source entity: a pydantic v2 discriminated union on `type`. Pure, no I/O.

Credentials never live in a source: sources point at a secret by name (`secret_ref`).
Every other unknown key is rejected by `extra="forbid"`; the model has no dict-typed
fields, so there is no nested place for a credential to hide and no key blacklist to evade.
"""

import re
import unicodedata
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
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


def normalize_name(name: str) -> str:
    """Canonical form used for case/whitespace-insensitive uniqueness."""
    return unicodedata.normalize("NFC", name.strip()).casefold()


def _slug(value: str) -> str:
    try:
        return validate_slug(value)
    except ValidationError as exc:
        raise ValueError(exc.message) from None


def _utc(value: datetime) -> datetime:
    try:
        return ensure_utc(value)
    except ValidationError as exc:
        raise ValueError(exc.message) from None


def _has_control_chars(value: str) -> bool:
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


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
    if any(ch.isspace() for ch in value) or _has_control_chars(value):
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
    if _has_control_chars(value):
        raise ValueError("path must not contain control characters")
    return value


# ---- field types ----------------------------------------------------------------------

Slug = Annotated[str, AfterValidator(_slug)]
Name = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=100)
]
Credibility = Annotated[int, Field(strict=True, ge=1, le=5)]
Timestamp = Annotated[datetime, AfterValidator(_utc)]
SecretRef = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_.:/-]{1,128}$")]
HttpUrl = Annotated[str, StringConstraints(strict=True), AfterValidator(_http_url)]
StreamUrl = Annotated[str, StringConstraints(strict=True), AfterValidator(_stream_url)]
WatchDir = Annotated[str, StringConstraints(strict=True), AfterValidator(_watch_dir)]
Keyword = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=100)
]
InstagramHandle = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9._]{1,30}$")]
Language = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=16)
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

    @property
    def normalized_name(self) -> str:
        return normalize_name(self.name)


class NewsRssSource(SourceBase):
    type: Literal["news_rss"]
    mode: Literal["recorded"] = "recorded"
    feed_url: HttpUrl
    poll_interval_s: int = Field(default=600, strict=True, ge=120)
    keywords: list[Keyword] | None = Field(default=None, max_length=50)


class InstagramSource(SourceBase):
    type: Literal["instagram"]
    mode: Literal["recorded"] = "recorded"
    handle: InstagramHandle
    secret_ref: SecretRef  # required for Instagram
    include_stories: bool = Field(default=True, strict=True)
    include_posts: bool = Field(default=True, strict=True)
    poll_interval_s: int = Field(default=1800, strict=True, ge=900)


class LiveStreamSource(SourceBase):
    type: Literal["live_stream"]
    mode: Literal["streaming"] = "streaming"
    stream_url: StreamUrl
    chunk_seconds: int = Field(default=30, strict=True, ge=10, le=60)
    language: Language = "es"


class RecordedAudioSource(SourceBase):
    type: Literal["recorded_audio"]
    mode: Literal["recorded"] = "recorded"
    feed_url: HttpUrl | None = None
    watch_dir: WatchDir | None = None
    poll_interval_s: int = Field(default=1800, strict=True, ge=1)

    @model_validator(mode="after")
    def _exactly_one_origin(self) -> "RecordedAudioSource":
        if (self.feed_url is None) == (self.watch_dir is None):
            raise ValueError("exactly one of feed_url or watch_dir is required")
        return self


Source = Annotated[
    NewsRssSource | InstagramSource | LiveStreamSource | RecordedAudioSource,
    Field(discriminator="type"),
]
SOURCE_TYPES: tuple[str, ...] = ("news_rss", "instagram", "live_stream", "recorded_audio")

_ADAPTER: TypeAdapter[Any] = TypeAdapter(Source)


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


def parse_source(data: Any) -> SourceBase:
    """Validate raw input into the right Source variant or raise `ValidationError`."""
    try:
        return _ADAPTER.validate_python(data)
    except _PydanticValidationError as exc:
        message, fields = _describe_errors(exc)
        raise ValidationError(message, fields) from None
