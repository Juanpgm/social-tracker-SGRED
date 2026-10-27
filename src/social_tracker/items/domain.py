"""Content items: the normalized record every collector produces. Pure, no I/O.

An item is one article, post, comment or transcript segment fetched from a source (CI-2). It
carries the source id AND a snapshot of the source name, so deleting a source never orphans or
deletes what was collected from it (SR-9). Item text is untrusted: nothing here interprets it.

Dedup keys (CI-3): `(source_id, external_id)` and `(source_id, content_hash)`; the hash covers the
title and text only, so the same story under another URL is the same content.

Retention (CI-10): raw items are kept for `DEFAULT_RETENTION_DAYS` days, counted from `fetched_at`
(when WE stored them, which a feed cannot forge). An item is purged only when it is STRICTLY
older than the retention: at exactly N days it is still kept, one microsecond later it is not.
Zero means "purge once classified" (nothing is time-based then).
"""

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError
from social_tracker.shared.ids import validate_slug

MAX_TEXT_BYTES = 100 * 1024  # CI-8: longer text is truncated (see `truncate_utf8`) and flagged
MAX_TITLE_CHARS = 1000
MAX_EXTERNAL_ID_CHARS = 512
MAX_URL_CHARS = 2048
MAX_AUTHOR_CHARS = 200
MAX_RAW_PAYLOAD_BYTES = 256 * 1024
DEFAULT_RETENTION_DAYS = 30
MAX_RETENTION_DAYS = 3650
MAX_SEARCH_TERMS = 16
MAX_SEARCH_TERM_CHARS = 64

_FLAG_RE = re.compile(r"[a-z_]{1,32}")
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_LANGUAGE_RE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*")
_MAX_LANGUAGE_CHARS = 35
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\n]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_ANY_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W_]+")


class ItemKind(StrEnum):
    POST = "post"
    COMMENT = "comment"
    ARTICLE = "article"
    SEGMENT = "segment"


class ItemStatus(StrEnum):
    """Pipeline stage of an item. Later stages only ever add values."""

    NEW = "new"
    CLASSIFIED = "classified"


# ---- text helpers ---------------------------------------------------------------------


def normalize_text(raw: str) -> str:
    """NFC text with line breaks unified, control characters and odd spaces turned into plain
    spaces, runs of spaces collapsed, at most one blank line in a row, and no outer whitespace.
    Accents, emoji and joiners are untouched. Idempotent."""
    text = raw.encode("utf-8", errors="replace").decode("utf-8")  # a lone surrogate becomes "?"
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_SPACE_RE.sub(" ", _CONTROL_RE.sub(" ", text))
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def truncate_utf8(text: str, max_bytes: int) -> tuple[str, bool]:
    """`text` cut to at most `max_bytes` bytes of UTF-8, never inside a character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def _canonical(text: str) -> str:
    return _ANY_SPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def content_hash(title: str, text: str) -> str:
    """SHA-256 of the title and text, ignoring case, whitespace and Unicode composition."""
    joined = _canonical(title) + "\n" + _canonical(text)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def item_id(source_id: str, external_id: str) -> str:
    digest = hashlib.sha256(f"{validate_slug(source_id)}\0{external_id}".encode()).hexdigest()
    return "itm_" + digest[:32]


def raw_json(payload: Mapping[str, Any]) -> str:
    """The one way a raw payload is written down: compact, key-sorted, real characters."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_bytes(payload: Mapping[str, Any]) -> int:
    return len(raw_json(payload).encode("utf-8"))


def bounded_raw_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """`payload` shrunk to `MAX_RAW_PAYLOAD_BYTES`: the longest text value is halved until the
    JSON fits. Deterministic. Returns the payload and whether anything was cut."""
    shrunk = dict(payload)
    truncated = False
    while _json_bytes(shrunk) > MAX_RAW_PAYLOAD_BYTES:
        key = max(
            (k for k, v in shrunk.items() if isinstance(v, str) and v),
            key=lambda k: (len(shrunk[k]), k),
            default=None,
        )
        if key is None:
            raise ValidationError("raw payload cannot be reduced to the size limit")
        shrunk[key] = shrunk[key][: len(shrunk[key]) // 2]
        truncated = True
    return shrunk, truncated


# ---- retention ------------------------------------------------------------------------


def validate_retention_days(days: Any) -> int:
    if isinstance(days, bool) or not isinstance(days, int) or not 0 <= days <= MAX_RETENTION_DAYS:
        raise ValidationError(
            f"retention days must be a whole number from 0 to {MAX_RETENTION_DAYS}"
        )
    return days


def retention_cutoff(now: datetime, days: int) -> datetime:
    """Items fetched strictly before this instant are expired. Not defined for zero days."""
    if validate_retention_days(days) == 0:
        raise ValidationError("a retention of 0 days has no time cutoff")
    return ensure_utc(now) - timedelta(days=days)


# ---- search ---------------------------------------------------------------------------


def plain_words(text: str) -> list[str]:
    """Lowercase words of `text` with accents removed and punctuation dropped: how both the
    search index and a search query see text."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return _WORD_RE.findall(plain)


def search_terms(query: Any) -> tuple[str, ...]:
    """The plain words of a user's search text, bounded in count and length. Operators (`AND`,
    `NEAR`, `*`, quotes...) are just words or nothing. This is the only way user text reaches the
    full-text index."""
    if not isinstance(query, str):
        raise ValidationError("the search text must be a string")
    return tuple(word[:MAX_SEARCH_TERM_CHARS] for word in plain_words(query)[:MAX_SEARCH_TERMS])


DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = 1000


def validate_limit(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIST_LIMIT:
        raise ValidationError(f"limit must be a whole number from 1 to {MAX_LIST_LIMIT}")
    return limit


# ---- entity ---------------------------------------------------------------------------


def _require_text(value: Any, what: str, max_chars: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > max_chars or (not value and not allow_empty):
        raise ValidationError(f"{what} must be text of at most {max_chars} characters")
    if "\x00" in value:
        raise ValidationError(f"{what} must not contain NUL characters")
    return value


def _optional_url(value: Any) -> str | None:
    if value is None:
        return None
    text = _require_text(value, "url", MAX_URL_CHARS)
    scheme, sep, rest = text.partition("://")
    authority = re.match(r"[^/?#]*", rest)
    userinfo = "@" in (authority.group() if authority else "")
    if (
        not sep
        or scheme.lower() not in ("http", "https")
        or not rest
        or userinfo
        or any(ch.isspace() or ord(ch) < 32 for ch in text)
    ):
        raise ValidationError("url must be an http(s) address without credentials or whitespace")
    return text


def _flags(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple | list) or not all(
        isinstance(f, str) and _FLAG_RE.fullmatch(f) for f in value
    ):
        raise ValidationError("flags must be lowercase names")
    return tuple(sorted(set(value)))


def _coerce[E: StrEnum](enum_cls: type[E], value: Any, what: str) -> E:
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        raise ValidationError(f"unknown item {what}") from None


@dataclass(frozen=True)
class Item:
    id: str
    source_id: str
    source_name: str
    kind: ItemKind
    external_id: str
    parent_id: str | None
    url: str | None
    title: str
    author: str | None
    text: str
    language: str | None
    published_at: datetime
    fetched_at: datetime
    content_hash: str
    flags: tuple[str, ...]
    status: ItemStatus
    raw_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        validate_slug(self.id)
        validate_slug(self.source_id)
        _require_text(self.source_name, "source name", 100)
        set_(self, "kind", _coerce(ItemKind, self.kind, "kind"))
        _require_text(self.external_id, "external id", MAX_EXTERNAL_ID_CHARS)
        if self.parent_id is not None:
            validate_slug(self.parent_id)
        set_(self, "url", _optional_url(self.url))
        _require_text(self.title, "title", MAX_TITLE_CHARS, allow_empty=True)
        if self.author is not None:
            _require_text(self.author, "author", MAX_AUTHOR_CHARS)
        _require_text(self.text, "text", MAX_TEXT_BYTES, allow_empty=True)
        try:
            text_bytes = len(self.text.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValidationError("text must be valid Unicode") from None
        if text_bytes > MAX_TEXT_BYTES:
            raise ValidationError(f"text must be at most {MAX_TEXT_BYTES} bytes")
        if self.language is not None and (
            not isinstance(self.language, str)
            or len(self.language) > _MAX_LANGUAGE_CHARS
            or not _LANGUAGE_RE.fullmatch(self.language)
        ):
            raise ValidationError("language must be a language tag")
        set_(self, "published_at", ensure_utc(self.published_at))
        set_(self, "fetched_at", ensure_utc(self.fetched_at))
        if not isinstance(self.content_hash, str) or not _HASH_RE.fullmatch(self.content_hash):
            raise ValidationError("content hash must be 64 lowercase hex digits")
        set_(self, "flags", _flags(self.flags))
        set_(self, "status", _coerce(ItemStatus, self.status, "status"))
        if not isinstance(self.raw_payload, Mapping):
            raise ValidationError("raw payload must be a mapping")
        try:
            size = _json_bytes(self.raw_payload)
            copied = json.loads(raw_json(self.raw_payload))
        except (TypeError, ValueError, UnicodeEncodeError):
            raise ValidationError("raw payload must be JSON") from None
        if size > MAX_RAW_PAYLOAD_BYTES:
            raise ValidationError(f"raw payload must be at most {MAX_RAW_PAYLOAD_BYTES} bytes")
        set_(self, "raw_payload", copied)


@dataclass(frozen=True)
class AddReport:
    """What storing a batch did: how many items were new and how many were already known."""

    new: int
    duplicate: int
