"""The source forms: which fields each type shows, how text becomes domain input and how domain
errors come back as inline messages.

The domain model is the only validator. This module turns form text into the values the domain
expects (numbers, booleans, lists) and passes anything it cannot convert through unchanged, so the
domain rejects it with its own message. Nothing here decides whether a value is acceptable.

A field marked `sensitive` is never written back into a re-rendered form after it failed
validation, and the secret reference is never written back at all: either may be a pasted
credential.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from social_tracker.control.domain import validate_reason
from social_tracker.shared.errors import ConflictError, ValidationError
from social_tracker.sources.domain import Source

ACTOR = "web:local"
DEFAULT_REASON = "via web UI"

_INT = re.compile(r"[+-]?[0-9]{1,18}")
_UNKNOWN_FIELD_PREFIX = "<source>: "
_VALUE_ERROR_PREFIX = "Value error, "
REF_FIELD = "secret_ref"
VERSION_FIELD = "updated_at"  # hidden on the edit form: the version the editor started from
URL_SCHEMES = frozenset({"http", "https", "rtmp", "icecast"})
QUERY_MASK = "***"

# How a blank input is sent: REQUIRED as an empty value (the domain rejects it), DEFAULT not at
# all (the stored or default value stays), CLEAR as null on an edit and not at all on an add.
REQUIRED, DEFAULT, CLEAR = "required", "default", "clear"

TYPE_LABELS = {
    "news_rss": "News RSS feed",
    "instagram": "Instagram account",
    "live_stream": "Live audio stream",
    "recorded_audio": "Recorded audio",
}


@dataclass(frozen=True)
class FormField:
    name: str
    label: str
    parse: str = "text"  # text | int | bool | lines
    blank: str = REQUIRED
    hint: str = ""
    sensitive: bool = False

    @property
    def required(self) -> bool:
        return self.blank == REQUIRED and self.parse != "bool"


ID_FIELD = FormField(
    "id", "Source id", hint="3 to 64 characters: a-z, 0-9 and _. It cannot change later."
)
NAME = FormField("name", "Name", hint="Up to 100 characters, unique ignoring case.")
CREDIBILITY = FormField("credibility", "Credibility", "int", hint="A whole number from 1 to 5.")
REASON_FIELD = FormField("reason", "Reason", hint="Kept in the audit log.")
POLL = FormField(
    "poll_interval_s", "Poll interval (seconds)", "int", DEFAULT, "Leave blank for the default."
)

TYPE_FIELDS: dict[str, tuple[FormField, ...]] = {
    "news_rss": (
        FormField(
            "feed_url",
            "Feed URL",
            hint="An http or https RSS or Atom address.",
            sensitive=True,
        ),
        POLL,
        FormField("keywords", "Keywords", "lines", CLEAR, "One per line. Leave blank for none."),
    ),
    "instagram": (
        FormField("handle", "Instagram handle", hint="Letters, digits, dots and underscores."),
        FormField(
            REF_FIELD,
            "Secret reference",
            hint="Where the login lives, as env:NAME, keyring:NAME or file:NAME. "
            "Never the password itself.",
            sensitive=True,
        ),
        FormField("include_stories", "Collect stories", "bool"),
        FormField("include_posts", "Collect posts", "bool"),
        POLL,
    ),
    "live_stream": (
        FormField(
            "stream_url",
            "Stream URL",
            hint="A direct audio stream: http, https, rtmp or icecast.",
            sensitive=True,
        ),
        FormField("chunk_seconds", "Chunk length (seconds)", "int", DEFAULT, "10 to 60."),
        FormField("language", "Language", "text", DEFAULT, "A language tag such as es or es-CO."),
    ),
    "recorded_audio": (
        FormField(
            "feed_url",
            "Feed URL",
            blank=CLEAR,
            hint="A podcast-style feed. Give this or a directory, not both.",
            sensitive=True,
        ),
        FormField(
            "watch_dir", "Watched directory", blank=CLEAR, hint="A local folder of audio files."
        ),
        POLL,
    ),
}
CHECKBOXES = {"include_stories", "include_posts"}
URL_FIELDS = frozenset({"feed_url", "stream_url"})
INITIAL_VALUES = {"credibility": "3", "reason": DEFAULT_REASON}
INITIAL_CHECKED = dict.fromkeys(CHECKBOXES, True)


def fields_for(source_type: str, *, editing: bool) -> tuple[FormField, ...]:
    head = (NAME, CREDIBILITY) if editing else (ID_FIELD, NAME, CREDIBILITY)
    return (*head, *TYPE_FIELDS[source_type])


def reason_of(form: Mapping[str, str]) -> str:
    """The operator's reason, or the default when the form carries none (row buttons)."""
    return form.get("reason", DEFAULT_REASON)


def _value(field: FormField, form: Mapping[str, str], *, editing: bool) -> tuple[bool, Any]:
    """`(send, value)` for one field of a submitted form."""
    if field.parse == "bool":
        return True, field.name in form
    raw = form.get(field.name, "").strip()
    if not raw:
        if field.blank == DEFAULT or (field.name == REF_FIELD and editing):
            return False, None  # the stored or default value stays
        if field.blank == CLEAR:
            return editing, None  # an edit empties the field; an add leaves it unset
    if field.parse == "int":
        return True, int(raw) if _INT.fullmatch(raw) else raw
    if field.parse == "lines":
        return True, [line.strip() for line in raw.splitlines() if line.strip()]
    return True, raw


def build_data(source_type: str, form: Mapping[str, str], *, editing: bool) -> dict[str, Any]:
    """Domain input from a submitted form. Only the fields the type offers are read."""
    data: dict[str, Any] = {}
    for field in fields_for(source_type, editing=editing):
        send, value = _value(field, form, editing=editing)
        if send:
            data[field.name] = value
    if not editing:
        data["type"] = source_type
    return data


@dataclass(frozen=True)
class FormErrors:
    """Inline messages by field plus the ones that belong to the whole form."""

    fields: dict[str, str]
    form: list[str]

    def __bool__(self) -> bool:
        return bool(self.fields or self.form)


NO_ERRORS = FormErrors({}, [])


def check_reason(form: Mapping[str, str]) -> FormErrors:
    try:
        validate_reason(reason_of(form))
    except ValidationError as exc:
        return FormErrors({"reason": exc.message}, [])
    return NO_ERRORS


def map_validation(exc: ValidationError, names: set[str]) -> FormErrors:
    """Split the domain message (`field: message; field: message`) over the form's fields."""
    fields: dict[str, str] = {}
    general: list[str] = []
    for part in exc.message.split("; "):
        location, sep, message = part.partition(": ")
        field = location.split(".")[0]
        if sep and field in names:
            fields.setdefault(field, message.removeprefix(_VALUE_ERROR_PREFIX))
        else:
            text = part.removeprefix(_UNKNOWN_FIELD_PREFIX).removeprefix(_VALUE_ERROR_PREFIX)
            general.append(text)
    return FormErrors(fields, general)


def map_conflict(exc: ConflictError) -> FormErrors:
    words = exc.message.split()
    if "name" in words:
        return FormErrors({"name": exc.message}, [])
    if "id" in words:
        return FormErrors({"id": exc.message}, [])
    return FormErrors({}, [exc.message])


def is_plain_url(value: str) -> bool:
    """True for an address that carries nothing worth hiding: a known scheme and a host, with no
    login, query string or fragment (any of which may hold a credential)."""
    try:
        parts = urlsplit(value)
        host = parts.hostname
    except ValueError:
        return False
    return (
        parts.scheme in URL_SCHEMES
        and bool(host)
        and "@" not in parts.netloc
        and not parts.query
        and not parts.fragment
    )


def mask_url(url: str) -> str:
    """The address as an edit form shows it: the query string (where tokens live) is replaced."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.query:
        return url
    netloc = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, netloc, parts.path, QUERY_MASK, ""))


def keep_masked_urls(source: Source, form: Mapping[str, str]) -> dict[str, str]:
    """A form field still holding the mask of the stored address means "unchanged": put the stored
    address back, so saving never replaces it with the mask."""
    kept = dict(form)
    for name in URL_FIELDS:
        stored = getattr(source, name, None)
        if stored and form.get(name, "").strip() == mask_url(stored):
            kept[name] = stored
    return kept


def without_unvalidated_secrets(source_type: str, form: Mapping[str, str]) -> dict[str, str]:
    """The form as typed, minus any sensitive value that was never validated (storage failed
    first): an address is kept only when it is plain, and the secret reference never."""
    safe = dict(form)
    safe.pop(REF_FIELD, None)
    for field in fields_for(source_type, editing=False):
        if field.sensitive and not (
            field.name in URL_FIELDS and is_plain_url(safe.get(field.name, ""))
        ):
            safe.pop(field.name, None)
    return safe


def parse_version(raw: str | None) -> datetime | None:
    """The `updated_at` a form was opened at, or None when it is missing, malformed or naive."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def values_from_source(source: Source) -> tuple[dict[str, str], dict[str, bool]]:
    """What an edit form starts with. The secret reference is deliberately absent."""
    values: dict[str, str] = {"reason": DEFAULT_REASON}
    checked: dict[str, bool] = {}
    for name, value in source.model_dump().items():
        if name == REF_FIELD or value is None:
            continue
        if isinstance(value, bool):
            if name in CHECKBOXES:
                checked[name] = value
        elif isinstance(value, tuple):
            values[name] = "\n".join(value)
        elif name in URL_FIELDS:
            values[name] = mask_url(str(value))
        else:
            values[name] = str(value)
    return values, checked


@dataclass(frozen=True)
class FieldView:
    spec: FormField
    value: str
    checked: bool
    error: str


def view_fields(
    source_type: str,
    values: Mapping[str, str],
    errors: Mapping[str, str],
    *,
    editing: bool,
    checked: Mapping[str, bool],
) -> list[FieldView]:
    """The fields as the template draws them: the value to put back (never the secret reference,
    never a sensitive value that failed validation) and the inline error."""
    views: list[FieldView] = []
    for spec in (*fields_for(source_type, editing=editing), REASON_FIELD):
        error = errors.get(spec.name, "")
        hidden = spec.name == REF_FIELD or (spec.sensitive and bool(error))
        value = "" if hidden else values.get(spec.name, "")
        views.append(FieldView(spec, value, checked.get(spec.name, False), error))
    return views
