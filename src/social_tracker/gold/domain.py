"""A labeled text of the gold set. Pure, no I/O.

The gold set is what a person decided a text IS (one of the six labels of the cascade), kept on
this machine to measure the cascade and to tune the alert threshold. It never leaves it and is
never committed: the texts may be real content collected from real sources.

An example is its text (normalized like an item text: control characters become spaces, runs of
white space collapse), its label, a short note, the credibility of the source it came from (a
prior, 1 to 5), an optional address, who added it and when. Its id is made from the text, so the
same text is the same example whoever adds it and however many times: adding it again with the
same label changes nothing, with another label is a conflict the person resolves.
"""

import hashlib
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from social_tracker.classification.domain import Label
from social_tracker.items.domain import normalize_text, optional_url
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError
from social_tracker.shared.text import has_control_chars, has_unsafe_chars

MAX_GOLD_TEXT_CHARS = 20_000  # what the cascade reads of a text (`max_text_chars`)
MAX_NOTE_CHARS = 500
MAX_ACTOR_CHARS = 100
MAX_GOLD_EXAMPLES = 5000
GOLD_ID_HEX = 16


def _canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def gold_id_for(text: str) -> str:
    """The id of the example a text is: the same words in the same case, however spaced."""
    digest = hashlib.sha256(_canonical(text).encode("utf-8", "replace")).hexdigest()
    return "gld_" + digest[:GOLD_ID_HEX]


def _label(value: Any) -> Label:
    try:
        return Label(value)
    except (ValueError, TypeError):
        raise ValidationError("unknown label") from None


def _plain(value: Any, what: str, low: int, high: int) -> str:
    if (
        not isinstance(value, str)
        or not low <= len(value) <= high
        or has_control_chars(value)
        or has_unsafe_chars(value)
    ):
        raise ValidationError(f"{what} must be {low} to {high} printable characters")
    return value


@dataclass(frozen=True)
class GoldExample:
    id: str
    text: str
    label: Label
    note: str
    credibility: int
    url: str | None
    added_at: datetime
    actor: str

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValidationError("the text of an example must not be empty")
        if len(self.text) > MAX_GOLD_TEXT_CHARS:
            raise ValidationError(
                f"the text of an example is at most {MAX_GOLD_TEXT_CHARS} characters"
            )
        try:
            self.text.encode("utf-8")
        except UnicodeEncodeError:
            raise ValidationError("the text of an example must be valid Unicode") from None
        if self.id != gold_id_for(self.text):
            raise ValidationError("the id of an example is made from its text")
        set_(self, "label", _label(self.label))
        _plain(self.note, "the note", 0, MAX_NOTE_CHARS)
        if (
            isinstance(self.credibility, bool)
            or not isinstance(self.credibility, int)
            or not 1 <= self.credibility <= 5
        ):
            raise ValidationError("credibility must be a whole number from 1 to 5")
        if self.url is not None:
            optional_url(self.url)
        set_(self, "added_at", ensure_utc(self.added_at))
        if not isinstance(self.actor, str) or not self.actor.strip():
            raise ValidationError("an example needs the person who added it")
        _plain(self.actor, "the actor", 1, MAX_ACTOR_CHARS)


def make_example(
    *,
    text: Any,
    label: Any,
    note: Any,
    credibility: Any,
    url: Any,
    added_at: datetime,
    actor: Any,
) -> GoldExample:
    """The example a text and a label make: the text is normalized first, the id follows."""
    if not isinstance(text, str):
        raise ValidationError("the text of an example must be text")
    cleaned = normalize_text(text)
    if len(cleaned) > MAX_GOLD_TEXT_CHARS:
        raise ValidationError(f"the text of an example is at most {MAX_GOLD_TEXT_CHARS} characters")
    return GoldExample(
        id=gold_id_for(cleaned) if cleaned.strip() else "gld_empty",
        text=cleaned,
        label=_label(label),
        note=note,
        credibility=credibility,
        url=url,
        added_at=added_at,
        actor=actor,
    )
