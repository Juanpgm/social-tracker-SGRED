"""YAML form of a gold set: strict, bounded, and the same shape as the seed file of the repository.

`dump_examples` writes what `gold export` prints. `load_samples` reads that file back and also the
synthetic seed (`tests/data/gold_examples.yaml`) for `classify calibrate --file`: each example has
an `id`, a `text` and `expect` with a `label` (a name, or a list of acceptable names, the first
being the expectation) or `in_scope: false` (an off-topic text, expected to be out of scope);
`credibility` and `url` are optional and every other key is ignored. Loading goes through
`shared.strict_yaml` (no tags, anchors, merge keys, duplicate keys or second documents; capped in
events and depth); messages name an example by its position and never echo what the file says.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import yaml

from social_tracker.classification.calibration import Sample
from social_tracker.classification.domain import Label
from social_tracker.gold.domain import MAX_GOLD_EXAMPLES, MAX_GOLD_TEXT_CHARS, GoldExample
from social_tracker.items.domain import optional_url
from social_tracker.shared.errors import ValidationError
from social_tracker.shared.strict_yaml import PlainDumper, Rejected, parse_document

MAX_FILE_BYTES = 4 * 1024 * 1024
_MAX_EVENTS = 400_000
_MAX_DEPTH = 8
_HEADER = (
    "# Gold set exported by social-tracker: it may hold real text, do not commit it.\n"
    "# `social-tracker classify calibrate --file <this file>` measures the cascade against it.\n"
)


def dump_examples(examples: Sequence[GoldExample]) -> str:
    """The examples as YAML, in the order given (real characters, nothing aliased)."""
    rows = []
    for example in examples:
        row: dict[str, Any] = {"id": example.id, "kind": "labeled", "text": example.text}
        if example.note:
            row["note"] = example.note
        row["credibility"] = example.credibility
        if example.url:
            row["url"] = example.url
        row["expect"] = {
            "in_scope": example.label is not Label.OUT_OF_SCOPE,
            "label": example.label.value,
        }
        rows.append(row)
    body = yaml.dump(
        {"version": 1, "examples": rows},
        Dumper=PlainDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=MAX_GOLD_TEXT_CHARS,
    )
    return _HEADER + body


def _labels(expect: Any, position: int) -> tuple[Label, ...]:
    """The acceptable labels of an example; empty for one that only says it is in scope (the seed
    has some that only state whether the text should be escalated: nothing to score a label by)."""
    if not isinstance(expect, dict):
        raise ValidationError(f"example {position} needs an `expect` mapping")
    raw = expect.get("label")
    if raw is None:
        return (Label.OUT_OF_SCOPE,) if expect.get("in_scope") is False else ()
    names = raw if isinstance(raw, list) else [raw]
    if not names:
        raise ValidationError(f"example {position} has an empty list of labels")
    try:
        return tuple(Label(name) for name in names)
    except (ValueError, TypeError):
        raise ValidationError(f"example {position} has an unknown label") from None


def _sample(entry: Any, position: int) -> Sample | None:
    """The sample of one entry, None for an entry with nothing to score a label by."""
    if not isinstance(entry, dict):
        raise ValidationError(f"example {position} must be a mapping")
    sample_id, text = entry.get("id"), entry.get("text")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValidationError(f"example {position} needs an id")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(f"example {position} needs a text")
    if len(text) > MAX_GOLD_TEXT_CHARS:
        raise ValidationError(f"example {position} has a text longer than the cascade reads")
    credibility = entry.get("credibility", 3)
    url = entry.get("url")
    try:
        if url is not None:
            optional_url(url)
        labels = _labels(entry.get("expect"), position)
        if not labels:
            return None
        return Sample(sample_id, text, labels, credibility, url)
    except ValidationError as exc:
        if exc.message.startswith("example "):
            raise
        raise ValidationError(f"example {position} is not valid") from None


@dataclass(frozen=True)
class Loaded:
    """The labeled texts of a file and how many entries were left out for having no label."""

    samples: tuple[Sample, ...]
    unlabeled: int


def load_samples(text: str) -> tuple[Sample, ...]:
    """The labeled texts of a YAML document, or `ValidationError`."""
    return load_document(text).samples


def load_document(text: str) -> Loaded:
    """Like `load_samples`, also saying how many entries had no expected label."""
    if not isinstance(text, str):
        raise ValidationError("the gold file must be text")
    try:
        document: Any = parse_document(text, max_events=_MAX_EVENTS, max_depth=_MAX_DEPTH)
    except Rejected as exc:
        raise ValidationError(exc.message) from None
    if not isinstance(document, dict) or not isinstance(document.get("examples"), list):
        raise ValidationError("the gold file needs an `examples` list")
    entries = document["examples"]
    if len(entries) > MAX_GOLD_EXAMPLES:
        raise ValidationError(f"the gold file has more than {MAX_GOLD_EXAMPLES} examples")
    found = [_sample(entry, i) for i, entry in enumerate(entries, start=1)]
    samples = tuple(s for s in found if s is not None)
    if len({s.id for s in samples}) != len(samples):
        raise ValidationError("two examples of the gold file have the same id")
    return Loaded(samples, len(found) - len(samples))
