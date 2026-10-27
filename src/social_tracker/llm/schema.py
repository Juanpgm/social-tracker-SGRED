"""The strict schema of what the model may answer (RC-9, RC-12). No I/O.

The text of the model is untrusted: it is parsed as JSON and validated against a closed schema
(four labels, a finite confidence from 0 to 1, a rationale, up to five evidence strings, nothing
else). Prose around the JSON, a code fence, a second object, a duplicate key, an unknown field,
a wrong type, a non-finite number or a truncated document is `invalid_output`. The one thing that
is repaired and not refused is the cosmetic: the rationale and the evidence strings are cleaned of
hidden characters and capped, because they are only ever shown, never obeyed.
"""

import json
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from social_tracker.llm.domain import (
    FALLBACK_RATIONALE,
    JUDGE_LABELS,
    MAX_EVIDENCE_CHARS,
    MAX_EVIDENCE_ITEMS,
    JudgeError,
    JudgeErrorKind,
    JudgeResponse,
    sanitize_text,
)

MAX_OUTPUT_CHARS: Final = 65_536
_MAX_RATIONALE_INPUT: Final = 2000  # longer is not a rationale; up to here it is capped to 300
_MAX_EVIDENCE_INPUT: Final = 1000
RATIONALE_CHARS: Final = 300

JUDGE_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": [label.value for label in JUDGE_LABELS]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": RATIONALE_CHARS},
        "evidence": {
            "type": "array",
            "items": {"type": "string", "maxLength": MAX_EVIDENCE_CHARS},
            "maxItems": MAX_EVIDENCE_ITEMS,
        },
    },
    "required": ["label", "confidence", "rationale", "evidence"],
    "additionalProperties": False,
}

_LabelName = Literal[
    "unintentional_rumor", "deliberate_disinformation", "recycled_content", "no_risk"
]


_Evidence = Annotated[str, Field(max_length=_MAX_EVIDENCE_INPUT)]


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    label: _LabelName
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    rationale: str = Field(min_length=1, max_length=_MAX_RATIONALE_INPUT)
    evidence: list[_Evidence] = Field(max_length=MAX_EVIDENCE_ITEMS)


def _invalid(why: str, raw: object = "") -> JudgeError:
    return JudgeError(
        JudgeErrorKind.INVALID_OUTPUT, why, raw_output=raw if isinstance(raw, str) else ""
    )


def _no_constant(name: str) -> Any:
    raise ValueError(name)


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate key")
    return dict(pairs)


def parse_judge_output(raw: object, *, model: str) -> JudgeResponse:
    """The verdict in `raw`, or `JudgeError(INVALID_OUTPUT)`. The message names the kind of
    problem and never repeats what the model wrote."""
    if not isinstance(raw, str) or not raw.strip():
        raise _invalid("empty or non-text output", raw)
    if len(raw) > MAX_OUTPUT_CHARS:
        raise _invalid("output too large", raw)
    try:
        document = json.loads(raw, object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    except (ValueError, RecursionError):
        raise _invalid("not a single JSON document", raw) from None
    if not isinstance(document, dict):
        raise _invalid("not a JSON object", raw)
    try:
        verdict = _Verdict.model_validate(document)
    except ValidationError:
        raise _invalid("does not match the schema", raw) from None
    rationale = sanitize_text(verdict.rationale, RATIONALE_CHARS) or FALLBACK_RATIONALE
    evidence = tuple(
        cleaned
        for entry in verdict.evidence
        if (cleaned := sanitize_text(entry, MAX_EVIDENCE_CHARS))
    )
    return JudgeResponse(
        label=verdict.label,  # type: ignore[arg-type]
        confidence=verdict.confidence,
        rationale=rationale,
        evidence=evidence,
        model=model,
        raw_output=raw,
    )
