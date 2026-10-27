"""Vocabulary of the LLM escalation (RC-7, RC-9, RC-12). Pure, no I/O.

The language model is the last stage of the cascade. It is asked only about an item the CPU stages
could not settle, it sees the item text as untrusted data next to the structured evidence found so
far, and all it can ever change is a label, a confidence and a short rationale: the answer is
validated against a strict schema (see `schema`) and anything else is invalid.

Network policy: the model runs on this machine. The only plain-http address the application
accepts is the loopback one (`parse_loopback_url`), and the item text is sent nowhere else.
"""

import math
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Final

from social_tracker.classification.domain import (
    MAX_RATIONALE_CHARS,
    Classification,
    Label,
    Signal,
    Stage,
)
from social_tracker.shared.errors import DomainError, ValidationError

DEFAULT_BASE_URL: Final = "http://127.0.0.1:11434"
DEFAULT_MODEL: Final = "st-gemma4-e4b"
DEFAULT_FALLBACK_MODEL: Final = "st-qwen35-4b"
PROMPT_VERSION: Final = "judge-1"
LOOPBACK_ADDRESS: Final = "127.0.0.1"

# The labels the model may answer with (RC-3 and `no_risk`); `unclassified` and `out_of_scope`
# are never a model's call.
JUDGE_LABELS: Final = (
    Label.UNINTENTIONAL_RUMOR,
    Label.DELIBERATE_DISINFORMATION,
    Label.RECYCLED_CONTENT,
    Label.NO_RISK,
)
MAX_EVIDENCE_ITEMS: Final = 5
MAX_EVIDENCE_CHARS: Final = 200
MAX_RAW_OUTPUT_CHARS: Final = 2048
LOW_CONFIDENCE_FLOOR: Final = 0.5  # an LLM label under this is always flagged low_confidence
FALLBACK_RATIONALE: Final = "judged by the language model without a stated reason"

_URL_RE = re.compile(r"(?i)http://(127\.0\.0\.1|localhost)(?::([0-9]{1,5}))?/?")
_MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}")


def _number(value: object, low: float, high: float) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and low <= value <= high
    )


def _whole(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


@dataclass(frozen=True)
class LoopbackEndpoint:
    """A validated address of this machine. `connect_host` is ALWAYS the literal IPv4 loopback
    address: a name such as `localhost` is never looked up, so it cannot be made to point
    elsewhere."""

    host: str
    port: int

    @property
    def connect_host(self) -> str:
        return LOOPBACK_ADDRESS

    @property
    def url(self) -> str:
        return f"http://{self.connect_host}:{self.port}"


def parse_loopback_url(value: object) -> LoopbackEndpoint:
    """`http://127.0.0.1[:port]` or `http://localhost[:port]` and nothing else: no other scheme,
    host, credentials, path, query or fragment. Raises `ValidationError`."""
    match = _URL_RE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValidationError("the language model address must be http://127.0.0.1 or localhost")
    port = int(match.group(2)) if match.group(2) else 80
    if not 1 <= port <= 65535:
        raise ValidationError("the language model port must be 1..65535")
    return LoopbackEndpoint(match.group(1).lower(), port)


def _model_name(value: object, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not _MODEL_RE.fullmatch(value):
        raise ValidationError("a model name is 1..100 letters, digits or . _ : / -")


@dataclass(frozen=True)
class LlmSettings:
    """What the escalation may be tuned by.

    - `base_url`: loopback only. `model`/`fallback_model`: tried in that order when the first is
      not installed or cannot be loaded (`None`: no fallback).
    - `keep_alive_s`: how long the server keeps the model loaded after a call (the GPU is freed
      when it expires). `call_deadline_s`: wall-clock limit of one call, body included.
    - `hourly_cap`: judgments started per UTC hour. `queue_max`: items waiting for a judgment;
      the oldest beyond it are dropped and counted. `batch_limit`: items judged per worker pass.
    - `max_prompt_chars`: item text longer than this is read at both ends and flagged.
      `max_response_bytes`: a larger answer is refused. `max_output_tokens`: generation cap.
    - `retry_base_s`/`retry_cap_s`: backoff after a failed call, doubling per failure.
      `max_failures`: item-specific failures after which the item is parked for good.
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    fallback_model: str | None = DEFAULT_FALLBACK_MODEL
    keep_alive_s: int = 300
    call_deadline_s: float = 60.0
    hourly_cap: int = 200
    queue_max: int = 2000
    batch_limit: int = 20
    max_prompt_chars: int = 6000
    max_response_bytes: int = 262_144
    max_output_tokens: int = 512
    retry_base_s: int = 60
    retry_cap_s: int = 3600
    max_failures: int = 3

    def __post_init__(self) -> None:
        parse_loopback_url(self.base_url)
        _model_name(self.model)
        _model_name(self.fallback_model, optional=True)
        valid = (
            _whole(self.keep_alive_s, 1, 86_400)
            and _number(self.call_deadline_s, 0.0, 600.0)
            and self.call_deadline_s > 0
            and _whole(self.hourly_cap, 1, 100_000)
            and _whole(self.queue_max, 1, 1_000_000)
            and _whole(self.batch_limit, 1, 1000)
            and _whole(self.max_prompt_chars, 100, 100_000)
            and _whole(self.max_response_bytes, 1024, 16_777_216)
            and _whole(self.max_output_tokens, 1, 8192)
            and _whole(self.retry_base_s, 1, 86_400)
            and _whole(self.retry_cap_s, 1, 604_800)
            and self.retry_cap_s >= self.retry_base_s
            and _whole(self.max_failures, 1, 100)
        )
        if not valid:
            raise ValidationError("invalid language model settings")

    @property
    def endpoint(self) -> LoopbackEndpoint:
        return parse_loopback_url(self.base_url)

    @property
    def models(self) -> tuple[str, ...]:
        """The models to try, in order."""
        if self.fallback_model and self.fallback_model != self.model:
            return (self.model, self.fallback_model)
        return (self.model,)


# ---- sanitising ----

_HIDDEN = frozenset({"Cf", "Cs", "Zl", "Zp"})


def _blankable(ch: str) -> bool:
    return ord(ch) < 32 or 127 <= ord(ch) < 160


def _visible(value: str, *, keep_newlines: bool) -> str:
    """`value` without hidden characters (format, surrogates, line and paragraph separators) and
    with every control character turned into a blank (a newline stays one if asked)."""
    out: list[str] = []
    for ch in value.replace("\r\n", "\n").replace("\r", "\n"):
        if ch == "\n" and keep_newlines:
            out.append(ch)
        elif unicodedata.category(ch) in _HIDDEN:
            continue
        elif _blankable(ch):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def sanitize_text(value: str, limit: int) -> str:
    """Text from the model made safe to store and show: hidden characters removed, controls and
    runs of whitespace collapsed to single blanks, at most `limit` characters, no trailing blank."""
    return " ".join(_visible(value, keep_newlines=False).split())[:limit].rstrip()


def prompt_text(title: str, text: str, max_chars: int) -> tuple[str, bool]:
    """Title and text joined, cleaned (nothing hidden, no control character but the newline) and
    bounded: text over `max_chars` is read at both ends and flagged. Deterministic."""
    joined = _visible(f"{title}\n{text}" if title else text, keep_newlines=True)
    if len(joined) <= max_chars:
        return joined, False
    head = max_chars // 2
    tail = max_chars - head
    return f"{joined[:head]}\n{joined[len(joined) - tail :]}", True


# ---- errors ----


class JudgeErrorKind(StrEnum):
    UNREACHABLE = "unreachable"  # nothing is listening, or the connection could not be made
    TIMEOUT = "timeout"  # the call, body included, outlived its deadline
    SERVER_ERROR = "server_error"  # the server answered 5xx
    BAD_RESPONSE = "bad_response"  # not the shape of an answer, too large, redirected, truncated
    MODEL_NOT_FOUND = "model_not_found"  # no model to answer (also: it cannot be loaded)
    INVALID_OUTPUT = "invalid_output"  # the text of the model is not a valid verdict (RC-9)


_ITEM_SPECIFIC = frozenset(
    {JudgeErrorKind.TIMEOUT, JudgeErrorKind.SERVER_ERROR, JudgeErrorKind.BAD_RESPONSE}
)


class JudgeError(DomainError):
    """A judge that could not answer. `item_specific` failures may be the fault of the item (it
    makes the model crash or never finish) and count towards parking it; the others (nothing
    listening, no model) say nothing about the item. The message is a fixed phrase: never item
    text, never model output."""

    def __init__(self, kind: JudgeErrorKind, detail: str = "") -> None:
        super().__init__(detail or kind.value)
        self.kind = kind
        self.code = f"llm_{kind.value}"

    @property
    def item_specific(self) -> bool:
        return self.kind in _ITEM_SPECIFIC


# ---- request and response ----


def _text(value: object, low: int, high: int, what: str) -> str:
    if (
        not isinstance(value, str)
        or not low <= len(value) <= high
        or _visible(value, keep_newlines=False) != value
        or (low > 0 and not value.strip())
    ):
        raise ValidationError(f"{what} must be {low} to {high} printable characters")
    return value


_STATUS_RE = re.compile(r"[a-z_]{1,32}")
_SLUG_RE = re.compile(r"[a-z0-9_]{3,64}")


@dataclass(frozen=True)
class JudgeEvidence:
    """What the earlier stages found, as structured data for the prompt: no item text.
    `source_credibility` (1..5) is a prior, never a reason for a label."""

    prior_label: Label
    prior_confidence: float
    prior_stage: Stage
    prior_rationale: str
    topics: tuple[str, ...]
    signals: tuple[Signal, ...]
    catalog_status: str
    source_credibility: int
    near_copy: bool = False
    recycled_before: bool = False

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "prior_label", Label(self.prior_label))
        set_(self, "prior_stage", Stage(self.prior_stage))
        valid = (
            _number(self.prior_confidence, 0.0, 1.0)
            and isinstance(self.prior_rationale, str)
            and isinstance(self.topics, tuple)
            and all(isinstance(t, str) for t in self.topics)
            and isinstance(self.signals, tuple)
            and all(isinstance(s, Signal) for s in self.signals)
            and isinstance(self.catalog_status, str)
            and _STATUS_RE.fullmatch(self.catalog_status) is not None
            and _whole(self.source_credibility, 1, 5)
            and isinstance(self.near_copy, bool)
            and isinstance(self.recycled_before, bool)
        )
        if not valid:
            raise ValidationError("invalid evidence for the language model")


@dataclass(frozen=True)
class JudgeRequest:
    """One question for the judge: the item text (untrusted data, already cleaned and bounded,
    `text_truncated` says it was cut) and the evidence found so far."""

    item_id: str
    text: str
    text_truncated: bool
    evidence: JudgeEvidence

    def __post_init__(self) -> None:
        valid = (
            isinstance(self.item_id, str)
            and _SLUG_RE.fullmatch(self.item_id) is not None
            and isinstance(self.text, str)
            and isinstance(self.text_truncated, bool)
            and isinstance(self.evidence, JudgeEvidence)
        )
        if not valid:
            raise ValidationError("invalid request for the language model")


@dataclass(frozen=True)
class JudgeResponse:
    """A valid verdict of the judge: one of the four labels, a confidence in [0, 1], a short
    printable rationale and up to five short evidence strings. `model` is who answered and
    `raw_output` what it said, truncated, kept for the audit only."""

    label: Label
    confidence: float
    rationale: str
    evidence: tuple[str, ...] = field(default=())
    model: str = ""
    raw_output: str = ""

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        try:
            label = Label(self.label)
        except ValueError:
            raise ValidationError("the judge gave an unknown label") from None
        if label not in JUDGE_LABELS:
            raise ValidationError("the judge may only give one of four labels")
        set_(self, "label", label)
        if not _number(self.confidence, 0.0, 1.0):
            raise ValidationError("the confidence of the judge must be a number from 0 to 1")
        _text(self.rationale, 1, MAX_RATIONALE_CHARS, "the rationale")
        if not isinstance(self.evidence, tuple) or len(self.evidence) > MAX_EVIDENCE_ITEMS:
            raise ValidationError(f"at most {MAX_EVIDENCE_ITEMS} evidence strings")
        for entry in self.evidence:
            _text(entry, 1, MAX_EVIDENCE_CHARS, "an evidence string")
        if not isinstance(self.model, str) or not isinstance(self.raw_output, str):
            raise ValidationError("model and raw output must be text")
        set_(self, "raw_output", self.raw_output[:MAX_RAW_OUTPUT_CHARS])


def evidence_of(verdict: Classification) -> JudgeEvidence:
    """The structured evidence for the prompt, taken from the stored verdict."""
    return JudgeEvidence(
        prior_label=verdict.label,
        prior_confidence=verdict.confidence,
        prior_stage=verdict.stage,
        prior_rationale=verdict.rationale,
        topics=tuple(t.value for t in verdict.topics),
        signals=verdict.signals,
        catalog_status=verdict.catalog_status,
        source_credibility=verdict.source_credibility_used,
        near_copy=verdict.near_of is not None,
        recycled_before=verdict.recycled_of is not None,
    )


# ---- time ----


def backoff_delay(failures: int, base_s: int, cap_s: int) -> timedelta:
    """How long to wait after the `failures`-th consecutive failure: `base_s` doubling, capped."""
    if not _whole(failures, 1, 10**9):
        raise ValidationError("a backoff needs at least one failure")
    exponent = min(failures - 1, 40)
    return timedelta(seconds=min(cap_s, base_s * 2**exponent))


def hour_start(moment: datetime) -> datetime:
    """The UTC hour a moment falls in: the unit of the hourly budget."""
    if not isinstance(moment, datetime) or moment.tzinfo is None or moment.utcoffset() is None:
        raise ValidationError("a budget hour needs a timezone-aware moment")
    return moment.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


# ---- how a judgment lands ----


def llm_low_confidence(confidence: float, alert_min: float) -> bool:
    """RC-11 plus the floor of this stage: an LLM label under 0.5, or under the alert threshold,
    is applied but flagged `low_confidence`."""
    return confidence < LOW_CONFIDENCE_FLOOR or confidence < alert_min


def judged(prior: Classification, answer: JudgeResponse, alert_min: float) -> Classification:
    """The verdict after a valid judgment: the label, confidence and rationale of the model,
    stage `llm`, out of the queue. Evidence, topics, catalog outcome, credibility used and the
    links are the earlier stages' and stay. The credibility of the source is never applied to a
    judged confidence."""
    return replace(
        prior,
        label=answer.label,
        confidence=answer.confidence,
        stage=Stage.LLM,
        rationale=answer.rationale,
        low_confidence=llm_low_confidence(answer.confidence, alert_min),
        needs_llm=False,
        priority=0.0,
    )


def kept_after_invalid(prior: Classification) -> Classification:
    """RC-9: the model answered twice with something that is not a valid verdict. The judgment
    contributes nothing (it is `unclassified`): the verdict of the earlier stage stays as it was,
    out of the queue and flagged low confidence."""
    return replace(prior, needs_llm=False, low_confidence=True, priority=0.0)


def reused_by_duplicate(
    duplicate: Classification, original: Classification, alert_min: float
) -> Classification:
    """An exact copy that was waiting for its original: it takes the judged label and confidence
    of the original, keeps its OWN source credibility (the prior it was recorded with) and says
    so. It was not judged itself, so its stage stays `dedup`."""
    note = f"duplicate of {original.item_id}: reused its judged {original.label.value} verdict"
    return replace(
        duplicate,
        label=original.label,
        confidence=original.confidence,
        rationale=note[:MAX_RATIONALE_CHARS],
        low_confidence=llm_low_confidence(original.confidence, alert_min),
        awaiting_original=False,
    )


def released_duplicate(duplicate: Classification) -> Classification:
    """The original left the queue without a verdict to copy: the duplicate keeps the provisional
    verdict it already has and stops waiting."""
    return replace(duplicate, awaiting_original=False)
