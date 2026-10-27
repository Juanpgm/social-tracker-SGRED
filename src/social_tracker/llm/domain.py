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
PROMPT_VERSION: Final = "judge-2"
JUDGE_SEED: Final = 0  # seed of the first attempt; a retry adds its attempt number
LOOPBACK_ADDRESS: Final = "127.0.0.1"

# The labels the model may answer with (RC-3 and `no_risk`); `unclassified` and `out_of_scope`
# are never a model's call.
JUDGE_LABELS: Final = (
    Label.UNINTENTIONAL_RUMOR,
    Label.DELIBERATE_DISINFORMATION,
    Label.RECYCLED_CONTENT,
    Label.NO_RISK,
)
MIN_PROMPT_TOKENS: Final = 2048  # room the instruction and a short text need next to the output
CHARS_PER_TOKEN: Final = 2.5  # a conservative estimate: real text is about 3 to 4
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
      `max_failures`: item-specific failures after which the item is parked.
    - `load_deadline_s`: wall-clock limit of a call that has to load the model first (it is not
      resident); a call to a resident model keeps `call_deadline_s`. `num_ctx`: the context
      window sent to the server; the prompt is cut to fit it.
    - `pass_budget_s`: wall-clock budget of one escalation pass (the item count bounds it too).
    - `parked_ttl_s`: a parked item re-enters the queue after this long (`llm requeue-parked`
      does it at once). `source_share`: the share of `queue_max` one source may hold when the
      queue overflows; the excess of a source over it is dropped first.
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
    load_deadline_s: float = 180.0
    num_ctx: int = 8192
    pass_budget_s: float = 120.0
    parked_ttl_s: int = 86_400
    source_share: float = 0.25

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
            and _number(self.load_deadline_s, 0.0, 1800.0)
            and self.load_deadline_s > 0
            and _whole(self.num_ctx, 2048, 262_144)
            and self.num_ctx - self.max_output_tokens >= MIN_PROMPT_TOKENS
            and _number(self.pass_budget_s, 0.0, 86_400.0)
            and self.pass_budget_s > 0
            and _whole(self.parked_ttl_s, 60, 2_592_000)
            and _number(self.source_share, 0.0, 1.0)
            and self.source_share > 0
        )
        if not valid:
            raise ValidationError("invalid language model settings")

    @property
    def endpoint(self) -> LoopbackEndpoint:
        return parse_loopback_url(self.base_url)

    @property
    def cold_deadline_s(self) -> float:
        """The wall-clock limit of a call that loads the model first."""
        return max(self.load_deadline_s, self.call_deadline_s)

    @property
    def longest_judgment_s(self) -> float:
        """The longest one judgment can take: a cold first call, then the one warm retry of an
        invalid answer. A lease must outlive it."""
        return self.cold_deadline_s + self.call_deadline_s

    @property
    def source_cap(self) -> int:
        """How many waiting items one source may hold when the queue overflows."""
        return max(1, int(self.queue_max * self.source_share))

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
    TIMEOUT = "timeout"  # the call, body included, outlived its deadline (model resident)
    LOAD_TIMEOUT = "load_timeout"  # the same, on a call that had to load the model first
    SERVER_ERROR = "server_error"  # the server answered 5xx
    BAD_RESPONSE = "bad_response"  # not the shape of an answer, too large, redirected, 4xx
    MODEL_NOT_FOUND = "model_not_found"  # no model to answer (also: it cannot be loaded)
    INVALID_OUTPUT = "invalid_output"  # the text of the model is not a valid verdict (RC-9)
    NOT_OLLAMA = "not_ollama"  # what listens at the address does not identify as Ollama


# Failures that say something about the SERVER and nothing about the item: they back the whole
# judge off and never count towards parking an item. A status of 400/401/403/429/5xx on the chat
# call (`status` is set) belongs here too: a refused, rate-limited or crashed server is no verdict
# on the item it was asked about. All of them happen before the model produced an answer, so the
# hourly reservation of the call is given back.
_JUDGE_LEVEL = frozenset(
    {
        JudgeErrorKind.UNREACHABLE,
        JudgeErrorKind.LOAD_TIMEOUT,
        JudgeErrorKind.SERVER_ERROR,
        JudgeErrorKind.MODEL_NOT_FOUND,
        JudgeErrorKind.NOT_OLLAMA,
    }
)
# The classes of an invalid answer, as fixed codes: a retry names the class to the model, never
# what the model wrote.
INVALID_PROBLEMS: Final = ("empty", "too_large", "not_json", "not_object", "schema", "cut")


class JudgeError(DomainError):
    """A judge that could not answer. `judge_level` failures are the server's (down, refusing,
    crashing, slow to load) and never count against the item. `item_specific` ones MAY be the
    fault of the item (a warm call that never finishes, a body that is not an answer): the
    service counts them against the item only when the judge has been answering other items.
    The message is a fixed phrase: never item text, never model output. `status` is the HTTP
    status of a refused call; `problem` the class of an invalid answer (`INVALID_PROBLEMS`)."""

    def __init__(
        self,
        kind: JudgeErrorKind,
        detail: str = "",
        *,
        raw_output: str = "",
        status: int | None = None,
        problem: str = "",
    ) -> None:
        super().__init__(detail or kind.value)
        self.kind = kind
        self.raw_output = raw_output[
            :MAX_RAW_OUTPUT_CHARS
        ]  # for the audit; not part of the message
        self.status = status
        self.problem = problem
        self.code = f"llm_{kind.value}"

    @property
    def judge_level(self) -> bool:
        return self.kind in _JUDGE_LEVEL or (
            self.kind is JudgeErrorKind.BAD_RESPONSE and self.status is not None
        )

    @property
    def item_specific(self) -> bool:
        return not self.judge_level and self.kind is not JudgeErrorKind.INVALID_OUTPUT

    @property
    def refundable(self) -> bool:
        """The call failed before the model gave anything: its reservation is given back."""
        return self.judge_level


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
    `text_truncated` says it was cut) and the evidence found so far. A retry of an invalid answer
    has `attempt` 1 and names the class of the problem (`previous_problem`), never the text."""

    item_id: str
    text: str
    text_truncated: bool
    evidence: JudgeEvidence
    attempt: int = 0
    previous_problem: str = ""

    def __post_init__(self) -> None:
        valid = (
            isinstance(self.item_id, str)
            and _SLUG_RE.fullmatch(self.item_id) is not None
            and isinstance(self.text, str)
            and isinstance(self.text_truncated, bool)
            and isinstance(self.evidence, JudgeEvidence)
            and _whole(self.attempt, 0, 3)
            and (self.previous_problem == "" or self.previous_problem in INVALID_PROBLEMS)
            and (self.attempt > 0 or self.previous_problem == "")
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

MAX_LLM_CONFIDENCE: Final = 0.95  # a model is never more sure than this
SUSPICIOUS_NO_RISK_CAP: Final = 0.7  # ... nor when it clears what the earlier stages suspected
STRONG_EVIDENCE: Final = 0.6  # a risk signal or prior at least this strong is not waved away
REQUEUED_PRIORITY_FLOOR: Final = 0.1  # a copy that has to be judged on its own queues last
_SUSPICIOUS_SIGNALS: Final = frozenset({"obfuscated_text"})
_REVIEW_NOTE: Final = " [needs review: earlier signals disagree]"
_RISK = frozenset(
    {Label.DELIBERATE_DISINFORMATION, Label.UNINTENTIONAL_RUMOR, Label.RECYCLED_CONTENT}
)


def llm_low_confidence(confidence: float, alert_min: float) -> bool:
    """RC-11 plus the floor of this stage: an LLM label under 0.5, or under the alert threshold,
    is applied but flagged `low_confidence`."""
    return confidence < LOW_CONFIDENCE_FLOOR or confidence < alert_min


def contradicts_prior(prior: Classification, answer: JudgeResponse) -> bool:
    """A `no_risk` from the model that clears something the earlier stages found suspicious: an
    obfuscated or injection-style text, a risk signal (or a risk verdict) of at least 0.6, or a
    claim the official catalog contradicts. A risk answer never contradicts: it only adds to the
    suspicion. Such a `no_risk` is applied but never final (see `judged`)."""
    if answer.label is not Label.NO_RISK:
        return False
    if prior.catalog_status == "mismatch":
        return True
    if prior.label in _RISK and prior.confidence >= STRONG_EVIDENCE:
        return True
    for signal in prior.signals:
        if signal.name in _SUSPICIOUS_SIGNALS or signal.name.startswith("injection"):
            return True
        if signal.name == "catalog_mismatch":
            return True
        if signal.label in _RISK and signal.weight >= STRONG_EVIDENCE:
            return True
    return False


def judged(
    prior: Classification, answer: JudgeResponse, alert_min: float, now: datetime | None = None
) -> Classification:
    """The verdict after a valid judgment: the label, confidence and rationale of the model,
    stage `llm`, out of the queue. Evidence, topics, catalog outcome, credibility used and the
    links are the earlier stages' and stay. The credibility of the source is never applied to a
    judged confidence.

    Guard rails on an answer that is valid but suspicious: the confidence is at most 0.95, and a
    `no_risk` that clears what the earlier stages found suspicious (`contradicts_prior`) is at
    most 0.7, flagged `low_confidence` and noted for review, so it is never taken as final."""
    confidence = min(answer.confidence, MAX_LLM_CONFIDENCE)
    rationale = answer.rationale
    flagged = False
    if contradicts_prior(prior, answer):
        confidence = min(confidence, SUSPICIOUS_NO_RISK_CAP)
        rationale = rationale[: MAX_RATIONALE_CHARS - len(_REVIEW_NOTE)].rstrip() + _REVIEW_NOTE
        flagged = True
    return replace(
        prior,
        label=answer.label,
        confidence=confidence,
        stage=Stage.LLM,
        rationale=rationale,
        low_confidence=flagged or llm_low_confidence(confidence, alert_min),
        needs_llm=False,
        priority=0.0,
        **_settlement("llm", now),
    )


def _settlement(reason: str, now: datetime | None) -> dict[str, object]:
    if now is None:
        return {}
    return {"settled_at": now, "settle_reason": reason}


def left_unjudged(
    prior: Classification, now: datetime | None = None, reason: str = "invalid"
) -> Classification:
    """The verdict of the earlier stage, out of the queue and flagged low confidence, because no
    judgment will come: RC-9 (the model answered twice with something that is not a valid verdict,
    so its judgment is `unclassified` and contributes nothing), an item parked after repeated
    failures, or one dropped from a full queue. `reason` is `invalid` or `parked`."""
    return replace(
        prior, needs_llm=False, low_confidence=True, priority=0.0, **_settlement(reason, now)
    )


def reused_by_duplicate(
    duplicate: Classification,
    original: Classification,
    alert_min: float,
    now: datetime | None = None,
) -> Classification:
    """An exact copy that was waiting for its original: it takes the judged label and confidence
    of the original, keeps its OWN source credibility (the prior it was recorded with) and says
    so. It was not judged itself, so its stage stays `dedup`. Only a copy whose text is identical
    to the original's may be settled this way (`text_identity`)."""
    note = f"duplicate of {original.item_id}: reused its judged {original.label.value} verdict"
    return replace(
        duplicate,
        label=original.label,
        confidence=original.confidence,
        rationale=note[:MAX_RATIONALE_CHARS],
        low_confidence=original.low_confidence
        or llm_low_confidence(original.confidence, alert_min),
        awaiting_original=False,
        **_settlement("duplicate", now),
    )


def released_duplicate(
    duplicate: Classification, now: datetime | None = None, reason: str = "duplicate"
) -> Classification:
    """The original left the queue without a verdict to copy (invalid answer, parked, dropped):
    the duplicate keeps the provisional verdict it already has, stops waiting and gets the same
    low-confidence treatment as the original (`left_unjudged`)."""
    return replace(
        duplicate, awaiting_original=False, low_confidence=True, **_settlement(reason, now)
    )


def requeued_copy(copy: Classification, *, near_of: str | None, priority: float) -> Classification:
    """A waiting copy that cannot take its original's verdict (its text differs by a character:
    punctuation, a look-alike letter, another address) or whose original is gone: it becomes a
    normal queued item judged on its own. The link to the original stays as a near copy, which is
    only a prior for the judge; `None` drops it."""
    return replace(
        copy,
        duplicate_of=None,
        near_of=near_of,
        awaiting_original=False,
        needs_llm=True,
        priority=max(priority, REQUEUED_PRIORITY_FLOOR),
        settled_at=None,
        settle_reason=None,
    )


# ---- what the repositories keep ----

JUDGMENT_OUTCOMES: Final = ("judged", "invalid", "reused")
ESCALATION_STATES: Final = ("escalation_failed", "parked", "dropped")
_ERROR_CODE_RE = re.compile(r"[a-z0-9_]{1,64}")


def _aware(value: object, what: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{what} must be a timezone-aware time")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class QueueEntry:
    """An item waiting for the judge, in the order it is served."""

    item_id: str
    priority: float
    classified_at: datetime


@dataclass(frozen=True)
class Judgment:
    """What is kept of one judgment, for the audit and for reproducibility. `outcome` is
    `judged` (a valid answer), `invalid` (RC-9: two invalid answers; no label) or `reused` (an
    exact copy that took the answer of its original; `reused_from` names it, and the label is
    absent when that original was invalid). The prior columns are the verdict the earlier stage
    had given. `raw_output` is truncated model text, never logged."""

    item_id: str
    outcome: str
    label: str | None
    confidence: float | None
    rationale: str | None
    evidence: tuple[str, ...]
    llm_model: str | None
    prompt_version: str
    raw_output: str | None
    prior_label: str
    prior_confidence: float
    prior_stage: str
    reused_from: str | None
    judged_at: datetime

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        self._check_shape()
        if self.outcome == "judged" and None in (
            self.label,
            self.confidence,
            self.rationale,
            self.llm_model,
        ):
            raise ValidationError("a judged answer has a label, confidence, rationale and model")
        if self.outcome == "invalid" and self.label is not None:
            raise ValidationError("an invalid answer has no label")
        if (self.outcome == "reused") != (self.reused_from is not None):
            raise ValidationError("only a reused judgment names its original")
        set_(self, "judged_at", _aware(self.judged_at, "the judgment time"))

    def _check_shape(self) -> None:
        if not isinstance(self.item_id, str) or _SLUG_RE.fullmatch(self.item_id) is None:
            raise ValidationError("a judgment needs an item id")
        if self.outcome not in JUDGMENT_OUTCOMES:
            raise ValidationError("unknown judgment outcome")
        if self.label is not None and self.label not in {x.value for x in JUDGE_LABELS}:
            raise ValidationError("unknown judged label")
        if self.confidence is not None and not _number(self.confidence, 0.0, 1.0):
            raise ValidationError("the judged confidence must be from 0 to 1")
        if self.rationale is not None:
            _text(self.rationale, 1, MAX_RATIONALE_CHARS, "the rationale")
        if not isinstance(self.evidence, tuple) or len(self.evidence) > MAX_EVIDENCE_ITEMS:
            raise ValidationError("the evidence is a short tuple of strings")
        for entry in self.evidence:
            _text(entry, 1, MAX_EVIDENCE_CHARS, "an evidence string")
        if self.llm_model is not None:
            _model_name(self.llm_model)
        if not isinstance(self.prompt_version, str) or not 1 <= len(self.prompt_version) <= 64:
            raise ValidationError("the prompt version must be short text")
        if self.raw_output is not None and (
            not isinstance(self.raw_output, str) or len(self.raw_output) > MAX_RAW_OUTPUT_CHARS
        ):
            raise ValidationError("the raw output is kept truncated")
        if self.prior_label not in {x.value for x in Label}:
            raise ValidationError("unknown prior label")
        if not _number(self.prior_confidence, 0.0, 1.0):
            raise ValidationError("the prior confidence must be from 0 to 1")
        if self.prior_stage not in {x.value for x in Stage}:
            raise ValidationError("unknown prior stage")
        if self.reused_from is not None and (
            not isinstance(self.reused_from, str)
            or _SLUG_RE.fullmatch(self.reused_from) is None
            or self.reused_from == self.item_id
        ):
            raise ValidationError("a reused judgment names another item")


@dataclass(frozen=True)
class EscalationState:
    """Bookkeeping of an item that did not get a judgment (yet): `escalation_failed` (retry after
    `next_attempt_at`), `parked` (gave up for good) or `dropped` (pushed out of a full queue).
    `attempts` counts every failure in a row (it sets the backoff), `item_failures` those that may
    be the fault of the item (it sets the parking). `last_error` is a fixed code, never text."""

    item_id: str
    state: str
    attempts: int
    next_attempt_at: datetime | None
    last_error: str | None
    updated_at: datetime
    item_failures: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or _SLUG_RE.fullmatch(self.item_id) is None:
            raise ValidationError("a state needs an item id")
        if not _whole(self.item_failures, 0, 1_000_000):
            raise ValidationError("item failures must be a whole number")
        if self.state not in ESCALATION_STATES:
            raise ValidationError("unknown escalation state")
        if not _whole(self.attempts, 0, 1_000_000):
            raise ValidationError("attempts must be a whole number")
        if self.last_error is not None and (
            not isinstance(self.last_error, str)
            or _ERROR_CODE_RE.fullmatch(self.last_error) is None
        ):
            raise ValidationError("the last error is a short code")
        if self.next_attempt_at is not None:
            object.__setattr__(
                self, "next_attempt_at", _aware(self.next_attempt_at, "the next attempt")
            )
        object.__setattr__(self, "updated_at", _aware(self.updated_at, "the update time"))


@dataclass(frozen=True)
class EscalationCounts:
    judged: int
    invalid: int
    reused: int
    failed_waiting: int
    parked: int
    dropped: int
    queued: int
