"""The gold service (application layer): labeling texts and measuring the cascade with them.

Labeling keeps texts a person decided on in the local database: the same text is the same
example, adding it again with the same label changes nothing and with another label is a
conflict. `export_text` writes the set as YAML (the shape `classify calibrate --file` reads).

`calibrate` runs the CPU cascade (the topic filter and the rules, see `classification.calibration`)
over labeled samples and reports; it is a measurement, never an action: it changes no verdict, no
setting, no file and sends nothing anywhere. When a judge is given (`--with-llm`, never the
default) the texts the cascade would escalate are put to the local language model, one at a time,
through the judge handed in; the answer replaces the cascade's for that text in the report only.
Three failures in a row (a model server that is down or slow) end the asking: the rest is scored by
the cascade alone, so a dead server costs three timeouts and not the whole set.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from social_tracker.classification.calibration import (
    CalibrationReport,
    Prediction,
    Sample,
    evaluate,
    predict,
)
from social_tracker.classification.domain import ClassifierSettings, Label, Stage
from social_tracker.classification.service import TopicConfigService
from social_tracker.classification.topics import TopicMatcher
from social_tracker.control.domain import validate_actor
from social_tracker.gold.domain import (
    MAX_GOLD_EXAMPLES,
    GoldExample,
    make_example,
)
from social_tracker.gold.ports import GoldRepository
from social_tracker.gold.yaml_io import dump_examples
from social_tracker.llm.domain import JudgeError, JudgeEvidence, JudgeResponse
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import ConflictError, NotFoundError, ValidationError
from social_tracker.shared.ids import is_valid_slug
from social_tracker.shared.uow import UnitOfWork

MAX_CONSECUTIVE_FAILURES = 3
DEFAULT_LIST_LIMIT = 100
MAX_RATIONALE_CHARS = 300

# What `calibrate` is given to ask the model: an id for the request (never the example's own),
# the text and the evidence of the rules. It answers, returns None when the model is busy, or
# raises `JudgeError` (`EscalationService.judge_text` is one).
SampleJudge = Callable[[str, str, JudgeEvidence], JudgeResponse | None]


@dataclass(frozen=True)
class AddResult:
    """The example a text is, and whether this call stored it (False: it was already there)."""

    example: GoldExample
    added: bool


@dataclass(frozen=True)
class CalibrationRun:
    """The report, and what the asking of the model came to: texts put to it, answered, failed
    and given up because another process held the model."""

    report: CalibrationReport
    asked: int = 0
    answered: int = 0
    failed: int = 0
    busy: int = 0


class GoldService:
    def __init__(
        self,
        repo: GoldRepository,
        topics: TopicConfigService,
        uow: UnitOfWork,
        clock: Clock,
        *,
        settings: ClassifierSettings | None = None,
        max_examples: int = MAX_GOLD_EXAMPLES,
    ) -> None:
        self._repo = repo
        self._topics = topics
        self._uow = uow
        self._clock = clock
        self._settings = settings if settings is not None else ClassifierSettings()
        self._max = max_examples

    # ---- labeling ----

    def add(
        self,
        text: str,
        label: Label | str,
        *,
        actor: str,
        note: str = "",
        credibility: int = 3,
        url: str | None = None,
    ) -> AddResult:
        """Label a text. Returns the stored example (the existing one when the text was there)."""
        example = make_example(
            text=text,
            label=label,
            note=note,
            credibility=credibility,
            url=url,
            added_at=self._clock.now(),
            actor=validate_actor(actor),
        )
        with self._uow.transaction():
            existing = self._repo.get(example.id)
            if existing is not None:
                if existing.label is not example.label:
                    raise ConflictError(
                        "this text is already in the gold set with another label; "
                        "remove it first to label it again"
                    )
                return AddResult(existing, added=False)
            if self._repo.count() >= self._max:
                raise ConflictError(f"the gold set is full ({self._max} examples)")
            self._repo.add(example)
        return AddResult(example, added=True)

    def get(self, example_id: str) -> GoldExample:
        found = self._repo.get(example_id) if is_valid_slug(example_id) else None
        if found is None:
            raise NotFoundError("gold example not found")
        return found

    def remove(self, example_id: str) -> None:
        if not is_valid_slug(example_id) or not self._repo.remove(example_id):
            raise NotFoundError("gold example not found")

    def count(self) -> int:
        return self._repo.count()

    def list(
        self, *, label: Label | None = None, limit: int = DEFAULT_LIST_LIMIT, offset: int = 0
    ) -> "list[GoldExample]":
        return self._repo.list(label=label, limit=limit, offset=offset)

    # ---- export ----

    def _all(self, label: Label | None = None) -> "list[GoldExample]":
        return self._repo.list(label=label, limit=self._max)

    def export_text(self, *, label: Label | None = None) -> str:
        """The set (or the examples with one label) as YAML, in the order they were added."""
        return dump_examples(self._all(label))

    def samples(self) -> tuple[Sample, ...]:
        """The whole local set as the samples a calibration takes."""
        return tuple(Sample(e.id, e.text, (e.label,), e.credibility, e.url) for e in self._all())

    # ---- measuring ----

    def calibrate(
        self,
        samples: Sequence[Sample],
        *,
        current_threshold: float,
        judge: SampleJudge | None = None,
    ) -> CalibrationRun:
        """Run the cascade over `samples` and report. With a `judge` the texts the cascade would
        escalate are put to the model. Nothing is stored or changed."""
        config = self._topics.current().config
        matcher = TopicMatcher(config)
        year = self._clock.now().year
        pairs: list[tuple[Sample, Prediction]] = []
        asked = answered = failed = busy = streak = 0
        for index, sample in enumerate(samples, start=1):
            prediction = predict(
                sample,
                matcher=matcher,
                config=config,
                settings=self._settings,
                reference_year=year,
            )
            if judge is not None and prediction.escalated and streak < MAX_CONSECUTIVE_FAILURES:
                asked += 1
                try:
                    answer = judge(f"gold_{index:05d}", sample.text, _evidence(sample, prediction))
                except (JudgeError, ValidationError):
                    failed += 1
                    streak += 1
                else:
                    streak = 0
                    if answer is None:
                        busy += 1
                    else:
                        answered += 1
                        prediction = replace(
                            prediction,
                            label=answer.label,
                            confidence=answer.confidence,
                            stage=Stage.LLM,
                            rationale=answer.rationale,
                            answered_by_llm=True,
                        )
            pairs.append((sample, prediction))
        report = evaluate(pairs, current_threshold=current_threshold)
        return CalibrationRun(report, asked, answered, failed, busy)


def _evidence(sample: Sample, prediction: Prediction) -> JudgeEvidence:
    return JudgeEvidence(
        prior_label=prediction.label,
        prior_confidence=prediction.confidence,
        prior_stage=prediction.stage,
        prior_rationale=prediction.rationale[:MAX_RATIONALE_CHARS],
        topics=tuple(t.value for t in prediction.topics),
        signals=prediction.signals,
        catalog_status="unavailable",
        source_credibility=sample.credibility,
    )
