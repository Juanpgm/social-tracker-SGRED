"""A world around the escalation service: real services and repositories (in memory or SQLite)
and a scripted judge that never touches a network."""

import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from social_tracker.classification.domain import Classification, ClassifierSettings
from social_tracker.items.domain import Item, ItemStatus
from social_tracker.llm.domain import (
    PROMPT_VERSION,
    JudgeError,
    JudgeRequest,
    JudgeResponse,
    Judgment,
    LlmSettings,
)
from social_tracker.llm.service import EscalationService
from social_tracker.persistence.llm_repo import SqliteEscalationRepository
from social_tracker.shared.uow import UnitOfWork
from tests.classification.harness import World, memory_world, sqlite_world
from tests.fakes.escalation_repository import InMemoryEscalationRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.llm.support import response, verdict


class ScriptedJudge:
    """Answers from a script (responses, errors, or callables taking the request), records every
    request, and counts how many calls are in flight at once."""

    def __init__(self, *script: Any) -> None:
        self.script: deque[Any] = deque(script)
        self.default: Any = None
        self.requests: list[JudgeRequest] = []
        self.on_call: Callable[[JudgeRequest], None] | None = None
        self.models: tuple[str, ...] | Exception = ("st-gemma4-e4b:latest", "st-qwen35-4b:latest")
        self.identity: Exception | None = None  # what `identify` raises, if anything
        self.identified = 0
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.requests.append(request)
        try:
            if self.on_call is not None:
                self.on_call(request)
            step = self.script.popleft() if self.script else self.default
            if callable(step):
                step = step(request)
            if step is None:
                step = response()
            if isinstance(step, Exception):
                raise step
            return step
        finally:
            with self._lock:
                self.in_flight -= 1

    def list_models(self) -> tuple[str, ...]:
        if isinstance(self.models, Exception):
            raise self.models
        return self.models

    def identify(self) -> None:
        self.identified += 1
        if self.identity is not None:
            raise self.identity

    @property
    def calls(self) -> int:
        return len(self.requests)


class TrackedUow:
    """Wraps a unit of work and says whether a transaction is open right now (any thread)."""

    def __init__(self, inner: UnitOfWork) -> None:
        self._inner = inner
        self.depth = 0
        self._lock = threading.Lock()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._inner.transaction():
            with self._lock:
                self.depth += 1
            try:
                yield
            finally:
                with self._lock:
                    self.depth -= 1

    def snapshot(self):
        return self._inner.snapshot()


@dataclass
class LlmWorld:
    base: World
    escalations: Any
    uow: TrackedUow
    judge: ScriptedJudge
    service: EscalationService
    stop: threading.Event = field(default_factory=threading.Event)
    db: Any = None

    @property
    def clock(self):
        return self.base.clock

    @property
    def classifications(self):
        return self.base.classifications

    @property
    def items(self):
        return self.base.items

    def ready(self, source_id: str = "src_news", **options: Any) -> str:
        return self.base.ready(source_id, **options)

    def seed(
        self,
        n: int | str,
        *,
        text: str | None = None,
        title: str = "",
        source: str = "src_news",
        status: ItemStatus = ItemStatus.CLASSIFIED,
        **changes: Any,
    ) -> Classification:
        """An item that is classified, with a verdict waiting for the LLM."""
        body = f"Mañana habrá un temblor fuerte en Cali {n}" if text is None else text
        item: Item = self.base.add(f"guid-{n}", body, title=title, source_id=source)
        stored = verdict(item.id.removeprefix("itm_"), **changes)
        assert self.items.set_status(item.id, status)
        assert self.classifications.add(stored)
        return stored

    def verdict(self, stored: Classification) -> Classification:
        found = self.classifications.get(stored.item_id)
        assert found is not None
        return found

    def escalate(self, **kwargs: Any):
        return self.service.escalate(**kwargs)

    def prime_success(self, n: int = 9000) -> None:
        """Record that the judge answered some other item a moment ago: from then on a failure
        that may be the item's own counts against the item (it does not while the judge answers
        nobody)."""
        self.escalations.add_judgment(
            Judgment(
                item_id=f"itm_{n:08d}",
                outcome="judged",
                label="no_risk",
                confidence=0.9,
                rationale="an earlier judgment",
                evidence=(),
                llm_model="st-gemma4-e4b",
                prompt_version=PROMPT_VERSION,
                raw_output=None,
                prior_label="unintentional_rumor",
                prior_confidence=0.5,
                prior_stage="rules",
                reused_from=None,
                judged_at=self.clock.now(),
            )
        )

    def close(self) -> None:
        self.base.close()


def build(
    base: World,
    escalations: Any,
    uow: Any,
    judge: ScriptedJudge,
    settings: LlmSettings | None,
    *,
    holder: str = "test-holder",
) -> LlmWorld:
    tracked = TrackedUow(uow)
    stop = threading.Event()
    service = EscalationService(
        base.items,
        base.control,
        base.classifications,
        escalations,
        judge,
        tracked,
        base.clock,
        settings=settings or LlmSettings(),
        should_stop=stop.is_set,
        holder=holder,
    )
    return LlmWorld(base, escalations, tracked, judge, service, stop)


def memory_llm(
    settings: LlmSettings | None = None,
    *,
    judge: ScriptedJudge | None = None,
    classifier_settings: ClassifierSettings | None = None,
) -> LlmWorld:
    base = memory_world(settings=classifier_settings)
    escalations = InMemoryEscalationRepository(base.classifications, base.items)
    uow = InMemoryUnitOfWork(
        [base.sources_repo, base.control_repo, base.items, base.classifications, escalations]
    )
    return build(base, escalations, uow, judge or ScriptedJudge(), settings)


def sqlite_llm(
    path: Path,
    settings: LlmSettings | None = None,
    *,
    judge: ScriptedJudge | None = None,
    holder: str = "test-holder",
    classifier_settings: ClassifierSettings | None = None,
) -> LlmWorld:
    base = sqlite_world(path, settings=classifier_settings)
    escalations = SqliteEscalationRepository(base.db)
    world = build(base, escalations, base.uow, judge or ScriptedJudge(), settings, holder=holder)
    world.db = base.db
    return world


def error(kind: str, detail: str = "boom") -> JudgeError:
    from social_tracker.llm.domain import JudgeErrorKind

    return JudgeError(JudgeErrorKind(kind), detail)
