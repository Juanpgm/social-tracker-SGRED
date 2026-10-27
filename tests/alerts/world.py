"""A SQLite world around the alert service: real repositories, a fixed clock, items with verdicts
put straight into the store (no need to run the cascade to test what is made of its output)."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from social_tracker.alerts.service import AlertService

from social_tracker.alerts.domain import Alert, AlertFilter, AlertSettings, SyncReport
from social_tracker.classification.domain import (
    Classification,
    Label,
    Signal,
    Stage,
    Topic,
)
from social_tracker.items.domain import Item, ItemStatus
from social_tracker.persistence.alerts_repo import SqliteAlertRepository
from tests.classification.harness import PUBLISHED
from tests.llm.harness import LlmWorld, ScriptedJudge, sqlite_llm


@dataclass
class AlertWorld:
    llm: LlmWorld
    repo: SqliteAlertRepository
    service: AlertService
    settings: AlertSettings

    @property
    def clock(self):
        return self.llm.clock

    @property
    def db(self):
        return self.llm.db

    @property
    def control(self):
        return self.llm.base.control

    def put(
        self,
        name: str,
        *,
        label: Label = Label.UNINTENTIONAL_RUMOR,
        confidence: float = 0.6,
        text: str | None = None,
        title: str = "",
        source: str = "src_news",
        published: datetime = PUBLISHED,
        fetched: datetime | None = None,
        catalog: str = "unavailable",
        needs_llm: bool = False,
        signals: tuple[Signal, ...] | None = None,
        topics: tuple[Topic, ...] = (Topic.EARTHQUAKE,),
        stage: Stage = Stage.RULES,
        source_name: str | None = None,
        **verdict: Any,
    ) -> Item:
        """An item, classified, with the verdict given."""
        body = f"Texto de prueba {name} sobre un temblor en Cali" if text is None else text
        extra: dict[str, Any] = {} if fetched is None else {"fetched_at": fetched}
        if source_name is not None:
            extra["source_name"] = source_name
        item = self.llm.base.add(
            f"guid-{name}", body, title=title, source_id=source, published_at=published, **extra
        )
        self._classify(item, label, confidence, catalog, needs_llm, signals, topics, stage, verdict)
        return item

    def _classify(
        self,
        item: Item,
        label: Label,
        confidence: float,
        catalog: str,
        needs_llm: bool,
        signals: tuple[Signal, ...] | None,
        topics: tuple[Topic, ...],
        stage: Stage,
        extra: dict[str, Any],
    ) -> Classification:
        fields: dict[str, Any] = {
            "item_id": item.id,
            "label": label,
            "confidence": confidence,
            "stage": stage,
            "rationale": "prediction_claim",
            "classified_at": self.clock.now(),
            "classifier_version": "cascade-2/topics-0",
            "signals": (Signal("prediction_claim", Label.UNINTENTIONAL_RUMOR, 0.45),)
            if signals is None
            else signals,
            "topics": topics,
            "needs_llm": needs_llm,
            "priority": 0.5 if needs_llm else 0.0,
            "catalog_status": catalog,
        }
        fields.update(extra)
        stored = Classification(**fields)
        assert self.llm.classifications.add(stored)
        assert self.llm.items.set_status(item.id, ItemStatus.CLASSIFIED)
        return stored

    def refresh(self, **kwargs: Any) -> SyncReport:
        return self.service.refresh(**kwargs)

    def alerts(self, **filters: Any) -> list[Alert]:
        return self.repo.list(AlertFilter(**filters), limit=200, offset=0)

    def only(self, **filters: Any) -> Alert:
        found = self.alerts(**filters)
        assert len(found) == 1, found
        return found[0]

    def close(self) -> None:
        self.llm.close()


def alert_world(
    tmp_path: Path,
    settings: AlertSettings | None = None,
    *,
    judge: ScriptedJudge | None = None,
    name: str = "alerts.db",
) -> AlertWorld:
    llm = sqlite_llm(tmp_path / name, judge=judge)
    return attach(llm, settings)


def attach(llm: LlmWorld, settings: AlertSettings | None = None) -> AlertWorld:
    """An alert service over the same database as an existing world."""
    from social_tracker.persistence.llm_repo import SqliteEscalationRepository

    chosen = settings or AlertSettings()
    repo = SqliteAlertRepository(llm.db)
    service = AlertService(
        repo,
        llm.classifications,
        llm.items,
        llm.base.sources_repo,
        SqliteEscalationRepository(llm.db),
        llm.base.control,
        llm.base.uow,
        llm.clock,
        settings=chosen,
    )
    return AlertWorld(llm, repo, service, chosen)


HOUR = timedelta(hours=1)
