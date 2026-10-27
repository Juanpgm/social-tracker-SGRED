"""A complete in-memory or SQLite world around the classifier, with Spanish sample texts."""

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from social_tracker.classification.domain import Classification, ClassifierSettings
from social_tracker.classification.ports import CatalogCheck
from social_tracker.classification.service import ClassifierService, TopicConfigService
from social_tracker.classification.topics import TopicConfig
from social_tracker.classification.yaml_io import default_topic_config
from social_tracker.control.service import ControlService
from social_tracker.items.domain import Item
from social_tracker.persistence.classification_repo import (
    SqliteClassificationRepository,
    SqliteSignatureRepository,
    SqliteTopicConfigRepository,
)
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import FixedClock
from social_tracker.sources.service import SourceRegistry
from tests.actors import WHO
from tests.fakes.classification_repositories import (
    InMemoryClassificationRepository,
    InMemorySignatureRepository,
    InMemoryTopicConfigRepository,
)
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.item_repository import InMemoryItemRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.items.factories import make_item
from tests.sources.factories import payload

START = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
PUBLISHED = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)

PREDICTION = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
REPORT = "El SGC reportó un sismo de magnitud 4.2 en Cali"
OFF_TOPIC = "El equipo ganó el partido de ayer en el estadio de la ciudad"
CRIME = "Hubo un robo en un banco del centro de Cali"
ATTRIBUTION = "Según la Secretaría de Gestión del Riesgo, evacuen la zona ya"
BENIGN_QUAKE = "El sismo de anoche fue sentido por varios vecinos de Cali"

_VOCABULARY = [f"palabra{n}" for n in range(300)]


def long_text(seed: int, words: int = 70) -> str:
    """A quake article of `words` distinct-ish words: stable per seed, far from other seeds."""
    rng = random.Random(seed)
    body = " ".join(rng.choice(_VOCABULARY) for _ in range(words))
    return f"El sismo de ayer en Cali dejó este reporte: {body}"


@dataclass
class World:
    registry: SourceRegistry
    control: ControlService
    control_repo: Any
    items: Any
    sources_repo: Any
    classifications: Any
    signatures: Any
    topic_repo: Any
    uow: Any
    clock: FixedClock
    topics: TopicConfigService
    service: ClassifierService
    db: Database | None = None

    def add_source(
        self,
        source_id: str = "src_news",
        *,
        credibility: int = 3,
        enable: bool = True,
        analysis: bool = True,
    ) -> str:
        data = payload(
            "news_rss", id=source_id, name=f"Name of {source_id}", credibility=credibility
        )
        self.registry.add(data, **WHO)
        if enable:
            self.registry.enable(source_id, **WHO)
        if analysis:
            self.control.set_source_analysis(source_id, True, **WHO)
        return source_id

    def ready(self, source_id: str = "src_news", **options: Any) -> str:
        """Global analysis on and one enabled source with analysis on: the working condition."""
        self.control.set_global_analysis(True, **WHO)
        return self.add_source(source_id, **options)

    def add(
        self,
        external_id: str,
        text: str,
        *,
        title: str = "",
        source_id: str = "src_news",
        published_at: datetime = PUBLISHED,
        **overrides: Any,
    ) -> Item:
        item = make_item(
            external_id,
            source_id=source_id,
            title=title,
            text=text,
            published_at=published_at,
            **overrides,
        )
        report = self.items.add_new([item])
        assert report.new == 1, report
        return item

    def verdict(self, item: Item) -> Classification | None:
        return self.classifications.get(item.id)

    def status(self, item: Item):
        return self.items.get(item.id).status

    def close(self) -> None:
        if self.db is not None:
            self.db.close()


def _build(
    sources: Any,
    control_repo: Any,
    items: Any,
    classifications: Any,
    signatures: Any,
    topic_repo: Any,
    uow: Any,
    db: Database | None,
    *,
    settings: ClassifierSettings | None,
    catalog: CatalogCheck | None,
    default: TopicConfig | None,
    clock: FixedClock | None = None,
) -> World:
    clock = clock or FixedClock(START)
    registry = SourceRegistry(sources, control_repo, uow, clock)
    control = ControlService(control_repo, sources, uow, clock)
    topics = TopicConfigService(
        topic_repo, control_repo, uow, clock, default or default_topic_config()
    )
    service = ClassifierService(
        items,
        sources,
        control,
        classifications,
        signatures,
        topics,
        uow,
        clock,
        catalog=catalog,
        settings=settings,
    )
    return World(
        registry,
        control,
        control_repo,
        items,
        sources,
        classifications,
        signatures,
        topic_repo,
        uow,
        clock,
        topics,
        service,
        db,
    )


def memory_world(
    *,
    settings: ClassifierSettings | None = None,
    catalog: CatalogCheck | None = None,
    default: TopicConfig | None = None,
    items: Any = None,
    classifications: Any = None,
) -> World:
    sources, control_repo = InMemorySourceRepository(), InMemoryControlRepository()
    items = items if items is not None else InMemoryItemRepository()
    classifications = classifications or InMemoryClassificationRepository()
    signatures = InMemorySignatureRepository()
    topic_repo = InMemoryTopicConfigRepository()
    uow = InMemoryUnitOfWork(
        [sources, control_repo, items, classifications, signatures, topic_repo]
    )
    return _build(
        sources,
        control_repo,
        items,
        classifications,
        signatures,
        topic_repo,
        uow,
        None,
        settings=settings,
        catalog=catalog,
        default=default,
    )


def sqlite_world(
    path: Path,
    *,
    settings: ClassifierSettings | None = None,
    catalog: CatalogCheck | None = None,
    default: TopicConfig | None = None,
    clock: FixedClock | None = None,
    **db_options: Any,
) -> World:
    db = Database(path, **db_options)
    return _build(
        SqliteSourceRepository(db),
        SqliteControlRepository(db),
        SqliteItemRepository(db),
        SqliteClassificationRepository(db),
        SqliteSignatureRepository(db),
        SqliteTopicConfigRepository(db),
        SqliteUnitOfWork(db),
        db,
        settings=settings,
        catalog=catalog,
        default=default,
        clock=clock,
    )


def days(n: float) -> timedelta:
    return timedelta(days=n)
