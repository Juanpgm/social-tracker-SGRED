"""A complete in-memory or SQLite world around the collector, with a scripted fetcher."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from social_tracker.collector.domain import FetchRequest, FetchResponse
from social_tracker.collector.service import CollectorService
from social_tracker.control.domain import CollectionState, ControlEvent, SourceState
from social_tracker.control.service import ControlService
from social_tracker.items.ports import ItemRepository
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import FixedClock
from social_tracker.sources.service import SourceRegistry
from tests.actors import WHO
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.item_repository import InMemoryItemRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.sources.factories import payload

START = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
FEED_URL = "https://news.example.org/rss.xml"


def rss(*entries: tuple[str, str, str], channel_extra: str = "") -> bytes:
    """An RSS 2.0 document with `(guid, title, body)` entries."""
    items = "".join(
        f"<item><guid>{guid}</guid><title>{title}</title><link>https://news.example.org/{guid}</link>"
        f"<description>{body}</description><pubDate>Sun, 01 Mar 2026 07:00:00 GMT</pubDate></item>"
        for guid, title, body in entries
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>t</title>'
        f"{channel_extra}{items}</channel></rss>"
    ).encode()


FEED = rss(("a", "Sismo en Cali", "Un temblor sacudio la ciudad"), ("b", "Otra noticia", "Texto b"))


class FakeFetcher:
    """Plays a script, one step per call, repeating the last step: a response, bytes (a 200
    body), an exception to raise, or a function of the request returning any of those."""

    def __init__(self, *script: Any) -> None:
        self._script = list(script) or [FEED]
        self.requests: list[FetchRequest] = []

    def fetch(self, request: FetchRequest) -> FetchResponse:
        self.requests.append(request)
        step = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if callable(step):
            step = step(request)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, bytes):
            return FetchResponse(step)
        return step

    def then(self, *script: Any) -> None:
        self._script = list(script)


@dataclass
class World:
    registry: SourceRegistry
    control: ControlService
    control_repo: Any
    items: ItemRepository
    sources_repo: Any
    fetcher: FakeFetcher
    clock: FixedClock
    collector: CollectorService
    uow: Any
    db: Database | None = None

    def add_source(self, source_id: str = "src_news", *, enable: bool = True, **fields: Any) -> str:
        fields = {"feed_url": FEED_URL, **fields}
        data = payload("news_rss", id=source_id, name=f"Name of {source_id}", **fields)
        self.registry.add(data, **WHO)
        if enable:
            self.registry.enable(source_id, **WHO)
        return source_id

    def analysis_on(self) -> None:
        self.control.set_global_analysis(True, **WHO)

    def state(self, source_id: str = "src_news") -> SourceState:
        return self.control.get_state(source_id)

    def events(self, source_id: str = "src_news") -> list[ControlEvent]:
        return self.control_repo.list_control_events(source_id)

    def kinds(self, source_id: str = "src_news") -> list[str]:
        return [e.kind for e in self.events(source_id)]

    def details(self, kind: str, source_id: str = "src_news") -> list[dict[str, Any]]:
        return [json.loads(e.detail) for e in self.events(source_id) if e.kind == kind]

    def ready(self, source_id: str = "src_news", **fields: Any) -> str:
        """An enabled source with global analysis on: the normal working condition."""
        self.analysis_on()
        return self.add_source(source_id, **fields)

    def wait_out_backoff(self, source_id: str = "src_news") -> None:
        retry = self.state(source_id).next_run_at
        assert retry is not None
        if retry > self.clock.now():
            self.clock.advance(retry - self.clock.now())

    def close(self) -> None:
        if self.db is not None:
            self.db.close()


def _assemble(
    sources: Any,
    control: Any,
    items: ItemRepository,
    uow: Any,
    fetcher: FakeFetcher,
    db: Database | None,
    jitter: Callable[[float], float] | None,
) -> World:
    clock = FixedClock(START)
    registry = SourceRegistry(sources, control, uow, clock)
    control_service = ControlService(control, sources, uow, clock)
    kwargs = {"jitter": jitter} if jitter else {}
    collector = CollectorService(
        sources, control_service, control, items, fetcher, uow, clock, **kwargs
    )
    return World(
        registry, control_service, control, items, sources, fetcher, clock, collector, uow, db
    )


def memory_world(fetcher: FakeFetcher | None = None, jitter=None) -> World:
    sources, control, items = (
        InMemorySourceRepository(),
        InMemoryControlRepository(),
        InMemoryItemRepository(),
    )
    uow = InMemoryUnitOfWork([sources, control, items])
    return _assemble(sources, control, items, uow, fetcher or FakeFetcher(), None, jitter)


def sqlite_world(path: Path, fetcher: FakeFetcher | None = None, jitter=None, **db_kwargs) -> World:
    db = Database(path, **db_kwargs)
    sources, control, items = (
        SqliteSourceRepository(db),
        SqliteControlRepository(db),
        SqliteItemRepository(db),
    )
    return _assemble(
        sources, control, items, SqliteUnitOfWork(db), fetcher or FakeFetcher(), db, jitter
    )


def idle_state(source_id: str = "src_news") -> SourceState:
    return SourceState(source_id, CollectionState.IDLE)
