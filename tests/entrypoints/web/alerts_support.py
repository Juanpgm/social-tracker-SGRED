"""Alerts for the web tests: items with stored verdicts, raised into alerts by the real service."""

from contextlib import closing
from datetime import timedelta
from pathlib import Path
from typing import Any

from social_tracker.alerts.domain import Alert, AlertFilter
from social_tracker.bootstrap import DB_PATH_ENV
from social_tracker.classification.domain import Classification, Label, Signal, Stage, Topic
from social_tracker.persistence.classification_repo import SqliteClassificationRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from tests.entrypoints.web.conftest import Web
from tests.items.factories import NOW, make_item

RUMOR = Label.UNINTENTIONAL_RUMOR
DELIBERATE = Label.DELIBERATE_DISINFORMATION
RECYCLED = Label.RECYCLED_CONTENT


def plant(
    web: Web,
    name: str,
    *,
    label: Label = RUMOR,
    confidence: float = 0.6,
    hours: int = 0,
    text: str | None = None,
    title: str = "",
    source_id: str = "src_one",
    source_name: str = "City News",
    topics: tuple[Topic, ...] = (Topic.EARTHQUAKE,),
    catalog: str = "unavailable",
    rationale: str = "prediction_claim",
    **item: Any,
) -> str:
    """An item and its verdict, straight into the database; returns the item id."""
    made = make_item(
        name,
        source_id=source_id,
        source_name=source_name,
        title=title,
        text=text or f"Texto {name}: manana habra un temblor fuerte en Cali",
        published_at=NOW + timedelta(hours=hours),
        **item,
    )
    verdict = Classification(
        item_id=made.id,
        label=label,
        confidence=confidence,
        stage=Stage.RULES,
        rationale=rationale,
        classified_at=NOW,
        classifier_version="cascade-2/topics-0",
        signals=(Signal("prediction_claim", RUMOR, 0.45),),
        topics=topics,
        catalog_status=catalog,
    )
    with Database(Path(web.env[DB_PATH_ENV])) as db:
        SqliteItemRepository(db).add_new([made])
        SqliteClassificationRepository(db).add(verdict)
    return made.id


def raise_alerts(web: Web) -> list[Alert]:
    """Run the alert step and return every alert, newest first."""
    with closing(web.services()) as services:
        services.alerts.refresh()
        return list(services.alerts.list(AlertFilter(), limit=200).alerts)


def alert_for(web: Web, item_id: str) -> Alert:
    with closing(web.services()) as services:
        found = [
            a for a in services.alerts.list(AlertFilter(), limit=200).alerts if a.item_id == item_id
        ]
    assert found, item_id
    return found[0]


def events_of(web: Web, alert_id: str) -> list[Any]:
    with closing(web.services()) as services:
        return list(services.alerts.detail(alert_id).events)
