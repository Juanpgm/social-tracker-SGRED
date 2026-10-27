"""Builders for valid items."""

from datetime import UTC, datetime, timedelta
from typing import Any

from social_tracker.items.domain import Item, ItemKind, ItemStatus, content_hash, item_id

NOW = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def make_item(
    external_id: str = "guid-1",
    *,
    source_id: str = "src_news",
    title: str = "Quake felt in Cali",
    text: str = "A magnitude 4.2 tremor shook the city.",
    published_at: datetime = NOW,
    fetched_at: datetime | None = None,
    **overrides: Any,
) -> Item:
    fields: dict[str, Any] = {
        "id": item_id(source_id, external_id),
        "source_id": source_id,
        "source_name": "City News",
        "kind": ItemKind.ARTICLE,
        "external_id": external_id,
        "parent_id": None,
        "url": f"https://example.org/{external_id}",
        "title": title,
        "author": None,
        "text": text,
        "language": None,
        "published_at": published_at,
        "fetched_at": fetched_at or published_at + timedelta(minutes=1),
        "content_hash": content_hash(title, text),
        "flags": (),
        "status": ItemStatus.NEW,
        "raw_payload": {"guid": external_id},
    }
    fields.update(overrides)
    return Item(**fields)
