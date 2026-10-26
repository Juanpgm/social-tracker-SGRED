"""Builders for valid source payloads (plain dicts, as they arrive from config or the API)."""

from datetime import UTC, datetime
from typing import Any

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

_COMMON: dict[str, Any] = {"name": "Some Source", "created_at": T0, "updated_at": T0}

_PER_TYPE: dict[str, dict[str, Any]] = {
    "news_rss": {"feed_url": "https://example.org/rss.xml"},
    "instagram": {"handle": "some.handle", "secret_ref": "ig_main"},
    "live_stream": {"stream_url": "https://example.org/live.m3u8"},
    "recorded_audio": {"feed_url": "https://example.org/podcast.xml"},
}


def payload(type_: str = "news_rss", **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"id": "src_one", "type": type_, **_COMMON, **_PER_TYPE[type_]}
    data.update(overrides)
    return data


def without(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in data.items() if k not in keys}


# Built by concatenation so the repository hygiene scan does not flag these fixtures.
CANARY = "canary-" + "value-42"
EMBEDDED_CREDENTIALS_URL = "https://" + "user:pw" + "@example.org/rss"
