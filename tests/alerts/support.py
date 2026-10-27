"""Builders for alert domain objects."""

from datetime import UTC, datetime, timedelta
from typing import Any

from social_tracker.alerts.domain import (
    AlertMember,
    Evidence,
    Severity,
    alert_id_for,
)
from social_tracker.classification.domain import Label

T0 = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
HASH = "a" * 64


def make_evidence(**overrides: Any) -> Evidence:
    fields: dict[str, Any] = {
        "catalog_status": "unavailable",
        "stage": "rules",
        "classifier_version": "cascade-2/topics-0",
        "signals": (("prediction_claim", "unintentional_rumor", 0.45),),
        "topics": ("earthquake",),
    }
    fields.update(overrides)
    return Evidence(**fields)


def make_member(
    item_id: str = "itm_one",
    *,
    alert_id: str | None = None,
    severity: Severity = Severity.MEDIUM,
    confidence: float = 0.6,
    active: bool = True,
    occurred_at: datetime = T0,
    **overrides: Any,
) -> AlertMember:
    fields: dict[str, Any] = {
        "alert_id": alert_id or alert_id_for(item_id),
        "item_id": item_id,
        "source_id": "src_news",
        "source_name": "City News",
        "content_hash": HASH,
        "excerpt": "Manana habra un temblor fuerte en Cali",
        "label": Label.UNINTENTIONAL_RUMOR,
        "confidence": confidence,
        "stage": "rules",
        "rationale": "prediction_claim",
        "severity": severity,
        "tags": ("earthquake",),
        "evidence": make_evidence(),
        "active": active,
        "occurred_at": occurred_at,
        "linked_at": occurred_at + timedelta(minutes=1),
        "event_age_s": 0,
        "settled_at": None,
    }
    fields.update(overrides)
    return AlertMember(**fields)
