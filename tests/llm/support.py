"""Builders shared by the LLM tests."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx2

from social_tracker.classification.domain import Classification, Label, Signal, Stage, Topic
from social_tracker.llm.domain import JudgeEvidence, JudgeRequest, JudgeResponse

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def with_identity(handler: Callable[[httpx2.Request], httpx2.Response]):
    """A scripted Ollama: the root page says who it is, everything else is the handler's."""

    def wrapped(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET" and request.url.path == "/":
            return httpx2.Response(200, text="Ollama is running")
        return handler(request)

    return wrapped


def item_id(n: int | str = 1) -> str:
    return f"itm_{n:0>8}" if isinstance(n, int) else f"itm_{n}"


def verdict(n: int | str = 1, **changes: Any) -> Classification:
    """A rules verdict waiting for the LLM."""
    fields: dict[str, Any] = {
        "item_id": item_id(n),
        "label": Label.UNINTENTIONAL_RUMOR,
        "confidence": 0.55,
        "stage": Stage.RULES,
        "rationale": "unintentional_rumor candidate from rules: prediction_claim",
        "classified_at": NOW,
        "classifier_version": "cascade-2/topics-0",
        "signals": (Signal("prediction_claim", Label.UNINTENTIONAL_RUMOR, 0.45),),
        "topics": (Topic.EARTHQUAKE,),
        "needs_llm": True,
        "priority": 0.5,
        "catalog_status": "no_claim",
    }
    fields.update(changes)
    return Classification(**fields)


def evidence(**changes: Any) -> JudgeEvidence:
    fields: dict[str, Any] = {
        "prior_label": Label.UNINTENTIONAL_RUMOR,
        "prior_confidence": 0.55,
        "prior_stage": Stage.RULES,
        "prior_rationale": "rumor candidate",
        "topics": ("earthquake",),
        "signals": (Signal("prediction_claim", Label.UNINTENTIONAL_RUMOR, 0.45),),
        "catalog_status": "no_claim",
        "source_credibility": 3,
    }
    fields.update(changes)
    return JudgeEvidence(**fields)


def request(text: str = "Mañana habrá un temblor fuerte en Cali", **changes: Any) -> JudgeRequest:
    fields: dict[str, Any] = {
        "item_id": item_id(),
        "text": text,
        "text_truncated": False,
        "evidence": evidence(),
    }
    fields.update(changes)
    return JudgeRequest(**fields)


def response(**changes: Any) -> JudgeResponse:
    fields: dict[str, Any] = {
        "label": Label.NO_RISK,
        "confidence": 0.9,
        "rationale": "the text reports a past event without a prediction",
        "evidence": ("past tense",),
        "model": "st-gemma4-e4b",
        "raw_output": "{}",
    }
    fields.update(changes)
    return JudgeResponse(**fields)
