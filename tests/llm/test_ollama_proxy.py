"""The adapter ignores the proxy variables of the environment (a real socket, loopback only)."""

import json

import pytest

from social_tracker.classification.domain import Label
from social_tracker.llm.domain import LlmSettings
from social_tracker.llm.ollama import OllamaJudge
from tests.llm.fake_ollama import FakeOllama, closed_port, reply
from tests.llm.support import request

CHAT = {
    "model": "st-gemma4-e4b",
    "message": {
        "role": "assistant",
        "content": json.dumps(
            {"label": "recycled_content", "confidence": 0.77, "rationale": "old", "evidence": []}
        ),
    },
    "done": True,
    "done_reason": "stop",
}


@pytest.fixture
def server():
    with FakeOllama() as fake:
        yield fake


def test_a_proxy_in_the_environment_is_ignored(server: FakeOllama, monkeypatch: pytest.MonkeyPatch):
    dead = f"http://127.0.0.1:{closed_port()}"
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, dead)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    server.script.steps.append(reply(200, CHAT))
    assert (
        OllamaJudge(LlmSettings(base_url=server.url)).judge(request()).label
        is Label.RECYCLED_CONTENT
    )
