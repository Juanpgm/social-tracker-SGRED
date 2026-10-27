"""The Ollama adapter across a real loopback socket, against the stand-in server (never Ollama)."""

import json
import time

import pytest

from social_tracker.classification.domain import Label
from social_tracker.llm.domain import JudgeError, JudgeErrorKind, LlmSettings
from social_tracker.llm.ollama import OllamaJudge
from tests.llm.fake_ollama import FakeOllama, closed_port, redirect, reply, silence, trickle
from tests.llm.support import request

VERDICT = {
    "label": "recycled_content",
    "confidence": 0.77,
    "rationale": "an old photo presented as today's",
    "evidence": ["same image as 2019"],
}
CHAT = {
    "model": "st-gemma4-e4b",
    "message": {"role": "assistant", "content": json.dumps(VERDICT)},
    "done": True,
    "done_reason": "stop",
}


# Not a performance bound: a call that ignores its deadline would be seen here long before the
# test runner gives up; a slow machine does not get near it.
HANG_GUARD_S = 30


def judge_for(server: FakeOllama, **settings) -> OllamaJudge:
    judge = OllamaJudge(LlmSettings(base_url=server.url, **settings))
    return judge


@pytest.fixture
def server():
    with FakeOllama() as fake:
        yield fake


def test_a_real_exchange_returns_the_validated_verdict(server: FakeOllama):
    server.script.steps.append(reply(200, CHAT, {"Content-Type": "application/json"}))
    answer = judge_for(server).judge(request("texto"))
    assert (answer.label, answer.confidence) == (Label.RECYCLED_CONTENT, 0.77)
    (seen,) = server.requests
    assert (seen.method, seen.path) == ("POST", "/api/chat")
    assert seen.headers["Host"] == f"127.0.0.1:{server.port}"
    assert "Cookie" not in seen.headers
    assert "Authorization" not in seen.headers
    assert seen.json["stream"] is False
    assert seen.json["options"]["temperature"] == 0


def test_the_name_localhost_still_connects_to_the_literal_loopback_address(server: FakeOllama):
    server.script.steps.append(reply(200, CHAT))
    judge = OllamaJudge(LlmSettings(base_url=f"http://localhost:{server.port}"))
    assert judge.judge(request()).label is Label.RECYCLED_CONTENT
    assert server.requests[0].headers["Host"] == f"127.0.0.1:{server.port}"


def test_nothing_listening_is_unreachable_and_fast():
    judge = OllamaJudge(LlmSettings(base_url=f"http://127.0.0.1:{closed_port()}"))
    started = time.monotonic()
    with pytest.raises(JudgeError) as caught:
        judge.judge(request())
    assert caught.value.kind is JudgeErrorKind.UNREACHABLE
    assert time.monotonic() - started < HANG_GUARD_S


def test_a_redirect_is_not_followed(server: FakeOllama):
    server.script.steps.append(redirect("/steal"))
    server.script.default = reply(200, CHAT)  # what following the redirect would have found
    with pytest.raises(JudgeError) as caught:
        judge_for(server).judge(request())
    assert caught.value.kind is JudgeErrorKind.BAD_RESPONSE
    assert [r.path for r in server.requests] == ["/api/chat"]


def test_a_slow_loris_body_is_cut_by_the_deadline_of_the_call(server: FakeOllama):
    server.script.steps.append(trickle(total=500, pause_s=0.15))
    # The first call loads the model, so the load deadline is the one that applies.
    judge = judge_for(server, call_deadline_s=0.6, load_deadline_s=0.6)
    started = time.monotonic()
    with pytest.raises(JudgeError) as caught:
        judge.judge(request())
    assert caught.value.kind is JudgeErrorKind.LOAD_TIMEOUT
    assert time.monotonic() - started < HANG_GUARD_S


def test_a_server_that_never_answers_is_a_timeout(server: FakeOllama):
    server.script.steps.append(silence(3))
    judge = judge_for(server, call_deadline_s=0.5, load_deadline_s=0.5)
    started = time.monotonic()
    with pytest.raises(JudgeError) as caught:
        judge.judge(request())
    assert caught.value.kind is JudgeErrorKind.LOAD_TIMEOUT
    assert time.monotonic() - started < HANG_GUARD_S


def test_a_5xx_then_a_200_is_an_error_then_a_verdict(server: FakeOllama):
    server.script.steps.extend([reply(503, {"error": "busy"}), reply(200, CHAT)])
    judge = judge_for(server)
    with pytest.raises(JudgeError) as caught:
        judge.judge(request())
    assert caught.value.kind is JudgeErrorKind.SERVER_ERROR
    assert judge.judge(request()).label is Label.RECYCLED_CONTENT


def test_a_missing_model_falls_back_over_the_wire(server: FakeOllama):
    server.script.steps.extend(
        [reply(404, {"error": "model 'st-gemma4-e4b' not found"}), reply(200, CHAT)]
    )
    answer = judge_for(server).judge(request())
    assert answer.model == "st-qwen35-4b"
    assert [r.json["model"] for r in server.requests] == ["st-gemma4-e4b", "st-qwen35-4b"]
    assert server.requests[0].json["think"] is False
    assert server.requests[1].json["think"] is False


def test_the_model_list_over_the_wire(server: FakeOllama):
    server.script.steps.append(reply(200, {"models": [{"name": "st-gemma4-e4b:latest"}]}))
    assert judge_for(server).list_models() == ("st-gemma4-e4b:latest",)
    assert (server.requests[0].method, server.requests[0].path) == ("GET", "/api/tags")
