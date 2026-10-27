"""The Ollama adapter against scripted transports: what it sends, what it refuses, how it fails."""

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx2
import pytest

from social_tracker.classification.domain import Label
from social_tracker.llm.domain import JudgeError, JudgeErrorKind, LlmSettings
from social_tracker.llm.ollama import OllamaJudge
from social_tracker.llm.prompt import build_messages
from social_tracker.llm.schema import JUDGE_JSON_SCHEMA
from tests.llm.support import request

GOOD = {
    "label": "deliberate_disinformation",
    "confidence": 0.81,
    "rationale": "attributes an evacuation order to the secretariat without a source",
    "evidence": ["no official link"],
}


def envelope(content: Any = GOOD, **extra: Any) -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content)
    return {
        "model": "st-gemma4-e4b",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": "stop",
        **extra,
    }


def ok(content: Any = GOOD, **extra: Any) -> httpx2.Response:
    return httpx2.Response(200, json=envelope(content, **extra))


def make(handler: Callable[[httpx2.Request], httpx2.Response], **settings: Any) -> OllamaJudge:
    return OllamaJudge(LlmSettings(**settings), transport=httpx2.MockTransport(handler))


def fails(kind: JudgeErrorKind, judge: OllamaJudge) -> JudgeError:
    with pytest.raises(JudgeError) as caught:
        judge.judge(request())
    assert caught.value.kind is kind
    return caught.value


# ---- what is sent ----


def test_a_call_posts_the_prompt_with_the_schema_deterministic_options_and_keep_alive():
    seen: list[httpx2.Request] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen.append(req)
        return ok()

    answer = make(handler).judge(request("texto"))
    assert (answer.label, answer.confidence) == (Label.DELIBERATE_DISINFORMATION, 0.81)
    assert answer.model == "st-gemma4-e4b"
    (sent,) = seen
    assert (sent.method, sent.url.path) == ("POST", "/api/chat")
    assert (sent.url.host, sent.url.port) == ("127.0.0.1", 11434)
    body = json.loads(sent.content)
    assert body["model"] == "st-gemma4-e4b"
    assert body["stream"] is False
    assert body["format"] == JUDGE_JSON_SCHEMA
    assert body["messages"] == build_messages(request("texto"))
    assert body["keep_alive"] == "300s"
    assert body["options"]["temperature"] == 0
    assert body["options"]["num_predict"] == 512
    assert "think" not in body  # a Gemma model has no thinking switch to turn off


def test_the_same_request_sends_the_same_bytes():
    bodies: list[bytes] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        bodies.append(req.content)
        return ok()

    judge = make(handler)
    judge.judge(request())
    judge.judge(request())
    assert bodies[0] == bodies[1]


def test_a_qwen_model_is_asked_not_to_think():
    seen: list[dict] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(req.content))
        return ok()

    make(handler, model="st-qwen35-4b", fallback_model=None).judge(request())
    assert seen[0]["think"] is False


def test_keep_alive_and_token_cap_follow_the_settings():
    seen: list[dict] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(req.content))
        return ok()

    make(handler, keep_alive_s=45, max_output_tokens=64).judge(request())
    assert seen[0]["keep_alive"] == "45s"
    assert seen[0]["options"]["num_predict"] == 64


def test_the_connection_is_made_to_the_literal_loopback_address_even_for_localhost():
    hosts: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        hosts.append(req.url.host)
        return ok()

    make(handler, base_url="http://localhost:11434").judge(request())
    assert hosts == ["127.0.0.1"]


def test_no_credentials_cookies_or_compression_are_sent():
    seen: list[httpx2.Request] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen.append(req)
        return httpx2.Response(200, json=envelope(), headers={"Set-Cookie": "a=b"})

    judge = make(handler)
    judge.judge(request())
    judge.judge(request())
    for sent in seen:
        assert "authorization" not in sent.headers
        assert "cookie" not in sent.headers
        assert sent.headers["accept-encoding"] == "identity"


# ---- the answer ----


def test_the_envelope_content_is_validated_by_the_strict_schema():
    answer = make(lambda r: ok(GOOD)).judge(request())
    assert answer.rationale.startswith("attributes an evacuation")
    assert answer.raw_output == json.dumps(GOOD)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        '{"label": "no_risk", "confidence": 0.9',
        "Here you go: " + json.dumps(GOOD),
        json.dumps({**GOOD, "label": "totally_safe"}),
        json.dumps({**GOOD, "confidence": 7}),
        json.dumps({**GOOD, "extra": 1}),
        json.dumps({**GOOD, "confidence": "high"}),
        '{"label":"no_risk","confidence":NaN,"rationale":"r","evidence":[]}',
        "[]",
    ],
)
def test_output_that_breaks_the_schema_is_invalid_output(content: str):
    fails(JudgeErrorKind.INVALID_OUTPUT, make(lambda r: ok(content)))


def test_an_answer_cut_by_the_token_cap_is_invalid_output_even_if_it_parses():
    fails(JudgeErrorKind.INVALID_OUTPUT, make(lambda r: ok(GOOD, done_reason="length")))


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[]",
        b"{}",
        json.dumps({"message": "text"}).encode(),
        json.dumps({"message": {"role": "assistant"}}).encode(),
        json.dumps({"message": {"role": "assistant", "content": 5}}).encode(),
        json.dumps({"done": True}).encode(),
        json.dumps({"message": {"content": "{}"}, "done": False}).encode(),
    ],
)
def test_a_body_that_is_not_a_chat_answer_is_a_bad_response(body: bytes):
    fails(JudgeErrorKind.BAD_RESPONSE, make(lambda r: httpx2.Response(200, content=body)))


# ---- failures of the network ----


def test_nothing_listening_is_unreachable():
    def handler(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=req)

    fails(JudgeErrorKind.UNREACHABLE, make(handler))


def test_a_connect_timeout_is_unreachable_but_a_read_timeout_is_a_timeout():
    def connect(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectTimeout("slow", request=req)

    def read(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=req)

    fails(JudgeErrorKind.UNREACHABLE, make(connect))
    fails(JudgeErrorKind.TIMEOUT, make(read))


def test_a_connection_dropped_in_the_middle_is_a_bad_response():
    def handler(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.RemoteProtocolError("closed", request=req)

    fails(JudgeErrorKind.BAD_RESPONSE, make(handler))


@pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
def test_a_5xx_is_a_server_error(status: int):
    fails(
        JudgeErrorKind.SERVER_ERROR,
        make(lambda r: httpx2.Response(status, json={"error": "boom"})),
    )


def test_a_5xx_then_a_200_succeeds_on_the_next_call():
    answers = iter([httpx2.Response(503, json={"error": "busy"}), ok()])
    judge = make(lambda r: next(answers))
    fails(JudgeErrorKind.SERVER_ERROR, judge)
    assert judge.judge(request()).label is Label.DELIBERATE_DISINFORMATION


@pytest.mark.parametrize("status", [400, 401, 403, 405, 413, 429])
def test_other_client_errors_are_a_bad_response(status: int):
    fails(JudgeErrorKind.BAD_RESPONSE, make(lambda r: httpx2.Response(status, json={"error": "x"})))


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_refused_and_not_followed(status: int):
    hits: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        hits.append(req.url.path)
        return httpx2.Response(status, headers={"Location": "http://example.org/steal"})

    error = fails(JudgeErrorKind.BAD_RESPONSE, make(handler))
    assert hits == ["/api/chat"]
    assert "example.org" not in str(error)


# ---- model not found and the fallback ----


def test_a_missing_model_falls_back_to_the_second_one_once():
    models: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        model = json.loads(req.content)["model"]
        models.append(model)
        if model == "st-gemma4-e4b":
            return httpx2.Response(404, json={"error": "model 'st-gemma4-e4b' not found"})
        return ok()

    answer = make(handler).judge(request())
    assert models == ["st-gemma4-e4b", "st-qwen35-4b"]
    assert answer.model == "st-qwen35-4b"


def test_a_model_that_cannot_be_loaded_also_falls_back():
    def handler(req: httpx2.Request) -> httpx2.Response:
        if json.loads(req.content)["model"] == "st-gemma4-e4b":
            return httpx2.Response(500, json={"error": "unable to load model: out of memory"})
        return ok()

    assert make(handler).judge(request()).model == "st-qwen35-4b"


def test_the_fallback_gets_the_qwen_thinking_switch():
    seen: list[dict] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        body = json.loads(req.content)
        seen.append(body)
        if body["model"] == "st-gemma4-e4b":
            return httpx2.Response(404, json={"error": "model not found"})
        return ok()

    make(handler).judge(request())
    assert "think" not in seen[0]
    assert seen[1]["think"] is False


def test_both_models_missing_is_model_not_found_after_exactly_two_calls():
    calls: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        calls.append(json.loads(req.content)["model"])
        return httpx2.Response(404, json={"error": "model not found"})

    fails(JudgeErrorKind.MODEL_NOT_FOUND, make(handler))
    assert calls == ["st-gemma4-e4b", "st-qwen35-4b"]


def test_without_a_fallback_a_missing_model_is_one_call_and_model_not_found():
    calls: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        calls.append("x")
        return httpx2.Response(404, json={"error": "model not found"})

    fails(JudgeErrorKind.MODEL_NOT_FOUND, make(handler, fallback_model=None))
    assert calls == ["x"]


def test_the_same_name_twice_is_not_tried_twice():
    calls: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        calls.append("x")
        return httpx2.Response(404, json={"error": "model not found"})

    fails(JudgeErrorKind.MODEL_NOT_FOUND, make(handler, model="same", fallback_model="same"))
    assert calls == ["x"]


def test_a_failure_that_is_not_about_the_model_does_not_fall_back():
    calls: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        calls.append("x")
        return httpx2.Response(503, json={"error": "busy"})

    fails(JudgeErrorKind.SERVER_ERROR, make(handler))
    assert calls == ["x"]


def test_invalid_output_from_the_first_model_is_not_a_reason_to_fall_back():
    calls: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        calls.append("x")
        return ok("garbage")

    fails(JudgeErrorKind.INVALID_OUTPUT, make(handler))
    assert calls == ["x"]


# ---- size and time ----


def test_a_body_larger_than_the_cap_is_refused_by_its_declared_length():
    big = b"x" * 2048
    fails(
        JudgeErrorKind.BAD_RESPONSE,
        make(lambda r: httpx2.Response(200, content=big), max_response_bytes=1024),
    )


class _Chunks(httpx2.SyncByteStream):
    def __init__(self, chunks: Iterator[bytes], on_chunk: Callable[[], None] = lambda: None):
        self._chunks = chunks
        self._on_chunk = on_chunk

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self._on_chunk()
            yield chunk


def test_a_body_larger_than_the_cap_without_a_declared_length_is_refused_while_reading():
    read: list[int] = []

    def chunks() -> Iterator[bytes]:
        for _ in range(1000):
            read.append(1)
            yield b"y" * 512

    def handler(req: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=_Chunks(chunks()))

    fails(JudgeErrorKind.BAD_RESPONSE, make(handler, max_response_bytes=1024))
    assert len(read) < 10  # it stopped reading soon after the cap


def test_a_body_exactly_at_the_cap_is_read():
    body = json.dumps(envelope()).encode()
    padded = body + b" " * (1024 - len(body))
    assert len(padded) == 1024
    answer = make(lambda r: httpx2.Response(200, content=padded), max_response_bytes=1024).judge(
        request()
    )
    assert answer.label is Label.DELIBERATE_DISINFORMATION


def test_a_body_one_byte_over_the_cap_is_refused():
    body = json.dumps(envelope()).encode()
    padded = body + b" " * (1025 - len(body))
    fails(
        JudgeErrorKind.BAD_RESPONSE,
        make(lambda r: httpx2.Response(200, content=padded), max_response_bytes=1024),
    )


def test_a_compressed_answer_is_refused():
    fails(
        JudgeErrorKind.BAD_RESPONSE,
        make(lambda r: httpx2.Response(200, content=b"x", headers={"content-encoding": "gzip"})),
    )


def test_a_slow_trickle_hits_the_wall_clock_deadline_not_the_per_read_timeout():
    clock = {"now": 0.0}

    def tick() -> None:
        clock["now"] += 20.0  # every chunk "takes" 20 seconds

    def handler(req: httpx2.Request) -> httpx2.Response:
        stream = _Chunks(iter([b"{"] * 100), tick)
        return httpx2.Response(200, stream=stream)

    judge = OllamaJudge(
        LlmSettings(call_deadline_s=60),
        transport=httpx2.MockTransport(handler),
        monotonic=lambda: clock["now"],
    )
    fails(JudgeErrorKind.TIMEOUT, judge)
    assert clock["now"] <= 100.0  # gave up after three or four chunks, not a hundred


def test_the_deadline_is_per_call_not_cumulative():
    clock = {"now": 0.0}

    def handler(req: httpx2.Request) -> httpx2.Response:
        clock["now"] += 50.0
        return ok()

    judge = OllamaJudge(
        LlmSettings(call_deadline_s=60),
        transport=httpx2.MockTransport(handler),
        monotonic=lambda: clock["now"],
    )
    assert judge.judge(request()).label is Label.DELIBERATE_DISINFORMATION
    assert judge.judge(request()).label is Label.DELIBERATE_DISINFORMATION


def test_the_fallback_shares_the_deadline_of_the_call():
    clock = {"now": 0.0}
    models: list[str] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        models.append(json.loads(req.content)["model"])
        clock["now"] += 70.0
        return httpx2.Response(404, json={"error": "model not found"})

    judge = OllamaJudge(
        LlmSettings(call_deadline_s=60),
        transport=httpx2.MockTransport(handler),
        monotonic=lambda: clock["now"],
    )
    fails(JudgeErrorKind.TIMEOUT, judge)
    assert models == ["st-gemma4-e4b"]  # no time left for the second model


# ---- the model list ----


def test_the_installed_models_come_from_api_tags():
    seen: list[httpx2.Request] = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen.append(req)
        return httpx2.Response(
            200, json={"models": [{"name": "st-gemma4-e4b:latest"}, {"name": "other"}]}
        )

    assert make(handler).list_models() == ("st-gemma4-e4b:latest", "other")
    assert (seen[0].method, seen[0].url.path) == ("GET", "/api/tags")


@pytest.mark.parametrize(
    "payload",
    [{"models": "x"}, {"models": [1]}, {"models": [{"nope": 1}]}, [], {"other": []}],
)
def test_a_model_list_of_the_wrong_shape_is_a_bad_response(payload: Any):
    judge = make(lambda r: httpx2.Response(200, json=payload))
    with pytest.raises(JudgeError) as caught:
        judge.list_models()
    assert caught.value.kind is JudgeErrorKind.BAD_RESPONSE


def test_an_empty_model_list_is_a_valid_answer():
    assert make(lambda r: httpx2.Response(200, json={"models": []})).list_models() == ()


def test_the_model_list_fails_like_a_call_when_nothing_listens():
    def handler(req: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=req)

    with pytest.raises(JudgeError) as caught:
        make(handler).list_models()
    assert caught.value.kind is JudgeErrorKind.UNREACHABLE


def test_error_messages_never_carry_the_item_text_or_the_model_output():
    secret = "ITEM-TEXT-SECRET"
    judge = make(lambda r: ok("garbage " + secret))
    with pytest.raises(JudgeError) as caught:
        judge.judge(request(secret))
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)


@pytest.mark.parametrize("status", [301, 302, 307, 400, 404, 500])
def test_an_error_status_is_an_error_even_with_a_body_that_looks_like_an_answer(status: int):
    """The status decides, not the body: a redirect or an error page that carries a valid verdict
    is never taken for one."""
    seen = []

    def handler(req: httpx2.Request) -> httpx2.Response:
        seen.append(req.url.path)
        return httpx2.Response(
            status, json=envelope(), headers={"Location": "http://example.org/x"}
        )

    judge = make(handler, fallback_model=None)
    with pytest.raises(JudgeError) as caught:
        judge.judge(request())
    assert caught.value.kind in {
        JudgeErrorKind.BAD_RESPONSE,
        JudgeErrorKind.SERVER_ERROR,
        JudgeErrorKind.MODEL_NOT_FOUND,
    }
    assert seen == ["/api/chat"]
