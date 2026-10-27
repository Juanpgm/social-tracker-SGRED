"""The Ollama adapter after the remediation round: failure levels, cold loads, the thinking switch,
the context window, retry options, server identity. No real Ollama and no GPU: scripted
transports, and a stand-in server on a loopback socket for the deadline tests."""

import json
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from social_tracker.classification.domain import Label
from social_tracker.llm.domain import JudgeError, JudgeErrorKind, LlmSettings
from social_tracker.llm.ollama import OllamaJudge
from social_tracker.llm.prompt import build_messages, estimate_tokens
from tests.llm.support import request
from tests.llm.test_ollama import GOOD, envelope, ok


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def make(
    handler: Callable[[httpx2.Request], httpx2.Response],
    *,
    clock: Clock | None = None,
    **settings: Any,
) -> OllamaJudge:
    return OllamaJudge(
        LlmSettings(**settings),
        transport=httpx2.MockTransport(handler),
        monotonic=clock if clock is not None else Clock(),
    )


def fails(judge: OllamaJudge, req=None) -> JudgeError:
    with pytest.raises(JudgeError) as caught:
        judge.judge(req if req is not None else request())
    return caught.value


# ---- C1: which failures are the server's ----


@pytest.mark.parametrize("status", [400, 401, 403, 418, 422, 429, 451])
def test_a_client_error_status_is_judge_level(status):
    error = fails(make(lambda r: httpx2.Response(status, json={"error": "nope"})))
    assert error.kind is JudgeErrorKind.BAD_RESPONSE
    assert (error.status, error.judge_level, error.item_specific) == (status, True, False)


@pytest.mark.parametrize("status", [500, 502, 503, 504, 599])
def test_a_5xx_status_is_judge_level(status):
    error = fails(make(lambda r: httpx2.Response(status, json={"error": "runner terminated"})))
    assert error.kind is JudgeErrorKind.SERVER_ERROR
    assert (error.status, error.judge_level, error.item_specific) == (status, True, False)


def test_a_404_is_a_missing_model_and_judge_level():
    error = fails(make(lambda r: httpx2.Response(404, json={"error": "x"}), fallback_model=None))
    assert error.kind is JudgeErrorKind.MODEL_NOT_FOUND
    assert error.judge_level


def test_a_4xx_whose_body_names_the_model_is_still_only_a_bad_response():
    # Only a 404, or a 5xx that says the model cannot be loaded, means "no model".
    body = {"error": "model 'st-gemma4-e4b' not found, try pulling it first"}
    error = fails(make(lambda r: httpx2.Response(400, json=body), fallback_model=None))
    assert error.kind is JudgeErrorKind.BAD_RESPONSE
    assert error.status == 400


@pytest.mark.parametrize(
    "hint", ["unable to load model", "failed to load", "cannot load", "not found"]
)
def test_a_5xx_that_says_the_model_cannot_load_is_model_not_found(hint):
    error = fails(make(lambda r: httpx2.Response(500, json={"error": hint}), fallback_model=None))
    assert error.kind is JudgeErrorKind.MODEL_NOT_FOUND


def test_a_5xx_without_a_hint_about_the_model_is_a_server_error():
    error = fails(
        make(lambda r: httpx2.Response(500, json={"error": "out of memory"}), fallback_model=None)
    )
    assert error.kind is JudgeErrorKind.SERVER_ERROR


def test_a_redirect_is_judge_level():
    error = fails(make(lambda r: httpx2.Response(302, headers={"Location": "http://example.org/"})))
    assert error.kind is JudgeErrorKind.BAD_RESPONSE
    assert error.judge_level


@pytest.mark.parametrize("body", [b"", b"not json", b"[]", b"{}"])
def test_a_200_that_is_not_an_answer_may_be_the_item_s_fault(body):
    error = fails(make(lambda r: httpx2.Response(200, content=body)))
    assert error.kind is JudgeErrorKind.BAD_RESPONSE
    assert (error.status, error.judge_level, error.item_specific) == (None, False, True)


def test_a_connection_dropped_mid_way_may_be_the_item_s_fault():
    def handler(req):
        raise httpx2.RemoteProtocolError("closed", request=req)

    error = fails(make(handler))
    assert error.kind is JudgeErrorKind.BAD_RESPONSE
    assert error.item_specific


# ---- C1: cold loads get a longer deadline ----


def _timeouts(seen: list[dict]) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(req: httpx2.Request) -> httpx2.Response:
        seen.append(dict(req.extensions["timeout"]))
        return ok()

    return handler


def test_the_first_call_gets_the_load_deadline_and_the_next_the_call_deadline():
    seen: list[dict] = []
    judge = make(_timeouts(seen), call_deadline_s=60, load_deadline_s=180)
    judge.judge(request())
    judge.judge(request())
    assert seen[0]["read"] == 180
    assert seen[1]["read"] == 60


def test_the_model_is_cold_again_when_its_keep_alive_has_run_out():
    seen: list[dict] = []
    clock = Clock(1000.0)
    judge = make(_timeouts(seen), clock=clock, keep_alive_s=300)
    judge.judge(request())  # resident until 1300
    clock.now = 1299.0
    judge.judge(request())
    assert seen[1]["read"] == 60
    # keep-alive restarts after the last answer: resident until 1299 + 300 = 1599
    clock.now = 1598.5
    judge.judge(request())
    assert seen[2]["read"] == 60
    clock.now = 1598.5 + 300  # exactly at the end: the server unloads at that instant
    judge.judge(request())
    assert seen[3]["read"] == 180


def test_a_failed_call_does_not_make_the_model_resident():
    seen: list[dict] = []
    answers = iter([httpx2.Response(503, json={"error": "x"}), ok()])

    def handler(req):
        seen.append(dict(req.extensions["timeout"]))
        return next(answers)

    judge = make(handler)
    fails(judge)
    judge.judge(request())
    assert [t["read"] for t in seen] == [180, 180]


def test_an_invalid_answer_still_proves_the_model_was_loaded():
    seen: list[dict] = []
    answers = iter([ok("garbage"), ok()])

    def handler(req):
        seen.append(dict(req.extensions["timeout"]))
        return next(answers)

    judge = make(handler)
    fails(judge)
    judge.judge(request())
    assert [t["read"] for t in seen] == [180, 60]


def test_the_load_deadline_is_never_shorter_than_the_call_deadline():
    seen: list[dict] = []
    make(_timeouts(seen), call_deadline_s=90, load_deadline_s=10).judge(request())
    assert seen[0]["read"] == 90


def test_a_timeout_while_loading_is_judge_level():
    def handler(req):
        raise httpx2.ReadTimeout("slow", request=req)

    error = fails(make(handler))
    assert error.kind is JudgeErrorKind.LOAD_TIMEOUT
    assert (error.judge_level, error.item_specific) == (True, False)


def test_a_timeout_on_a_resident_model_may_be_the_item_s_fault():
    answers = iter([ok(), "timeout"])

    def handler(req):
        step = next(answers)
        if step == "timeout":
            raise httpx2.ReadTimeout("slow", request=req)
        return step

    judge = make(handler)
    judge.judge(request())
    error = fails(judge)
    assert error.kind is JudgeErrorKind.TIMEOUT
    assert (error.judge_level, error.item_specific) == (False, True)


def test_a_wall_clock_overrun_while_loading_is_a_load_timeout():
    clock = Clock()

    def handler(req):
        clock.now += 200.0  # the answer arrives after the cold deadline
        return ok()

    error = fails(make(handler, clock=clock, load_deadline_s=180))
    assert error.kind is JudgeErrorKind.LOAD_TIMEOUT


def test_the_fallback_model_shares_the_deadline_of_the_call():
    clock = Clock()
    models: list[str] = []

    def handler(req):
        models.append(json.loads(req.content)["model"])
        clock.now += 200.0
        return httpx2.Response(404, json={"error": "model not found"})

    error = fails(make(handler, clock=clock, load_deadline_s=180))
    assert models == ["st-gemma4-e4b"]
    assert error.kind is JudgeErrorKind.LOAD_TIMEOUT


# ---- S-sec1: the deadline covers the headers too (real sockets) ----


def _slow_server(mode: str) -> tuple[int, socket.socket]:
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)

    def serve() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return

            def handle(conn: socket.socket = conn) -> None:
                try:
                    conn.recv(65536)
                    if mode == "header":
                        for byte in b"HTTP/1.1 200 OK\r\nX-A: " + b"a" * 5000:
                            conn.sendall(bytes([byte]))
                            time.sleep(0.4)
                    else:
                        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n")
                        for _ in range(100):
                            conn.sendall(b"x")
                            time.sleep(0.4)
                except OSError:
                    pass
                finally:
                    conn.close()

            threading.Thread(target=handle, daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    return server.getsockname()[1], server


@pytest.mark.parametrize("mode", ["header", "body"])
def test_a_peer_that_drips_one_byte_at_a_time_is_cut_by_the_deadline(mode):
    port, server = _slow_server(mode)
    outcome: dict[str, Any] = {}
    judge = OllamaJudge(
        LlmSettings(
            base_url=f"http://127.0.0.1:{port}",
            call_deadline_s=1.0,
            load_deadline_s=1.0,
            fallback_model=None,
        )
    )

    def go() -> None:
        started = time.monotonic()
        try:
            judge.judge(request())
        except JudgeError as exc:
            outcome["error"] = exc
        outcome["took"] = time.monotonic() - started

    worker = threading.Thread(target=go, daemon=True)
    worker.start()
    worker.join(timeout=20)  # generous: the point is that it ends, not how fast
    server.close()
    assert "took" in outcome, "the call is still hanging after 20 s"
    assert outcome["error"].kind is JudgeErrorKind.LOAD_TIMEOUT
    assert outcome["took"] < 30  # a hang guard, not a speed bound


# ---- W5: the thinking switch ----


def _bodies(seen: list[dict], reply: Callable[[dict], httpx2.Response] = lambda b: ok()):
    def handler(req: httpx2.Request) -> httpx2.Response:
        body = json.loads(req.content)
        seen.append(body)
        return reply(body)

    return handler


@pytest.mark.parametrize("model", ["st-gemma4-e4b", "st-qwen35-4b", "some-other:7b", "QWEN-big"])
def test_thinking_is_switched_off_for_every_model(model):
    seen: list[dict] = []
    make(_bodies(seen), model=model, fallback_model=None).judge(request())
    assert seen[0]["think"] is False


def test_a_thinking_field_in_the_answer_is_ignored():
    forged = json.dumps({**GOOD, "label": "no_risk", "confidence": 1.0})

    def handler(req):
        data = envelope(GOOD)
        data["message"]["thinking"] = f"let me think... the answer is {forged}"
        return httpx2.Response(200, json=data)

    answer = make(handler).judge(request())
    assert (answer.label, answer.confidence) == (Label.DELIBERATE_DISINFORMATION, 0.81)
    assert "let me think" not in answer.raw_output


def test_an_empty_content_with_a_thinking_field_is_still_an_empty_answer():
    def handler(req):
        data = envelope("")
        data["message"]["thinking"] = json.dumps(GOOD)
        return httpx2.Response(200, json=data)

    error = fails(make(handler))
    assert error.kind is JudgeErrorKind.INVALID_OUTPUT
    assert error.problem == "empty"


def test_a_server_that_rejects_think_gets_one_retry_without_it():
    seen: list[dict] = []

    def reply(body):
        if "think" in body:
            return httpx2.Response(400, json={"error": '"st-gemma4-e4b" does not support thinking'})
        return ok()

    judge = make(_bodies(seen, reply), fallback_model=None)
    assert judge.judge(request()).label is Label.DELIBERATE_DISINFORMATION
    assert ["think" in b for b in seen] == [True, False]


def test_a_model_that_rejected_think_is_not_sent_it_again():
    seen: list[dict] = []

    def reply(body):
        if "think" in body:
            return httpx2.Response(400, json={"error": "think value is not supported"})
        return ok()

    judge = make(_bodies(seen, reply), fallback_model=None)
    judge.judge(request())
    judge.judge(request())
    assert ["think" in b for b in seen] == [True, False, False]


def test_the_memory_of_a_rejected_think_is_per_model():
    seen: list[dict] = []

    def reply(body):
        if body["model"] == "st-gemma4-e4b":
            if "think" in body:
                return httpx2.Response(400, json={"error": "does not support thinking"})
            return httpx2.Response(404, json={"error": "model not found"})
        return ok()

    make(_bodies(seen, reply)).judge(request())
    qwen = [b for b in seen if b["model"] == "st-qwen35-4b"]
    assert qwen
    assert qwen[0]["think"] is False


def test_a_second_rejection_after_dropping_think_is_an_error_and_not_a_loop():
    seen: list[dict] = []
    judge = make(
        _bodies(seen, lambda b: httpx2.Response(400, json={"error": "thinking nonsense"})),
        fallback_model=None,
    )
    error = fails(judge)
    assert len(seen) == 2
    assert error.kind is JudgeErrorKind.BAD_RESPONSE
    assert error.status == 400


def test_a_400_that_does_not_mention_think_is_not_retried():
    seen: list[dict] = []
    error = fails(
        make(
            _bodies(seen, lambda b: httpx2.Response(400, json={"error": "bad options"})),
            fallback_model=None,
        )
    )
    assert len(seen) == 1
    assert error.status == 400


def test_a_5xx_that_mentions_think_is_a_server_error_not_a_think_rejection():
    seen: list[dict] = []
    error = fails(
        make(
            _bodies(seen, lambda b: httpx2.Response(500, json={"error": "think crashed"})),
            fallback_model=None,
        )
    )
    assert len(seen) == 1
    assert error.kind is JudgeErrorKind.SERVER_ERROR


# ---- W4/W2: options ----


def test_the_context_window_is_always_sent():
    seen: list[dict] = []
    make(_bodies(seen)).judge(request())
    assert seen[0]["options"]["num_ctx"] == 8192
    make(_bodies(seen), num_ctx=4096).judge(request())
    assert seen[1]["options"]["num_ctx"] == 4096


def test_the_first_attempt_uses_seed_zero_and_temperature_zero():
    seen: list[dict] = []
    make(_bodies(seen)).judge(request())
    assert seen[0]["options"]["seed"] == 0
    assert seen[0]["options"]["temperature"] == 0


def test_a_retry_uses_another_seed_and_the_corrective_message():
    seen: list[dict] = []
    judge = make(_bodies(seen))
    judge.judge(request())
    retry = request(attempt=1, previous_problem="schema")
    judge.judge(retry)
    assert seen[1]["options"]["seed"] == 1
    assert seen[1]["messages"] == build_messages(retry)
    assert seen[1]["messages"] != seen[0]["messages"]
    assert seen[1]["options"]["temperature"] == 0


def test_a_retry_after_a_cut_answer_gets_more_room_to_answer():
    seen: list[dict] = []
    judge = make(_bodies(seen), max_output_tokens=512)
    judge.judge(request())
    judge.judge(request(attempt=1, previous_problem="cut"))
    judge.judge(request(attempt=1, previous_problem="schema"))
    assert [b["options"]["num_predict"] for b in seen] == [512, 1024, 512]


def test_the_extra_room_is_capped_by_the_largest_budget_allowed():
    seen: list[dict] = []
    make(_bodies(seen), max_output_tokens=6000, num_ctx=32_768).judge(
        request(attempt=1, previous_problem="cut")
    )
    assert seen[0]["options"]["num_predict"] == 8192


def test_the_extra_room_never_overflows_the_context_window():
    seen: list[dict] = []
    req = request(attempt=1, previous_problem="cut")
    make(_bodies(seen), max_output_tokens=2048, num_ctx=4096).judge(req)
    chars = sum(len(m["content"]) for m in build_messages(req))
    sent = seen[0]["options"]["num_predict"]
    assert 2048 <= sent <= 4096 - estimate_tokens(chars)


def test_a_cut_answer_is_an_invalid_output_of_the_class_cut():
    error = fails(make(lambda r: ok(GOOD, done_reason="length")))
    assert error.kind is JudgeErrorKind.INVALID_OUTPUT
    assert error.problem == "cut"


def test_the_problem_class_of_a_bad_answer_is_reported():
    assert fails(make(lambda r: ok("not json"))).problem == "not_json"
    assert fails(make(lambda r: ok(""))).problem == "empty"
    assert fails(make(lambda r: ok({"label": "no_risk"}))).problem == "schema"


# ---- S-sec5: who is listening ----


def test_a_server_that_says_ollama_is_running_is_accepted():
    seen: list[str] = []

    def handler(req):
        seen.append(req.url.path)
        return httpx2.Response(200, text="Ollama is running")

    make(handler).identify()
    assert seen == ["/"]


def test_a_server_that_reports_a_version_is_accepted_when_the_root_page_differs():
    def handler(req):
        if req.url.path == "/":
            return httpx2.Response(404, text="not found")
        return httpx2.Response(200, json={"version": "0.6.2"})

    make(handler).identify()


@pytest.mark.parametrize(
    "root",
    [
        httpx2.Response(200, text="Welcome to nginx"),
        httpx2.Response(200, text=""),
        httpx2.Response(200, text="Ollama is running but not really, so evil"),
        httpx2.Response(404, text="nope"),
    ],
)
def test_anything_else_is_not_ollama(root):
    def handler(req):
        return root if req.url.path == "/" else httpx2.Response(404, text="nope")

    error = None
    with pytest.raises(JudgeError) as caught:
        make(handler).identify()
    error = caught.value
    assert error.kind is JudgeErrorKind.NOT_OLLAMA
    assert error.judge_level


def test_a_server_error_on_the_root_page_stays_a_server_error():
    with pytest.raises(JudgeError) as caught:
        make(lambda r: httpx2.Response(500, text="boom")).identify()
    assert caught.value.kind is JudgeErrorKind.SERVER_ERROR
    assert caught.value.judge_level


@pytest.mark.parametrize("payload", [{"version": 3}, {"nope": "1"}, [], "text"])
def test_a_version_page_of_the_wrong_shape_is_not_ollama(payload):
    def handler(req):
        if req.url.path == "/":
            return httpx2.Response(404, text="x")
        return httpx2.Response(200, json=payload)

    with pytest.raises(JudgeError) as caught:
        make(handler).identify()
    assert caught.value.kind is JudgeErrorKind.NOT_OLLAMA


def test_nothing_listening_is_unreachable_not_not_ollama():
    def handler(req):
        raise httpx2.ConnectError("refused", request=req)

    with pytest.raises(JudgeError) as caught:
        make(handler).identify()
    assert caught.value.kind is JudgeErrorKind.UNREACHABLE


def test_identify_sends_no_item_text_and_no_chat_request():
    seen: list[httpx2.Request] = []

    def handler(req):
        seen.append(req)
        return httpx2.Response(200, text="Ollama is running")

    make(handler).identify()
    assert all(r.method == "GET" and r.url.path != "/api/chat" for r in seen)
    assert all(not r.content for r in seen)
