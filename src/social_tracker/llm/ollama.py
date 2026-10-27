"""The Ollama adapter: the judge over the HTTP API of a model server on THIS machine.

Network policy. The address was validated in `LlmSettings` (plain http, `127.0.0.1` or `localhost`
only) and the connection is always made to the literal address 127.0.0.1: a name is never looked
up. This is the only plain-http client of the application and it cannot reach another host. On top
of that, per call: no redirect is followed, no proxy and no environment setting is honoured
(`trust_env=False`), no cookie or credential is sent or kept, nothing is compressed, the body is
read with a size cap and a wall-clock deadline of the whole call (a body trickled byte by byte
runs into it), and the answer is validated against the strict schema (`schema`).

What is sent: `POST /api/chat` with `stream: false`, the JSON schema as `format`, temperature 0
and a fixed seed, `keep_alive` (the model is unloaded from the GPU that long after the last call)
and, for a Qwen model, `think: false`. If the first model is not installed or cannot be loaded the
second one is tried once; any other failure is reported as it is.

Errors carry fixed phrases: never the item text, never what the model wrote.
"""

import json
import time
from collections.abc import Callable, Iterator
from typing import Any, Final

import httpx2

from social_tracker.llm.domain import (
    JudgeError,
    JudgeErrorKind,
    JudgeRequest,
    JudgeResponse,
    LlmSettings,
)
from social_tracker.llm.prompt import build_messages
from social_tracker.llm.schema import JUDGE_JSON_SCHEMA, parse_judge_output

_CHAT: Final = "/api/chat"
_TAGS: Final = "/api/tags"
_ERROR_BODY_BYTES: Final = 2048
_MAX_MODELS: Final = 200
_CONNECT_TIMEOUT_S: Final = 5.0
_SEED: Final = 0
_NOT_FOUND_HINTS: Final = ("not found", "unable to load", "failed to load", "cannot load")


def _error(kind: JudgeErrorKind, why: str = "") -> JudgeError:
    return JudgeError(kind, why)


def _wants_thinking_off(model: str) -> bool:
    """Qwen3 models think unless told not to; Gemma has no such switch and rejects it."""
    return "qwen" in model.lower()


def _chunks(response: httpx2.Response) -> Iterator[bytes]:
    """The raw body, chunk by chunk. A response that a transport already read whole (a test
    double does that) yields its content once."""
    try:
        yield from response.iter_raw()
    except httpx2.StreamConsumed:
        yield response.content


class OllamaJudge:
    def __init__(
        self,
        settings: LlmSettings,
        *,
        transport: httpx2.BaseTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._monotonic = monotonic
        endpoint = settings.endpoint
        self._client = httpx2.Client(
            base_url=endpoint.url,  # the literal loopback address, whatever the name was
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            http2=False,
            headers={"Accept": "application/json", "Accept-Encoding": "identity"},
            timeout=httpx2.Timeout(settings.call_deadline_s, connect=_CONNECT_TIMEOUT_S),
        )

    def close(self) -> None:
        self._client.close()

    # ---- the port ----

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        deadline = self._monotonic() + self._settings.call_deadline_s
        messages = build_messages(request)
        models = self._settings.models
        for index, model in enumerate(models):
            try:
                return self._chat(model, messages, deadline)
            except JudgeError as exc:
                last = index == len(models) - 1
                if exc.kind is not JudgeErrorKind.MODEL_NOT_FOUND or last:
                    raise
                self._check_deadline(deadline)
        raise _error(JudgeErrorKind.MODEL_NOT_FOUND)  # pragma: no cover - models is never empty

    def list_models(self) -> tuple[str, ...]:
        deadline = self._monotonic() + self._settings.call_deadline_s
        body = self._exchange("GET", _TAGS, None, deadline)
        try:
            data = json.loads(body)
            listed = data["models"]
            names = tuple(str(entry["name"]) for entry in listed)
        except (ValueError, TypeError, KeyError):
            raise _error(JudgeErrorKind.BAD_RESPONSE, "not a model list") from None
        return names[:_MAX_MODELS]

    # ---- one call ----

    def _chat(self, model: str, messages: list[dict[str, str]], deadline: float) -> JudgeResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": JUDGE_JSON_SCHEMA,
            "keep_alive": f"{self._settings.keep_alive_s}s",
            "options": {
                "temperature": 0,
                "seed": _SEED,
                "num_predict": self._settings.max_output_tokens,
            },
        }
        if _wants_thinking_off(model):
            payload["think"] = False
        body = self._exchange("POST", _CHAT, payload, deadline)
        content, truncated = self._content_of(body)
        if truncated:
            raise JudgeError(
                JudgeErrorKind.INVALID_OUTPUT,
                "the answer was cut by the token limit",
                raw_output=content,
            )
        return parse_judge_output(content, model=model)

    @staticmethod
    def _content_of(body: bytes) -> tuple[str, bool]:
        try:
            data = json.loads(body)
            message = data["message"]
            content = message["content"]
            done = data["done"]
        except (ValueError, TypeError, KeyError):
            raise _error(JudgeErrorKind.BAD_RESPONSE, "not a chat answer") from None
        if not isinstance(content, str) or done is not True:
            raise _error(JudgeErrorKind.BAD_RESPONSE, "not a finished chat answer")
        return content, data.get("done_reason") == "length"

    # ---- the exchange ----

    def _check_deadline(self, deadline: float) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise _error(JudgeErrorKind.TIMEOUT, "the call outlived its deadline")
        return remaining

    def _exchange(self, method: str, path: str, payload: Any, deadline: float) -> bytes:
        remaining = self._check_deadline(deadline)
        timeout = httpx2.Timeout(remaining, connect=min(_CONNECT_TIMEOUT_S, remaining))
        try:
            with self._client.stream(method, path, json=payload, timeout=timeout) as response:
                self._client.cookies.clear()
                return self._read(response, path, deadline)
        except JudgeError:
            raise
        except (httpx2.ConnectError, httpx2.ConnectTimeout):
            raise _error(JudgeErrorKind.UNREACHABLE, "nothing answered at the address") from None
        except httpx2.TimeoutException:
            raise _error(JudgeErrorKind.TIMEOUT, "the server did not answer in time") from None
        except (httpx2.HTTPError, httpx2.StreamError):
            raise _error(JudgeErrorKind.BAD_RESPONSE, "the exchange broke off") from None

    def _read(self, response: httpx2.Response, path: str, deadline: float) -> bytes:
        status = response.status_code
        if 300 <= status < 400:
            raise _error(JudgeErrorKind.BAD_RESPONSE, "a redirect was refused")
        if status >= 400:
            self._refuse(response, path, status, deadline)
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise _error(JudgeErrorKind.BAD_RESPONSE, "an encoded body was refused")
        limit = self._settings.max_response_bytes
        declared = response.headers.get("content-length", "").strip()
        if declared.isascii() and declared.isdigit() and int(declared) > limit:
            raise _error(JudgeErrorKind.BAD_RESPONSE, "the body is larger than allowed")
        return self._body(response, limit, deadline)

    def _body(self, response: httpx2.Response, limit: int, deadline: float) -> bytes:
        parts: list[bytes] = []
        size = 0
        for chunk in _chunks(response):
            size += len(chunk)
            if size > limit:
                raise _error(JudgeErrorKind.BAD_RESPONSE, "the body is larger than allowed")
            parts.append(chunk)
            self._check_deadline(deadline)
        return b"".join(parts)

    def _refuse(self, response: httpx2.Response, path: str, status: int, deadline: float) -> None:
        """Raise the error for an HTTP failure. A short piece of the error body is read only to
        tell a missing or unloadable model from other failures; it is never kept or shown."""
        hint = b""
        try:
            for chunk in _chunks(response):
                hint += chunk
                if len(hint) >= _ERROR_BODY_BYTES:
                    break
                self._check_deadline(deadline)
        except httpx2.HTTPError:
            hint = b""
        lowered = hint[:_ERROR_BODY_BYTES].decode("utf-8", "replace").lower()
        names_model = any(h in lowered for h in _NOT_FOUND_HINTS)
        if path == _CHAT and (status == 404 or (status >= 500 and names_model)):
            raise _error(JudgeErrorKind.MODEL_NOT_FOUND, "no model could answer")
        if status >= 500:
            raise _error(JudgeErrorKind.SERVER_ERROR, f"the server answered {status}")
        raise _error(JudgeErrorKind.BAD_RESPONSE, f"the server answered {status}")
