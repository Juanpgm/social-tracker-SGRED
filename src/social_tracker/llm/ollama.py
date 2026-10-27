"""The Ollama adapter: the judge over the HTTP API of a model server on THIS machine.

Network policy. The address was validated in `LlmSettings` (plain http, `127.0.0.1` or `localhost`
only) and the connection is always made to the literal address 127.0.0.1: a name is never looked
up. This is the only plain-http client of the application and it cannot reach another host. On top
of that, per call: no redirect is followed, no proxy and no environment setting is honoured
(`trust_env=False`), no cookie or credential is sent or kept, nothing is compressed, and the whole
exchange, response HEADERS included, runs under one wall-clock deadline that is enforced inside
every connect, read and write (`shared.iodeadline`, the layer the feed fetcher uses): a peer that
drips one byte at a time, in the headers or in the body, runs into it. The body is read with a size
cap and the answer is validated against the strict schema (`schema`).

What is sent: `POST /api/chat` with `stream: false`, the JSON schema as `format`, temperature 0, a
seed (0, and the attempt number on a retry, so a retry can differ), `num_ctx`, `num_predict`,
`keep_alive` (the model is unloaded from the GPU that long after the last call) and `think: false`
for every model: if the server rejects the field for a model, that model is asked once more
without it and remembered. A `thinking` field in the answer is never read. If the first model is
not installed or cannot be loaded the second one is tried once; any other failure is reported as it
is.

A call to a model that is probably not resident (never answered, or its keep-alive has run out) has
the longer `load_deadline_s`; a timeout then is a `LOAD_TIMEOUT`, which says nothing about the item.

Before any item text is sent the service asks `identify()`: the address must answer like Ollama
(`Ollama is running` at `/`, or a version at `/api/version`), because a loopback port can be held
by any local process.

Errors carry fixed phrases: never the item text, never what the model wrote.
"""

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Final

import httpx2

from social_tracker.llm.domain import (
    JUDGE_SEED,
    JudgeError,
    JudgeErrorKind,
    JudgeRequest,
    JudgeResponse,
    LlmSettings,
)
from social_tracker.llm.prompt import build_messages, estimate_tokens
from social_tracker.llm.schema import JUDGE_JSON_SCHEMA, parse_judge_output
from social_tracker.shared.iodeadline import DeadlineTransport

_CHAT: Final = "/api/chat"
_TAGS: Final = "/api/tags"
_ROOT: Final = "/"
_VERSION: Final = "/api/version"
_ROOT_BANNER: Final = b"Ollama is running"
_ERROR_BODY_BYTES: Final = 2048
_MAX_MODELS: Final = 200
_CONNECT_TIMEOUT_S: Final = 5.0
_IDENTIFY_DEADLINE_S: Final = 10.0
_MAX_PREDICT: Final = 8192
_NOT_FOUND_HINTS: Final = ("not found", "unable to load", "failed to load", "cannot load")
_THINK_STATUSES: Final = (400, 422)


def _error(kind: JudgeErrorKind, why: str = "", *, status: int | None = None) -> JudgeError:
    return JudgeError(kind, why, status=status)


class _ThinkRejected(Exception):
    """The server refused the `think` field for this model (an internal signal, never escapes)."""


@dataclass
class _Budget:
    """The wall-clock limit of one exchange and whether the model had to be loaded for it."""

    deadline: float
    cold: bool = False


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
        self._active = _Budget(0.0)  # the budget the deadline layer enforces right now
        self._warm_until: dict[str, float] = {}  # model -> when the server unloads it
        self._no_think: set[str] = set()  # models that rejected the `think` field
        endpoint = settings.endpoint
        if transport is None:
            transport = DeadlineTransport(
                httpx2.create_ssl_context(trust_env=False), self._remaining
            )
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

    def _remaining(self) -> float:
        return self._active.deadline - self._monotonic()

    # ---- the port ----

    def judge(self, request: JudgeRequest) -> JudgeResponse:
        settings = self._settings
        start = self._monotonic()
        models = settings.models
        cold = self._is_cold(models[0], start)
        budget = _Budget(start + (settings.cold_deadline_s if cold else settings.call_deadline_s))
        budget.cold = cold
        messages = build_messages(request)
        options = self._options(request, messages)
        for index, model in enumerate(models):
            if index and self._is_cold(model, self._monotonic()):
                # The fallback has to be loaded too: it may use what is left of the hard cap that
                # also bounds the lease of the caller.
                budget = _Budget(start + settings.cold_deadline_s, cold=True)
            try:
                return self._chat(model, messages, options, budget)
            except JudgeError as exc:
                last = index == len(models) - 1
                if exc.kind is not JudgeErrorKind.MODEL_NOT_FOUND or last:
                    raise
                self._check_deadline(budget)
        raise _error(JudgeErrorKind.MODEL_NOT_FOUND)  # pragma: no cover - models is never empty

    def list_models(self) -> tuple[str, ...]:
        budget = _Budget(self._monotonic() + self._settings.call_deadline_s)
        body = self._exchange("GET", _TAGS, None, budget)
        try:
            data = json.loads(body)
            listed = data["models"]
            names = tuple(str(entry["name"]) for entry in listed)
        except (ValueError, TypeError, KeyError):
            raise _error(JudgeErrorKind.BAD_RESPONSE, "not a model list") from None
        return names[:_MAX_MODELS]

    def identify(self) -> None:
        """Check that what listens at the address is Ollama, before any item text is sent. Raises
        `NOT_OLLAMA` if it is something else, and the usual errors if nothing answers."""
        limit = min(self._settings.call_deadline_s, _IDENTIFY_DEADLINE_S)
        budget = _Budget(self._monotonic() + limit)
        try:
            if self._exchange("GET", _ROOT, None, budget).strip() == _ROOT_BANNER:
                return
        except JudgeError as exc:
            if exc.kind is not JudgeErrorKind.BAD_RESPONSE:
                raise
        try:
            data = json.loads(self._exchange("GET", _VERSION, None, budget))
            version = data["version"]
        except JudgeError as exc:
            if exc.kind is not JudgeErrorKind.BAD_RESPONSE:
                raise
        except (ValueError, TypeError, KeyError):
            pass
        else:
            if isinstance(version, str) and version:
                return
        raise _error(JudgeErrorKind.NOT_OLLAMA, "the address does not answer like Ollama")

    # ---- one call ----

    def _is_cold(self, model: str, now: float) -> bool:
        """The server unloads a model `keep_alive_s` after its last answer: from that instant on
        the next call has to load it again."""
        return now >= self._warm_until.get(model, 0.0)

    def _options(self, request: JudgeRequest, messages: list[dict[str, str]]) -> dict[str, Any]:
        settings = self._settings
        predict = settings.max_output_tokens
        if request.previous_problem == "cut":
            # The answer was cut by the token cap: allow it once to be longer, as far as the
            # context window leaves room next to the prompt.
            room = settings.num_ctx - estimate_tokens(sum(len(m["content"]) for m in messages))
            predict = min(2 * predict, _MAX_PREDICT, max(predict, room))
        return {
            "temperature": 0,
            "seed": JUDGE_SEED + request.attempt,
            "num_predict": predict,
            "num_ctx": settings.num_ctx,
        }

    def _chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any],
        budget: _Budget,
    ) -> JudgeResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": JUDGE_JSON_SCHEMA,
            "keep_alive": f"{self._settings.keep_alive_s}s",
            "options": options,
        }
        if model not in self._no_think:
            payload["think"] = False
        try:
            body = self._exchange("POST", _CHAT, payload, budget)
        except _ThinkRejected:
            # The server does not know the field for this model: ask once more without it, and
            # remember. A second rejection (of something else that mentions it) is an error.
            if "think" not in payload:
                raise _error(
                    JudgeErrorKind.BAD_RESPONSE, "the request was refused", status=400
                ) from None
            self._no_think.add(model)
            del payload["think"]
            try:
                body = self._exchange("POST", _CHAT, payload, budget)
            except _ThinkRejected:
                raise _error(
                    JudgeErrorKind.BAD_RESPONSE, "the request was refused", status=400
                ) from None
        content, truncated = self._content_of(body)
        self._warm_until[model] = self._monotonic() + self._settings.keep_alive_s
        if truncated:
            raise JudgeError(
                JudgeErrorKind.INVALID_OUTPUT,
                "the answer was cut by the token limit",
                raw_output=content,
                problem="cut",
            )
        return parse_judge_output(content, model=model)

    @staticmethod
    def _content_of(body: bytes) -> tuple[str, bool]:
        """The text of the answer. Only `message.content` is read: a `thinking` field, if there
        is one, is never looked at."""
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

    def _timeout(self, budget: _Budget, why: str) -> JudgeError:
        kind = JudgeErrorKind.LOAD_TIMEOUT if budget.cold else JudgeErrorKind.TIMEOUT
        return _error(kind, why)

    def _check_deadline(self, budget: _Budget) -> float:
        remaining = budget.deadline - self._monotonic()
        if remaining <= 0:
            raise self._timeout(budget, "the call outlived its deadline")
        return remaining

    def _exchange(self, method: str, path: str, payload: Any, budget: _Budget) -> bytes:
        remaining = self._check_deadline(budget)
        self._active = budget
        timeout = httpx2.Timeout(remaining, connect=min(_CONNECT_TIMEOUT_S, remaining))
        try:
            with self._client.stream(method, path, json=payload, timeout=timeout) as response:
                self._client.cookies.clear()
                return self._read(response, path, budget)
        except (JudgeError, _ThinkRejected):
            raise
        except (httpx2.ConnectError, httpx2.ConnectTimeout):
            raise _error(JudgeErrorKind.UNREACHABLE, "nothing answered at the address") from None
        except httpx2.TimeoutException:
            raise self._timeout(budget, "the server did not answer in time") from None
        except (httpx2.HTTPError, httpx2.StreamError):
            raise _error(JudgeErrorKind.BAD_RESPONSE, "the exchange broke off") from None

    def _read(self, response: httpx2.Response, path: str, budget: _Budget) -> bytes:
        status = response.status_code
        if 300 <= status < 400:
            raise _error(JudgeErrorKind.BAD_RESPONSE, "a redirect was refused", status=status)
        if status >= 400:
            self._refuse(response, path, status, budget)
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in ("", "identity"):
            raise _error(JudgeErrorKind.BAD_RESPONSE, "an encoded body was refused")
        limit = self._settings.max_response_bytes
        declared = response.headers.get("content-length", "").strip()
        if declared.isascii() and declared.isdigit() and int(declared) > limit:
            raise _error(JudgeErrorKind.BAD_RESPONSE, "the body is larger than allowed")
        return self._body(response, limit, budget)

    def _body(self, response: httpx2.Response, limit: int, budget: _Budget) -> bytes:
        parts: list[bytes] = []
        size = 0
        for chunk in _chunks(response):
            size += len(chunk)
            if size > limit:
                raise _error(JudgeErrorKind.BAD_RESPONSE, "the body is larger than allowed")
            parts.append(chunk)
            self._check_deadline(budget)
        return b"".join(parts)

    def _refuse(self, response: httpx2.Response, path: str, status: int, budget: _Budget) -> None:
        """Raise the error for an HTTP failure. A short piece of the error body is read only to
        tell a missing or unloadable model, or a refused `think` field, from other failures; it is
        never kept or shown. Every status carries on the error: a 400/401/403/429/5xx is the
        server's trouble (judge level), not a verdict on the item."""
        hint = b""
        try:
            for chunk in _chunks(response):
                hint += chunk
                if len(hint) >= _ERROR_BODY_BYTES:
                    break
                self._check_deadline(budget)
        except httpx2.HTTPError:
            hint = b""
        lowered = hint[:_ERROR_BODY_BYTES].decode("utf-8", "replace").lower()
        names_model = any(h in lowered for h in _NOT_FOUND_HINTS)
        if path == _CHAT and (status == 404 or (status >= 500 and names_model)):
            raise _error(JudgeErrorKind.MODEL_NOT_FOUND, "no model could answer", status=status)
        if status >= 500:
            raise _error(
                JudgeErrorKind.SERVER_ERROR, f"the server answered {status}", status=status
            )
        if path == _CHAT and status in _THINK_STATUSES and "think" in lowered:
            raise _ThinkRejected
        raise _error(JudgeErrorKind.BAD_RESPONSE, f"the server answered {status}", status=status)
