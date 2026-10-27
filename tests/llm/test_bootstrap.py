"""LLM settings from the environment and the wiring of the escalation."""

import threading
from pathlib import Path

import pytest

from social_tracker import bootstrap
from social_tracker.bootstrap import (
    DB_PATH_ENV,
    LLM_BASE_URL_ENV,
    LLM_BATCH_ENV,
    LLM_FALLBACK_MODEL_ENV,
    LLM_HOURLY_CAP_ENV,
    LLM_KEEP_ALIVE_ENV,
    LLM_MAX_FAILURES_ENV,
    LLM_MODEL_ENV,
    LLM_PROMPT_CHARS_ENV,
    LLM_QUEUE_MAX_ENV,
    LLM_RETRY_BASE_ENV,
    LLM_RETRY_CAP_ENV,
    LLM_TIMEOUT_ENV,
    build_container,
    build_llm,
    build_worker,
    parse_llm_settings,
)
from social_tracker.llm.domain import JudgeError, JudgeErrorKind, LlmSettings
from social_tracker.llm.ollama import OllamaJudge
from social_tracker.shared.errors import ValidationError
from tests.collector.harness import FakeFetcher
from tests.llm.harness import ScriptedJudge


def test_no_variable_means_the_defaults():
    assert parse_llm_settings({}) == LlmSettings()
    blank = {name: "  " for name in (LLM_BASE_URL_ENV, LLM_MODEL_ENV, LLM_HOURLY_CAP_ENV)}
    assert parse_llm_settings(blank) == LlmSettings()


def test_every_variable_is_read():
    env = {
        LLM_BASE_URL_ENV: "http://localhost:9999",
        LLM_MODEL_ENV: "my-model",
        LLM_FALLBACK_MODEL_ENV: "my-backup:7b",
        LLM_KEEP_ALIVE_ENV: "120",
        LLM_TIMEOUT_ENV: "45",
        LLM_HOURLY_CAP_ENV: "150",
        LLM_QUEUE_MAX_ENV: "500",
        LLM_BATCH_ENV: "7",
        LLM_MAX_FAILURES_ENV: "5",
        LLM_RETRY_BASE_ENV: "30",
        LLM_RETRY_CAP_ENV: "900",
        LLM_PROMPT_CHARS_ENV: "4000",
    }
    s = parse_llm_settings(env)
    assert s.base_url == "http://localhost:9999"
    assert (s.model, s.fallback_model) == ("my-model", "my-backup:7b")
    assert (s.keep_alive_s, s.call_deadline_s) == (120, 45.0)
    assert (s.hourly_cap, s.queue_max, s.batch_limit, s.max_failures) == (150, 500, 7, 5)
    assert (s.retry_base_s, s.retry_cap_s, s.max_prompt_chars) == (30, 900, 4000)


def test_the_fallback_can_be_switched_off():
    assert parse_llm_settings({LLM_FALLBACK_MODEL_ENV: "none"}).fallback_model is None
    assert parse_llm_settings({LLM_FALLBACK_MODEL_ENV: "NONE"}).fallback_model is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (LLM_BASE_URL_ENV, "https://127.0.0.1:11434"),
        (LLM_BASE_URL_ENV, "http://example.org:11434"),
        (LLM_BASE_URL_ENV, "http://10.0.0.1:11434"),
        (LLM_BASE_URL_ENV, "http://127.0.0.1:11434/api"),
        (LLM_BASE_URL_ENV, "http://user@127.0.0.1:11434"),
        (LLM_BASE_URL_ENV, "http://localhost.evil.example"),
        (LLM_BASE_URL_ENV, "127.0.0.1:11434"),
        (LLM_BASE_URL_ENV, "http://127.0.0.1:99999"),
        (LLM_MODEL_ENV, "has space"),
        (LLM_MODEL_ENV, "x" * 101),
        (LLM_FALLBACK_MODEL_ENV, "bad;name"),
        (LLM_KEEP_ALIVE_ENV, "0"),
        (LLM_KEEP_ALIVE_ENV, "86401"),
        (LLM_KEEP_ALIVE_ENV, "5m"),
        (LLM_TIMEOUT_ENV, "0"),
        (LLM_TIMEOUT_ENV, "601"),
        (LLM_TIMEOUT_ENV, "1.5"),
        (LLM_HOURLY_CAP_ENV, "0"),
        (LLM_HOURLY_CAP_ENV, "100001"),
        (LLM_HOURLY_CAP_ENV, "-1"),
        (LLM_HOURLY_CAP_ENV, "two hundred"),
        (LLM_QUEUE_MAX_ENV, "0"),
        (LLM_BATCH_ENV, "1001"),
        (LLM_MAX_FAILURES_ENV, "0"),
        (LLM_RETRY_BASE_ENV, "0"),
        (LLM_RETRY_CAP_ENV, "604801"),
        (LLM_PROMPT_CHARS_ENV, "99"),
        (LLM_PROMPT_CHARS_ENV, "100001"),
    ],
)
def test_an_invalid_value_is_refused_naming_the_variable_and_never_echoing_the_value(name, value):
    with pytest.raises(ValidationError) as caught:
        parse_llm_settings({name: value})
    assert name in str(caught.value)
    assert len(value) < 5 or value not in str(caught.value)  # short ones are in the range text


def test_a_retry_cap_under_the_base_is_refused():
    with pytest.raises(ValidationError) as caught:
        parse_llm_settings({LLM_RETRY_BASE_ENV: "120", LLM_RETRY_CAP_ENV: "60"})
    assert LLM_RETRY_CAP_ENV in str(caught.value)


@pytest.mark.parametrize(
    ("name", "value", "field", "expected"),
    [
        (LLM_HOURLY_CAP_ENV, "1", "hourly_cap", 1),
        (LLM_HOURLY_CAP_ENV, "100000", "hourly_cap", 100000),
        (LLM_TIMEOUT_ENV, "1", "call_deadline_s", 1.0),
        (LLM_TIMEOUT_ENV, "600", "call_deadline_s", 600.0),
        (LLM_BATCH_ENV, "1000", "batch_limit", 1000),
        (LLM_KEEP_ALIVE_ENV, "86400", "keep_alive_s", 86400),
        (LLM_PROMPT_CHARS_ENV, "100", "max_prompt_chars", 100),
    ],
)
def test_the_ends_of_every_range_are_accepted(name, value, field, expected):
    assert getattr(parse_llm_settings({name: value}), field) == expected


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {DB_PATH_ENV: str(tmp_path / "llm.db")}


def test_an_invalid_variable_stops_the_start_before_the_database_is_opened(tmp_path: Path):
    path = tmp_path / "never.db"
    with pytest.raises(ValidationError):
        build_container({DB_PATH_ENV: str(path), LLM_BASE_URL_ENV: "http://example.org"})
    assert not path.exists()


def test_the_container_never_creates_a_network_client(env, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an HTTP client was created")

    monkeypatch.setattr(bootstrap, "OllamaJudge", forbidden)
    container = build_container(env)
    try:
        # reading what the escalation kept needs no judge; asking one is answered "unreachable"
        assert container.escalation.counts().judged == 0
        with pytest.raises(JudgeError) as caught:
            container.escalation._judge.judge(None)  # type: ignore[arg-type]
        assert caught.value.kind is JudgeErrorKind.UNREACHABLE
        assert container.llm_settings == LlmSettings()
    finally:
        container.close()


def test_build_llm_wires_the_real_adapter_with_the_settings_of_the_environment(env):
    env[LLM_MODEL_ENV] = "other-model"
    built = build_llm(env, stop=threading.Event())
    try:
        assert isinstance(built.judge, OllamaJudge)
        assert built.settings.model == "other-model"
        assert built.service is not built.container.escalation
    finally:
        built.close()


def test_build_llm_takes_a_judge_for_tests_and_tooling(env):
    judge = ScriptedJudge()
    built = build_llm(env, judge=judge)
    try:
        assert built.judge is judge
        assert built.service.status().reachable is True
    finally:
        built.close()


def test_the_worker_escalates_with_the_real_adapter_by_default(env):
    built = build_worker(env)
    try:
        assert built.worker._escalate is not None
    finally:
        built.close()


def test_a_replaced_fetcher_leaves_the_escalation_off_so_no_test_reaches_a_model(env):
    built = build_worker(env, fetcher=FakeFetcher())
    try:
        assert built.worker._escalate is None
    finally:
        built.close()


def test_a_replaced_fetcher_with_a_judge_escalates_with_that_judge(env):
    judge = ScriptedJudge()
    built = build_worker(env, fetcher=FakeFetcher(), judge=judge)
    try:
        assert built.worker._escalate is not None
        built.control.set_global_analysis(True, actor="test", reason="test")
        summary = built.worker.run_pass()
        assert summary.escalation is not None
        assert judge.calls == 0  # nothing was queued
    finally:
        built.close()


def test_a_stop_requested_before_the_pass_skips_the_escalation(env):
    stop = threading.Event()
    judge = ScriptedJudge()
    built = build_worker(env, fetcher=FakeFetcher(), judge=judge, stop=stop)
    try:
        built.control.set_global_analysis(True, actor="test", reason="test")
        stop.set()
        assert built.worker.run_pass().escalation is None
    finally:
        built.close()
