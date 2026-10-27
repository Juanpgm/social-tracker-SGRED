"""`classify escalate`, `llm status` and `llm ping`: the escalation from the shell.

No test starts or reaches a model server: the judge is a script, and a few tests use the stand-in
server of `tests.llm.fake_ollama` on the loopback interface.
"""

import re

import pytest

from social_tracker.classification.domain import Label
from social_tracker.llm.domain import PROMPT_VERSION, JudgeError, JudgeErrorKind
from tests.entrypoints.conftest import WHY, Cli, waiting

# ---- classify escalate ----


def test_escalate_judges_the_queued_items_and_reports_what_it_did(queued: Cli, judge):
    before = waiting(queued)
    assert before >= 1
    result = queued.ok("classify", "escalate")
    assert f"judged {before} item(s)" in result.stdout
    assert judge.calls == before
    assert waiting(queued) == 0
    assert "stage: llm" in queued.ok("classify", "show", judge.requests[0].item_id).stdout


def test_a_dry_run_says_what_would_happen_and_changes_nothing(queued: Cli, judge):
    before = waiting(queued)
    result = queued.ok("classify", "escalate", "--dry-run")
    assert f"dry run: {before} item(s) would be sent" in result.stdout
    assert judge.calls == 0
    assert waiting(queued) == before


def test_the_limit_bounds_the_batch(queued: Cli, judge):
    total = waiting(queued)
    assert total >= 2
    result = queued.ok("classify", "escalate", "--limit", "1")
    assert "judged 1 item(s)" in result.stdout
    assert waiting(queued) == total - 1


@pytest.mark.parametrize("limit", ["0", "-1", "1001", "x", "2.5"])
def test_a_bad_limit_is_a_usage_error(queued: Cli, judge, limit):
    result = queued("classify", "escalate", "--limit", limit)
    assert result.exit_code == 2
    assert judge.calls == 0


def test_nothing_eligible_is_exit_three(cli: Cli, judge):
    cli.add()
    result = cli("classify", "escalate")
    assert result.exit_code == 3
    assert "no source is eligible" in result.stderr
    assert result.stdout == ""


def test_with_global_analysis_off_nothing_is_sent(queued: Cli, judge):
    queued.ok("analysis", "off", "--reason", WHY)
    result = queued("classify", "escalate")
    assert result.exit_code == 3
    assert judge.calls == 0


def test_a_judge_that_is_down_is_exit_three_and_the_items_are_kept_for_later(queued: Cli, judge):
    judge.default = JudgeError(JudgeErrorKind.UNREACHABLE)
    before = waiting(queued)
    result = queued("classify", "escalate")
    assert result.exit_code == 3  # nothing could be judged: the judge is not there
    assert "not reachable" in result.stderr
    assert "1 failed" in result.stdout
    assert waiting(queued) == before  # still queued, with a backoff


def test_invalid_answers_are_reported_and_leave_the_cpu_verdicts(queued: Cli, judge):
    garbage = JudgeError(JudgeErrorKind.INVALID_OUTPUT, raw_output="garbage")
    # the judge answers the first item, so the next one's double failure is that item's own
    judge.script.extend([None, garbage, garbage])
    result = queued.ok("classify", "escalate", "--limit", "2")
    assert "1 invalid" in result.stdout
    shown = queued.ok("classify", "show", judge.requests[1].item_id).stdout
    assert "stage: rules" in shown
    assert "llm: two invalid answers" in shown


def test_the_hourly_cap_is_reported(queued: Cli, judge):
    result = queued.ok("classify", "escalate", env={"SOCIAL_TRACKER_LLM_HOURLY_CAP": "1"})
    assert "judged 1 item(s)" in result.stdout
    assert "hourly cap" in result.stdout


def test_a_bad_setting_is_refused_before_anything_is_asked(queued: Cli, judge):
    result = queued(
        "classify", "escalate", env={"SOCIAL_TRACKER_LLM_BASE_URL": "http://evil.example"}
    )
    assert result.exit_code == 1
    assert "SOCIAL_TRACKER_LLM_BASE_URL" in result.stderr
    assert judge.calls == 0


# ---- classify show / stats ----


def test_show_names_the_model_and_the_prompt_version_of_a_judged_item(queued: Cli, judge):
    queued.ok("classify", "escalate", "--limit", "1")
    item = judge.requests[0].item_id
    shown = queued.ok("classify", "show", item).stdout
    assert f"llm: judged by st-gemma4-e4b (prompt {PROMPT_VERSION})" in shown
    assert "label: no_risk" in shown


def test_show_of_a_failed_item_tells_when_it_will_be_tried_again(queued: Cli, judge):
    judge.default = JudgeError(JudgeErrorKind.UNREACHABLE)
    queued("classify", "escalate", "--limit", "1")
    shown = queued.ok("classify", "show", judge.requests[0].item_id).stdout
    assert "escalation_failed" in shown
    assert "next attempt" in shown


def test_stats_count_the_llm_stage(queued: Cli, judge):
    queued.ok("classify", "escalate", "--limit", "1")
    out = queued.ok("classify", "stats").stdout
    assert "llm judged: 1" in out
    assert "llm dropped: 0" in out
    assert "llm parked: 0" in out


def test_stats_report_dropped_items(queued: Cli, judge):
    queued.ok(
        "classify",
        "escalate",
        env={"SOCIAL_TRACKER_LLM_QUEUE_MAX": "1", "SOCIAL_TRACKER_LLM_HOURLY_CAP": "1"},
    )
    out = queued.ok("classify", "stats").stdout
    assert re.search(r"llm dropped: [1-9]", out)


# ---- llm status ----


def test_status_shows_reachability_models_queue_budget_and_counts(queued: Cli, judge):
    out = queued.ok("llm", "status").stdout
    assert "server: http://127.0.0.1:11434" in out
    assert "reachable: yes" in out
    assert "installed: st-gemma4-e4b:latest, st-qwen35-4b:latest" in out
    assert "missing: none" in out
    assert re.search(r"waiting: [1-9]", out)
    assert "budget this hour: 0 of 200" in out
    assert "global analysis: on" in out


def test_status_names_the_models_that_are_not_installed(queued: Cli, judge):
    judge.models = ("st-qwen35-4b:latest",)
    assert "missing: st-gemma4-e4b" in queued.ok("llm", "status").stdout


def test_status_of_a_server_that_is_down_says_so_and_still_exits_zero(queued: Cli, judge):
    judge.models = JudgeError(JudgeErrorKind.UNREACHABLE)
    out = queued.ok("llm", "status").stdout
    assert "reachable: no (llm_unreachable)" in out
    assert re.search(r"waiting: [1-9]", out)


def test_status_neutralises_control_characters_in_what_a_server_says(queued: Cli, judge):
    judge.models = ("evil\x1b[31mmodel\x07",)
    out = queued.ok("llm", "status").stdout
    assert "\x1b" not in out
    assert "\x07" not in out


# ---- llm ping ----


def test_ping_reports_the_model_that_answered(cli: Cli, judge):
    out = cli.ok("llm", "ping").stdout
    assert "ok: st-gemma4-e4b answered" in out
    assert judge.calls == 1


def test_ping_that_fails_is_exit_one_with_the_reason(cli: Cli, judge):
    judge.default = JudgeError(JudgeErrorKind.UNREACHABLE)
    result = cli("llm", "ping")
    assert result.exit_code == 1
    assert "llm_unreachable" in result.stderr
    assert result.stdout == ""


def test_ping_touches_no_verdict(queued: Cli, judge):
    before = waiting(queued)
    queued.ok("llm", "ping")
    assert waiting(queued) == before
    assert "budget this hour: 0 of 200" in queued.ok("llm", "status").stdout


def test_the_llm_group_lists_its_commands(cli: Cli):
    out = cli.ok("llm", "--help").stdout
    assert "status" in out
    assert "ping" in out


# ---- the real adapter across the loopback, against the stand-in server ----


def test_status_and_ping_over_a_real_loopback_socket(queued: Cli):
    import json

    from tests.llm.fake_ollama import FakeOllama, reply

    verdict = {"label": "no_risk", "confidence": 0.9, "rationale": "fine", "evidence": []}
    chat = {
        "model": "st-gemma4-e4b",
        "message": {"role": "assistant", "content": json.dumps(verdict)},
        "done": True,
        "done_reason": "stop",
    }
    with FakeOllama() as server:
        server.script.default = None
        banner = reply(200, "Ollama is running")  # who is there, asked before anything else
        server.script.steps.extend(
            [
                banner,
                reply(200, {"models": [{"name": "st-gemma4-e4b:latest"}]}),
                banner,
                reply(200, chat),
            ]
        )
        env = {"SOCIAL_TRACKER_LLM_BASE_URL": server.url}
        status = queued.ok("llm", "status", env=env).stdout
        ping = queued.ok("llm", "ping", env=env).stdout
        assert "reachable: yes" in status
        assert "missing: st-qwen35-4b" in status
        assert "ok: st-gemma4-e4b answered" in ping
        assert [r.path for r in server.requests] == ["/", "/api/tags", "/", "/api/chat"]
        assert Label.NO_RISK.value in ping
