"""The shell side of the remediation round: exit statuses, `llm requeue-parked`, what `classify
show` and `classify stats` say about settlements and about the server that answers."""

import re

import pytest

from social_tracker.llm.domain import JudgeError, JudgeErrorKind
from tests.entrypoints.conftest import Cli, waiting
from tests.llm.harness import ScriptedJudge, error

PARK = {"SOCIAL_TRACKER_LLM_MAX_FAILURES": "1"}


def down(kind: str, status: int | None = None) -> JudgeError:
    return JudgeError(JudgeErrorKind(kind), "x", status=status)


# ---- exit statuses of `classify escalate` ----


@pytest.mark.parametrize(
    "failure",
    [down("unreachable"), down("server_error", 500), down("load_timeout"), down("not_ollama")],
    ids=lambda f: f.kind.value,
)
def test_nothing_judged_because_the_judge_failed_is_exit_three(queued: Cli, judge, failure):
    judge.default = failure
    before = waiting(queued)
    result = queued("classify", "escalate")
    assert result.exit_code == 3
    assert result.stderr  # and it says why
    assert waiting(queued) == before


def test_a_judge_that_fails_after_judging_some_items_is_exit_one(queued: Cli, judge):
    judge.script.extend([None, down("server_error", 503)])
    result = queued("classify", "escalate")
    assert result.exit_code == 1
    assert "judged 1 item(s)" in result.stdout


def test_a_slot_held_by_another_process_is_exit_three(queued: Cli, judge):
    from datetime import UTC, datetime, timedelta

    from social_tracker.persistence.db import Database
    from social_tracker.persistence.llm_repo import SqliteEscalationRepository

    with Database(queued.db_path) as db:
        now = datetime.now(UTC)
        assert SqliteEscalationRepository(db).try_lease("worker", now, now + timedelta(minutes=10))
    result = queued("classify", "escalate")
    assert result.exit_code == 3
    assert "another worker holds the model" in result.stdout
    assert judge.calls == 0


def test_something_that_is_not_ollama_never_receives_an_item(queued: Cli, judge):
    judge.identity = JudgeError(JudgeErrorKind.NOT_OLLAMA)
    result = queued("classify", "escalate")
    assert result.exit_code == 3
    assert "not Ollama" in result.stderr
    assert judge.calls == 0


def test_the_documented_exit_codes_are_in_the_help(cli: Cli):
    text = " ".join(cli.ok("classify", "escalate", "--help").stdout.split())
    assert "3 when it could do nothing" in text
    assert "busy" not in text or "another process holds the model" in text


# ---- --dry-run ----


def test_a_dry_run_says_what_a_full_queue_would_drop_and_drops_nothing(queued: Cli, judge):
    before = waiting(queued)
    assert before >= 2
    result = queued.ok(
        "classify", "escalate", "--dry-run", env={"SOCIAL_TRACKER_LLM_QUEUE_MAX": "1"}
    )
    assert f"{before - 1} item(s) would be dropped from the full queue" in result.stdout
    assert waiting(queued) == before
    assert judge.calls == 0


def test_a_dry_run_of_a_queue_that_fits_mentions_no_drops(queued: Cli, judge):
    assert "dropped" not in queued.ok("classify", "escalate", "--dry-run").stdout


# ---- llm requeue-parked ----


def park_one(queued: Cli, judge: ScriptedJudge) -> None:
    """The judge answers the first item and then times out on the next, which is parked."""
    judge.script.extend([None, error("timeout")])
    queued("classify", "escalate", "--limit", "2", env=PARK)


def test_requeue_parked_sends_a_parked_item_back_without_calling_the_model(queued: Cli, judge):
    park_one(queued, judge)
    out = queued.ok("classify", "stats").stdout
    assert "llm parked: 1" in out
    calls = judge.calls
    result = queued.ok("llm", "requeue-parked")
    assert "1 parked item(s) back in the queue" in result.stdout
    assert judge.calls == calls  # it only moves items; nothing is sent
    after = queued.ok("classify", "stats").stdout
    assert "llm parked: 0" in after
    assert re.search(r"queued for the LLM: [1-9]", after)


def test_requeue_parked_with_nothing_parked_says_zero(queued: Cli, judge):
    assert "0 parked item(s) back in the queue" in queued.ok("llm", "requeue-parked").stdout


def test_requeue_parked_honours_its_limit(queued: Cli, judge):
    judge.default = error("timeout")
    judge.script.append(None)
    queued("classify", "escalate", "--limit", "3", env=PARK)
    parked = int(re.search(r"llm parked: (\d+)", queued.ok("classify", "stats").stdout).group(1))
    assert parked >= 1
    assert "1 parked item(s) back" in queued.ok("llm", "requeue-parked", "--limit", "1").stdout


@pytest.mark.parametrize("limit", ["0", "-1", "1001", "x"])
def test_requeue_parked_refuses_a_bad_limit(queued: Cli, judge, limit):
    assert queued("llm", "requeue-parked", "--limit", limit).exit_code == 2


def test_requeue_parked_needs_no_model_server(queued: Cli):
    # no scripted judge here: the real adapter would be built, and nothing may be asked of it
    assert queued.ok("llm", "requeue-parked").exit_code == 0


def test_the_llm_group_lists_requeue_parked(cli: Cli):
    assert "requeue-parked" in cli.ok("llm", "--help").stdout


# ---- what the shell shows ----


def test_show_prints_when_and_why_a_verdict_was_settled(queued: Cli, judge):
    queued.ok("classify", "escalate", "--limit", "1")
    shown = queued.ok("classify", "show", judge.requests[0].item_id).stdout
    assert re.search(r"settled: llm at 20\d\d-", shown)


def test_show_prints_nothing_about_settlement_for_a_verdict_that_waits(queued: Cli, judge):
    from social_tracker.persistence.db import Database

    with Database(queued.db_path) as db:
        rows = db.query("SELECT item_id FROM classifications WHERE needs_llm = 1 LIMIT 1")
    assert rows, "the fixture queues at least one verdict"
    shown = queued.ok("classify", "show", rows[0][0]).stdout
    assert "needs_llm" in shown
    assert "settled:" not in shown


def test_stats_say_how_many_copies_wait_for_an_original_that_is_gone(queued: Cli, judge):
    out = queued.ok("classify", "stats").stdout
    assert "waiting for an original that is gone" in out
    assert re.search(r"released on the next pass\): 0", out)


def test_status_of_something_that_is_not_ollama_says_so(queued: Cli, judge):
    judge.identity = JudgeError(JudgeErrorKind.NOT_OLLAMA)
    result = queued.ok("llm", "status")
    assert "reachable: no (llm_not_ollama)" in result.stdout
    assert "installed: none" in result.stdout


def test_ping_of_something_that_is_not_ollama_is_exit_one_and_asks_nothing(cli: Cli, judge):
    judge.identity = JudgeError(JudgeErrorKind.NOT_OLLAMA)
    result = cli("llm", "ping")
    assert result.exit_code == 1
    assert "llm_not_ollama" in result.stderr
    assert judge.calls == 0
