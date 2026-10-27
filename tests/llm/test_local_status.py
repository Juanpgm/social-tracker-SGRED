"""`local_status`: what the web page may show about the model without ever calling it."""

from datetime import timedelta

from social_tracker.llm.domain import JudgeError, JudgeErrorKind
from tests.llm.harness import ScriptedJudge, memory_llm
from tests.llm.support import response


class Forbidden(ScriptedJudge):
    """A judge that fails the test the moment anything asks it anything."""

    def judge(self, request):
        raise AssertionError("the model was asked")

    def list_models(self):
        raise AssertionError("the server was asked")

    def identify(self):
        raise AssertionError("the server was asked")


def test_the_local_status_never_asks_the_judge_or_the_server():
    w = memory_llm(judge=Forbidden())
    w.ready()
    w.seed(1)
    status = w.service.local_status()
    assert status.queued == 1


def test_it_reports_the_queue_the_budget_and_the_counts_from_what_is_stored():
    w = memory_llm()
    w.ready()
    w.seed(1)
    w.seed(2)
    w.seed(3, priority=0.9)
    w.judge.default = response()
    w.service.escalate(limit=1)
    status = w.service.local_status()
    assert status.queued == 2
    assert status.used_this_hour == 1
    assert status.counts.judged == 1
    assert status.analysis_on is True
    assert status.hourly_cap == 200
    assert status.configured == ("st-gemma4-e4b", "st-qwen35-4b")
    assert status.base_url == "http://127.0.0.1:11434"


def test_the_budget_of_a_new_hour_starts_again():
    w = memory_llm()
    w.ready()
    w.seed(1)
    w.service.escalate()
    assert w.service.local_status().used_this_hour == 1
    w.clock.advance(timedelta(hours=1))
    assert w.service.local_status().used_this_hour == 0


def test_with_analysis_off_it_says_so():
    w = memory_llm()
    assert w.service.local_status().analysis_on is False


def test_a_judge_that_is_down_changes_nothing_here():
    w = memory_llm()
    w.ready()
    w.judge.identity = JudgeError(JudgeErrorKind.UNREACHABLE, "down")
    assert w.service.local_status().configured
