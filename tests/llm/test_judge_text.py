"""`judge_text`: the judge asked about a text that is not an item (the calibration of the gold set).
It leaves no verdict, no judgment, no budget used and no trace but the answer."""

import pytest

from social_tracker.classification.domain import Label
from social_tracker.llm.domain import JudgeError, JudgeErrorKind, LlmSettings
from tests.llm.harness import memory_llm
from tests.llm.support import evidence, response


def ask(w, text="Mañana habrá un temblor fuerte en Cali", item_id="gold_00001"):
    return w.service.judge_text(item_id, text, evidence())


def test_it_returns_the_answer_of_the_judge_for_the_text_and_the_evidence():
    w = memory_llm()
    w.judge.default = response(label=Label.UNINTENTIONAL_RUMOR, confidence=0.7)
    answer = ask(w)
    assert (answer.label, answer.confidence) == (Label.UNINTENTIONAL_RUMOR, 0.7)
    (request,) = w.judge.requests
    assert request.item_id == "gold_00001"
    assert request.text == "Mañana habrá un temblor fuerte en Cali"
    assert request.evidence == evidence()
    assert request.attempt == 0


def test_it_checks_who_answers_before_sending_the_text():
    w = memory_llm()
    ask(w)
    assert w.judge.identified == 1
    w2 = memory_llm()
    w2.judge.identity = JudgeError(JudgeErrorKind.NOT_OLLAMA, "something else")
    with pytest.raises(JudgeError) as caught:
        ask(w2)
    assert caught.value.kind is JudgeErrorKind.NOT_OLLAMA
    assert w2.judge.requests == []  # the text went nowhere


def test_nothing_is_stored_and_no_budget_is_used():
    w = memory_llm()
    before = w.escalations.counts()
    ask(w)
    ask(w)
    assert w.escalations.counts() == before
    assert w.service.local_status().used_this_hour == 0
    assert w.escalations.get_judgment("gold_00001") is None


def test_it_works_with_global_analysis_off():
    w = memory_llm()
    assert w.service.local_status().analysis_on is False
    assert ask(w) is not None


def test_when_another_process_holds_the_model_it_gives_up_with_none():
    w = memory_llm()
    w.escalations.try_lease("another-process", w.clock.now(), w.clock.now().replace(year=2099))
    assert ask(w) is None
    assert w.judge.requests == []


def test_the_slot_is_given_back_even_when_the_judge_fails():
    w = memory_llm()
    w.judge.script.append(JudgeError(JudgeErrorKind.UNREACHABLE, "down"))
    with pytest.raises(JudgeError):
        ask(w)
    assert ask(w) is not None  # the next one can take the slot


def test_an_answer_that_breaks_the_schema_is_asked_again_once():
    w = memory_llm()
    w.judge.script.append(JudgeError(JudgeErrorKind.INVALID_OUTPUT, "bad", problem="not_json"))
    w.judge.script.append(response(label=Label.NO_RISK))
    answer = ask(w)
    assert answer.label is Label.NO_RISK
    assert [r.attempt for r in w.judge.requests] == [0, 1]
    assert w.judge.requests[1].previous_problem == "not_json"


def test_two_invalid_answers_in_a_row_are_an_error_not_a_guess():
    w = memory_llm()
    bad = JudgeError(JudgeErrorKind.INVALID_OUTPUT, "bad", problem="not_json")
    w.judge.script.extend([bad, bad])
    with pytest.raises(JudgeError) as caught:
        ask(w)
    assert caught.value.kind is JudgeErrorKind.INVALID_OUTPUT
    assert len(w.judge.requests) == 2


@pytest.mark.parametrize(
    ("limit_chars", "text_len", "cut"), [(100, 99, False), (100, 100, False), (100, 101, True)]
)
def test_a_long_text_is_cut_at_the_boundary_and_flagged(limit_chars, text_len, cut):
    w = memory_llm(LlmSettings(max_prompt_chars=limit_chars))
    ask(w, text="x" * text_len)
    (request,) = w.judge.requests
    assert request.text_truncated is cut
    assert len(request.text) <= limit_chars + 1  # both ends are kept, with a break between


@pytest.mark.parametrize("kind", [JudgeErrorKind.UNREACHABLE, JudgeErrorKind.TIMEOUT])
def test_a_judge_that_fails_raises_its_error_unchanged(kind):
    w = memory_llm()
    w.judge.script.append(JudgeError(kind, "boom"))
    with pytest.raises(JudgeError) as caught:
        ask(w)
    assert caught.value.kind is kind


def test_a_text_the_judge_cannot_be_asked_about_is_a_validation_error():
    from social_tracker.shared.errors import ValidationError

    w = memory_llm()
    with pytest.raises(ValidationError):
        w.service.judge_text("NOT A SLUG", "x", evidence())
    assert w.judge.requests == []
