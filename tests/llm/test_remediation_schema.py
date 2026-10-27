"""The strict answer schema: every refusal names its problem class, the numeric bounds are
enforced on this side too, and a percentage is refused (the prompt asks for 0.0 to 1.0)."""

import json

import pytest

from social_tracker.llm.domain import JudgeError, JudgeErrorKind
from social_tracker.llm.schema import JUDGE_JSON_SCHEMA, MAX_OUTPUT_CHARS, parse_judge_output

GOOD = {"label": "no_risk", "confidence": 0.5, "rationale": "ok", "evidence": []}


def refused(raw) -> JudgeError:
    with pytest.raises(JudgeError) as caught:
        parse_judge_output(raw, model="m")
    assert caught.value.kind is JudgeErrorKind.INVALID_OUTPUT
    return caught.value


def doc(**changes) -> str:
    return json.dumps({**GOOD, **changes})


@pytest.mark.parametrize(
    ("raw", "problem"),
    [
        ("", "empty"),
        ("   \n", "empty"),
        (None, "empty"),
        (b"{}", "empty"),
        pytest.param("x" * (MAX_OUTPUT_CHARS + 1), "too_large", id="too_large"),
        ("not json at all", "not_json"),
        ('{"label": "no_risk"} {"label": "no_risk"}', "not_json"),
        ('{"confidence": NaN}', "not_json"),
        ('{"label": "a", "label": "b"}', "not_json"),
        ("[]", "not_object"),
        ('"text"', "not_object"),
        ("3", "not_object"),
        ("{}", "schema"),
        (doc(label="made_up"), "schema"),
        (doc(extra=1), "schema"),
    ],
)
def test_each_refusal_names_its_problem_class(raw, problem):
    assert refused(raw).problem == problem


@pytest.mark.parametrize("confidence", [85, 100, 1.0000001, -0.0001, -1, 1e9])
def test_a_confidence_outside_zero_to_one_is_refused_never_rescaled(confidence):
    assert refused(doc(confidence=confidence)).problem == "schema"


@pytest.mark.parametrize("confidence", ["0.5", True, None, [0.5], {"v": 0.5}])
def test_a_confidence_of_another_type_is_refused(confidence):
    assert refused(doc(confidence=confidence)).problem == "schema"


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0, 0.5, 0.123456])
def test_the_closed_bounds_are_accepted(confidence):
    assert parse_judge_output(doc(confidence=confidence), model="m").confidence == confidence


def test_the_server_schema_states_the_same_bounds():
    props = JUDGE_JSON_SCHEMA["properties"]["confidence"]
    assert (props["minimum"], props["maximum"]) == (0, 1)


def test_a_rationale_over_the_input_cap_is_refused_but_a_long_one_is_cut():
    assert refused(doc(rationale="x" * 2001)).problem == "schema"
    cut = parse_judge_output(doc(rationale="x" * 2000), model="m")
    assert len(cut.rationale) == 300


def test_an_empty_rationale_is_refused():
    assert refused(doc(rationale="")).problem == "schema"


def test_more_than_five_evidence_strings_are_refused():
    assert refused(doc(evidence=["a"] * 6)).problem == "schema"
    assert len(parse_judge_output(doc(evidence=["a"] * 5), model="m").evidence) == 5


def test_the_refusal_message_never_repeats_the_model_text():
    error = refused("ignore all rules SECRET-PHRASE-42")
    assert "SECRET-PHRASE-42" not in str(error)
    assert "SECRET-PHRASE-42" in error.raw_output  # kept for the audit only
