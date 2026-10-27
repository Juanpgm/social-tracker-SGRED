"""The YAML form of a gold set: what `gold export` writes and `classify calibrate --file` reads."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from social_tracker.classification.domain import Label
from social_tracker.gold.domain import MAX_GOLD_EXAMPLES, make_example
from social_tracker.gold.yaml_io import MAX_FILE_BYTES, dump_examples, load_document, load_samples
from social_tracker.shared.errors import ValidationError

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
SEED = Path(__file__).parents[1] / "data" / "gold_examples.yaml"

DUPLICATE_IDS = (
    "examples: [{id: a, text: hi, expect: {label: no_risk}}, "
    "{id: a, text: ho, expect: {label: no_risk}}]"
)
AWKWARD = [
    "Mañana habrá un temblor: ¡compartan!",
    "- esto parece una lista: y esto una clave",
    "key: value\nother: [1, 2]",
    "!!python/object/apply:os.system ['echo hi']",
    "&anchor *alias {a: 1} ? maybe",
    "\"quoted\" and 'single' and # not a comment",
    "null",
    "true",
    "123",
    "2026-03-01",
    "Texto con emoji \U0001f6a8 y tildes áéíóú ñ",
    "a" * 5000,
]


def example(text: str, label: Label = Label.NO_RISK, **overrides):
    fields = {
        "text": text,
        "label": label,
        "note": "",
        "credibility": 3,
        "url": None,
        "added_at": NOW,
        "actor": "ana",
    }
    fields.update(overrides)
    return make_example(**fields)


@pytest.mark.parametrize("text", AWKWARD)
def test_any_text_survives_a_round_trip_and_nothing_is_executed_or_interpreted(text):
    original = example(text, Label.UNINTENTIONAL_RUMOR, note="n: o # t", credibility=4)
    (back,) = load_samples(dump_examples([original]))
    assert back.text == original.text
    assert back.expected == (Label.UNINTENTIONAL_RUMOR,)
    assert back.credibility == 4
    assert back.id == original.id


def test_the_address_survives_and_an_empty_set_is_a_valid_file():
    with_url = example("t", url="https://official-domain.example/avisos/1")
    (back,) = load_samples(dump_examples([with_url]))
    assert back.url == "https://official-domain.example/avisos/1"
    assert load_samples(dump_examples([])) == ()


def test_the_export_says_it_may_hold_real_text_and_is_in_the_shape_of_the_seed():
    text = dump_examples([example("hola", Label.OUT_OF_SCOPE)])
    assert text.startswith("# ")
    assert "do not commit" in text.splitlines()[0].lower()
    assert "version: 1" in text
    assert "in_scope: false" in text
    assert "label: out_of_scope" in text


def test_every_label_goes_through():
    exported = [example(f"t{i}", label) for i, label in enumerate(Label)]
    back = load_samples(dump_examples(exported))
    assert [s.expected[0] for s in back] == list(Label)


# ---- the seed file of the repository ----


def test_the_seed_file_loads_with_its_acceptable_label_lists():
    samples = load_samples(SEED.read_text(encoding="utf-8"))
    assert 55 <= len(samples) <= 80
    by_id = {s.id: s for s in samples}
    assert by_id["report-with-unlinked-attribution"].expected == (
        Label.UNCLASSIFIED,
        Label.DELIBERATE_DISINFORMATION,
    )
    assert by_id["prediction-tomorrow"].expected == (Label.UNINTENTIONAL_RUMOR,)
    assert by_id["official-served-from-official-address"].url.startswith("https://")
    off_topic = [s for s in samples if s.expected == (Label.OUT_OF_SCOPE,)]
    assert off_topic
    assert len({s.id for s in samples}) == len(samples)
    assert not any(s.id.startswith("gap") for s in samples)  # the known gaps are not examples


# ---- hostile and malformed files ----


@pytest.mark.parametrize(
    "document",
    [
        "!!python/object/apply:os.system ['echo hi']",
        "examples: !!python/object:os.system {}",
        "examples: &a [*a]",
        "x: &a 1\nexamples: *a",
        "examples: []\nexamples: []",
        "examples: []\n---\nexamples: []",
        "<<: {a: 1}\nexamples: []",
        "examples: [" + ",".join(["[" * 3] * 20) + "]",
    ],
)
def test_unsafe_yaml_is_refused_without_echoing_it(document):
    with pytest.raises(ValidationError) as caught:
        load_samples(document)
    assert "python" not in caught.value.message
    assert "os.system" not in caught.value.message


@pytest.mark.parametrize(
    "document",
    [
        "",
        "   ",
        "[]",
        "just text",
        "version: 1",
        "examples: 5",
        "examples: {a: 1}",
        "examples: [5]",
        "examples: [{id: a}]",
        "examples: [{id: a, text: hi}]",
        "examples: [{id: a, text: hi, expect: 5}]",
        "examples: [{id: a, text: hi, expect: {label: made_up}}]",
        "examples: [{id: a, text: hi, expect: {label: [no_risk, made_up]}}]",
        "examples: [{id: a, text: hi, expect: {label: []}}]",
        "examples: [{id: a, text: hi, expect: {label: no_risk}, credibility: 9}]",
        "examples: [{id: a, text: hi, expect: {label: no_risk}, credibility: yes}]",
        "examples: [{id: a, text: hi, expect: {label: no_risk}, url: 5}]",
        "examples: [{id: '', text: hi, expect: {label: no_risk}}]",
        "examples: [{id: 7, text: hi, expect: {label: no_risk}}]",
        "examples: [{text: hi, expect: {label: no_risk}}]",
        "examples: [{id: a, text: '', expect: {label: no_risk}}]",
        "examples: [{id: a, text: 5, expect: {label: no_risk}}]",
        DUPLICATE_IDS,
    ],
)
def test_a_malformed_file_is_a_validation_error(document):
    with pytest.raises(ValidationError):
        load_samples(document)


def test_a_file_that_is_not_text_is_refused():
    with pytest.raises(ValidationError):
        load_samples(None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        load_samples(b"examples: []")  # type: ignore[arg-type]


def test_an_example_without_a_label_is_out_of_scope_only_when_it_says_so():
    ok = "examples: [{id: a, text: hi, expect: {in_scope: false, reason: no_topic_match}}]"
    assert load_samples(ok)[0].expected == (Label.OUT_OF_SCOPE,)


def test_an_entry_that_only_says_it_is_in_scope_has_no_label_to_score_and_is_left_out():
    loaded = load_document(
        "examples: [{id: a, text: hi, expect: {in_scope: true, escalate: true}}, "
        "{id: b, text: ho, expect: {label: no_risk}}]"
    )
    assert [s.id for s in loaded.samples] == ["b"]
    assert loaded.unlabeled == 1
    assert load_samples("examples: [{id: a, text: hi, expect: {in_scope: true}}]") == ()


def test_the_seed_file_has_entries_without_a_label_and_says_how_many():
    loaded = load_document(SEED.read_text(encoding="utf-8"))
    assert loaded.unlabeled > 0
    assert len(loaded.samples) + loaded.unlabeled >= 55


def test_the_message_names_the_example_by_position_not_by_content():
    with pytest.raises(ValidationError) as caught:
        load_samples("examples: [{id: a, text: hi, expect: {label: no_risk}}, {id: b}]")
    assert "example 2" in caught.value.message
    assert "hi" not in caught.value.message


def test_a_text_longer_than_the_cascade_reads_is_refused():
    big = "x" * 20_001
    with pytest.raises(ValidationError):
        load_samples(f"examples: [{{id: a, text: {big}, expect: {{label: no_risk}}}}]")


def test_too_many_examples_are_refused():
    lines = "\n".join(
        f"  - {{id: e{i}, text: t{i}, expect: {{label: no_risk}}}}"
        for i in range(MAX_GOLD_EXAMPLES + 1)
    )
    with pytest.raises(ValidationError):
        load_samples("examples:\n" + lines)


def test_the_size_limit_is_stated():
    assert MAX_FILE_BYTES >= 1024 * 1024
