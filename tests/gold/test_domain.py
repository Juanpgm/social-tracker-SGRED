"""A labeled text of the gold set: what is accepted, how it is identified, how it is cleaned."""

from datetime import UTC, datetime

import pytest

from social_tracker.classification.domain import Label
from social_tracker.gold.domain import (
    MAX_GOLD_TEXT_CHARS,
    MAX_NOTE_CHARS,
    GoldExample,
    gold_id_for,
    make_example,
)
from social_tracker.shared.errors import ValidationError

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
TEXT = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"


def make(**overrides):
    fields = {
        "text": TEXT,
        "label": Label.UNINTENTIONAL_RUMOR,
        "note": "a prediction chain",
        "credibility": 3,
        "url": None,
        "added_at": NOW,
        "actor": "ana",
    }
    fields.update(overrides)
    return make_example(**fields)


def test_an_example_has_a_stable_id_made_from_its_text():
    a, b = make(), make(label=Label.NO_RISK, note="")
    assert a.id == b.id == gold_id_for(TEXT)
    assert a.id.startswith("gld_")
    assert len(a.id) == 20
    assert make(text=TEXT + " otra").id != a.id


def test_case_and_spacing_do_not_make_a_different_text_but_the_words_do():
    assert gold_id_for(TEXT) == gold_id_for("  " + TEXT.replace(" ", "   ") + "\n")
    assert gold_id_for(TEXT) != gold_id_for(TEXT.upper())


def test_the_text_is_normalized_like_an_item_text():
    example = make(text="a\x00b\x1b c\t\td\r\ne")
    assert example.text == "a b c d\ne"
    assert "\x00" not in example.text
    assert "\x1b" not in example.text
    assert "\r" not in example.text


def test_a_lone_surrogate_cannot_be_stored():
    example = make(text="a" + chr(0xD800) + "b")
    example.text.encode("utf-8")


@pytest.mark.parametrize("label", list(Label))
def test_every_label_of_the_cascade_can_be_the_expected_one(label):
    assert make(label=label).label is label


def test_a_label_is_given_by_name_too():
    assert make(label="no_risk").label is Label.NO_RISK


@pytest.mark.parametrize("bad", ["", "rumor", "NO_RISK", None, 3, "no_risk "])
def test_an_unknown_label_is_refused(bad):
    with pytest.raises(ValidationError):
        make(label=bad)


@pytest.mark.parametrize("text", ["", "   ", "\n\t", None, 5, "x" * (MAX_GOLD_TEXT_CHARS + 1)])
def test_a_text_that_is_empty_or_too_long_is_refused(text):
    with pytest.raises(ValidationError):
        make(text=text)


def test_the_text_may_be_exactly_as_long_as_allowed():
    assert len(make(text="x" * MAX_GOLD_TEXT_CHARS).text) == MAX_GOLD_TEXT_CHARS


@pytest.mark.parametrize("length", [0, 1, MAX_NOTE_CHARS])
def test_a_note_up_to_the_limit_is_kept(length):
    assert len(make(note="n" * length).note) == length


@pytest.mark.parametrize(
    "note",
    ["n" * (MAX_NOTE_CHARS + 1), "bell\x07", "multi\nline", None, 5, "hid" + chr(0x202E) + "den"],
)
def test_a_note_must_be_short_plain_text(note):
    with pytest.raises(ValidationError):
        make(note=note)


@pytest.mark.parametrize("credibility", [1, 2, 3, 4, 5])
def test_credibility_is_one_to_five(credibility):
    assert make(credibility=credibility).credibility == credibility


@pytest.mark.parametrize("credibility", [0, 6, -1, 3.5, "3", True, None])
def test_a_credibility_out_of_range_is_refused(credibility):
    with pytest.raises(ValidationError):
        make(credibility=credibility)


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        (None, True),
        ("https://official-domain.example/avisos/1", True),
        ("http://example.org/a", True),
        ("ftp://example.org/a", False),
        ("javascript:alert(1)", False),
        ("https://" + "user:pw" + "@example.org/a", False),
        ("https://example.org/a b", False),
        ("", False),
        (5, False),
    ],
)
def test_the_address_of_an_example_is_a_plain_http_link_or_nothing(url, ok):
    if ok:
        assert make(url=url).url == url
    else:
        with pytest.raises(ValidationError):
            make(url=url)


@pytest.mark.parametrize("actor", ["", "  ", "a" * 101, "bell\x07", None])
def test_an_actor_is_required_and_plain(actor):
    with pytest.raises(ValidationError):
        make(actor=actor)


def test_the_time_must_be_aware():
    with pytest.raises(ValidationError):
        make(added_at=datetime(2026, 3, 1))


def test_a_built_example_validates_itself_again():
    good = make()
    with pytest.raises(ValidationError):
        GoldExample(**{**good.__dict__, "id": "gld_wrong"})
    with pytest.raises(ValidationError):
        GoldExample(**{**good.__dict__, "text": "other text"})
