"""Fields shared by every source type: id, name, credibility, enabled, timestamps, mode."""

from datetime import date, datetime, timedelta, timezone
from typing import get_type_hints

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError as PydanticValidationError

from social_tracker.shared.errors import ValidationError
from social_tracker.sources import domain as sources_domain
from social_tracker.sources.domain import Source, normalize_name, parse_source
from tests.sources.factories import CANARY, T0, payload, without

TYPES = ["news_rss", "instagram", "live_stream", "recorded_audio"]


@pytest.mark.parametrize("type_", TYPES)
def test_minimal_valid_payload_gets_documented_defaults(type_: str):
    source = parse_source(payload(type_))
    assert source.type == type_
    assert source.credibility == 3
    assert source.enabled is False
    assert source.mode == ("streaming" if type_ == "live_stream" else "recorded")


def test_unknown_type_is_rejected():
    with pytest.raises(ValidationError, match="type"):
        parse_source(payload(type="telegram"))


def test_missing_type_is_rejected():
    with pytest.raises(ValidationError, match="type"):
        parse_source(without(payload(), "type"))


@pytest.mark.parametrize("bad", [None, 5, "", "News_RSS"])
def test_non_matching_type_values_are_rejected(bad):
    with pytest.raises(ValidationError, match="type"):
        parse_source(payload(type=bad))


@pytest.mark.parametrize("bad", [None, [], "src", 42])
def test_non_mapping_input_is_rejected(bad):
    with pytest.raises(ValidationError):
        parse_source(bad)


# ---- id -------------------------------------------------------------------------------


@pytest.mark.parametrize("slug", ["abc", "a" * 64, "a_1"])
def test_id_accepts_boundaries(slug: str):
    assert parse_source(payload(id=slug)).id == slug


@pytest.mark.parametrize(
    "slug",
    [
        "",
        "ab",
        "a" * 65,
        "Abc",
        "a-b",
        "a b",
        "café",
        "abc\n",
        None,
        123,
        b"src_one",
        bytearray(b"src_one"),
        True,
    ],
)
def test_id_rejects_bad_slugs(slug):
    with pytest.raises(ValidationError, match="id"):
        parse_source(payload(id=slug))


# ---- name -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["a", "a" * 100, "  padded  ", "Café 新聞", "Ñ" * 100])
def test_name_accepts_boundaries_and_trims(name: str):
    assert parse_source(payload(name=name)).name == name.strip()


def test_name_is_trimmed_before_the_length_check():
    padded = "  " + "a" * 100 + "  "
    assert parse_source(payload(name=padded)).name == "a" * 100


@pytest.mark.parametrize(
    "name", ["", " ", "\t\n  ", "a" * 101, "  " + "a" * 101 + "  ", None, 7, b"bytes-name"]
)
def test_name_rejects_bad_values(name):
    with pytest.raises(ValidationError, match="name"):
        parse_source(payload(name=name))


@pytest.mark.parametrize(
    "name",
    [
        "\x00",
        "\x00a",
        "a\x00",
        "a\x00b",
        "a\x01b",
        "a\x1fb",
        "a\x7fb",
        "a\x80b",
        "a\x85b",
        "a\x9fb",
        "a\tb",
        "a\nb",
        "a\rb",
        "a\x1bb",
        "\x1fa",  # not whitespace for pydantic, so never trimmed away
        "a\x1f",
    ],
)
def test_name_rejects_control_characters_anywhere_inside(name: str):
    with pytest.raises(ValidationError, match="name"):
        parse_source(payload(name=name))


@pytest.mark.parametrize("name", ["a b", "a b", "café", "\U0001f4f0"])
def test_name_accepts_non_control_oddities(name: str):
    assert parse_source(payload(name=name)).name == name


def test_name_still_trims_surrounding_tabs_and_newlines():
    # Whitespace is trimmed before the control-character check; only inner ones are refused.
    assert parse_source(payload(name="\t a \n")).name == "a"


def test_missing_name_is_rejected():
    with pytest.raises(ValidationError, match="name"):
        parse_source(without(payload(), "name"))


def test_normalized_name_is_case_and_whitespace_insensitive():
    a = parse_source(payload(name="  Radio NACIONAL "))
    b = parse_source(payload(name="radio nacional"))
    assert a.normalized_name == b.normalized_name == "radio nacional"


def test_normalized_name_handles_unicode_case_and_composition():
    assert normalize_name("STRASSE") == normalize_name("straße")  # casefold: ss
    assert normalize_name("Café") == normalize_name("Café")  # NFC


# NFC then casefold is not a fixed point: casefolding can create sequences that NFC reorders.
# U+1E9B U+0323 casefolds to U+1E61 U+0323, which NFC recomposes to U+1E69 on the next pass.
IDEMPOTENCE_COUNTEREXAMPLES = [
    "\u1e9b\u0323",
    "\u1e9b\u0323\u0323",
    "x\u1e9b\u0323y",
    "\u1e61\u0323",
    "\u1e69",
    "s\u0323\u0307",
    "S\u0307\u0323",
    "\u0130",  # dotted capital I casefolds to i + combining dot
    "\ufb03",  # ffi ligature
    "\u1fd3",
    "\u0345\u0301",
]


@pytest.mark.parametrize("name", IDEMPOTENCE_COUNTEREXAMPLES, ids=ascii)
def test_normalize_name_is_idempotent_on_known_hard_cases(name: str):
    once = normalize_name(name)
    assert normalize_name(once) == once


def test_the_original_counterexample_normalises_to_the_composed_form():
    assert normalize_name("\u1e9b\u0323") == "\u1e69"


@settings(max_examples=2000, deadline=None)
@given(st.text(max_size=30))
def test_normalize_name_is_idempotent_for_arbitrary_unicode(name: str):
    once = normalize_name(name)
    assert normalize_name(once) == once


@settings(max_examples=1000, deadline=None)
@given(
    st.text(
        alphabet=st.sampled_from(
            "sS\u1e9b\u1e61\u1e69\u0323\u0307\u0345\u0301\u0130\u03a3\u03c2\u00df\ufb03 "
        ),
        max_size=12,
    )
)
def test_normalize_name_is_idempotent_for_combining_and_casefold_heavy_text(name: str):
    once = normalize_name(name)
    assert normalize_name(once) == once


def test_a_source_is_found_by_its_own_stored_normalized_name():
    source = parse_source(payload(name="\u1e9b\u0323"))
    assert normalize_name(source.normalized_name) == source.normalized_name


def test_normalization_that_never_settles_is_an_explicit_error(monkeypatch):
    flip = iter("abcdefghijklmnopqrstuvwxyz" * 3)
    monkeypatch.setattr(sources_domain, "_normalize_once", lambda text: next(flip))
    with pytest.raises(ValidationError, match="normali"):
        normalize_name("anything")


def test_normalization_gives_up_after_a_small_bounded_number_of_passes(monkeypatch):
    calls: list[str] = []

    def never_settles(text: str) -> str:
        calls.append(text)
        return text + "x"

    monkeypatch.setattr(sources_domain, "_normalize_once", never_settles)
    with pytest.raises(ValidationError):
        normalize_name("a")
    assert 2 <= len(calls) <= 10


def test_normalized_name_keeps_distinct_names_distinct():
    assert normalize_name("Radio A") != normalize_name("Radio B")


# ---- credibility ----------------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
def test_credibility_accepts_1_to_5(value: int):
    assert parse_source(payload(credibility=value)).credibility == value


@pytest.mark.parametrize("value", [0, 6, -1, 3.5, 3.0, None, "3", "high", True, False, [], 10**30])
def test_credibility_rejects_everything_else(value):
    with pytest.raises(ValidationError, match="credibility"):
        parse_source(payload(credibility=value))


# ---- enabled --------------------------------------------------------------------------


def test_enabled_can_be_set_true():
    assert parse_source(payload(enabled=True)).enabled is True


@pytest.mark.parametrize("value", [None, 1, "true", "yes", 0])
def test_enabled_rejects_non_bool(value):
    with pytest.raises(ValidationError, match="enabled"):
        parse_source(payload(enabled=value))


# ---- timestamps -----------------------------------------------------------------------


def test_timestamps_are_normalized_to_utc():
    plus3 = timezone(timedelta(hours=3))
    src = parse_source(payload(created_at=datetime(2026, 1, 1, 15, 0, tzinfo=plus3)))
    assert src.created_at == T0
    assert src.created_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
def test_naive_timestamps_are_rejected(field: str):
    with pytest.raises(ValidationError, match=field):
        parse_source(payload(**{field: datetime(2026, 1, 1)}))


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
def test_naive_iso_string_is_rejected(field: str):
    with pytest.raises(ValidationError, match=field):
        parse_source(payload(**{field: "2026-01-01T12:00:00"}))


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
def test_timestamps_are_required(field: str):
    with pytest.raises(ValidationError, match=field):
        parse_source(without(payload(), field))


def test_aware_iso_string_is_accepted():
    # Documented: ISO-8601 strings are accepted because config files and JSON carry them.
    assert parse_source(payload(created_at="2026-01-01T12:00:00Z")).created_at == T0
    assert parse_source(payload(created_at="2026-01-01T15:00:00+03:00")).created_at == T0


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
@pytest.mark.parametrize(
    "bad",
    [
        1767268800,  # integer epoch
        1767268800.5,  # float epoch
        "1767268800",  # numeric string
        True,
        b"2026-01-01T12:00:00Z",  # bytes
        None,
        "",
        "yesterday",
        "2026-01-01",  # date only is naive
        date(2026, 1, 1),
    ],
)
def test_timestamps_reject_epochs_bytes_and_non_iso_text(field: str, bad):
    with pytest.raises(ValidationError, match=field):
        parse_source(payload(**{field: bad}))


def test_updated_before_created_is_rejected():
    with pytest.raises(ValidationError, match="updated_at"):
        parse_source(payload(updated_at=T0 - timedelta(microseconds=1)))


def test_updated_equal_to_created_is_accepted():
    assert parse_source(payload(updated_at=T0)).updated_at == T0


# ---- mode / extras / immutability -----------------------------------------------------


@pytest.mark.parametrize(
    ("type_", "mode"),
    [
        ("news_rss", "streaming"),
        ("instagram", "streaming"),
        ("recorded_audio", "streaming"),
        ("live_stream", "recorded"),
        ("news_rss", "bogus"),
        ("live_stream", None),
    ],
)
def test_mode_must_match_type(type_: str, mode):
    with pytest.raises(ValidationError, match="mode"):
        parse_source(payload(type_, mode=mode))


@pytest.mark.parametrize(
    ("type_", "mode"), [("news_rss", "recorded"), ("live_stream", "streaming")]
)
def test_explicit_matching_mode_is_accepted(type_: str, mode: str):
    assert parse_source(payload(type_, mode=mode)).mode == mode


@pytest.mark.parametrize("type_", TYPES)
def test_unknown_extra_keys_are_rejected_with_a_generic_message(type_: str):
    with pytest.raises(ValidationError, match="unknown field") as info:
        parse_source(payload(type_, surprise=1))
    assert "surprise" not in str(info.value)
    assert "surprise" not in info.value.fields


def test_mistyped_secret_looking_key_is_never_echoed():
    leaky = "my_pas" + "sword_typo"
    with pytest.raises(ValidationError) as info:
        parse_source(payload("news_rss", **{leaky: CANARY}))
    assert leaky not in str(info.value)
    assert leaky not in info.value.fields
    assert CANARY not in str(info.value)


def test_huge_unknown_key_does_not_inflate_the_error():
    with pytest.raises(ValidationError) as info:
        parse_source(payload("news_rss", **{"k" * 100_000: 1}))
    assert len(info.value.message) < 200


def test_unknown_key_next_to_a_known_problem_reports_only_the_known_field():
    with pytest.raises(ValidationError) as info:
        parse_source(payload("news_rss", surprise=1, credibility=9))
    assert info.value.fields == ("credibility",)
    assert "credibility" in info.value.message
    assert "unknown field" in info.value.message


def test_unknown_type_value_is_not_echoed():
    bad_type = "tele" + "gram-" + CANARY
    with pytest.raises(ValidationError, match="type") as info:
        parse_source(payload(type=bad_type))
    assert CANARY not in str(info.value)


def test_source_is_immutable():
    source = parse_source(payload())
    with pytest.raises(PydanticValidationError, match="frozen"):
        source.enabled = True  # type: ignore[misc]


def test_error_message_for_multiple_problems_lists_each_field():
    with pytest.raises(ValidationError) as info:
        parse_source(payload(id="X", credibility=9))
    assert "id" in info.value.message
    assert "credibility" in info.value.message
    assert info.value.fields == ("id", "credibility")


def test_fields_name_the_top_level_field_for_nested_problems():
    with pytest.raises(ValidationError) as info:
        parse_source(payload("news_rss", keywords=["ok", ""]))
    assert info.value.fields == ("keywords",)


def test_problems_without_a_field_use_a_source_placeholder_and_no_fields():
    with pytest.raises(ValidationError, match="<source>") as info:
        parse_source("not a mapping")
    assert info.value.fields == ()


def test_repeated_problems_on_one_field_are_listed_once():
    with pytest.raises(ValidationError) as info:
        parse_source(payload("recorded_audio", feed_url="", watch_dir=""))
    assert len(set(info.value.fields)) == len(info.value.fields)


# ---- typing / with_changes ------------------------------------------------------------


def test_parse_source_is_typed_as_the_source_union():
    assert get_type_hints(parse_source)["return"] == Source


def test_with_changes_returns_a_new_validated_source():
    original = parse_source(payload(credibility=2))
    changed = original.with_changes({"credibility": 5, "name": "Renamed"})
    assert changed is not original
    assert (changed.credibility, changed.name) == (5, "Renamed")
    assert original.credibility == 2
    assert changed.id == original.id


def test_with_changes_without_arguments_is_an_equal_copy():
    original = parse_source(payload("instagram"))
    assert original.with_changes({}) == original


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"credibility": 99}, "credibility"),
        ({"name": ""}, "name"),
        ({"id": "X"}, "id"),
        ({"updated_at": T0 - timedelta(days=1)}, "updated_at"),
        ({"enabled": "yes"}, "enabled"),
        ({"surprise": 1}, "unknown field"),
        ({"feed_url": "https://x.org/rss?api_key=1"}, "feed_url"),
    ],
)
def test_with_changes_revalidates_like_parse_source(changes: dict, field: str):
    original = parse_source(payload())
    with pytest.raises(ValidationError, match=field):
        original.with_changes(changes)
    assert original.credibility == 3  # the original is untouched


def test_with_changes_keeps_per_type_rules():
    ig = parse_source(payload("instagram"))
    with pytest.raises(ValidationError, match="poll_interval_s"):
        ig.with_changes({"poll_interval_s": 10})
    audio = parse_source(payload("recorded_audio"))
    with pytest.raises(ValidationError, match="exactly one"):
        audio.with_changes({"watch_dir": "/data/audio"})


@pytest.mark.parametrize("type_", TYPES)
def test_dump_and_json_round_trip_preserve_equality(type_: str):
    source = parse_source(payload(type_))
    assert parse_source(source.model_dump()) == source
    assert parse_source(source.model_dump(mode="json")) == source


HIDDEN_TEXT = ["\ud800x", "x\udfff", "a\u200bb", "\u202e", "a\u2028b", "a\u2029b", "a\ufeffb"]


@pytest.mark.parametrize("value", HIDDEN_TEXT, ids=ascii)
def test_name_rejects_surrogates_format_characters_and_separators(value: str):
    with pytest.raises(ValidationError) as caught:
        parse_source(payload(name=value))
    assert caught.value.fields == ("name",)


def test_a_name_made_only_of_format_characters_is_blank_and_rejected():
    with pytest.raises(ValidationError, match="name"):
        parse_source(payload(name="\u200b\u200b"))


@pytest.mark.parametrize("name", ["Diario El País", "cafe\u0301", "Mañana \U0001f600"], ids=ascii)
def test_name_keeps_accents_combining_marks_and_emoji(name: str):
    assert parse_source(payload(name=name)).name == name


@pytest.mark.parametrize("value", HIDDEN_TEXT, ids=ascii)
def test_keywords_reject_surrogates_format_characters_and_separators(value: str):
    with pytest.raises(ValidationError) as caught:
        parse_source(payload("news_rss", keywords=["fine", value]))
    assert caught.value.fields == ("keywords",)


def test_keywords_keep_accents_combining_marks_and_emoji():
    words = ["incendio", "niño", "cafe\u0301", "\U0001f525"]
    assert parse_source(payload("news_rss", keywords=words)).keywords == tuple(words)


@pytest.mark.parametrize(
    "key", ["self", "cls", "changes", "__class__", "model_config", "model_fields", "id "]
)
def test_with_changes_treats_every_key_as_data_even_one_named_like_a_parameter(key: str):
    original = parse_source(payload())
    with pytest.raises(ValidationError) as caught:
        original.with_changes({key: "x"})
    assert key not in caught.value.message
    assert caught.value.fields == ()


@pytest.mark.parametrize("bad", [None, "name", ["name"], 5, [("name", "x")]], ids=repr)
def test_with_changes_requires_a_mapping(bad):
    with pytest.raises(ValidationError):
        parse_source(payload()).with_changes(bad)


def test_with_changes_takes_the_mapping_positionally_only():
    original = parse_source(payload())
    with pytest.raises(TypeError):
        original.with_changes(name="Renamed")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        original.with_changes(changes={"name": "Renamed"})  # type: ignore[call-arg]


def test_with_changes_rejects_non_string_keys():
    with pytest.raises(ValidationError):
        parse_source(payload()).with_changes({1: "x"})  # type: ignore[dict-item]


def test_with_changes_does_not_mutate_the_mapping_it_was_given():
    changes = {"name": "Renamed"}
    parse_source(payload()).with_changes(changes)
    assert changes == {"name": "Renamed"}
