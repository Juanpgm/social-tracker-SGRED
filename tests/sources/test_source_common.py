"""Fields shared by every source type: id, name, credibility, enabled, timestamps, mode."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError as PydanticValidationError

from social_tracker.shared.errors import ValidationError
from social_tracker.sources.domain import normalize_name, parse_source
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
    ["", "ab", "a" * 65, "Abc", "a-b", "a b", "café", "abc\n", None, 123],
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
    assert parse_source(payload(created_at="2026-01-01T12:00:00Z")).created_at == T0


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
