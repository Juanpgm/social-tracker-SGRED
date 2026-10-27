"""YAML codec for source files: strict, bounded and free of secrets."""

import time
from datetime import UTC, datetime
from typing import Any

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from social_tracker.shared.errors import ValidationError
from social_tracker.sources.domain import SOURCE_TYPES, Source, parse_source
from social_tracker.sources.yaml_io import (
    MAX_FILE_BYTES,
    SourceFileError,
    SourceFileProblem,
    dump_sources,
    load_sources,
)
from tests.sources.factories import CANARY, EMBEDDED_CREDENTIALS_URL, T0, payload

NOW = datetime(2026, 5, 5, 5, 5, tzinfo=UTC)


def src(type_: str = "news_rss", **overrides: Any) -> Source:
    return parse_source(payload(type_, **overrides))


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def doc(*entries: dict[str, Any]) -> str:
    return yaml.dump({"sources": list(entries)}, Dumper=_NoAliasDumper, sort_keys=False)


def problems_of(text: str, **kwargs: Any) -> tuple[SourceFileProblem, ...]:
    with pytest.raises(SourceFileError) as caught:
        load_sources(text, **kwargs)
    return caught.value.problems


ALL_TYPES = [
    src("news_rss", id="src_news", name="News", keywords=["alpha", "beta"], enabled=True),
    src("instagram", id="src_insta", name="Insta", include_stories=False),
    src("live_stream", id="src_live", name="Live", language="en", chunk_seconds=45),
    src("recorded_audio", id="src_audio", name="Audio", feed_url=None, watch_dir="/data/audio"),
]


# ---- dump ----------------------------------------------------------------------------


def test_dump_has_a_pinned_golden_layout():
    assert dump_sources([src()]) == (
        "sources:\n"
        "- id: src_one\n"
        "  type: news_rss\n"
        "  name: Some Source\n"
        "  mode: recorded\n"
        "  credibility: 3\n"
        "  enabled: false\n"
        "  feed_url: https://example.org/rss.xml\n"
        "  poll_interval_s: 600\n"
        "  created_at: '2026-01-01T12:00:00+00:00'\n"
        "  updated_at: '2026-01-01T12:00:00+00:00'\n"
    )


def test_dump_of_nothing_is_an_empty_list_that_loads_back_to_nothing():
    text = dump_sources([])
    assert text == "sources: []\n"
    assert load_sources(text) == []


def test_dump_is_sorted_by_id_and_independent_of_input_order():
    forward = dump_sources(ALL_TYPES)
    assert dump_sources(reversed(ALL_TYPES)) == forward
    ids = [entry["id"] for entry in yaml.safe_load(forward)["sources"]]
    assert ids == sorted(ids) == ["src_audio", "src_insta", "src_live", "src_news"]


def test_dump_is_deterministic_across_calls():
    assert dump_sources(ALL_TYPES) == dump_sources(ALL_TYPES)


@pytest.mark.parametrize("source", ALL_TYPES, ids=SOURCE_TYPES[::-1])
def test_each_source_type_round_trips_exactly(source: Source):
    assert load_sources(dump_sources([source])) == [source]


def test_all_types_together_round_trip_in_id_order():
    assert load_sources(dump_sources(ALL_TYPES)) == sorted(ALL_TYPES, key=lambda s: s.id)


def test_unicode_names_and_urls_round_trip_without_escaping():
    source = src(name="Ñandú Noticias ✓ 日本", id="src_uni")
    text = dump_sources([source])
    assert "Ñandú Noticias ✓ 日本" in text
    assert load_sources(text) == [source]


def test_dump_never_emits_python_tags_even_for_tuples():
    text = dump_sources(ALL_TYPES)
    assert "!!" not in text
    assert "python" not in text


def test_dump_emits_only_the_secret_reference_never_a_value():
    text = dump_sources([ALL_TYPES[1]])
    entry = yaml.safe_load(text)["sources"][0]
    assert entry["secret_ref"] == "ig_main"
    forbidden = {"password", "token", "cookie", "secret", "api_key", "session"}
    assert forbidden.isdisjoint(entry)


def test_dump_omits_unset_optional_fields():
    entry = yaml.safe_load(dump_sources([src()]))["sources"][0]
    assert "keywords" not in entry
    assert "secret_ref" not in entry


@pytest.mark.parametrize("bad", [{"id": "src_one"}, "x", None, 5])
def test_dump_rejects_items_that_are_not_sources(bad: Any):
    with pytest.raises(ValidationError):
        dump_sources([bad])


def test_dump_rejects_a_non_iterable():
    with pytest.raises(ValidationError):
        dump_sources(None)  # type: ignore[arg-type]


def test_dump_rejects_duplicate_ids_instead_of_writing_a_file_that_cannot_load():
    with pytest.raises(ValidationError) as caught:
        dump_sources([src(), src(name="Other")])
    assert caught.value.fields == ("id",)


# ---- load: document shape -------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\n\t\n", "# only a comment\n", "---\n", "~", "null"],
    ids=["empty", "spaces", "blank-lines", "comment", "bare-doc", "tilde", "null"],
)
def test_empty_documents_are_rejected(text: str):
    (problem,) = problems_of(text)
    assert problem.code == "validation_error"
    assert problem.index is None
    assert problem.fields == ("sources",)


@pytest.mark.parametrize(
    "text",
    ["- a\n- b\n", "just text", "42", "true", "[1, 2]"],
    ids=["list", "scalar-text", "number", "bool", "flow-list"],
)
def test_a_document_that_is_not_a_mapping_is_rejected(text: str):
    (problem,) = problems_of(text)
    assert problem.fields == ("sources",)
    assert "mapping" in problem.message


def test_missing_sources_key_is_rejected_without_echoing_the_other_key():
    (problem,) = problems_of(f"{CANARY}: 1\n")
    assert problem.fields == ("sources",)
    assert CANARY not in problem.message


@pytest.mark.parametrize(
    "text",
    ["sources:\n", "sources: null\n", "sources: 5\n", "sources: text\n", "sources: {a: 1}\n"],
    ids=["empty-value", "null", "number", "text", "mapping"],
)
def test_sources_that_is_not_a_list_is_rejected(text: str):
    (problem,) = problems_of(text)
    assert problem.fields == ("sources",)
    assert "list" in problem.message


def test_extra_top_level_keys_are_rejected_without_being_echoed():
    exc = pytest.raises(SourceFileError, load_sources, f"sources: []\n{CANARY}: 1\n").value
    assert CANARY not in str(exc) + "".join(p.message for p in exc.problems)
    assert exc.problems[0].index is None


def test_null_and_scalar_entries_are_reported_by_index():
    entries = problems_of("sources:\n- ~\n- text\n- 7\n- [1]\n")
    assert [(p.index, p.fields) for p in entries] == [(0, ()), (1, ()), (2, ()), (3, ())]
    assert all("mapping" in p.message for p in entries)


def test_multiple_yaml_documents_are_rejected():
    (problem,) = problems_of("sources: []\n---\nsources: []\n")
    assert problem.index is None


@pytest.mark.parametrize(
    "text", ["sources: [", "sources: {a: b", "sources:\n  - a: [\n", "\tsources: []", "a: b: c"]
)
def test_malformed_yaml_gives_a_generic_error_that_never_echoes_the_text(text: str):
    (problem,) = problems_of(text)
    assert problem.code == "validation_error"
    assert "YAML" in problem.message
    assert text.strip()[:6] not in problem.message


def test_malformed_yaml_does_not_leak_secrets_from_the_text():
    text = f"sources: [\n  password: {CANARY}"
    exc = pytest.raises(SourceFileError, load_sources, text).value
    assert CANARY not in repr(exc.problems) + str(exc) + exc.message


@pytest.mark.parametrize("value", [None, b"sources: []", 42, ["sources: []"]])
def test_non_text_input_is_rejected(value: Any):
    with pytest.raises(SourceFileError):
        load_sources(value)


# ---- load: unsafe YAML ----------------------------------------------------------------


@pytest.mark.parametrize(
    "tag",
    [
        "!!python/object/apply:os.system ['echo pwned']",
        "!!python/object:builtins.object {}",
        "!!python/name:os.system",
        "!!python/module:os",
        "!custom tagged",
        "!!binary aGVsbG8=",
        "!!set {a, b}",
    ],
)
def test_python_object_tags_and_other_custom_tags_are_rejected(tag: str):
    text = f"sources:\n- id: src_one\n  name: {tag}\n"
    problems = problems_of(text)
    assert [p.index for p in problems] == [None]
    assert "pwned" not in repr(problems)


def test_aliases_are_rejected():
    text = "x: &a [1, 2]\nsources:\n- *a\n"
    (problem,) = problems_of(text)
    assert "alias" in problem.message


def test_anchors_are_rejected_even_when_unused():
    (problem,) = problems_of("sources: []\nx: &unused 1\n")
    assert "anchor" in problem.message


def test_merge_keys_are_not_honoured():
    entry = {k: v for k, v in payload("news_rss").items() if k != "feed_url"}
    text = doc(entry)
    text += "  <<: {feed_url: 'https://example.org/rss.xml'}\n"
    (problem,) = problems_of(text)
    assert problem.index is None


def test_a_billion_laughs_document_is_rejected_quickly_without_expanding():
    levels = ["a: &a0 [lol, lol, lol, lol, lol, lol, lol, lol, lol]"]
    for i in range(1, 10):
        refs = ", ".join([f"*a{i - 1}"] * 9)
        levels.append(f"b{i}: &a{i} [{refs}]")
    text = "\n".join(levels) + "\nsources: []\n"
    started = time.perf_counter()
    with pytest.raises(SourceFileError):
        load_sources(text)
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize("depth", [20, 200, 5000])
def test_deeply_nested_documents_are_rejected_not_recursed_into(depth: int):
    text = "sources: " + "[" * depth + "]" * depth + "\n"
    started = time.perf_counter()
    with pytest.raises(SourceFileError):
        load_sources(text)
    assert time.perf_counter() - started < 2.0


def test_deep_block_nesting_is_rejected():
    lines = [f"{'  ' * i}k{i}:" for i in range(60)]
    with pytest.raises(SourceFileError):
        load_sources("sources:\n- " + "\n".join(lines) + "\n")


def test_too_many_entries_are_rejected_before_any_are_validated():
    text = "sources:\n" + "- {}\n" * 5000
    started = time.perf_counter()
    (problem,) = problems_of(text)
    assert problem.index is None
    assert "entries" in problem.message
    assert time.perf_counter() - started < 2.0


def test_a_flood_of_scalars_is_bounded():
    text = "sources: [" + ",".join(["1"] * 200_000) + "]"
    assert len(text.encode()) < MAX_FILE_BYTES
    with pytest.raises(SourceFileError):
        load_sources(text)


def test_duplicate_keys_in_an_entry_are_rejected():
    text = "sources:\n- id: src_one\n  id: src_two\n"
    (problem,) = problems_of(text)
    assert "duplicate" in problem.message
    assert "src_" not in problem.message


def test_duplicate_top_level_keys_are_rejected():
    (problem,) = problems_of("sources: []\nsources: []\n")
    assert "duplicate" in problem.message


# ---- load: size, encoding -------------------------------------------------------------


def test_a_file_of_exactly_the_limit_loads():
    body = dump_sources([src()])
    pad = MAX_FILE_BYTES - len(body.encode())
    text = body + "#" * (pad - 1) + "\n"
    assert len(text.encode()) == MAX_FILE_BYTES
    assert load_sources(text) == [src()]


def test_a_file_one_byte_over_the_limit_is_rejected():
    body = dump_sources([src()])
    text = body + "#" * (MAX_FILE_BYTES - len(body.encode())) + "\n"
    assert len(text.encode()) == MAX_FILE_BYTES + 1
    (problem,) = problems_of(text)
    assert "1000000" in problem.message


def test_the_limit_counts_bytes_not_characters():
    text = "# " + "é" * (MAX_FILE_BYTES // 2) + "\nsources: []\n"
    assert len(text) < MAX_FILE_BYTES < len(text.encode())
    with pytest.raises(SourceFileError):
        load_sources(text)


def test_a_byte_order_mark_is_ignored():
    assert load_sources("﻿" + dump_sources(ALL_TYPES)) == sorted(ALL_TYPES, key=lambda s: s.id)


def test_windows_line_endings_load():
    text = dump_sources(ALL_TYPES).replace("\n", "\r\n")
    assert load_sources(text) == sorted(ALL_TYPES, key=lambda s: s.id)


def test_control_characters_in_the_text_are_rejected():
    with pytest.raises(SourceFileError):
        load_sources("sources: []\n\x00")


# ---- load: entry validation -----------------------------------------------------------


def test_every_invalid_entry_is_reported_together_with_its_index_and_fields():
    bad_interval = payload("news_rss", id="src_a", name="A", poll_interval_s=1)
    unknown_key = payload("news_rss", id="src_b", name="B", **{CANARY: "x"})
    good = payload("news_rss", id="src_good", name="Good")
    bad_type = payload("news_rss", id="src_d", name="D", type="telegram")
    no_name = {k: v for k, v in payload("news_rss", id="src_e").items() if k != "name"}
    problems = problems_of(doc(bad_interval, unknown_key, good, bad_type, no_name))
    assert [(p.index, p.fields) for p in problems] == [
        (0, ("poll_interval_s",)),
        (1, ()),
        (3, ("type",)),
        (4, ("name",)),
    ]
    assert all(p.code == "validation_error" for p in problems)
    assert CANARY not in repr(problems)


def test_the_aggregate_error_lists_the_known_fields_of_all_problems():
    texts = doc(
        payload("news_rss", id="src_a", name="A", poll_interval_s=1),
        payload("news_rss", id="src_b", name="", feed_url="ftp://x"),
    )
    exc = pytest.raises(SourceFileError, load_sources, texts).value
    assert exc.code == "validation_error"
    assert exc.fields == ("poll_interval_s", "name", "feed_url")
    assert isinstance(exc, ValidationError)


@pytest.mark.parametrize("key", ["password", "token", "cookie", "api_key", "session_id", "secret"])
def test_inline_secret_keys_are_rejected_without_echoing_key_or_value(key: str):
    entry = payload("news_rss", **{key: CANARY})
    (problem,) = problems_of(doc(entry))
    assert problem.index == 0
    assert problem.fields == ()
    assert CANARY not in problem.message
    assert key not in problem.message


def test_urls_carrying_secrets_in_the_query_are_rejected():
    entry = payload("news_rss", feed_url=f"https://example.org/rss?api_key={CANARY}")
    (problem,) = problems_of(doc(entry))
    assert problem.fields == ("feed_url",)
    assert CANARY not in problem.message


def test_urls_with_embedded_credentials_are_rejected():
    (problem,) = problems_of(doc(payload("news_rss", feed_url=EMBEDDED_CREDENTIALS_URL)))
    assert problem.fields == ("feed_url",)
    assert "user:pw" not in problem.message


def test_secret_references_are_the_only_way_to_point_at_a_secret():
    loaded = load_sources(doc(payload("instagram", secret_ref="ig_main")))
    assert loaded[0].secret_ref == "ig_main"


def test_instagram_without_a_secret_reference_is_rejected():
    entry = {k: v for k, v in payload("instagram").items() if k != "secret_ref"}
    (problem,) = problems_of(doc(entry))
    assert problem.fields == ("secret_ref",)


def test_duplicate_ids_and_names_inside_the_file_are_reported_with_the_other_problems():
    entries = [
        payload("news_rss", id="src_a", name="Alpha"),
        payload("news_rss", id="src_a", name="Beta"),  # duplicate id (1)
        payload("news_rss", id="src_c", name="  ALPHA "),  # duplicate name, case-insensitive (2)
        payload("news_rss", id="src_d", name="Delta", poll_interval_s=1),  # invalid (3)
    ]
    problems = problems_of(doc(*entries))
    assert [(p.index, p.code, p.fields) for p in problems] == [
        (1, "duplicate", ("id",)),
        (2, "duplicate", ("name",)),
        (3, "validation_error", ("poll_interval_s",)),
    ]
    assert "entry 0" in problems[0].message
    assert "entry 0" in problems[1].message


def test_invalid_booleans_and_numbers_are_not_coerced():
    cases = [
        ("enabled", "yes-please"),
        ("enabled", 1),
        ("credibility", 3.0),
        ("credibility", "3"),
        ("poll_interval_s", "600"),
    ]
    for key, value in cases:
        (problem,) = problems_of(doc(payload("news_rss", **{key: value})))
        assert problem.fields == (key,), (key, value)


def test_timestamps_accept_native_yaml_timestamps_and_quoted_iso_strings():
    native = doc(payload("news_rss", created_at=T0, updated_at=T0))
    quoted = doc(payload("news_rss", created_at=T0.isoformat(), updated_at=T0.isoformat()))
    assert load_sources(native) == load_sources(quoted) == [src()]


def test_naive_timestamps_are_rejected():
    entry = payload("news_rss", created_at="2026-01-01T12:00:00", updated_at="2026-01-01T12:00:00")
    (problem,) = problems_of(doc(entry))
    assert set(problem.fields) == {"created_at", "updated_at"}


def test_missing_timestamps_are_filled_from_the_default_timestamp():
    entry = {k: v for k, v in payload("news_rss").items() if k not in ("created_at", "updated_at")}
    (loaded,) = load_sources(doc(entry), default_timestamp=NOW)
    assert (loaded.created_at, loaded.updated_at) == (NOW, NOW)


def test_missing_updated_at_defaults_to_the_given_created_at():
    entry = {k: v for k, v in payload("news_rss").items() if k != "updated_at"}
    (loaded,) = load_sources(doc(entry), default_timestamp=NOW)
    assert (loaded.created_at, loaded.updated_at) == (T0, T0)


def test_missing_timestamps_without_a_default_are_rejected():
    entry = {k: v for k, v in payload("news_rss").items() if k not in ("created_at", "updated_at")}
    (problem,) = problems_of(doc(entry))
    assert set(problem.fields) == {"created_at", "updated_at"}


def test_a_naive_default_timestamp_is_rejected():
    entry = {k: v for k, v in payload("news_rss").items() if k not in ("created_at", "updated_at")}
    with pytest.raises(ValidationError):
        load_sources(doc(entry), default_timestamp=datetime(2026, 1, 1))  # noqa: DTZ001


def test_loaded_sources_are_order_preserving_file_order():
    entries = [payload("news_rss", id=f"src_{c}", name=c.upper()) for c in "cab"]
    assert [s.id for s in load_sources(doc(*entries))] == ["src_c", "src_a", "src_b"]


# ---- properties -----------------------------------------------------------------------

_names = st.text(
    alphabet=st.characters(blacklist_categories=("Cc", "Cs")), min_size=1, max_size=100
).filter(lambda t: t.strip() != "")
_keywords = st.lists(
    st.text(alphabet=st.characters(blacklist_categories=("Cc", "Cs")), min_size=1, max_size=40)
    .map(str.strip)
    .filter(bool),
    max_size=8,
)


@settings(max_examples=150, deadline=None)
@given(name=_names, keywords=st.none() | _keywords)
def test_arbitrary_names_and_keywords_survive_a_round_trip(name: str, keywords: list[str] | None):
    source = src("news_rss", name=name, keywords=keywords)
    assert load_sources(dump_sources([source])) == [source]
