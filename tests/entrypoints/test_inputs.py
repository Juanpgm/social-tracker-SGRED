"""Input helpers of the CLI: actor resolution, `--set` parsing, bounded file reading/writing."""

from pathlib import Path

import pytest

from social_tracker.entrypoints.inputs import (
    ACTOR_ENV,
    parse_fields,
    read_import_text,
    resolve_actor,
    write_export,
)
from social_tracker.shared.errors import ConflictError, ValidationError
from social_tracker.sources.yaml_io import MAX_FILE_BYTES

# ---- actor ----


def test_an_explicit_actor_wins_over_the_environment():
    assert resolve_actor("alice", {ACTOR_ENV: "bob"}) == "alice"


def test_an_explicit_blank_actor_is_passed_on_so_the_service_rejects_it():
    assert resolve_actor("", {ACTOR_ENV: "bob"}) == ""
    assert resolve_actor("   ", {}) == "   "


def test_the_environment_is_used_when_no_actor_is_given():
    assert resolve_actor(None, {ACTOR_ENV: " bob "}) == "bob"


@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_environment_actor_falls_back_to_the_os_user(
    monkeypatch: pytest.MonkeyPatch, blank: str
):
    monkeypatch.setattr("getpass.getuser", lambda: "carol")
    assert resolve_actor(None, {ACTOR_ENV: blank}) == "cli:carol"


def test_the_default_actor_is_cli_plus_the_os_user(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("getpass.getuser", lambda: "dave")
    assert resolve_actor(None, {}) == "cli:dave"


@pytest.mark.parametrize("failure", [OSError("no user"), KeyError("uid")])
def test_an_unknown_os_user_still_yields_a_valid_actor(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
):
    def fail() -> str:
        raise failure

    monkeypatch.setattr("getpass.getuser", fail)
    assert resolve_actor(None, {}) == "cli:unknown"


# ---- --set KEY=VALUE ----


def test_values_are_json_when_they_parse_and_plain_text_otherwise():
    fields = parse_fields(
        [
            "poll_interval_s=600",
            "feed_url=https://example.org/rss.xml",
            'keywords=["a","b"]',
            "include_stories=false",
            "secret_ref=null",
            "credibility=4",
        ]
    )
    assert fields == {
        "poll_interval_s": 600,
        "feed_url": "https://example.org/rss.xml",
        "keywords": ["a", "b"],
        "include_stories": False,
        "secret_ref": None,
        "credibility": 4,
    }


def test_a_quoted_value_stays_a_string_and_only_the_first_equals_splits():
    assert parse_fields(['name="123"', "feed_url=https://x.org/a?b=c"]) == {
        "name": "123",
        "feed_url": "https://x.org/a?b=c",
    }


def test_an_empty_value_is_an_empty_string_and_no_pairs_is_no_fields():
    assert parse_fields(["feed_url="]) == {"feed_url": ""}
    assert parse_fields([]) == {}


def test_a_value_that_is_too_deeply_nested_for_json_is_kept_as_text():
    deep = "[" * 100_000
    assert parse_fields([f"keywords={deep}"]) == {"keywords": deep}


@pytest.mark.parametrize("pair", ["novalue-ghp_leak", "=value", "", "  =x"])
def test_a_pair_without_a_key_or_equals_is_rejected_without_echoing_it(pair: str):
    with pytest.raises(ValidationError) as caught:
        parse_fields([pair])
    assert "KEY=VALUE" in caught.value.message
    assert "ghp_leak" not in caught.value.message


def test_a_repeated_key_is_rejected():
    with pytest.raises(ValidationError) as caught:
        parse_fields(["a_field=1", "a_field=2"])
    assert "more than once" in caught.value.message
    assert "a_field" not in caught.value.message


def test_reserved_keys_are_rejected_when_the_command_has_a_dedicated_option():
    with pytest.raises(ValidationError) as caught:
        parse_fields(["name=x"], reserved=frozenset({"name"}))
    assert "dedicated option" in caught.value.message
    assert parse_fields(["name=x"]) == {"name": "x"}


# ---- reading the import file ----


def test_a_plain_utf8_file_is_read(tmp_path: Path):
    path = tmp_path / "s.yaml"
    path.write_bytes("sources: []  # café\n".encode())
    assert read_import_text(path) == "sources: []  # café\n"


def test_a_utf8_bom_is_stripped(tmp_path: Path):
    path = tmp_path / "bom.yaml"
    path.write_bytes(b"\xef\xbb\xbfsources: []\n")
    assert read_import_text(path) == "sources: []\n"


def test_the_size_limit_is_inclusive_and_one_more_byte_is_refused(tmp_path: Path):
    at_limit = tmp_path / "at.yaml"
    at_limit.write_bytes(b"#" * MAX_FILE_BYTES)
    assert len(read_import_text(at_limit)) == MAX_FILE_BYTES
    over = tmp_path / "over.yaml"
    over.write_bytes(b"#" * (MAX_FILE_BYTES + 1))
    with pytest.raises(ValidationError) as caught:
        read_import_text(over)
    assert str(MAX_FILE_BYTES) in caught.value.message


def test_a_huge_file_is_never_read_beyond_the_limit(tmp_path: Path, monkeypatch):
    path = tmp_path / "huge.yaml"
    path.write_bytes(b"#" * (MAX_FILE_BYTES * 3))
    seen: list[int] = []
    real_open = Path.open

    def spying_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        real_read = handle.read

        def read(size=-1):
            seen.append(size)
            return real_read(size)

        handle.read = read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(Path, "open", spying_open)
    with pytest.raises(ValidationError):
        read_import_text(path)
    assert seen == [MAX_FILE_BYTES + 1]


@pytest.mark.parametrize("raw", [b"\xff\xfe\x00", b"sources: \xc3\x28", b"\x80"])
def test_invalid_utf8_is_a_clean_error(tmp_path: Path, raw: bytes):
    path = tmp_path / "bad.yaml"
    path.write_bytes(raw)
    with pytest.raises(ValidationError) as caught:
        read_import_text(path)
    assert "UTF-8" in caught.value.message


def test_an_empty_file_is_returned_as_empty_text(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_bytes(b"")
    assert read_import_text(path) == ""


def test_a_missing_file_and_a_directory_are_clean_errors(tmp_path: Path):
    for target in (tmp_path / "nope.yaml", tmp_path):
        with pytest.raises(ValidationError) as caught:
            read_import_text(target)
        assert caught.value.message.startswith("cannot read the import file")
        assert str(tmp_path) not in caught.value.message


# ---- writing the export ----


def test_export_writes_utf8_bytes_with_unix_newlines(tmp_path: Path):
    path = tmp_path / "out.yaml"
    write_export(path, "sources:\n- name: café\n", force=False)
    assert path.read_bytes() == "sources:\n- name: café\n".encode()


def test_export_refuses_to_overwrite_an_existing_file_and_leaves_it_untouched(tmp_path: Path):
    path = tmp_path / "out.yaml"
    path.write_text("precious", encoding="utf-8")
    with pytest.raises(ConflictError) as caught:
        write_export(path, "new", force=False)
    assert "--force" in caught.value.message
    assert path.read_text(encoding="utf-8") == "precious"


def test_force_overwrites_and_truncates(tmp_path: Path):
    path = tmp_path / "out.yaml"
    path.write_text("a much longer previous content", encoding="utf-8")
    write_export(path, "new", force=True)
    assert path.read_text(encoding="utf-8") == "new"


def test_export_to_a_missing_directory_or_over_a_directory_is_a_clean_error(tmp_path: Path):
    for target in (tmp_path / "missing" / "out.yaml", tmp_path):
        for force in (False, True):
            with pytest.raises((ValidationError, ConflictError)) as caught:
                write_export(target, "x", force=force)
            assert str(tmp_path) not in caught.value.message
