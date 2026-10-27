"""`social-tracker sources import` and `sources export`."""

from pathlib import Path

import pytest

from social_tracker.sources.service import SOURCE_ADDED, SOURCE_EDITED, SOURCES_IMPORTED
from social_tracker.sources.yaml_io import MAX_FILE_BYTES
from tests.entrypoints.conftest import WHY, Cli, table
from tests.entrypoints.test_cli_sources import PASTED, assert_clean_failure

TWO_SOURCES = """\
sources:
  - id: src_news
    type: news_rss
    name: City News
    feed_url: https://example.org/rss.xml
  - id: ig_main
    type: instagram
    name: Instagram Main
    handle: some.handle
    secret_ref: env:IG_MAIN
"""


def write(tmp_path: Path, content: str | bytes, name: str = "sources.yaml") -> Path:
    path = tmp_path / name
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def ids(cli: Cli) -> list[str]:
    return [row[0] for row in table(cli.ok("sources", "list").stdout)[1:]]


# ---- import ----


def test_import_adds_sources_and_audits_every_entry_with_actor_and_reason(cli: Cli, tmp_path: Path):
    path = write(tmp_path, TWO_SOURCES)
    result = cli.ok("sources", "import", str(path), "--reason", WHY, "--actor", "alice")
    assert result.stdout == "imported: 2 added, 0 updated, 0 unchanged\nadded: src_news, ig_main\n"
    assert result.stderr == ""
    assert ids(cli) == ["ig_main", "src_news"]
    assert {(d["actor"], d["reason"]) for d in cli.details(SOURCE_ADDED)} == {("alice", WHY)}
    (summary,) = cli.details(SOURCES_IMPORTED)
    assert (summary["actor"], summary["reason"], summary["added"]) == ("alice", WHY, 2)


def test_importing_the_same_file_again_changes_and_audits_nothing(cli: Cli, tmp_path: Path):
    path = write(tmp_path, TWO_SOURCES)
    cli.ok("sources", "import", str(path), "--reason", WHY)
    before = len(cli.events())
    again = cli.ok("sources", "import", str(path), "--reason", WHY)
    assert again.stdout == (
        "imported: 0 added, 0 updated, 2 unchanged\nunchanged: src_news, ig_main\n"
    )
    assert len(cli.events()) == before


def test_import_updates_wholesale_so_an_omitted_enabled_disables_the_source(
    cli: Cli, tmp_path: Path
):
    cli.add("src_news", "City News")
    cli.ok("sources", "enable", "src_news", "--reason", WHY)
    cli.ok("sources", "edit", "src_news", "--set", "credibility=5", "--reason", WHY)
    path = write(tmp_path, TWO_SOURCES)
    result = cli.ok("sources", "import", str(path), "--reason", WHY)
    assert "updated: src_news" in result.stdout
    row = table(cli.ok("sources", "list").stdout)[2]
    assert row[0:4] == ["src_news", "news_rss", "no", "disabled"]
    assert "credibility: 3" in cli.ok("sources", "export").stdout  # the edit was reset
    assert any("enabled" in d["changed"] for d in cli.details(SOURCE_EDITED))


def test_the_import_help_warns_that_entries_replace_stored_sources_wholesale(cli: Cli):
    text = " ".join(cli.ok("sources", "import", "--help").stdout.split())
    assert "replaces" in text
    assert "omit" in text
    assert "enabled" in text
    assert "false" in text
    assert "--dry-run" in text


def test_dry_run_reports_without_writing_and_needs_no_reason(cli: Cli, tmp_path: Path):
    path = write(tmp_path, TWO_SOURCES)
    result = cli.ok("sources", "import", str(path), "--dry-run")
    assert result.stdout == (
        "dry run, nothing written: would add 2, update 0, leave 0 unchanged\n"
        "added: src_news, ig_main\n"
    )
    assert cli.ok("sources", "list").stdout == "No sources registered.\n"
    assert cli.events() == []


def test_dry_run_still_rejects_an_invalid_file(cli: Cli, tmp_path: Path):
    path = write(tmp_path, "sources: [1]\n")
    assert "entry 0" in assert_clean_failure(cli("sources", "import", str(path), "--dry-run"))


def test_a_real_import_needs_a_reason(cli: Cli, tmp_path: Path):
    path = write(tmp_path, TWO_SOURCES)
    result = cli("sources", "import", str(path))
    assert result.exit_code == 2
    assert "--reason" in result.stderr
    assert not cli.db_path.exists()


@pytest.mark.parametrize("reason", ["", "  ", "bell\x07"])
def test_a_bad_reason_writes_nothing(cli: Cli, tmp_path: Path, reason: str):
    path = write(tmp_path, TWO_SOURCES)
    assert "reason" in assert_clean_failure(cli("sources", "import", str(path), "-r", reason))
    assert cli.events() == []
    assert ids(cli) == []


def test_import_conflicts_are_listed_one_per_line_and_nothing_is_written(cli: Cli, tmp_path: Path):
    cli.add("src_news", "City News")
    cli.add("src_other", "Other")
    conflicting = """\
sources:
  - id: src_new_one
    type: news_rss
    name: Fresh
    feed_url: https://example.org/a.xml
  - id: src_news
    type: live_stream
    name: Now a stream
    stream_url: https://example.org/live.m3u8
  - id: src_third
    type: news_rss
    name: OTHER
    feed_url: https://example.org/b.xml
"""
    path = write(tmp_path, conflicting)
    result = cli("sources", "import", str(path), "--reason", WHY)
    lines = assert_clean_failure(result).splitlines()
    assert lines[0] == "error: import rejected: 2 conflict(s) with existing sources"
    assert lines[1:] == [
        "  entry 1: type of an existing source is immutable",
        "  entry 2: name is used by another source",
    ]
    assert ids(cli) == ["src_news", "src_other"]  # all-or-nothing: src_new_one was not added


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "document is empty"),
        ("   \n\t\n", "document is empty"),
        ("just a string", "must be a mapping"),
        ("- a\n- b\n", "must be a mapping"),
        ("sources: not-a-list\n", "must be a list"),
        ("sources: [\n", "malformed"),
        ("sources: []\nextra: 1\n", "unexpected top-level key"),
        ("sources: []\n---\nsources: []\n", "multiple YAML documents"),
        ("sources: &a []\n", "anchors and aliases"),
        ("sources: !!python/object:os.system []\n", "tags are not allowed"),
        ("sources: [1]\n", "entry must be a mapping"),
        ("sources:\n  - id: src_one\n    id: src_two\n", "duplicate mapping key"),
        ("sources:\n  - id: src_one\n    type: tiktok\n    name: x\n", "unknown source type"),
    ],
)
def test_malformed_or_unsafe_documents_are_clean_errors(
    cli: Cli, tmp_path: Path, content: str, expected: str
):
    path = write(tmp_path, content)
    result = cli("sources", "import", str(path), "--reason", WHY)
    assert expected in assert_clean_failure(result)
    assert cli.events() == []


def test_every_problem_of_an_invalid_file_is_reported_and_secrets_are_not_echoed(
    cli: Cli, tmp_path: Path
):
    content = f"""\
sources:
  - id: src_news
    type: news_rss
    name: City News
    feed_url: https://example.org/rss.xml
    token: {PASTED}
  - id: src_valid
    type: news_rss
    name: Valid
    feed_url: https://example.org/valid.xml
  - id: src_valid
    type: news_rss
    name: Again
    feed_url: https://example.org/again.xml
  - id: ig_main
    type: instagram
    name: IG
    handle: h
    secret_ref: {PASTED}
"""
    result = cli("sources", "import", str(write(tmp_path, content)), "--reason", WHY)
    stderr = assert_clean_failure(result)
    assert "entry 0: <source>: unknown field" in stderr
    assert "entry 2: duplicate id: same as entry 1" in stderr
    assert "entry 3: secret_ref" in stderr
    assert PASTED not in result.stdout + result.stderr
    assert cli.events() == []


def test_a_missing_file_and_a_directory_are_clean_errors(cli: Cli, tmp_path: Path):
    for target in (tmp_path / "missing.yaml", tmp_path):
        result = cli("sources", "import", str(target), "--reason", WHY)
        assert "cannot read the import file" in assert_clean_failure(result)
        assert str(tmp_path) not in result.stderr


def test_a_file_of_exactly_the_size_limit_is_read_and_one_byte_more_is_refused(
    cli: Cli, tmp_path: Path
):
    at_limit = write(tmp_path, b"sources: []\n" + b"#" * (MAX_FILE_BYTES - 12), "at.yaml")
    assert at_limit.stat().st_size == MAX_FILE_BYTES
    ok = cli.ok("sources", "import", str(at_limit), "--reason", WHY)
    assert ok.stdout == "imported: 0 added, 0 updated, 0 unchanged\n"
    over = write(tmp_path, b"sources: []\n" + b"#" * (MAX_FILE_BYTES - 11), "over.yaml")
    assert over.stat().st_size == MAX_FILE_BYTES + 1
    result = cli("sources", "import", str(over), "--reason", WHY)
    assert f"larger than {MAX_FILE_BYTES} bytes" in assert_clean_failure(result)


def test_a_utf8_bom_is_accepted_but_a_hidden_character_inside_a_value_is_not(
    cli: Cli, tmp_path: Path
):
    bom = b"\xef\xbb\xbf"
    cli.ok("sources", "import", str(write(tmp_path, bom + TWO_SOURCES.encode())), "--reason", WHY)
    assert ids(cli) == ["ig_main", "src_news"]
    hidden = TWO_SOURCES.replace("City News", "City" + chr(0xFEFF) + "News")
    result = cli("sources", "import", str(write(tmp_path, hidden, "h.yaml")), "--reason", WHY)
    assert "hidden characters" in assert_clean_failure(result)


@pytest.mark.parametrize(
    "raw",
    [b"sources: \xff\xfe", b"\xff\xfes\x00o\x00", b"\x80\x81", b"sources: []\n\xc3\x28"],
)
def test_invalid_utf8_is_a_clean_error(cli: Cli, tmp_path: Path, raw: bytes):
    result = cli("sources", "import", str(write(tmp_path, raw)), "--reason", WHY)
    assert "not valid UTF-8" in assert_clean_failure(result)


def test_a_nul_byte_inside_the_document_is_a_clean_error(cli: Cli, tmp_path: Path):
    path = write(tmp_path, b"sources: []\n\x00")
    assert_clean_failure(cli("sources", "import", str(path), "--reason", WHY))


def test_non_ascii_names_survive_an_import_and_an_export_round_trip(cli: Cli, tmp_path: Path):
    content = TWO_SOURCES.replace("City News", "Diario de Cali ñáé")
    cli.ok("sources", "import", str(write(tmp_path, content)), "--reason", WHY)
    assert "Diario de Cali ñáé" in cli.ok("sources", "export").stdout


# ---- export ----


def test_exporting_an_empty_registry_gives_an_empty_list(cli: Cli):
    result = cli.ok("sources", "export")
    assert (result.stdout, result.stderr) == ("sources: []\n", "")


def test_export_prints_yaml_to_stdout_only_and_writes_no_audit_event(cli: Cli):
    cli.add()
    events_before = len(cli.events())
    result = cli.ok("sources", "export")
    assert result.stderr == ""
    assert result.stdout.startswith("sources:\n- id: src_news\n  type: news_rss\n")
    assert "feed_url: https://example.org/rss.xml" in result.stdout
    assert len(cli.events()) == events_before


def test_export_only_ever_shows_the_secret_reference_never_a_secret(cli: Cli, tmp_path: Path):
    cli.ok("sources", "import", str(write(tmp_path, TWO_SOURCES)), "--reason", WHY)
    exported = cli.ok("sources", "export").stdout
    assert "secret_ref: env:IG_MAIN" in exported
    assert "token" not in exported
    assert "password" not in exported


def test_what_export_writes_import_reads_back_unchanged(cli: Cli, tmp_path: Path):
    cli.ok("sources", "import", str(write(tmp_path, TWO_SOURCES)), "--reason", WHY)
    exported = write(tmp_path, cli.ok("sources", "export").stdout, "exported.yaml")
    again = cli.ok("sources", "import", str(exported), "--reason", WHY)
    assert again.stdout.startswith("imported: 0 added, 0 updated, 2 unchanged")


def test_export_to_a_file_refuses_to_overwrite_without_force(cli: Cli, tmp_path: Path):
    cli.add()
    target = tmp_path / "out.yaml"
    first = cli.ok("sources", "export", "--output", str(target))
    assert first.stdout == f"exported to {target}\n"
    assert target.read_text(encoding="utf-8").startswith("sources:")
    target.write_text("precious", encoding="utf-8")
    refused = cli("sources", "export", "--output", str(target))
    assert "--force" in assert_clean_failure(refused)
    assert target.read_text(encoding="utf-8") == "precious"
    forced = cli.ok("sources", "export", "-o", str(target), "--force")
    assert forced.stdout == f"exported to {target}\n"
    assert target.read_text(encoding="utf-8").startswith("sources:")


def test_export_to_a_missing_directory_is_a_clean_error(cli: Cli, tmp_path: Path):
    result = cli("sources", "export", "--output", str(tmp_path / "nope" / "out.yaml"))
    assert "cannot write the output file" in assert_clean_failure(result)
    assert str(tmp_path) not in result.stderr


def test_a_file_export_prints_nothing_on_stderr_and_no_yaml_on_stdout(cli: Cli, tmp_path: Path):
    cli.add()
    result = cli.ok("sources", "export", "--output", str(tmp_path / "o.yaml"))
    assert result.stderr == ""
    assert "feed_url" not in result.stdout


def test_exporting_a_registry_with_every_type_works(cli: Cli):
    cli.add("src_news", "City News")
    cli.ok("sources", "add", "live_one", "--type", "live_stream", "--name", "Live", "-r", WHY,
           "--set", "stream_url=https://example.org/live.m3u8")  # fmt: skip
    cli.ok("sources", "add", "audio_one", "--type", "recorded_audio", "--name", "Audio", "-r", WHY,
           "--set", "watch_dir=/srv/audio")  # fmt: skip
    exported = cli.ok("sources", "export").stdout
    for expected in ("type: news_rss", "type: live_stream", "type: recorded_audio"):
        assert expected in exported


def sources_file(prefix: str, count: int) -> str:
    entries = [
        f"  - id: {prefix}_{n:02d}\n    type: news_rss\n    name: Feed {n:02d}\n"
        f"    feed_url: https://example.org/{n}.xml\n"
        for n in range(count)
    ]
    return "sources:\n" + "".join(entries)


@pytest.mark.parametrize(
    ("conflicts", "more"),
    [(1, None), (20, None), (21, "  and 1 more"), (25, "  and 5 more")],
)
def test_at_most_twenty_import_conflicts_are_listed_followed_by_the_remaining_count(
    cli: Cli, tmp_path: Path, conflicts: int, more: str | None
):
    cli.ok("sources", "import", str(write(tmp_path, sources_file("old", 25))), "-r", WHY)
    # new ids, but the names of the first `conflicts` stored sources: one name conflict each
    clashing = sources_file("new", conflicts)
    path = write(tmp_path, clashing, "clash.yaml")
    lines = assert_clean_failure(cli("sources", "import", str(path), "-r", WHY)).splitlines()
    assert lines[0] == f"error: import rejected: {conflicts} conflict(s) with existing sources"
    listed = [line for line in lines[1:] if line.startswith("  entry ")]
    assert len(listed) == min(conflicts, 20)
    assert listed[0] == "  entry 0: name is used by another source"
    assert listed[-1] == f"  entry {min(conflicts, 20) - 1}: name is used by another source"
    assert lines[1 + len(listed) :] == ([more] if more else [])
