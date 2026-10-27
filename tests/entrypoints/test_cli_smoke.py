"""End-to-end smoke test: one operator session through the real CLI and a real SQLite file."""

import json
from pathlib import Path

from tests.entrypoints.conftest import Cli, table

IMPORT_FILE = """\
sources:
  - id: ig_main
    type: instagram
    name: Instagram Main
    handle: some.handle
    secret_ref: env:IG_MAIN
  - id: city_news
    type: news_rss
    name: City News (imported)
    feed_url: https://example.org/rss.xml
    enabled: true
"""


def test_an_operator_session_from_an_empty_database_to_a_running_analysis(cli: Cli, tmp_path: Path):
    # add -> list
    added = cli.ok(
        "sources", "add", "city_news", "--name", "City News", "--type", "news_rss",
        "--set", "feed_url=https://example.org/rss.xml", "--set", "poll_interval_s=300",
        "--reason", "first feed", "--actor", "ana",
    )  # fmt: skip
    assert added.stdout == "added city_news (news_rss)\n"
    assert table(cli.ok("sources", "list").stdout)[1:] == [
        ["city_news", "news_rss", "no", "disabled", "off", "City News"]
    ]

    # enable -> disable -> enable
    cli.ok("sources", "enable", "city_news", "--reason", "go live", "--actor", "ana")
    assert table(cli.ok("sources", "list").stdout)[1][2:4] == ["yes", "idle"]
    cli.ok("sources", "disable", "city_news", "--reason", "feed is down", "--actor", "ben")
    assert table(cli.ok("sources", "list").stdout)[1][2:4] == ["no", "disabled"]
    cli.ok("sources", "enable", "city_news", "--reason", "feed is back", "--actor", "ben")

    # import: dry run first (nothing changes), then for real
    path = tmp_path / "more.yaml"
    path.write_text(IMPORT_FILE, encoding="utf-8")
    dry = cli.ok("sources", "import", str(path), "--dry-run")
    assert dry.stdout.splitlines()[0] == (
        "dry run, nothing written: would add 1, update 1, leave 0 unchanged"
    )
    assert [row[0] for row in table(cli.ok("sources", "list").stdout)[1:]] == ["city_news"]
    real = cli.ok("sources", "import", str(path), "--reason", "bulk load", "--actor", "cy")
    assert real.stdout.splitlines()[0] == "imported: 1 added, 1 updated, 0 unchanged"
    rows = {row[0]: row for row in table(cli.ok("sources", "list").stdout)[1:]}
    assert rows["city_news"][5] == "City News (imported)"
    assert rows["city_news"][2:4] == ["yes", "idle"]  # the file said enabled: true
    assert rows["ig_main"][2:4] == ["no", "disabled"]  # omitted enabled defaults to false

    # export: the secret appears as a reference only, and it feeds the same registry back
    exported = cli.ok("sources", "export").stdout
    assert "secret_ref: env:IG_MAIN" in exported
    assert "poll_interval_s: 600" in exported  # the import reset the earlier 300
    out = tmp_path / "backup.yaml"
    cli.ok("sources", "export", "--output", str(out))
    assert out.read_text(encoding="utf-8") == exported

    # analysis off/on, globally and per source, then status
    cli.ok("analysis", "off", "--reason", "already paused", "--actor", "dee")  # no-op
    cli.ok("analysis", "on", "--source", "city_news", "--reason", "watch it", "--actor", "dee")
    cli.ok("analysis", "on", "--reason", "go", "--actor", "dee")
    status = cli.ok("status").stdout
    assert status == (
        f"database: {cli.db_path}\n"
        "global analysis: on\n"
        "alert threshold: 0.5\n"
        "sources: 2 total, 1 enabled\n"
        "collection states: disabled 1, idle 1\n"
        "effective analysis: 1 source(s)\n"
    )
    cli.ok("analysis", "off", "--reason", "end of shift", "--actor", "dee")
    assert "effective analysis: 0 source(s)" in cli.ok("status").stdout

    # every audit event of the session names who did it and why
    events = cli.events()
    assert events, "the session must have produced audit events"
    details = [json.loads(e.detail) for e in events]
    assert all(d["actor"] and d["reason"] for d in details), details
    assert [(e.kind, d["actor"], d["reason"]) for e, d in zip(events, details, strict=True)] == [
        ("source_added", "ana", "first feed"),
        ("source_enabled", "ana", "go live"),
        ("source_disabled", "ben", "feed is down"),
        ("source_enabled", "ben", "feed is back"),
        ("source_added", "cy", "bulk load"),
        ("source_edited", "cy", "bulk load"),
        ("sources_imported", "cy", "bulk load"),
        ("source_analysis_changed", "dee", "watch it"),
        ("global_analysis_changed", "dee", "go"),
        ("global_analysis_changed", "dee", "end of shift"),
    ]
