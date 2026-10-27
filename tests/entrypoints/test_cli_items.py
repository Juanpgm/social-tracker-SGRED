"""`social-tracker items list` and `items search`: reading collected items safely."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from tests.entrypoints.conftest import Cli
from tests.items.factories import make_item

NOW = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
HOSTILE = [
    '"',
    '""',
    '"unbalanced',
    "*",
    "-",
    "-minus",
    "AND",
    "OR",
    "NOT",
    "NEAR",
    "NEAR/2",
    "a AND",
    "col:term",
    "title:quake",
    "(",
    ")",
    "((quake)",
    "'; DROP TABLE items; --",
    "\U0001f30b",
    "^quake",
    "x" * 5000,
]


def seed(cli: Cli, *items) -> None:
    with Database(cli.db_path) as db:
        SqliteItemRepository(db).add_new(items)


def sample() -> list:
    return [
        make_item(
            "old",
            title="Old quake report",
            text="felt in Cali",
            published_at=NOW - timedelta(days=2),
        ),
        make_item("new", title="Nuevo sismo en Bogotá", text="sin heridos", published_at=NOW),
        make_item(
            "other",
            source_id="src_other",
            title="Economy news",
            text="the dollar rises",
            published_at=NOW - timedelta(days=1),
        ),
    ]


# ---- list ----


def test_an_empty_store_says_so(cli: Cli):
    result = cli.ok("items", "list")
    assert (result.stdout, result.stderr) == ("No items.\n", "")


def test_items_are_listed_newest_first_with_time_source_and_title(cli: Cli):
    seed(cli, *sample())
    lines = cli.ok("items", "list").stdout.splitlines()
    assert lines[0].split() == ["PUBLISHED", "SOURCE", "TITLE"]
    assert lines[1].split(None, 3) == ["2026-03-01", "08:00", "src_news", "Nuevo sismo en Bogotá"]
    assert [line.split()[3] for line in lines[1:]] == ["Nuevo", "Economy", "Old"]


def test_the_source_filter_and_the_limit_narrow_the_listing(cli: Cli):
    seed(cli, *sample())
    only = cli.ok("items", "list", "--source", "src_other").stdout
    assert "Economy news" in only
    assert "quake" not in only
    one = cli.ok("items", "list", "--limit", "1").stdout.splitlines()
    assert len(one) == 2  # header and one item


def test_an_unknown_source_has_no_items_but_is_not_an_error(cli: Cli):
    seed(cli, *sample())
    assert cli.ok("items", "list", "--source", "src_none").stdout == "No items.\n"


@pytest.mark.parametrize("limit", ["0", "-1", "1001", "abc", "1.5"])
def test_an_invalid_limit_is_a_usage_error(cli: Cli, limit: str):
    result = cli("items", "list", "--limit", limit)
    assert result.exit_code == 2


def test_the_limit_boundaries_are_accepted(cli: Cli):
    seed(cli, *sample())
    assert cli("items", "list", "--limit", "1").exit_code == 0
    assert cli("items", "list", "--limit", "1000").exit_code == 0


def test_an_item_without_a_title_shows_the_start_of_its_text(cli: Cli):
    seed(cli, make_item("a", title="", text="Only a body that is quite descriptive"))
    assert "Only a body that is quite descriptive" in cli.ok("items", "list").stdout


def test_long_titles_are_shortened_and_stay_on_one_line(cli: Cli):
    seed(cli, make_item("a", title="word " * 100, text="x"))
    line = cli.ok("items", "list").stdout.splitlines()[1]
    assert line.endswith("...")
    assert len(line) < 160


def test_terminal_control_sequences_in_titles_are_escaped_not_executed(cli: Cli):
    evil = "red\x1b[31m title second line\x07 " + chr(0x202E) + " reversed"
    seed(cli, make_item("a", title="placeholder", text="x", raw_payload={"x": 1}))
    with Database(cli.db_path) as db, db.transaction() as conn:
        conn.execute("UPDATE items SET title = ?", (evil.replace("\r\n", " "),))
    out = cli.ok("items", "list").stdout
    assert "\x1b" not in out
    assert "\x07" not in out
    assert chr(0x202E) not in out
    assert "\\x1b" in out  # shown as visible text, not obeyed
    assert "\\x07" in out
    assert "\\u202e" in out
    assert len(out.splitlines()) == 2


def test_listing_never_prints_urls_or_raw_payloads(cli: Cli):
    seed(cli, make_item("a", raw_payload={"description": "<script>x</script>"}))
    out = cli.ok("items", "list").stdout
    assert "example.org" not in out
    assert "<script>" not in out


# ---- search ----


def test_search_finds_whole_words_ignoring_case_and_accents(cli: Cli):
    seed(cli, *sample())
    for query in ("sismo", "SISMO", "bogota", "Bogotá"):
        out = cli.ok("items", "search", query).stdout
        assert "Nuevo sismo en Bogotá" in out
        assert "Economy" not in out


def test_search_with_no_match_says_so(cli: Cli):
    seed(cli, *sample())
    assert cli.ok("items", "search", "volcano").stdout == "No items found.\n"


def test_every_word_must_match_and_the_limit_applies(cli: Cli):
    seed(cli, *sample())
    assert "Old quake report" in cli.ok("items", "search", "quake cali").stdout
    assert cli.ok("items", "search", "quake bogota").stdout == "No items found.\n"
    assert len(cli.ok("items", "search", "the OR quake", "--limit", "1").stdout.splitlines()) <= 2


@pytest.mark.parametrize("query", HOSTILE, ids=lambda q: repr(q)[:18])
def test_search_text_is_never_interpreted_as_search_syntax(cli: Cli, query: str):
    seed(cli, make_item("a", title="AND OR NOT NEAR quake", text="plain text"))
    result = cli("items", "search", "--", query)
    assert result.exit_code == 0, result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert result.stderr == ""


def test_operator_words_are_ordinary_words(cli: Cli):
    seed(cli, *sample())
    assert cli.ok("items", "search", "--", "quake OR economy").stdout == "No items found.\n"
    assert cli.ok("items", "search", "--", "quake -zzz").stdout == "No items found.\n"


def test_a_search_without_words_finds_nothing(cli: Cli):
    seed(cli, *sample())
    assert cli.ok("items", "search", "--", "***").stdout == "No items found.\n"
    assert cli.ok("items", "search", "").stdout == "No items found.\n"


def test_items_commands_work_on_a_fresh_database_path(cli: Cli, tmp_path: Path):
    result = cli.ok(
        "items", "search", "x", env={"SOCIAL_TRACKER_DB_PATH": str(tmp_path / "n" / "new.db")}
    )
    assert result.stdout == "No items found.\n"
