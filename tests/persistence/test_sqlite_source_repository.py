"""SqliteSourceRepository: the shared contract plus everything specific to SQLite."""

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from social_tracker.persistence.db import Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.shared.errors import ConflictError, NotFoundError, StorageError, ValidationError
from social_tracker.sources.domain import (
    KEYWORD_MAX_LENGTH,
    KEYWORDS_MAX_ITEMS,
    MAX_PATH_LENGTH,
    MAX_URL_LENGTH,
    parse_source,
)
from social_tracker.sources.ports import SourceRepository
from tests.sources.factories import EMBEDDED_CREDENTIALS_URL, T0, payload
from tests.sources.repository_contract import SourceRepositoryContract, make_source


class TestSqliteSourceRepositoryContract(SourceRepositoryContract):
    """Every contract test, against a fresh database file per repository."""

    @pytest.fixture(autouse=True)
    def _databases(self, tmp_path: Path):
        self._root = tmp_path
        self._opened: list[Database] = []
        yield
        for db in self._opened:
            db.close()

    def make_repo(self) -> SourceRepository:
        db = Database(self._root / f"sources-{len(self._opened)}.db")
        self._opened.append(db)
        return SqliteSourceRepository(db)


# ---- helpers for the SQLite-specific tests ----


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "app.db")
    yield database
    database.close()


@pytest.fixture
def repo(db) -> SqliteSourceRepository:
    return SqliteSourceRepository(db)


def insert_raw(db: Database, **overrides: Any) -> None:
    """Insert a row bypassing both the repository and the CHECK constraints."""
    row: dict[str, Any] = {
        "id": "src_raw",
        "name": "Raw",
        "normalized_name": "raw",
        "type": "news_rss",
        "mode": "recorded",
        "credibility": 3,
        "enabled": 0,
        "secret_ref": None,
        "config_json": json.dumps({"feed_url": "https://example.org/rss.xml"}),
        "created_at": "2026-01-01T12:00:00+00:00",
        "updated_at": "2026-01-01T12:00:00+00:00",
    }
    row.update(overrides)
    db.query("PRAGMA ignore_check_constraints=ON")
    try:
        with db.transaction() as conn:
            marks = ", ".join("?" for _ in row)
            conn.execute(
                f"INSERT INTO sources ({', '.join(row)}) VALUES ({marks})", list(row.values())
            )
    finally:
        db.query("PRAGMA ignore_check_constraints=OFF")


def test_it_satisfies_the_port_and_exposes_nothing_else(repo):
    assert isinstance(repo, SourceRepository)
    public = {name for name in dir(repo) if not name.startswith("_")}
    assert public == {"add", "update", "get", "find_by_name_ci", "delete", "list"}


# ---- uniqueness ----


def test_duplicate_id_is_a_conflict_error_with_a_stable_code(repo):
    repo.add(make_source("src_one", "First"))
    with pytest.raises(ConflictError) as info:
        repo.add(make_source("src_one", "Second"))
    assert info.value.code == "conflict"
    assert repo.get("src_one").name == "First"


def test_duplicate_normalized_name_is_a_conflict_error(repo):
    repo.add(make_source("src_one", "Radio Uno"))
    with pytest.raises(ConflictError) as info:
        repo.add(make_source("src_two", "  RADIO UNO "))
    assert info.value.code == "conflict"
    assert [s.id for s in repo.list()] == ["src_one"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Caf\u00e9", "Cafe\u0301"),  # precomposed vs decomposed
        ("CAF\u00c9", "cafe\u0301"),
        ("Stra\u00dfe", "STRASSE"),  # casefold: sharp s -> ss
        ("\u212b", "\u00c5"),  # angstrom sign vs A-ring
    ],
)
def test_names_equal_after_nfc_and_casefold_collide(repo, first, second):
    repo.add(make_source("src_one", first))
    with pytest.raises(ConflictError):
        repo.add(make_source("src_two", second))
    assert repo.find_by_name_ci(second).id == "src_one"  # type: ignore[union-attr]


def test_failed_add_leaves_no_partial_row(repo, db):
    repo.add(make_source("src_one", "Alpha"))
    with pytest.raises(ConflictError):
        repo.add(make_source("src_two", "alpha"))
    assert db.query("SELECT count(*) FROM sources")[0][0] == 1


def test_failed_update_changes_nothing(repo):
    repo.add(make_source("src_one", "Alpha", credibility=2))
    repo.add(make_source("src_two", "Beta", credibility=4))
    with pytest.raises(ConflictError):
        repo.update(make_source("src_two", "ALPHA", credibility=5))
    assert repo.get("src_two").name == "Beta"
    assert repo.get("src_two").credibility == 4


@pytest.mark.parametrize("by", ["id", "name"])
def test_concurrent_adds_across_connections_exactly_one_wins(tmp_path, by):
    path = tmp_path / "shared.db"
    dbs = [Database(path), Database(path)]
    repos = [SqliteSourceRepository(d) for d in dbs]
    try:
        for trial in range(15):
            barrier = threading.Barrier(2)
            outcomes: list[object] = []
            lock = threading.Lock()

            def worker(index: int, trial=trial, barrier=barrier, outcomes=outcomes, lock=lock):
                same_id = f"src_{by}_{trial}"
                source = (
                    make_source(same_id, f"name {index} {trial}")
                    if by == "id"
                    else make_source(f"src_{index}_{trial}", f"Shared Name {trial}")
                )
                barrier.wait(timeout=5)
                try:
                    repos[index].add(source)
                    result: object = "won"
                except ConflictError:
                    result = "lost"
                except Exception as exc:  # surfaced by the assert below
                    result = exc
                with lock:
                    outcomes.append(result)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)
            assert sorted(map(str, outcomes)) == ["lost", "won"], outcomes
        assert len(repos[0].list()) == 15
    finally:
        for d in dbs:
            d.close()


def test_concurrent_adds_of_distinct_sources_all_succeed(repo):
    barrier = threading.Barrier(10)
    errors: list[BaseException] = []

    def worker(i: int):
        barrier.wait(timeout=5)
        try:
            repo.add(make_source(f"src_{i:02d}", f"Name {i}"))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert errors == []
    assert [s.id for s in repo.list()] == [f"src_{i:02d}" for i in range(10)]


# ---- round trips ----


def test_keywords_come_back_as_a_tuple_and_none_stays_none(repo):
    repo.add(make_source("src_kw", "With keywords", keywords=["b", "a", "b"]))
    repo.add(make_source("src_nokw", "Without keywords"))
    got = repo.get("src_kw")
    assert got.keywords == ("b", "a", "b")  # type: ignore[union-attr]
    assert isinstance(got.keywords, tuple)  # type: ignore[union-attr]
    assert repo.get("src_nokw").keywords is None  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "news_rss", "keywords": [], "poll_interval_s": 120},
        {"type": "instagram", "include_stories": False, "include_posts": False},
        {"type": "instagram", "poll_interval_s": 900, "secret_ref": "ref.name:v1/x-y"},
        {"type": "live_stream", "chunk_seconds": 10, "language": "pt-BR"},
        {"type": "live_stream", "stream_url": "rtmp://example.org/live", "chunk_seconds": 60},
        {"type": "recorded_audio", "feed_url": None, "watch_dir": "C:\\audio drop\\\u00e9"},
        {"type": "recorded_audio", "poll_interval_s": 1},
    ],
    ids=lambda o: str(o)[:50],
)
def test_every_type_specific_field_round_trips(repo, overrides):
    type_ = overrides.pop("type")
    source = parse_source(payload(type_, **overrides))
    repo.add(source)
    assert repo.get(source.id) == source
    assert repo.get(source.id).model_dump() == source.model_dump()


def test_common_fields_round_trip_including_disabled_and_enabled(repo):
    on = make_source("src_on", "On", enabled=True, credibility=5)
    off = make_source("src_off", "Off", enabled=False, credibility=1)
    repo.add(on)
    repo.add(off)
    assert repo.get("src_on") == on
    assert repo.get("src_off") == off
    assert repo.get("src_on").enabled is True
    assert repo.get("src_off").enabled is False


def test_timestamps_keep_microseconds(repo):
    moment = T0.replace(microsecond=123456)
    repo.add(make_source(created_at=moment, updated_at=moment))
    got = repo.get("src_one")
    assert (got.created_at, got.updated_at) == (moment, moment)


def test_update_replaces_the_whole_type_specific_payload(repo):
    repo.add(make_source("src_one", "Alpha", keywords=["x"]))
    replacement = parse_source(payload("live_stream", id="src_one", name="Alpha live"))
    repo.update(replacement)
    got = repo.get("src_one")
    assert got == replacement
    assert got.type == "live_stream"
    assert not hasattr(got, "keywords")


def test_very_long_but_valid_values_round_trip(repo):
    long_keywords = ["k" * KEYWORD_MAX_LENGTH for _ in range(KEYWORDS_MAX_ITEMS)]
    long_url = "https://example.org/" + "p" * (MAX_URL_LENGTH - len("https://example.org/"))
    source = parse_source(
        payload("news_rss", name="n" * 100, feed_url=long_url, keywords=long_keywords)
    )
    repo.add(source)
    assert repo.get(source.id) == source
    audio = parse_source(
        payload(
            "recorded_audio",
            id="src_audio",
            name="a",
            feed_url=None,
            watch_dir="d" * MAX_PATH_LENGTH,
        )
    )
    repo.add(audio)
    assert repo.get("src_audio") == audio


def test_id_boundaries_round_trip(repo):
    for source_id in ("abc", "z" * 64):
        repo.add(make_source(source_id, f"name {source_id}"))
    assert [s.id for s in repo.list()] == ["abc", "z" * 64]


def test_list_orders_by_id_bytes_not_by_insertion(repo):
    for source_id in ("src_b", "src_a2", "src_a10", "src_a_1", "src_aa", "src_a1"):
        repo.add(make_source(source_id, f"name {source_id}"))
    assert [s.id for s in repo.list()] == sorted(
        ["src_b", "src_a2", "src_a10", "src_a_1", "src_aa", "src_a1"]
    )


# ---- persistence across reopen ----


def test_data_survives_closing_and_reopening_the_database_file(tmp_path):
    path = tmp_path / "durable.db"
    originals = [
        make_source("src_one", "Alpha", keywords=["k1", "k2"]),
        parse_source(payload("instagram", id="src_ig", name="Insta")),
        parse_source(payload("live_stream", id="src_live", name="Live")),
        parse_source(payload("recorded_audio", id="src_audio", name="Audio")),
    ]
    first = Database(path)
    repo = SqliteSourceRepository(first)
    for source in originals:
        repo.add(source)
    repo.update(originals[0].with_changes(credibility=5))
    repo.delete("src_audio")
    first.close()

    second = Database(path)
    try:
        reopened = SqliteSourceRepository(second)
        expected = [originals[0].with_changes(credibility=5), originals[1], originals[2]]
        assert reopened.list() == sorted(expected, key=lambda s: s.id)
        assert reopened.find_by_name_ci("ALPHA").id == "src_one"  # type: ignore[union-attr]
        with pytest.raises(ConflictError):
            reopened.add(make_source("src_new", "alpha"))
    finally:
        second.close()


def test_two_repositories_on_one_file_share_data(tmp_path):
    path = tmp_path / "shared.db"
    with Database(path) as a, Database(path) as b:
        SqliteSourceRepository(a).add(make_source())
        assert SqliteSourceRepository(b).get("src_one") == make_source()


# ---- SQL injection ----

NASTY = [
    "'; DROP TABLE sources; --",
    "Robert'); DROP TABLE sources;--",
    '" OR "1"="1',
    "x' OR '1'='1",
    "%_[]\\*?",
    "1; PRAGMA writable_schema=ON",
]


@pytest.mark.parametrize("nasty", NASTY)
def test_sql_metacharacters_are_stored_verbatim_and_never_executed(repo, db, nasty):
    source = make_source(
        "src_one",
        nasty,
        feed_url="https://example.org/a?q=" + nasty.replace(" ", "+"),
        keywords=[nasty],
    )
    repo.add(source)
    assert repo.get("src_one") == source
    assert repo.get("src_one").name == nasty.strip()
    assert repo.find_by_name_ci(nasty).id == "src_one"  # type: ignore[union-attr]
    assert db.query("SELECT count(*) FROM sources")[0][0] == 1
    assert db.query("SELECT name FROM sqlite_master WHERE name = 'sources'") != []


@pytest.mark.parametrize("nasty", NASTY)
def test_sql_metacharacters_in_lookups_match_nothing_and_change_nothing(repo, db, nasty):
    repo.add(make_source("src_one", "Alpha"))
    repo.add(make_source("src_two", "Beta"))
    with pytest.raises(NotFoundError):
        repo.get(nasty)
    with pytest.raises(NotFoundError):
        repo.delete(nasty)
    assert repo.find_by_name_ci(nasty) is None
    assert [s.id for s in repo.list()] == ["src_one", "src_two"]


def test_an_embedded_nul_round_trips_but_a_leading_nul_is_rejected_by_the_schema(repo):
    # SQLite's length() stops at the first NUL, so a name that starts with one measures as empty
    # and trips the CHECK. The domain does not forbid NUL in names; the database is the net.
    inside = make_source("src_one", "ab\x00cd")
    repo.add(inside)
    assert repo.get("src_one") == inside
    with pytest.raises(StorageError):
        repo.add(make_source("src_two", "\x00leading"))
    assert [s.id for s in repo.list()] == ["src_one"]


# ---- error behaviour ----


def test_not_found_messages_do_not_echo_the_requested_id(repo):
    for action in (lambda: repo.get("TOP-SECRET-ID"), lambda: repo.delete("TOP-SECRET-ID")):
        with pytest.raises(NotFoundError) as info:
            action()
        assert "TOP-SECRET-ID" not in str(info.value)
        assert info.value.code == "not_found"


def test_non_source_arguments_are_rejected(repo):
    for bad in (None, {"id": "src_one"}, "src_one", 5):
        with pytest.raises(ValidationError):
            repo.add(bad)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            repo.update(bad)  # type: ignore[arg-type]
    assert repo.list() == []


def test_unbindable_ids_become_storage_errors_not_raw_sqlite_errors(repo):
    with pytest.raises(StorageError):
        repo.get(["src_one"])  # type: ignore[arg-type]


def test_a_closed_database_surfaces_as_a_storage_error(tmp_path):
    database = Database(tmp_path / "closed.db")
    repo = SqliteSourceRepository(database)
    database.close()
    for action in (
        lambda: repo.add(make_source()),
        lambda: repo.get("src_one"),
        lambda: repo.list(),
        lambda: repo.find_by_name_ci("x"),
        lambda: repo.delete("src_one"),
        lambda: repo.update(make_source()),
    ):
        with pytest.raises(StorageError):
            action()


# ---- malformed stored rows ----

MALFORMED = {
    "config is not json": {"config_json": "not json"},
    "config is an array": {"config_json": "[1, 2]"},
    "config is null": {"config_json": "null"},
    "config is deeply nested": {"config_json": "[" * 100_000 + "]" * 100_000},
    "config has an unknown key": {"config_json": '{"feed_url": "https://e.org/r", "evil": 1}'},
    "config lacks a required key": {"config_json": "{}"},
    "config overrides a common column": {
        "config_json": '{"feed_url": "https://e.org/r", "credibility": 1}'
    },
    "config has a bad value": {"config_json": '{"feed_url": "ftp://e.org/r"}'},
    "config embeds a credential url": {
        "config_json": json.dumps({"feed_url": EMBEDDED_CREDENTIALS_URL})
    },
    "unknown type": {"type": "podcast"},
    "credibility out of range": {"credibility": 99},
    "credibility is text": {"credibility": "high"},
    "enabled is not a flag": {"enabled": 7},
    "mode does not match the type": {"mode": "streaming"},
    "naive created_at": {"created_at": "2026-01-01T12:00:00"},
    "garbage updated_at": {"updated_at": "yesterday"},
    "updated before created": {"updated_at": "2025-01-01T00:00:00+00:00"},
    "invalid id": {"id": "Bad Id"},
    "blank name": {"name": ""},
    "instagram without secret": {
        "type": "instagram",
        "config_json": '{"handle": "some.handle"}',
        "secret_ref": None,
    },
}


@pytest.mark.parametrize("overrides", MALFORMED.values(), ids=MALFORMED.keys())
def test_a_malformed_stored_row_is_a_storage_error_on_every_read_path(repo, db, overrides):
    insert_raw(db, **overrides)
    row_id = db.query("SELECT id FROM sources")[0][0]
    with pytest.raises(StorageError) as info:
        repo.get(row_id)
    assert info.value.code == "storage_error"
    with pytest.raises(StorageError):
        repo.list()
    with pytest.raises(StorageError):
        repo.find_by_name_ci("raw")


def test_one_bad_row_never_yields_a_partial_list(repo, db):
    repo.add(make_source("src_good", "Good"))
    insert_raw(db, id="src_bad", name="Bad", normalized_name="bad", config_json="not json")
    with pytest.raises(StorageError):
        repo.list()
    assert repo.get("src_good").name == "Good"


def test_storage_error_messages_do_not_echo_stored_content(repo, db):
    insert_raw(
        db, config_json='{"feed_url": "https://TOP-SECRET.example.org/", "evil": "TOP-SECRET"}'
    )
    with pytest.raises(StorageError) as info:
        repo.get("src_raw")
    assert "TOP-SECRET" not in str(info.value)
