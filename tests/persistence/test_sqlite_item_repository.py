"""SqliteItemRepository: the shared contract plus everything specific to SQLite."""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from social_tracker.items.domain import AddReport, ItemStatus
from social_tracker.items.ports import ItemRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from social_tracker.shared.errors import StorageError
from tests.items.factories import NOW, make_item
from tests.items.repository_contract import ItemRepositoryContract


class TestSqliteItemRepositoryContract(ItemRepositoryContract):
    @pytest.fixture(autouse=True)
    def _databases(self, tmp_path: Path):
        self._root = tmp_path
        self._opened: list[Database] = []
        yield
        for db in self._opened:
            db.close()

    def make_repo(self) -> ItemRepository:
        db = Database(self._root / f"items-{len(self._opened)}.db")
        self._opened.append(db)
        return SqliteItemRepository(db)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "app.db")
    yield database
    database.close()


@pytest.fixture
def repo(db) -> SqliteItemRepository:
    return SqliteItemRepository(db)


def test_it_satisfies_the_port(repo):
    assert isinstance(repo, ItemRepository)


def test_the_index_follows_the_table_through_inserts_and_deletes(repo, db):
    repo.add_new(
        [make_item("a", title="alpha", text="first"), make_item("b", title="beta", text="second")]
    )
    assert [r[0] for r in db.query("SELECT rowid FROM items_fts WHERE items_fts MATCH 'alpha'")]
    with db.transaction() as conn:
        conn.execute("DELETE FROM items WHERE external_id = 'a'")
    assert db.query("SELECT rowid FROM items_fts WHERE items_fts MATCH 'alpha'") == []
    assert len(db.query("SELECT rowid FROM items_fts WHERE items_fts MATCH 'beta'")) == 1
    assert db.query("INSERT INTO items_fts(items_fts) VALUES ('integrity-check')") == []


def test_editing_the_text_reindexes_it(repo, db):
    item = make_item("a", title="alpha", text="first")
    repo.add_new([item])
    with db.transaction() as conn:
        conn.execute("UPDATE items SET text = 'gamma' WHERE id = ?", (item.id,))
    assert [i.id for i in repo.search("gamma")] == [item.id]
    assert repo.search("first") == []


def test_items_survive_the_deletion_of_their_source_row(repo, db):
    now = "2026-03-01T08:00:00+00:00"
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO sources (id, name, normalized_name, type, mode, credibility, enabled,"
            " config_json, created_at, updated_at)"
            " VALUES ('src_news', 'City News', 'city news', 'news_rss', 'recorded', 3, 0,"
            ' \'{"feed_url": "https://example.org/rss.xml"}\', ?, ?)',
            (now, now),
        )
    item = make_item("a")
    repo.add_new([item])
    with db.transaction() as conn:
        conn.execute("DELETE FROM sources WHERE id = 'src_news'")
    assert repo.get(item.id) == item
    assert repo.get(item.id).source_name == "City News"


def test_a_batch_is_all_or_nothing_when_the_database_refuses_a_row(repo, db):
    good = make_item("good", title="good", text="good")
    with db.transaction() as conn:
        conn.execute(
            "CREATE TRIGGER refuse_bad BEFORE INSERT ON items WHEN new.external_id = 'bad' "
            "BEGIN SELECT RAISE(ABORT, 'refused'); END"
        )
    bad = make_item("bad", title="bad", text="bad")
    with pytest.raises(StorageError):
        repo.add_new([good, bad])
    assert repo.count() == 0
    assert repo.search("good") == []


def test_two_threads_ingesting_the_same_batch_store_it_exactly_once(tmp_path):
    path = tmp_path / "race.db"
    first, second = Database(path), Database(path)
    repos = [SqliteItemRepository(first), SqliteItemRepository(second)]
    batch = [make_item(f"g{i}", title=f"title {i}", text=f"text {i}") for i in range(30)]
    barrier = threading.Barrier(2)
    reports: list[AddReport] = []
    errors: list[BaseException] = []

    def worker(index: int):
        barrier.wait(timeout=5)
        try:
            reports.append(repos[index].add_new(batch))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert errors == []
        assert sorted(r.new for r in reports) == [0, 30]
        assert sorted(r.duplicate for r in reports) == [0, 30]
        assert repos[0].count() == 30
    finally:
        first.close()
        second.close()


def test_a_database_locked_by_another_writer_is_a_storage_error_not_a_hang(tmp_path):
    path = tmp_path / "busy.db"
    database = Database(path, busy_timeout_ms=50)
    repo = SqliteItemRepository(database)
    blocker = sqlite3.connect(path, isolation_level=None)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        with pytest.raises(StorageError) as info:
            repo.add_new([make_item("a")])
        assert info.value.code == "storage_error"
        assert time.monotonic() - started < 5
        assert repo.count() == 0  # WAL readers are not blocked
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert repo.add_new([make_item("a")]) == AddReport(new=1, duplicate=0)  # recovers
    database.close()


def test_items_survive_closing_and_reopening_the_database(tmp_path):
    path = tmp_path / "durable.db"
    item = make_item("a", title="Sismo \U0001f30b", text="persistente")
    with Database(path) as first:
        SqliteItemRepository(first).add_new([item])
    with Database(path) as second:
        reopened = SqliteItemRepository(second)
        assert reopened.get(item.id) == item
        assert reopened.search("persistente") == [item]


def insert_raw(db: Database, **overrides: Any) -> None:
    row: dict[str, Any] = {
        "id": "itm_raw",
        "source_id": "src_news",
        "source_name": "City News",
        "kind": "article",
        "external_id": "raw-1",
        "parent_id": None,
        "url": None,
        "title": "t",
        "author": None,
        "text": "x",
        "language": None,
        "published_at": "2026-03-01T08:00:00+00:00",
        "fetched_at": "2026-03-01T08:00:00+00:00",
        "content_hash": "0" * 64,
        "flags": "[]",
        "status": "new",
        "raw_payload": "{}",
        **overrides,
    }
    db.query("PRAGMA ignore_check_constraints=ON")
    try:
        with db.transaction() as conn:
            marks = ", ".join("?" for _ in row)
            conn.execute(
                f"INSERT INTO items ({', '.join(row)}) VALUES ({marks})", list(row.values())
            )
    finally:
        db.query("PRAGMA ignore_check_constraints=OFF")


STORED_ROWS = {
    "unknown kind": {"kind": "video"},
    "unknown status value shape": {"status": "Bad Status"},
    "garbage timestamp": {"published_at": "yesterday"},
    "naive timestamp": {"fetched_at": "2026-03-01T08:00:00"},
    "bad hash": {"content_hash": "xyz"},
    "flags not an array": {"flags": "{}"},
    "flags not json": {"flags": "[oops"},
    "payload not json": {"raw_payload": "{oops"},
    "payload not an object": {"raw_payload": "[]"},
    "unsafe url": {"url": "javascript:alert(1)"},
    "invalid id": {"id": "Bad Id"},
}


@pytest.mark.parametrize("overrides", STORED_ROWS.values(), ids=STORED_ROWS.keys())
def test_a_tampered_stored_row_is_a_storage_error_not_a_half_valid_item(repo, db, overrides):
    repo.add_new([make_item("good", title="good", text="good")])
    insert_raw(db, **overrides)
    with pytest.raises(StorageError) as info:
        repo.list()
    assert info.value.code == "storage_error"


def test_a_closed_database_surfaces_as_a_storage_error(tmp_path):
    database = Database(tmp_path / "closed.db")
    repo = SqliteItemRepository(database)
    database.close()
    for action in (
        lambda: repo.add_new([make_item("a")]),
        lambda: repo.get("itm_a"),
        lambda: repo.list(),
        lambda: repo.search("quake"),
        lambda: repo.count(),
        lambda: repo.purge_fetched_before(NOW),
    ):
        with pytest.raises(StorageError):
            action()


def test_status_changes_are_stored_as_wire_values(repo, db):
    item = make_item("a")
    repo.add_new([item])
    repo.set_status(item.id, ItemStatus.CLASSIFIED)
    assert db.query("SELECT status FROM items")[0][0] == "classified"
