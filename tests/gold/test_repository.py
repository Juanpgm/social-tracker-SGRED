"""The SQLite gold set (migration 0011): keyed by text, ordered by arrival, strict on read."""

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from social_tracker.classification.domain import Label
from social_tracker.gold.domain import MAX_GOLD_TEXT_CHARS, make_example
from social_tracker.persistence.db import Database
from social_tracker.persistence.gold_repo import SqliteGoldRepository
from social_tracker.shared.errors import StorageError, ValidationError

NOW = datetime(2026, 3, 1, 12, 0, 0, 123456, tzinfo=UTC)


def make(text: str, label: Label = Label.NO_RISK, **overrides):
    fields = {
        "text": text,
        "label": label,
        "note": "",
        "credibility": 3,
        "url": None,
        "added_at": NOW,
        "actor": "ana",
    }
    fields.update(overrides)
    return make_example(**fields)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "gold.db")
    yield database
    database.close()


@pytest.fixture
def repo(db: Database) -> SqliteGoldRepository:
    return SqliteGoldRepository(db)


def test_an_example_comes_back_exactly_as_it_went_in(repo):
    original = make(
        "Mañana, \"temblor\": 'fuerte'\náé \U0001f6a8",
        Label.UNINTENTIONAL_RUMOR,
        note="a chain; seen twice",
        credibility=5,
        url="https://official-domain.example/avisos/1",
    )
    assert repo.add(original) is True
    assert repo.get(original.id) == original


def test_the_same_text_again_changes_nothing(repo):
    first = make("hola mundo", Label.NO_RISK, note="first")
    assert repo.add(first) is True
    assert repo.add(make("hola   mundo", Label.UNINTENTIONAL_RUMOR, note="second")) is False
    assert repo.get(first.id) == first
    assert repo.count() == 1


def test_an_unknown_example_is_none_and_removing_it_says_so(repo):
    assert repo.get("gld_" + "0" * 16) is None
    assert repo.remove("gld_" + "0" * 16) is False


def test_an_example_can_be_removed_once(repo):
    example = make("hola")
    repo.add(example)
    assert repo.remove(example.id) is True
    assert repo.remove(example.id) is False
    assert repo.get(example.id) is None
    assert repo.count() == 0


def test_examples_are_listed_in_the_order_they_were_added(repo):
    for text in ("uno", "dos", "tres", "cuatro"):
        repo.add(make(text))
    assert [e.text for e in repo.list(limit=10)] == ["uno", "dos", "tres", "cuatro"]


def test_removing_and_adding_again_goes_to_the_end(repo):
    for text in ("uno", "dos", "tres"):
        repo.add(make(text))
    repo.remove(make("uno").id)
    repo.add(make("uno"))
    assert [e.text for e in repo.list(limit=10)] == ["dos", "tres", "uno"]


@pytest.mark.parametrize(
    ("limit", "offset", "expected"),
    [(2, 0, ["a", "b"]), (2, 2, ["c", "d"]), (2, 4, ["e"]), (2, 5, []), (10, 0, list("abcde"))],
)
def test_pages_cover_the_set(repo, limit, offset, expected):
    for text in "abcde":
        repo.add(make(text))
    assert [e.text for e in repo.list(limit=limit, offset=offset)] == expected


def test_a_filter_by_label(repo):
    repo.add(make("a", Label.NO_RISK))
    repo.add(make("b", Label.UNINTENTIONAL_RUMOR))
    repo.add(make("c", Label.NO_RISK))
    assert [e.text for e in repo.list(label=Label.NO_RISK, limit=10)] == ["a", "c"]
    assert repo.list(label=Label.RECYCLED_CONTENT, limit=10) == []


@pytest.mark.parametrize(("limit", "offset"), [(0, 0), (-1, 0), (5001, 0), (5, -1), (True, 0)])
def test_a_page_out_of_range_is_refused(repo, limit, offset):
    with pytest.raises(ValidationError):
        repo.list(limit=limit, offset=offset)


def test_a_text_as_long_as_allowed_is_stored(repo):
    example = make("x" * MAX_GOLD_TEXT_CHARS)
    assert repo.add(example)
    assert len(repo.get(example.id).text) == MAX_GOLD_TEXT_CHARS


@pytest.mark.parametrize(
    "text",
    [
        "'; DROP TABLE gold_examples; --",
        "Robert'); DELETE FROM classifications; --",
        '" OR 1=1 --',
        "%_[]\\",
        "{{ 7*7 }} ${jndi:ldap://x}",
    ],
)
def test_sql_in_a_text_is_data_not_a_statement(repo, db, text):
    example = make(text)
    repo.add(example)
    assert repo.get(example.id).text == text
    assert repo.count() == 1
    assert db.query("SELECT count(*) FROM gold_examples")[0][0] == 1


def test_a_stored_row_that_no_longer_builds_an_example_is_a_storage_error(repo, db):
    example = make("hola")
    repo.add(example)
    with db.transaction() as conn:
        conn.execute("UPDATE gold_examples SET credibility = 5, note = ?", ("x" * 400,))
        conn.execute("UPDATE gold_examples SET text = 'altered text' WHERE id = ?", (example.id,))
    with pytest.raises(StorageError):
        repo.get(example.id)
    with pytest.raises(StorageError):
        repo.list(limit=10)


def test_the_database_refuses_nonsense_itself(db):
    with pytest.raises(StorageError), db.transaction() as conn:
        conn.execute(
            "INSERT INTO gold_examples (id, text, label, note, credibility, added_at, actor) "
            "VALUES ('gld_nothexnothexnoth', 'x', 'no_risk', '', 3, "
            "'2026-03-01T12:00:00+00:00', 'a')"
        )
    with pytest.raises(StorageError), db.transaction() as conn:
        conn.execute(
            "INSERT INTO gold_examples (id, text, label, note, credibility, added_at, actor) "
            "VALUES ('gld_0123456789abcdef', 'x', 'made_up', '', 3, "
            "'2026-03-01T12:00:00+00:00', 'a')"
        )
    with pytest.raises(StorageError), db.transaction() as conn:
        conn.execute(
            "INSERT INTO gold_examples (id, text, label, note, credibility, added_at, actor) "
            "VALUES ('gld_0123456789abcdef', 'x', 'no_risk', '', 9, "
            "'2026-03-01T12:00:00+00:00', 'a')"
        )


def test_it_survives_a_restart(tmp_path):
    path = tmp_path / "again.db"
    first = Database(path)
    SqliteGoldRepository(first).add(make("persistente", Label.RECYCLED_CONTENT))
    first.close()
    second = Database(path)
    try:
        (found,) = SqliteGoldRepository(second).list(limit=10)
        assert (found.text, found.label) == ("persistente", Label.RECYCLED_CONTENT)
    finally:
        second.close()


def test_two_writers_adding_the_same_text_store_it_once(tmp_path):
    path = tmp_path / "race.db"
    Database(path).close()
    results: list[bool] = []

    def add() -> None:
        with Database(path) as db:
            results.append(SqliteGoldRepository(db).add(make("el mismo texto")))

    threads = [threading.Thread(target=add) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False] * 5 + [True]


def test_only_a_gold_example_can_be_added(repo):
    with pytest.raises(ValidationError):
        repo.add("not an example")  # type: ignore[arg-type]
