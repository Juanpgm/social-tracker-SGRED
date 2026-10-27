"""The SQLite escalation repository: the shared contract plus what is specific to SQLite."""

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from social_tracker.persistence.classification_repo import SqliteClassificationRepository
from social_tracker.persistence.db import Database, load_migrations
from social_tracker.persistence.items_repo import SqliteItemRepository
from social_tracker.persistence.llm_repo import SqliteEscalationRepository
from social_tracker.shared.errors import StorageError
from tests.llm.repository_contracts import EscalationRepositoryContract, Rig, judgment, state
from tests.llm.support import NOW, item_id, verdict


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "llm.db")
    yield database
    database.close()


class TestSqliteEscalationRepositoryContract(EscalationRepositoryContract):
    @pytest.fixture(autouse=True)
    def _database(self, tmp_path: Path):
        self.db = Database(tmp_path / "contract.db")
        yield
        self.db.close()

    def make_rig(self) -> Rig:
        return Rig(
            SqliteEscalationRepository(self.db),
            SqliteClassificationRepository(self.db),
            SqliteItemRepository(self.db),
        )


def test_the_schema_is_the_ninth_migration():
    assert load_migrations()[-1].name == "llm_hardening"
    assert load_migrations()[-1].version == 9


_OLD_VERDICT = (
    "INSERT INTO classifications (item_id, label, confidence, stage, rationale, signals, topics, "
    "low_confidence, needs_llm, truncated, catalog_status, priority, source_credibility_used, "
    "classified_at, classifier_version) VALUES (?, 'unintentional_rumor', 0.55, 'rules', "
    "'rumor candidate', '[]', '[]', 0, ?, 0, 'no_claim', ?, 3, '2026-03-01T12:00:00+00:00', "
    "'cascade-2/topics-0')"
)


def test_a_database_at_the_previous_version_keeps_its_verdicts_when_migrated(tmp_path: Path):
    path = tmp_path / "old.db"
    old = Database(path, migrations=load_migrations()[:8])
    with old.transaction() as conn:  # the schema of the previous version has no settlement columns
        conn.execute(_OLD_VERDICT, (item_id(1), 1, 0.5))
    old.close()
    migrated = Database(path)
    try:
        kept = SqliteClassificationRepository(migrated).get(item_id(1))
        assert kept == verdict(1, rationale="rumor candidate", signals=(), topics=())
        assert (kept.settled_at, kept.settle_reason) == (None, None)
        assert SqliteEscalationRepository(migrated).counts().judged == 0
    finally:
        migrated.close()


def test_the_settlement_of_old_verdicts_is_rebuilt_from_what_the_judgments_say(tmp_path: Path):
    path = tmp_path / "backfill.db"
    old = Database(path, migrations=load_migrations()[:8])
    with old.transaction() as conn:
        for n in range(1, 8):
            conn.execute(_OLD_VERDICT, (item_id(n), 0 if n != 7 else 1, 0.5))
    esc = SqliteEscalationRepository(old)
    stamp = NOW + timedelta(minutes=2)
    esc.add_judgment(judgment(1, judged_at=stamp))
    esc.add_judgment(
        judgment(2, outcome="invalid", label=None, confidence=None, rationale=None, judged_at=stamp)
    )
    esc.add_judgment(judgment(3, outcome="reused", reused_from=item_id(1), judged_at=stamp))
    esc.put_state(state(4, state="parked", next_attempt_at=None, updated_at=stamp))
    esc.put_state(state(5, state="dropped", next_attempt_at=None, updated_at=stamp))
    esc.put_state(state(7, updated_at=stamp))  # still waiting for a retry
    old.close()
    migrated = Database(path)
    try:
        verdicts = SqliteClassificationRepository(migrated)
        got = {n: verdicts.get(item_id(n)) for n in range(1, 8)}
        assert [(got[n].settled_at, got[n].settle_reason) for n in (1, 2, 3, 4, 5)] == [
            (stamp, "llm"),
            (stamp, "invalid"),
            (stamp, "duplicate"),
            (stamp, "parked"),
            (stamp, "parked"),
        ]
        assert (got[6].settled_at, got[6].settle_reason) == (None, None)  # the cascade's own
        assert (got[7].settled_at, got[7].settle_reason) == (None, None)  # still queued
    finally:
        migrated.close()


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE classifications SET settled_at = '2026-03-01T12:00:00+00:00'",
        "UPDATE classifications SET settle_reason = 'llm'",
        "UPDATE classifications SET settled_at = 'yesterday', settle_reason = 'llm'",
        "UPDATE classifications SET settled_at = '2026-03-01T12:00:00+00:00', "
        "settle_reason = 'maybe'",
    ],
)
def test_a_settlement_that_breaks_a_constraint_is_refused_by_the_database(db: Database, sql: str):
    SqliteClassificationRepository(db).add(verdict(1))
    with pytest.raises(StorageError), db.transaction() as conn:
        conn.execute(sql)


def test_forgetting_a_verdict_also_forgets_its_judgment_and_bookkeeping(db: Database):
    verdicts, esc = SqliteClassificationRepository(db), SqliteEscalationRepository(db)
    for n in (1, 2, 3):
        verdicts.add(verdict(n))
        esc.add_judgment(judgment(n))
        esc.put_state(state(n))
    assert verdicts.delete(item_id(1)) is True
    assert verdicts.delete_many([item_id(2)]) == 1
    assert esc.get_judgment(item_id(1)) is None
    assert esc.get_state(item_id(2)) is None
    assert esc.get_judgment(item_id(2)) is None
    assert esc.get_judgment(item_id(3)) is not None  # a verdict that stays keeps its judgment
    assert esc.get_state(item_id(3)) is not None


def test_a_judgment_row_that_breaks_a_constraint_is_refused_by_the_database(db: Database):
    with pytest.raises(StorageError), db.transaction() as conn:
        conn.execute(
            "INSERT INTO llm_judgments (item_id, outcome, label, evidence, prompt_version, "
            "prior_label, prior_confidence, prior_stage, judged_at) "
            "VALUES ('itm_00000001', 'judged', NULL, '[]', 'judge-1', 'no_risk', 0.5, 'rules', "
            "'2026-03-01T12:00:00+00:00')"
        )


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO llm_budget VALUES ('2026-03-01T12:30:00+00:00', 1)",  # not on the hour
        "INSERT INTO llm_budget VALUES ('2026-03-01T12:00:00+00:00', -1)",
        "INSERT INTO llm_lease VALUES (2, 'a', '2026-03-01T12:00:00+00:00')",  # one slot only
        "INSERT INTO llm_escalations VALUES ('itm_00000001', 'weird', 0, 0, NULL, NULL, "
        "'2026-03-01T12:00:00+00:00')",
        "INSERT INTO llm_escalations VALUES ('X', 'parked', 0, 0, NULL, NULL, "
        "'2026-03-01T12:00:00+00:00')",
    ],
)
def test_the_last_line_of_defence_rejects_what_python_would_have_refused(db: Database, sql: str):
    with pytest.raises(StorageError), db.transaction() as conn:
        conn.execute(sql)


def test_a_tampered_stored_judgment_is_a_storage_error_not_a_half_valid_object(db: Database):
    esc = SqliteEscalationRepository(db)
    esc.add_judgment(judgment(1))
    with db.transaction() as conn:
        conn.execute("UPDATE llm_judgments SET evidence = '[1, 2]'")
    with pytest.raises(StorageError):
        esc.get_judgment(item_id(1))


def test_a_tampered_stored_state_is_a_storage_error(db: Database):
    esc = SqliteEscalationRepository(db)
    esc.put_state(state(1))
    with db.transaction() as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute("UPDATE llm_escalations SET last_error = 'Not A Code!'")
        conn.execute("PRAGMA ignore_check_constraints = OFF")
    with pytest.raises(StorageError):
        esc.get_state(item_id(1))


def test_two_connections_never_grant_more_than_the_cap(tmp_path: Path):
    first, second = Database(tmp_path / "two.db"), Database(tmp_path / "two.db")
    try:
        a, b = SqliteEscalationRepository(first), SqliteEscalationRepository(second)
        hour = NOW.replace(minute=0)
        granted = sum((a if n % 2 else b).reserve_call(hour, 7) for n in range(30))
        assert granted == 7
        assert a.calls_in_hour(hour) == b.calls_in_hour(hour) == 7
    finally:
        first.close()
        second.close()


def test_the_lease_is_shared_between_connections(tmp_path: Path):
    first, second = Database(tmp_path / "lease.db"), Database(tmp_path / "lease.db")
    try:
        a, b = SqliteEscalationRepository(first), SqliteEscalationRepository(second)
        until = NOW + timedelta(seconds=60)
        assert a.try_lease("proc-a", NOW, until) is True
        assert b.try_lease("proc-b", NOW, until) is False
        a.release_lease("proc-a")
        assert b.try_lease("proc-b", NOW, until) is True
    finally:
        first.close()
        second.close()


def test_the_queue_is_served_by_the_index_without_a_table_scan_of_the_verdicts(db: Database):
    plan = db.query(
        "EXPLAIN QUERY PLAN SELECT c.item_id FROM classifications c "
        "WHERE c.needs_llm = 1 ORDER BY c.priority DESC, c.classified_at, c.item_id"
    )
    text = " ".join(str(row["detail"]) for row in plan)
    assert "ix_classifications_queue" in text
    assert isinstance(db.query("SELECT 1")[0][0], int)
    assert sqlite3.sqlite_version_info >= (3, 35)  # ON CONFLICT ... DO UPDATE ... WHERE
