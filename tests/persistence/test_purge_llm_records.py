"""The retention purge and the records of the language model.

When an item is purged what the model said about it (the audit record, with its raw output and
evidence) goes with it, and an item whose verdict still waits for the model is not purged merely
because it is old: only when it is twice as old as the retention allows.
"""

from datetime import timedelta
from pathlib import Path

import pytest

from social_tracker.items.domain import ItemStatus
from social_tracker.items.service import ItemService
from social_tracker.persistence.classification_repo import SqliteClassificationRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from social_tracker.persistence.llm_repo import SqliteEscalationRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import FixedClock
from tests.items.factories import NOW, make_item
from tests.llm.repository_contracts import judgment, state
from tests.llm.support import verdict

FETCHED = NOW + timedelta(minutes=1)  # the factory's fetched_at


@pytest.fixture
def db(tmp_path: Path):
    database = Database(tmp_path / "purge.db")
    yield database
    database.close()


class Rig:
    def __init__(self, db: Database) -> None:
        self.items = SqliteItemRepository(db)
        self.verdicts = SqliteClassificationRepository(db)
        self.esc = SqliteEscalationRepository(db)

    def add(self, name: str, *, status=ItemStatus.CLASSIFIED, queued=False, fetched=FETCHED):
        item = make_item(name, text=f"text of {name}", fetched_at=fetched)
        assert self.items.add_new([item]).new == 1
        self.items.set_status(item.id, status)
        stored = verdict(
            item.id.removeprefix("itm_"),
            needs_llm=queued,
            priority=0.5 if queued else 0.0,
            **({} if queued else {"stage": "llm"}),
        )
        self.verdicts.add(stored)
        self.esc.add_judgment(judgment(item.id.removeprefix("itm_"), raw_output='{"label": "x"}'))
        self.esc.put_state(state(item.id.removeprefix("itm_")))
        return item


# ---- what the model said goes with the item ----


def test_a_time_purge_forgets_the_judgments_and_bookkeeping_of_the_purged_items_only(db):
    rig = Rig(db)
    old = rig.add("old", fetched=NOW - timedelta(days=60))
    new = rig.add("new", fetched=NOW)
    assert rig.items.purge_fetched_before(NOW - timedelta(days=30)) == 1
    assert rig.items.get(old.id) is None
    short_old, short_new = old.id.removeprefix("itm_"), new.id.removeprefix("itm_")
    assert rig.esc.get_judgment(f"itm_{short_old}") is None
    assert rig.esc.get_state(f"itm_{short_old}") is None
    assert rig.esc.get_judgment(f"itm_{short_new}") is not None  # the item that stays
    assert rig.esc.get_state(f"itm_{short_new}") is not None


def test_a_purge_once_classified_forgets_them_too(db):
    rig = Rig(db)
    done = rig.add("done")
    scope = rig.add("scope", status=ItemStatus.OUT_OF_SCOPE)
    kept = rig.add("kept", status=ItemStatus.CLASSIFICATION_FAILED)
    assert rig.items.purge_classified() == 2
    for gone in (done, scope):
        assert rig.esc.get_judgment(gone.id) is None
        assert rig.esc.get_state(gone.id) is None
    assert rig.esc.get_judgment(kept.id) is not None


def test_nothing_of_the_model_is_left_in_the_database_after_a_purge(db):
    rig = Rig(db)
    rig.add("a", fetched=NOW - timedelta(days=90))
    rig.items.purge_fetched_before(NOW - timedelta(days=30))
    rows = db.query("SELECT count(*) FROM llm_judgments WHERE raw_output IS NOT NULL")
    assert rows[0][0] == 0
    assert db.query("SELECT count(*) FROM llm_escalations")[0][0] == 0


# ---- an item that still waits for the model is not purged for being old ----


def test_a_purge_once_classified_skips_what_still_waits_for_the_model(db):
    rig = Rig(db)
    waiting = rig.add("waiting", queued=True)
    done = rig.add("done")
    assert rig.items.purge_classified() == 1
    assert rig.items.get(waiting.id) is not None
    assert rig.items.get(done.id) is None
    assert rig.esc.get_judgment(waiting.id) is not None  # nothing of its was forgotten


def test_it_is_purged_once_it_is_older_than_the_given_horizon(db):
    rig = Rig(db)
    horizon = NOW - timedelta(days=2)
    stale = rig.add("stale", queued=True, fetched=horizon - timedelta(seconds=1))
    edge = rig.add("edge", queued=True, fetched=horizon)
    fresh = rig.add("fresh", queued=True, fetched=NOW)
    assert rig.items.purge_classified(purge_queued_fetched_before=horizon) == 1
    assert rig.items.get(stale.id) is None  # strictly before the horizon
    assert rig.items.get(edge.id) is not None
    assert rig.items.get(fresh.id) is not None


def test_a_time_purge_skips_a_waiting_item_until_it_is_twice_as_old(db):
    rig = Rig(db)
    cutoff = NOW - timedelta(days=30)
    twice = NOW - timedelta(days=60)
    waiting = rig.add("waiting", queued=True, fetched=NOW - timedelta(days=45))
    ancient = rig.add("ancient", queued=True, fetched=twice - timedelta(seconds=1))
    plain = rig.add("plain", fetched=NOW - timedelta(days=45))
    assert rig.items.purge_fetched_before(cutoff, purge_queued_fetched_before=twice) == 2
    assert rig.items.get(waiting.id) is not None
    assert rig.items.get(ancient.id) is None
    assert rig.items.get(plain.id) is None


def test_a_time_purge_without_a_horizon_never_takes_a_waiting_item(db):
    rig = Rig(db)
    waiting = rig.add("waiting", queued=True, fetched=NOW - timedelta(days=900))
    assert rig.items.purge_fetched_before(NOW) == 0
    assert rig.items.get(waiting.id) is not None


def test_a_spared_item_leaves_no_tombstone(db):
    rig = Rig(db)
    waiting = rig.add("waiting", queued=True, fetched=NOW - timedelta(days=45))
    rig.items.purge_fetched_before(NOW - timedelta(days=30))
    assert (
        rig.items.add_new([make_item("waiting", text="text of waiting", fetched_at=NOW)]).duplicate
        == 1
    )
    assert rig.items.get(waiting.id) is not None


def test_a_settled_verdict_is_not_spared(db):
    rig = Rig(db)
    parked = rig.add("parked", fetched=NOW - timedelta(days=45))
    assert rig.items.purge_fetched_before(NOW - timedelta(days=30)) == 1
    assert rig.items.get(parked.id) is None


# ---- the service decides how old is old enough ----


def service(db: Database, retention_days: int, now=NOW) -> ItemService:
    return ItemService(
        SqliteItemRepository(db), SqliteUnitOfWork(db), FixedClock(now), retention_days
    )


def test_the_service_keeps_a_waiting_item_until_it_is_twice_as_old_as_the_retention(db):
    rig = Rig(db)
    waiting = rig.add("waiting", queued=True, fetched=NOW - timedelta(days=45))
    ancient = rig.add("ancient", queued=True, fetched=NOW - timedelta(days=61))
    edge = rig.add("edge", queued=True, fetched=NOW - timedelta(days=60))
    plain = rig.add("plain", fetched=NOW - timedelta(days=45))
    report = service(db, 30).purge_expired()
    assert report.deleted == 2
    assert rig.items.get(waiting.id) is not None
    assert (
        rig.items.get(edge.id) is not None
    )  # exactly twice: still kept, like the retention itself
    assert rig.items.get(ancient.id) is None
    assert rig.items.get(plain.id) is None


def test_with_a_retention_of_zero_a_waiting_item_gets_two_days_of_grace(db):
    rig = Rig(db)
    waiting = rig.add("waiting", queued=True, fetched=NOW - timedelta(hours=47))
    ancient = rig.add("ancient", queued=True, fetched=NOW - timedelta(hours=49))
    plain = rig.add("plain", fetched=NOW)
    assert service(db, 0).purge_expired().deleted == 2
    assert rig.items.get(waiting.id) is not None
    assert rig.items.get(ancient.id) is None
    assert rig.items.get(plain.id) is None
