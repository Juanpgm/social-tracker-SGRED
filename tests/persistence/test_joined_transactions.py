"""Repositories sharing one Database compose into one atomic unit through joined transactions.

This is what the application layer relies on: `sources.add(...)` followed by
`control.append_control_event(...)` (or `sources.delete(...)` followed by
`control.delete_source_state(...)`) succeed or fail together, even though each repository opens
its own `db.transaction()` internally.
"""

import pytest

from social_tracker.control.domain import CollectionState, ControlEvent, SourceState
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.shared.errors import ConflictError, NotFoundError
from tests.control.repository_contract import AT
from tests.sources.repository_contract import make_source


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "joined.db")
    yield database
    database.close()


@pytest.fixture
def sources(db) -> SqliteSourceRepository:
    return SqliteSourceRepository(db)


@pytest.fixture
def control(db) -> SqliteControlRepository:
    return SqliteControlRepository(db)


def event(kind: str, source_id: str | None = None) -> ControlEvent:
    return ControlEvent(at=AT, kind=kind, source_id=source_id)


def test_adding_a_source_and_recording_its_event_commit_together(db, sources, control):
    with db.transaction():
        sources.add(make_source("src_one", "One"))
        control.append_control_event(event("source_created", "src_one"))
        control.save_source_state(SourceState("src_one", CollectionState.IDLE))
    assert [s.id for s in sources.list()] == ["src_one"]
    assert [e.kind for e in control.list_control_events("src_one")] == ["source_created"]
    assert control.get_source_state("src_one") is not None


def test_an_outer_failure_undoes_every_repository_write(db, sources, control):
    with pytest.raises(RuntimeError), db.transaction():  # noqa: PT012 - the body is the scenario
        sources.add(make_source("src_one", "One"))
        control.append_control_event(event("source_created", "src_one"))
        control.save_source_state(SourceState("src_one", CollectionState.IDLE))
        raise RuntimeError("the service failed after the repositories succeeded")
    assert sources.list() == []
    assert control.list_control_events() == []
    assert control.get_source_state("src_one") is None


def test_a_failing_repository_call_undoes_the_ones_before_it(db, sources, control):
    sources.add(make_source("src_one", "One"))
    with pytest.raises(ConflictError), db.transaction():  # noqa: PT012 - the body is the scenario
        control.append_control_event(event("source_created", "src_two"))
        sources.add(make_source("src_two", "one"))  # same name as src_one once normalised
    assert control.list_control_events() == []
    assert [s.id for s in sources.list()] == ["src_one"]


def test_a_caught_inner_failure_keeps_the_other_writes_of_the_unit(db, sources, control):
    sources.add(make_source("src_one", "One"))
    with db.transaction():
        control.append_control_event(event("attempted", "src_two"))
        with pytest.raises(ConflictError):
            sources.add(make_source("src_two", "ONE"))
        control.append_control_event(event("rejected", "src_two"))
    assert [e.kind for e in control.list_control_events()] == ["attempted", "rejected"]
    assert [s.id for s in sources.list()] == ["src_one"]


def test_deleting_a_source_removes_its_state_but_keeps_its_events(db, sources, control):
    sources.add(make_source("src_one", "One"))
    sources.add(make_source("src_two", "Two"))
    control.save_source_state(SourceState("src_one", CollectionState.IDLE))
    control.save_source_state(SourceState("src_two", CollectionState.IDLE))
    control.append_control_event(event("source_created", "src_one"))
    with db.transaction():
        sources.delete("src_one")
        control.delete_source_state("src_one")
        control.append_control_event(event("source_deleted", "src_one"))
    assert [s.id for s in sources.list()] == ["src_two"]
    assert control.get_source_state("src_one") is None
    assert control.get_source_state("src_two") is not None
    assert [e.kind for e in control.list_control_events("src_one")] == [
        "source_created",
        "source_deleted",
    ]


def test_deleting_a_missing_source_leaves_the_state_untouched(db, sources, control):
    control.save_source_state(SourceState("src_ghost", CollectionState.IDLE))
    with pytest.raises(NotFoundError), db.transaction():  # noqa: PT012 - the body is the scenario
        control.delete_source_state("src_ghost")
        sources.delete("src_ghost")  # no such source: the whole unit rolls back
    assert control.get_source_state("src_ghost") is not None


def test_each_repository_call_is_still_atomic_on_its_own(sources, control):
    with pytest.raises(NotFoundError):
        sources.delete("src_nobody")
    control.delete_source_state("src_nobody")
    control.append_control_event(event("alone"))
    assert [e.kind for e in control.list_control_events()] == ["alone"]
