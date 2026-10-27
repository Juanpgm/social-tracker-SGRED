"""Reusable contract for any `UnitOfWork` implementation (in-memory fake and SQLite)."""

import threading
from dataclasses import dataclass

import pytest

from social_tracker.control.domain import ControlEvent
from social_tracker.control.ports import ControlRepository
from social_tracker.shared.errors import ConflictError
from social_tracker.shared.uow import UnitOfWork
from social_tracker.sources.ports import SourceRepository
from tests.control.repository_contract import AT
from tests.sources.repository_contract import make_source


@dataclass
class Env:
    uow: UnitOfWork
    sources: SourceRepository
    control: ControlRepository


class StopNow(BaseException):  # noqa: N818 - deliberately a BaseException, like KeyboardInterrupt
    """A `BaseException` that is not an `Exception`."""


class Boom(Exception):
    pass


def ev(kind: str) -> ControlEvent:
    return ControlEvent(at=AT, kind=kind)


class UnitOfWorkContract:
    def make_env(self) -> Env:
        raise NotImplementedError

    @pytest.fixture
    def env(self) -> Env:
        return self.make_env()

    def test_is_a_unit_of_work(self, env: Env):
        assert isinstance(env.uow, UnitOfWork)

    def test_writes_in_two_repositories_commit_together(self, env: Env):
        with env.uow.transaction():
            env.sources.add(make_source("src_one", "One"))
            env.control.append_control_event(ev("added"))
        assert [s.id for s in env.sources.list()] == ["src_one"]
        assert [e.kind for e in env.control.list_control_events()] == ["added"]

    def test_a_failure_rolls_back_every_repository_and_propagates_the_same_error(self, env: Env):
        error = Boom("x")
        with pytest.raises(Boom) as caught, env.uow.transaction():  # noqa: PT012
            env.sources.add(make_source("src_one", "One"))
            env.control.append_control_event(ev("added"))
            raise error
        assert caught.value is error
        assert env.sources.list() == []
        assert env.control.list_control_events() == []

    def test_a_repository_error_undoes_the_earlier_write_of_the_other_repository(self, env: Env):
        env.sources.add(make_source("src_one", "One"))
        with pytest.raises(ConflictError), env.uow.transaction():  # noqa: PT012
            env.control.append_control_event(ev("first"))
            env.sources.add(make_source("src_one", "Dup"))
        assert env.control.list_control_events() == []
        assert [s.name for s in env.sources.list()] == ["One"]

    def test_base_exceptions_roll_back_and_are_not_swallowed(self, env: Env):
        with pytest.raises(StopNow), env.uow.transaction():  # noqa: PT012
            env.sources.add(make_source("src_one", "One"))
            raise StopNow
        assert env.sources.list() == []

    def test_keyboard_interrupt_is_not_swallowed(self, env: Env):
        with pytest.raises(KeyboardInterrupt), env.uow.transaction():  # noqa: PT012
            env.control.append_control_event(ev("added"))
            raise KeyboardInterrupt
        assert env.control.list_control_events() == []

    def test_the_unit_is_reusable_after_a_failure(self, env: Env):
        with pytest.raises(Boom), env.uow.transaction():  # noqa: PT012
            env.sources.add(make_source("src_one", "One"))
            raise Boom
        with env.uow.transaction():
            env.sources.add(make_source("src_one", "One"))
        assert [s.id for s in env.sources.list()] == ["src_one"]

    def test_nested_blocks_join_and_commit_together(self, env: Env):
        with env.uow.transaction():
            env.sources.add(make_source("src_one", "One"))
            with env.uow.transaction():
                env.control.append_control_event(ev("inner"))
        assert len(env.sources.list()) == 1
        assert [e.kind for e in env.control.list_control_events()] == ["inner"]

    def test_an_outer_failure_undoes_work_that_a_nested_block_completed(self, env: Env):
        with pytest.raises(Boom), env.uow.transaction():  # noqa: PT012
            with env.uow.transaction():
                env.sources.add(make_source("src_one", "One"))
            raise Boom
        assert env.sources.list() == []

    def test_a_nested_failure_caught_outside_undoes_only_the_nested_writes(self, env: Env):
        with env.uow.transaction():
            env.sources.add(make_source("src_one", "One"))
            with pytest.raises(Boom), env.uow.transaction():  # noqa: PT012
                env.control.append_control_event(ev("inner"))
                raise Boom
            env.control.append_control_event(ev("outer"))
        assert [s.id for s in env.sources.list()] == ["src_one"]
        assert [e.kind for e in env.control.list_control_events()] == ["outer"]

    def test_a_nested_failure_that_escapes_undoes_everything(self, env: Env):
        with pytest.raises(Boom), env.uow.transaction():  # noqa: PT012
            env.sources.add(make_source("src_one", "One"))
            with env.uow.transaction():
                env.control.append_control_event(ev("inner"))
                raise Boom
        assert env.sources.list() == []
        assert env.control.list_control_events() == []

    def test_an_empty_block_changes_nothing(self, env: Env):
        with env.uow.transaction():
            pass
        assert env.sources.list() == []

    def test_other_threads_wait_for_the_owner_to_finish(self, env: Env):
        inside = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def owner() -> None:
            with env.uow.transaction():
                env.control.append_control_event(ev("owner"))
                inside.set()
                release.wait(5)
                order.append("owner-done")

        def other() -> None:
            inside.wait(5)
            with env.uow.transaction():
                order.append("other-entered")
                env.control.append_control_event(ev("other"))

        t1, t2 = threading.Thread(target=owner), threading.Thread(target=other)
        t1.start()
        t2.start()
        assert inside.wait(5)
        t2.join(0.3)
        assert t2.is_alive()  # still waiting for the owner
        release.set()
        t1.join(5)
        t2.join(5)
        assert order == ["owner-done", "other-entered"]
        assert [e.kind for e in env.control.list_control_events()] == ["owner", "other"]
