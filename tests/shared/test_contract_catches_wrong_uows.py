"""Meta-tests: the UnitOfWork contract must reject each deliberately wrong implementation."""

import copy
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import _LOCK_TYPES, InMemoryUnitOfWork
from tests.shared.uow_contract import Env, UnitOfWorkContract


def _capture(p: object) -> dict[str, Any]:
    return {k: copy.deepcopy(v) for k, v in vars(p).items() if not isinstance(v, _LOCK_TYPES)}


def _restore(snapshots: list[tuple[object, dict[str, Any]]]) -> None:
    for participant, state in snapshots:
        for name, value in state.items():
            setattr(participant, name, value)


class _Base:
    def __init__(self, participants: Sequence[object]) -> None:
        self._participants = tuple(participants)
        self._lock = threading.RLock()
        self._depth = 0


class NoRollback(_Base):
    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            yield


class Swallows(_Base):
    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            snaps = [(p, _capture(p)) for p in self._participants]
            try:
                yield
            except Exception:
                _restore(snaps)


class RollsBackOnlyExceptions(_Base):
    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            snaps = [(p, _capture(p)) for p in self._participants]
            try:
                yield
            except Exception:
                _restore(snaps)
                raise


class NoSavepoints(_Base):
    """Only the outermost block can roll back: a caught nested failure keeps its writes."""

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            snaps = [(p, _capture(p)) for p in self._participants] if self._depth == 0 else []
            self._depth += 1
            try:
                yield
            except BaseException:
                _restore(snaps)
                raise
            finally:
                self._depth -= 1


class RollsBackFirstRepositoryOnly(_Base):
    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            snaps = [(p, _capture(p)) for p in self._participants[:1]]
            try:
                yield
            except BaseException:
                _restore(snaps)
                raise


class WrapsErrors(_Base):
    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            snaps = [(p, _capture(p)) for p in self._participants]
            try:
                yield
            except Exception as exc:
                _restore(snaps)
                raise RuntimeError("wrapped") from exc
            except BaseException:
                _restore(snaps)
                raise


class NoLock(_Base):
    @contextmanager
    def transaction(self) -> Iterator[None]:
        snaps = [(p, _capture(p)) for p in self._participants]
        try:
            yield
        except BaseException:
            _restore(snaps)
            raise


class NestedFailureLeavesNestedWrites(_Base):
    """A failing nested block restores nothing and lets the outer block decide."""

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            outermost = self._depth == 0
            snaps = [(p, _capture(p)) for p in self._participants]
            self._depth += 1
            try:
                yield
            except BaseException:
                if outermost:
                    _restore(snaps)
                raise
            finally:
                self._depth -= 1


WRONG = {
    "no_rollback": (NoRollback, "test_a_failure_rolls_back_every_repository"),
    "swallows": (Swallows, "test_a_failure_rolls_back_every_repository"),
    "exceptions_only": (RollsBackOnlyExceptions, "test_keyboard_interrupt_is_not_swallowed"),
    "no_savepoints": (NoSavepoints, "test_a_nested_failure_caught_outside_undoes_only"),
    "first_repo_only": (RollsBackFirstRepositoryOnly, "test_a_failure_rolls_back_every"),
    "wraps_errors": (WrapsErrors, "test_a_failure_rolls_back_every_repository"),
    "no_lock": (NoLock, "test_other_threads_wait_for_the_owner_to_finish"),
    "nested_keeps_writes": (
        NestedFailureLeavesNestedWrites,
        "test_a_nested_failure_caught_outside_undoes_only",
    ),
}


def _contract_test_names() -> list[str]:
    return sorted(n for n in vars(UnitOfWorkContract) if n.startswith("test_"))


def _failures_of(factory: type) -> set[str]:
    failed: set[str] = set()
    for name in _contract_test_names():
        sources, control = InMemorySourceRepository(), InMemoryControlRepository()
        env = Env(factory([sources, control]), sources, control)
        try:
            getattr(UnitOfWorkContract(), name)(env)
        except BaseException:  # noqa: BLE001 - any failure, including a leaked BaseException
            failed.add(name)
    return failed


def test_the_reference_fake_passes_every_contract_test():
    assert _failures_of(InMemoryUnitOfWork) == set()


@pytest.mark.parametrize(("label", "factory", "expected"), [(k, *v) for k, v in WRONG.items()])
def test_each_wrong_unit_of_work_fails_the_test_that_targets_it(label, factory, expected):
    failed = _failures_of(factory)
    assert any(name.startswith(expected) for name in failed), (label, sorted(failed))
