"""Stateful property test: whatever sequence of operations runs, the audit log stays truthful.

After every step: replaying the event log reproduces the persisted settings, flags and states,
every event continues the chain of the previous one, and the log has exactly one event per
change the service reported (no-ops and rejected requests wrote nothing).
"""

import shutil
import tempfile
from collections.abc import Callable
from datetime import timedelta

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from social_tracker.control.domain import InvalidTransition
from social_tracker.control.service import ControlService
from social_tracker.persistence.db import Database
from tests.control import test_service as service_suites
from tests.control.service_contract import (
    OPERATIONS,
    WHO,
    Env,
    assert_chains_are_consistent,
    assert_log_reproduces_state,
)
from tests.control.wrong_services import (
    AuditsTheNewStateAsFrom,
    GlobalAlwaysAudits,
    StateThenEventOutsideTransaction,
    ThresholdAlwaysAudits,
)

IDS = ["src_a", "src_b", "src_c"]
THRESHOLDS = st.sampled_from([0.0, 0.1, 0.5, 0.5000000001, 0.9, 1.0]) | st.floats(0.0, 1.0)


class ControlMachine(RuleBasedStateMachine):
    open_env: Callable[[], tuple[Env, Callable[[], None]]]

    def __init__(self) -> None:
        super().__init__()
        self.env, self._close = self.open_env()
        for source_id in IDS:
            self.env.add_source(source_id)
        self.changes = 0

    def teardown(self) -> None:
        self._close()

    def _count(self, result) -> None:
        self.changes += int(result.changed)

    @rule(on=st.booleans())
    def set_global(self, on: bool) -> None:
        self._count(self.env.service.set_global_analysis(on, **WHO))

    @rule(source_id=st.sampled_from(IDS), on=st.booleans())
    def set_source_flag(self, source_id: str, on: bool) -> None:
        self._count(self.env.service.set_source_analysis(source_id, on, **WHO))

    @rule(
        source_id=st.sampled_from(IDS),
        operation=st.sampled_from(OPERATIONS),
        jitter=st.sampled_from([0.0, 7.5]),
    )
    def collection(self, source_id: str, operation: tuple, jitter: float) -> None:
        method, _ = operation
        kwargs = {"jitter_s": jitter} if method == "fail_collection" else {}
        try:
            result = getattr(self.env.service, method)(source_id, **WHO, **kwargs)
        except InvalidTransition:
            return
        self._count(result)

    @rule(minutes=st.integers(0, 90))
    def pass_time(self, minutes: int) -> None:
        self.env.clock.advance(timedelta(minutes=minutes))

    @rule(value=THRESHOLDS)
    def set_threshold(self, value: float) -> None:
        self._count(self.env.service.set_alert_min_confidence(value, **WHO))

    @invariant()
    def the_log_tells_the_truth(self) -> None:
        assert_log_reproduces_state(self.env)
        assert_chains_are_consistent(self.env)
        assert len(self.env.control.list_control_events()) == self.changes


def _in_memory(
    service_cls: type[ControlService] = ControlService,
) -> tuple[Env, Callable[[], None]]:
    return service_suites.TestInMemoryControlService().make_env(service_cls), lambda: None


def _sqlite() -> tuple[Env, Callable[[], None]]:
    root = tempfile.mkdtemp(prefix="stateful-")
    db = Database(f"{root}/state.db")

    def close() -> None:
        db.close()
        shutil.rmtree(root, ignore_errors=True)

    return service_suites.sqlite_env(db), close


class InMemoryMachine(ControlMachine):
    open_env = staticmethod(_in_memory)


class SqliteMachine(ControlMachine):
    open_env = staticmethod(_sqlite)


TestInMemoryStateful = InMemoryMachine.TestCase
TestInMemoryStateful.settings = settings(
    max_examples=120,
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
TestSqliteStateful = SqliteMachine.TestCase
TestSqliteStateful.settings = settings(
    max_examples=25,
    stateful_step_count=25,
    deadline=None,
    suppress_health_check=list(HealthCheck),
)


# ---- the invariant itself must be able to fail ------------------------------------------
# Deterministic scripts (not random search) so this proof can never be flaky.

ENABLE, START, _FINISH, FAIL = OPERATIONS[0], OPERATIONS[2], OPERATIONS[3], OPERATIONS[4]


def _machine(service_cls: type[ControlService]) -> ControlMachine:
    class Scripted(ControlMachine):
        open_env = staticmethod(lambda: _in_memory(service_cls))

    return Scripted()


def _double_global(machine: ControlMachine) -> None:
    machine.set_global(True)
    machine.set_global(True)


def _repeat_default_threshold(machine: ControlMachine) -> None:
    machine.set_threshold(0.5)


def _one_global_toggle(machine: ControlMachine) -> None:
    machine.set_global(True)


def _fail_a_collection(machine: ControlMachine) -> None:
    machine.collection("src_a", ENABLE, 0.0)
    machine.collection("src_a", START, 0.0)
    machine.collection("src_a", FAIL, 0.0)


@pytest.mark.parametrize(
    ("wrong", "script"),
    [
        (GlobalAlwaysAudits, _double_global),
        (ThresholdAlwaysAudits, _repeat_default_threshold),
        (AuditsTheNewStateAsFrom, _one_global_toggle),
        (StateThenEventOutsideTransaction, _fail_a_collection),
    ],
)
def test_the_invariant_fails_for_services_that_corrupt_the_log(wrong, script):
    machine = _machine(wrong)
    script(machine)
    with pytest.raises(AssertionError):
        machine.the_log_tells_the_truth()


def test_the_invariant_holds_for_the_same_scripts_on_the_real_service():
    for script in (
        _double_global,
        _repeat_default_threshold,
        _one_global_toggle,
        _fail_a_collection,
    ):
        machine = _machine(ControlService)
        script(machine)
        machine.the_log_tells_the_truth()
