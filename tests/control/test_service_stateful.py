"""Stateful property test over the registry AND the control service together.

Whatever sequence of operations runs, after every step:
- the audit log tells the truth: replaying it reproduces the persisted settings, flags and
  collection states (deleted sources included), every event continues the chain of the previous
  one, and the log has exactly one event per change the services reported;
- the single-source-of-truth invariant holds: `source.enabled` <=> collection state is not
  `disabled` (a missing state row is `disabled`), and a disabled source is never effectively
  analysed.
"""

import shutil
import tempfile
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from social_tracker.control.domain import InvalidTransition
from social_tracker.control.service import ControlService
from social_tracker.persistence.db import Database
from social_tracker.shared.errors import ConflictError, NotFoundError
from social_tracker.sources.domain import parse_source
from social_tracker.sources.service import SourceRegistry
from social_tracker.sources.yaml_io import dump_sources
from tests.actors import WHO
from tests.control import test_service as service_suites
from tests.control.service_contract import (
    OPERATIONS,
    Env,
    assert_chains_are_consistent,
    assert_enabled_matches_state,
    assert_log_reproduces_state,
)
from tests.sources.factories import payload
from tests.sources.registry_contract import data

IDS = ["src_a", "src_b", "src_c"]
THRESHOLDS = st.sampled_from([0.0, 0.1, 0.5, 0.5000000001, 0.9, 1.0]) | st.floats(0.0, 1.0)


class Wrapped:
    """Forwards to a control repository except for the named overrides (used by the proofs)."""

    def __init__(self, inner: Any, **overrides: Callable[..., Any]) -> None:
        self._inner, self._overrides = inner, overrides

    def __getattr__(self, name: str) -> Any:
        return self._overrides.get(name) or getattr(self._inner, name)


class ControlMachine(RuleBasedStateMachine):
    open_env: Callable[[], tuple[Env, Callable[[], None]]]

    def wrap_control(self, control: Any) -> Any:
        return control

    def __init__(self) -> None:
        super().__init__()
        self.env, self._close = self.open_env()
        env, control = self.env, self.wrap_control(self.env.control)
        self.registry = SourceRegistry(env.sources, control, env.uow, env.clock)
        self.service = ControlService(control, env.sources, env.uow, env.clock)
        env.service = self.service
        self.known: set[str] = set()
        self.changes = 0

    def teardown(self) -> None:
        self._close()

    def _count(self, result: Any) -> None:
        self.changes += int(result.changed)

    # ---- registry operations ----

    @rule(source_id=st.sampled_from(IDS), enabled=st.booleans())
    def add_source(self, source_id: str, enabled: bool) -> None:
        item = data(source_id, f"Name {source_id}", enabled=enabled)
        if source_id in self.known:
            with pytest.raises(ConflictError):
                self.registry.add(item, **WHO)
            return
        self.registry.add(item, **WHO)
        self.known.add(source_id)
        self.changes += 1

    def _flip(self, method: str, source_id: str) -> None:
        if source_id not in self.known:
            with pytest.raises(NotFoundError):
                getattr(self.registry, method)(source_id, **WHO)
            return
        self._count(getattr(self.registry, method)(source_id, **WHO))

    @rule(source_id=st.sampled_from(IDS))
    def enable_source(self, source_id: str) -> None:
        self._flip("enable", source_id)

    @rule(source_id=st.sampled_from(IDS))
    def disable_source(self, source_id: str) -> None:
        self._flip("disable", source_id)

    @rule(source_id=st.sampled_from(IDS))
    def delete_source(self, source_id: str) -> None:
        if source_id not in self.known:
            with pytest.raises(NotFoundError):
                self.registry.delete(source_id, **WHO)
            return
        self.registry.delete(source_id, **WHO)
        self.known.discard(source_id)
        self.changes += 1

    @rule(source_id=st.sampled_from(IDS), credibility=st.integers(1, 5))
    def edit_source(self, source_id: str, credibility: int) -> None:
        if source_id not in self.known:
            with pytest.raises(NotFoundError):
                self.registry.edit(source_id, {"credibility": credibility}, **WHO)
            return
        self._count(self.registry.edit(source_id, {"credibility": credibility}, **WHO))

    @rule(
        entries=st.lists(
            st.tuples(st.sampled_from(IDS), st.booleans()), max_size=3, unique_by=lambda e: e[0]
        )
    )
    def import_sources(self, entries: list[tuple[str, bool]]) -> None:
        sources = [
            parse_source(payload(id=sid, name=f"Name {sid}", enabled=on)) for sid, on in entries
        ]
        report = self.registry.import_sources(dump_sources(sources), **WHO)
        self.known.update(report.added)
        written = len(report.added) + len(report.updated)
        self.changes += written + (1 if written else 0)  # one event per entry + the summary

    # ---- control operations ----

    @rule(on=st.booleans())
    def set_global(self, on: bool) -> None:
        self._count(self.service.set_global_analysis(on, **WHO))

    @rule(source_id=st.sampled_from(IDS), on=st.booleans())
    def set_source_flag(self, source_id: str, on: bool) -> None:
        if source_id not in self.known:
            with pytest.raises(NotFoundError):
                self.service.set_source_analysis(source_id, on, **WHO)
            return
        self._count(self.service.set_source_analysis(source_id, on, **WHO))

    @rule(
        source_id=st.sampled_from(IDS),
        operation=st.sampled_from(OPERATIONS),
        jitter=st.sampled_from([0.0, 7.5]),
    )
    def collection(self, source_id: str, operation: tuple, jitter: float) -> None:
        method, _ = operation
        kwargs = {"jitter_s": jitter} if method == "fail_collection" else {}
        try:
            result = getattr(self.service, method)(source_id, **WHO, **kwargs)
        except (InvalidTransition, NotFoundError):
            return
        self._count(result)

    @rule(minutes=st.integers(0, 90))
    def pass_time(self, minutes: int) -> None:
        self.env.clock.advance(timedelta(minutes=minutes))

    @rule(value=THRESHOLDS)
    def set_threshold(self, value: float) -> None:
        self._count(self.service.set_alert_min_confidence(value, **WHO))

    # ---- invariants ----

    @invariant()
    def the_log_tells_the_truth(self) -> None:
        assert_log_reproduces_state(self.env)
        assert_chains_are_consistent(self.env)
        assert len(self.env.control.list_control_events()) == self.changes

    @invariant()
    def enabled_flag_and_collection_state_agree(self) -> None:
        assert_enabled_matches_state(self.env)

    def check_everything(self) -> None:
        self.the_log_tells_the_truth()
        self.enabled_flag_and_collection_state_agree()


def _in_memory() -> tuple[Env, Callable[[], None]]:
    return service_suites.TestInMemoryControlService().make_env(), lambda: None


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


# ---- the invariants must be able to fail ------------------------------------------------
# Deterministic scripts (not random search) against a control repository that misbehaves in one
# precise way, so this proof can never be flaky.


def _machine(**broken: Callable[..., Any]) -> ControlMachine:
    class Scripted(ControlMachine):
        open_env = staticmethod(_in_memory)

        def wrap_control(self, control: Any) -> Any:
            return Wrapped(control, **{name: fn(control) for name, fn in broken.items()})

    return Scripted()


def _drop(_: Any) -> Callable[..., None]:
    return lambda *args, **kwargs: None


def _tamper_from(control: Any) -> Callable[..., None]:
    def append(event: Any) -> None:
        from dataclasses import replace  # noqa: PLC0415

        if '"from":' in event.detail:
            event = replace(event, detail=event.detail.replace('"from":"', '"from":"x'))
        control.append_control_event(event)

    return append


def _script_add_enabled(machine: ControlMachine) -> None:
    machine.add_source("src_a", True)


def _script_enable_then_delete(machine: ControlMachine) -> None:
    machine.add_source("src_a", True)
    machine.delete_source("src_a")


def _script_global_on(machine: ControlMachine) -> None:
    machine.set_global(True)


BROKEN = [
    pytest.param(
        {"append_control_event": _drop},
        _script_global_on,
        "the_log_tells_the_truth",
        id="no-events",
    ),
    pytest.param(
        {"save_source_state": _drop},
        _script_add_enabled,
        "enabled_flag_and_collection_state_agree",
        id="state-writes-ignored",
    ),
    pytest.param(
        {"delete_source_state": _drop},
        _script_enable_then_delete,
        "the_log_tells_the_truth",
        id="state-kept-on-delete",
    ),
    pytest.param(
        {"append_control_event": _tamper_from},
        _script_global_on,
        "the_log_tells_the_truth",
        id="tampered-chain",
    ),
]


@pytest.mark.parametrize(("broken", "script", "check"), BROKEN)
def test_the_invariants_fail_for_a_store_that_misbehaves(broken, script, check):
    machine = _machine(**broken)
    script(machine)
    with pytest.raises(AssertionError):
        getattr(machine, check)()


@pytest.mark.parametrize("script", [p.values[1] for p in BROKEN], ids=lambda f: f.__name__)
def test_the_same_scripts_pass_the_invariants_on_the_real_stack(script):
    machine = _machine()
    script(machine)
    machine.check_everything()
