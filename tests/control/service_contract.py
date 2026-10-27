"""Behavioural contract of `ControlService`, run on in-memory and SQLite repositories.

Every test takes an `Env` so the same scenarios can be replayed against deliberately wrong
services (see the meta-test). Tests never look inside the service: they read the repositories.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

import pytest

from social_tracker.control.domain import (
    DEFAULT_ALERT_MIN_CONFIDENCE,
    MAX_ACTOR,
    MAX_REASON,
    SETTING_ALERT_MIN_CONFIDENCE,
    SETTING_GLOBAL_ANALYSIS_ENABLED,
    CollectionState,
    Event,
    InvalidTransition,
    SourceState,
    apply_event,
    source_analysis_key,
)
from social_tracker.control.ports import ControlRepository
from social_tracker.control.service import (
    COLLECTION_EVENT_KINDS,
    EVENT_ALERT_THRESHOLD,
    EVENT_GLOBAL_ANALYSIS,
    EVENT_SOURCE_ANALYSIS,
    CollectionChange,
    ControlService,
    FlagChange,
    SourceOverview,
    ThresholdChange,
)
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import NotFoundError, StorageError, ValidationError
from social_tracker.shared.uow import UnitOfWork
from social_tracker.sources.domain import parse_source
from social_tracker.sources.ports import SourceRepository
from social_tracker.sources.service import (
    SOURCE_ADDED,
    SOURCE_DELETED,
    SOURCE_DISABLED,
    SOURCE_EDITED,
    SOURCE_ENABLED,
)
from tests.actors import ACTOR, WHO, WHY
from tests.fakes.flaky_control import FlakyControl
from tests.sources.factories import CANARY, payload

START = datetime(2026, 7, 1, 8, 0, tzinfo=UTC)
MINUTE = timedelta(minutes=1)
COLLECTION_KINDS = frozenset(COLLECTION_EVENT_KINDS.values())
REGISTRY_STATE_KINDS = frozenset({SOURCE_ADDED, SOURCE_EDITED, SOURCE_ENABLED, SOURCE_DISABLED})

# (method name, domain event) for every collection operation the service exposes. Enabling and
# disabling a source belongs to the registry, so only the worker transitions are listed.
OPERATIONS = [
    ("start_collection", Event.START),
    ("finish_collection", Event.SUCCEED),
    ("fail_collection", Event.FAIL),
    ("retry_collection", Event.RETRY),
    ("reset_failed", Event.RESET),
]

STATES: dict[str, SourceState | None] = {
    "none": None,
    "disabled": SourceState("src_one", CollectionState.DISABLED),
    "idle": SourceState("src_one", CollectionState.IDLE),
    "collecting": SourceState("src_one", CollectionState.COLLECTING, 2),
    "backoff_future": SourceState("src_one", CollectionState.BACKOFF, 2, START + 10 * MINUTE),
    "backoff_due": SourceState("src_one", CollectionState.BACKOFF, 2, START - MINUTE),
    "failed": SourceState("src_one", CollectionState.FAILED, 5),
}


class OnlyExists:
    """A source repository that answers `exists` and fails the test if anything loads a source."""

    def __init__(self, inner: SourceRepository) -> None:
        self._inner = inner

    def exists(self, source_id: str) -> bool:
        return self._inner.exists(source_id)

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"a whole source must not be loaded just to check it exists: {name}")


class NoPerSourceLookups:
    """A control repository that fails the test on any per-source query: an overview must read
    the states and the flags in bulk, never once per source (N+1)."""

    def __init__(self, inner: ControlRepository) -> None:
        self._inner = inner

    def get_source_state(self, source_id: str) -> SourceState | None:
        raise AssertionError("overview must not query one source state at a time")

    def get_setting(self, key: str) -> str | None:
        assert not key.startswith("source_analysis."), "overview must not query one flag at a time"
        return self._inner.get_setting(key)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class Env:
    service: ControlService
    control: ControlRepository
    sources: SourceRepository
    uow: UnitOfWork
    clock: FixedClock
    service_cls: type[ControlService]
    seeded: dict[str, tuple[str, int, str | None]] = field(default_factory=dict)

    def service_with(self, *, control: Any = None, clock: Any = None) -> ControlService:
        return self.service_cls(
            control or self.control, self.sources, self.uow, clock or self.clock
        )

    def add_source(self, source_id: str = "src_one") -> None:
        """A registered, DISABLED source (the registry's default): no state row."""
        self.sources.add(parse_source(payload(id=source_id, name=f"Name {source_id}")))

    def enable_source(self, source_id: str = "src_one") -> None:
        """What `SourceRegistry.enable` leaves behind: the flag on and an idle state row.

        Written straight into the stores (no event) so control-service tests can start from a
        running source without going through the registry; `seeded` tells the log replay about it.
        """
        source = self.sources.get(source_id)
        self.sources.update(source.with_changes({"enabled": True}))
        self.control.save_source_state(SourceState(source_id, CollectionState.IDLE))
        self.seeded[source_id] = ("idle", 0, None)

    def events(self, source_id: str | None = None) -> list[tuple[str, str | None, dict[str, Any]]]:
        return [
            (e.kind, e.source_id, json.loads(e.detail) if e.detail else {})
            for e in self.control.list_control_events(source_id)
        ]

    def kinds(self) -> list[str]:
        return [kind for kind, _, _ in self.events()]


# ---- replaying the event log ---------------------------------------------------------
#
# The rule (also stated in the ControlService module docstring): replaying the log in order
# reproduces the persisted settings, per-source flags and collection states of every source that
# EXISTS. Registry events that move the state (`state_to` in their detail) are part of the log;
# `source_deleted` removes the state row and the flag row without a collection event, and carries
# the previous `state` and `analysis` value so the replay can follow it.

DISABLED_ROW = ("disabled", 0, None)


@dataclass
class Replayed:
    global_active: bool = False
    threshold: float = DEFAULT_ALERT_MIN_CONFIDENCE
    source_flags: dict[str, bool] = None  # type: ignore[assignment]
    states: dict[str, tuple[str, int, str | None]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.source_flags, self.states = {}, {}


def replay(env: Env) -> Replayed:
    result = Replayed()
    result.states.update(env.seeded)
    for kind, source_id, detail in env.events():
        if kind == EVENT_GLOBAL_ANALYSIS:
            result.global_active = detail["to"] == "active"
        elif kind == EVENT_SOURCE_ANALYSIS:
            result.source_flags[source_id] = detail["to"] == "on"
        elif kind == EVENT_ALERT_THRESHOLD:
            result.threshold = float(detail["to"])
        elif kind in COLLECTION_KINDS:
            result.states[source_id] = (detail["to"], detail["failures"], detail["next_run_at"])
        elif kind == SOURCE_DELETED:
            result.states.pop(source_id, None)
            result.source_flags.pop(source_id, None)
        elif kind in REGISTRY_STATE_KINDS and "state_to" in detail:
            # enable/disable/add/import moves are to idle or disabled: no failures, no backoff
            result.states[source_id] = (detail["state_to"], 0, None)
    return result


def mentioned_source_ids(env: Env) -> set[str]:
    ids = {e.source_id for e in env.control.list_control_events() if e.source_id}
    return ids | set(env.seeded) | {s.id for s in env.sources.list()}


def assert_log_reproduces_state(env: Env) -> None:
    """Replaying the audit log gives the persisted state of every source, deleted ones included
    (a deleted source has no state row and no flag row left)."""
    replayed = replay(env)
    assert (env.control.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "true") == (
        replayed.global_active
    )
    assert float(env.control.get_setting(SETTING_ALERT_MIN_CONFIDENCE)) == replayed.threshold
    for source_id in sorted(mentioned_source_ids(env)):
        flag = env.control.get_setting(source_analysis_key(source_id))
        assert (flag == "true") == replayed.source_flags.get(source_id, False), source_id
        stored = env.control.get_source_state(source_id)
        state, failures, next_run_at = replayed.states.get(source_id, DISABLED_ROW)
        if stored is None:
            assert (state, failures, next_run_at) == DISABLED_ROW, source_id
            continue
        assert (stored.state.value, stored.consecutive_failures) == (state, failures), source_id
        assert (stored.next_run_at.isoformat() if stored.next_run_at else None) == next_run_at


def assert_enabled_matches_state(env: Env) -> None:
    """The single-source-of-truth invariant: `source.enabled` <=> collection state != disabled
    (a missing state row counts as disabled), and effective analysis never holds for a
    disabled source."""
    for source in env.sources.list():
        row = env.control.get_source_state(source.id)
        state = row.state if row else CollectionState.DISABLED
        assert source.enabled == (state is not CollectionState.DISABLED), (source.id, state)
        if not source.enabled:
            assert env.service.effective_analysis(source.id) is False, source.id


def assert_chains_are_consistent(env: Env) -> None:
    """Each event's `from` is the previous event's `to` for the same target."""
    last: dict[tuple[str, str | None], str] = {
        ("collection", sid): row[0] for sid, row in env.seeded.items()
    }
    first = {
        EVENT_GLOBAL_ANALYSIS: "paused",
        EVENT_SOURCE_ANALYSIS: "off",
        EVENT_ALERT_THRESHOLD: str(DEFAULT_ALERT_MIN_CONFIDENCE),
    }
    for kind, source_id, detail in env.events():
        if kind == SOURCE_DELETED:
            collection = ("collection", source_id)
            assert (detail["state"] or "disabled") == last.get(collection, "disabled"), detail
            flag_key = (EVENT_SOURCE_ANALYSIS, source_id)
            was = {"true": "on", "false": "off", None: "off"}[detail["analysis"]]
            assert was == last.get(flag_key, "off"), detail
            last[collection] = "disabled"
            last[flag_key] = "off"
            continue
        if kind in REGISTRY_STATE_KINDS:
            if "state_to" in detail:
                key = ("collection", source_id)
                assert detail["state_from"] == last.get(key, "disabled"), (kind, detail)
                assert detail["state_from"] != detail["state_to"]
                last[key] = detail["state_to"]
            continue
        if kind in COLLECTION_KINDS:
            key, initial = ("collection", source_id), "disabled"
        elif kind in first:
            key, initial = (kind, source_id), first[kind]
        else:
            continue
        assert detail["from"] == last.get(key, initial), (kind, source_id, detail)
        assert detail["from"] != detail["to"] or kind in COLLECTION_KINDS
        last[key] = detail["to"]


def run_all(calls: list[Any]) -> list[tuple[str, Any]]:
    from tests.sources.registry_contract import race  # noqa: PLC0415

    return race(calls)


class ControlServiceContract:
    def make_env(self, service_cls: type[ControlService] = ControlService) -> Env:
        raise NotImplementedError

    @pytest.fixture
    def env(self) -> Env:
        env = self.make_env()
        env.add_source("src_one")
        return env

    def put(self, env: Env, name: str) -> None:
        """Seed the named state. A state other than disabled belongs to an ENABLED source (the
        invariant the registry keeps), so the source is enabled too."""
        state = STATES[name]
        if state is None:
            return
        if state.state is not CollectionState.DISABLED:
            source = env.sources.get(state.source_id)
            env.sources.update(source.with_changes({"enabled": True}))
        env.control.save_source_state(state)
        next_run = state.next_run_at.isoformat() if state.next_run_at else None
        env.seeded[state.source_id] = (state.state.value, state.consecutive_failures, next_run)

    # ---- fresh start ----

    def test_a_fresh_system_is_paused_with_analysis_off_and_the_default_threshold(self, env: Env):
        assert env.service.global_analysis_enabled() is False
        assert env.control.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"
        assert env.service.source_analysis_enabled("src_one") is False
        assert env.service.effective_analysis("src_one") is False
        assert env.service.alert_min_confidence() == 0.5
        assert env.service.get_state("src_one") == SourceState("src_one", CollectionState.DISABLED)
        assert env.control.list_control_events() == []

    # ---- global switch ----

    def test_turning_the_global_switch_on_writes_the_setting_and_one_exact_event(self, env: Env):
        result = env.service.set_global_analysis(True, **WHO)
        assert result == FlagChange(enabled=True, changed=True)
        assert env.service.global_analysis_enabled() is True
        assert env.control.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "true"
        stored = env.control.list_control_events()
        assert len(stored) == 1
        assert (stored[0].kind, stored[0].source_id, stored[0].at) == (
            EVENT_GLOBAL_ANALYSIS,
            None,
            START,
        )
        assert json.loads(stored[0].detail) == {
            "action": "set",
            "actor": ACTOR,
            "reason": WHY,
            "from": "paused",
            "to": "active",
        }

    def test_turning_it_off_again_records_the_reverse_transition(self, env: Env):
        env.service.set_global_analysis(True, **WHO)
        env.clock.advance(MINUTE)
        result = env.service.set_global_analysis(False, actor="ana", reason="done for today")
        assert result == FlagChange(enabled=False, changed=True)
        assert env.control.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"
        kind, _, detail = env.events()[-1]
        assert kind == EVENT_GLOBAL_ANALYSIS
        assert detail == {
            "action": "set",
            "actor": "ana",
            "reason": "done for today",
            "from": "active",
            "to": "paused",
        }
        assert env.control.list_control_events()[-1].at == START + MINUTE

    @pytest.mark.parametrize("on", [False, True])
    def test_a_redundant_request_is_a_noop_without_an_event(self, env: Env, on: bool):
        if on:
            env.service.set_global_analysis(True, **WHO)
        before = env.control.list_control_events()
        result = env.service.set_global_analysis(on, **WHO)
        assert result == FlagChange(enabled=on, changed=False)
        assert env.control.list_control_events() == before

    @pytest.mark.parametrize("bad", [1, 0, "true", "false", None, [], 1.0])
    def test_the_global_switch_takes_a_real_bool(self, env: Env, bad: Any):
        with pytest.raises(ValidationError) as caught:
            env.service.set_global_analysis(bad, **WHO)
        assert caught.value.fields == ("on",)
        assert env.control.list_control_events() == []

    def test_one_hundred_alternating_toggles_end_consistent_with_one_event_each(self, env: Env):
        for i in range(100):
            env.clock.advance(timedelta(seconds=1))
            env.service.set_global_analysis(i % 2 == 0, **WHO)
        assert env.kinds() == [EVENT_GLOBAL_ANALYSIS] * 100
        assert env.service.global_analysis_enabled() is False
        assert_log_reproduces_state(env)
        assert_chains_are_consistent(env)

    def test_one_hundred_identical_requests_leave_a_single_event(self, env: Env):
        for _ in range(100):
            env.service.set_global_analysis(True, **WHO)
        assert env.kinds() == [EVENT_GLOBAL_ANALYSIS]

    def test_a_failing_event_append_leaves_the_setting_unchanged(self, env: Env):
        flaky = FlakyControl(env.control, "append_control_event")
        with pytest.raises(StorageError):
            env.service_with(control=flaky).set_global_analysis(True, **WHO)
        assert env.control.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"
        assert env.control.list_control_events() == []

    def test_a_failing_setting_write_leaves_no_event(self, env: Env):
        flaky = FlakyControl(env.control, "set_setting")
        with pytest.raises(StorageError):
            env.service_with(control=flaky).set_global_analysis(True, **WHO)
        assert env.control.get_setting(SETTING_GLOBAL_ANALYSIS_ENABLED) == "false"
        assert env.control.list_control_events() == []

    def test_concurrent_identical_requests_produce_exactly_one_change(self, env: Env):
        for _ in range(6):
            fresh = self.make_env()
            fresh.add_source("src_one")
            results = run_all([partial(fresh.service.set_global_analysis, True, **WHO)] * 4)
            assert [kind for kind, _ in results] == ["ok"] * 4, results
            assert sorted(value.changed for _, value in results) == [False, False, False, True]
            assert fresh.kinds() == [EVENT_GLOBAL_ANALYSIS]

    # ---- who and why ----

    @pytest.mark.parametrize(
        ("field", "bad"),
        [
            ("actor", ""),
            ("actor", "   "),
            ("actor", None),
            ("actor", 5),
            ("actor", "x" * (MAX_ACTOR + 1)),
            ("actor", "a\x00b"),
            ("actor", "line\nbreak"),
            ("reason", ""),
            ("reason", " \t "),
            ("reason", None),
            ("reason", "x" * (MAX_REASON + 1)),
            ("reason", "bell\x07"),
            ("reason", "tab\there"),
            ("actor", "\ud800x"),  # a lone surrogate would crash a UTF-8 store
            ("actor", "​"),
            ("actor", "a‮b"),
            ("reason", " "),
            ("reason", "ok "),
            ("reason", "\udfff"),
        ],
    )
    def test_actor_and_reason_are_validated_for_every_kind_of_action(
        self, env: Env, field: str, bad: Any
    ):
        who = {**WHO, field: bad}
        calls = [
            partial(env.service.set_global_analysis, True, **who),
            partial(env.service.set_source_analysis, "src_one", True, **who),
            partial(env.service.start_collection, "src_one", **who),
            partial(env.service.set_alert_min_confidence, 0.9, **who),
        ]
        for call in calls:
            with pytest.raises(ValidationError) as caught:
                call()
            assert caught.value.fields == (field,)
        assert env.control.list_control_events() == []
        assert env.control.get_source_state("src_one") is None

    def test_a_redundant_request_with_an_invalid_reason_is_still_rejected(self, env: Env):
        with pytest.raises(ValidationError):
            env.service.set_global_analysis(False, actor=ACTOR, reason="")

    def test_actor_and_reason_have_to_be_given(self, env: Env):
        with pytest.raises(TypeError):
            env.service.set_global_analysis(True, reason=WHY)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            env.service.set_global_analysis(True, actor=ACTOR)  # type: ignore[call-arg]

    def test_the_longest_allowed_actor_and_reason_fit_in_an_event_even_when_escaped(self, env: Env):
        reason = '"' * MAX_REASON  # JSON escapes every quote: the worst case for the detail size
        actor = "\\" * MAX_ACTOR
        env.service.set_global_analysis(True, actor=actor, reason=reason)
        _, _, detail = env.events()[0]
        assert (detail["actor"], detail["reason"]) == (actor, reason)

    def test_unicode_actor_and_reason_are_stored_verbatim(self, env: Env):
        env.service.set_global_analysis(True, actor="José ✓", reason="rápido 日本")
        _, _, detail = env.events()[0]
        assert (detail["actor"], detail["reason"]) == ("José ✓", "rápido 日本")

    def test_errors_never_echo_the_actor_or_reason(self, env: Env):
        with pytest.raises(ValidationError) as caught:
            env.service.set_global_analysis(True, actor=ACTOR, reason=CANARY + "\x00")
        assert CANARY not in caught.value.message

    # ---- per-source analysis ----

    def test_the_source_flag_is_stored_per_source_with_an_exact_event(self, env: Env):
        env.add_source("src_two")
        result = env.service.set_source_analysis("src_one", True, **WHO)
        assert result == FlagChange(enabled=True, changed=True)
        assert env.service.source_analysis_enabled("src_one") is True
        assert env.service.source_analysis_enabled("src_two") is False
        assert env.control.get_setting(source_analysis_key("src_one")) == "true"
        assert env.events() == [
            (
                EVENT_SOURCE_ANALYSIS,
                "src_one",
                {"action": "set", "actor": ACTOR, "reason": WHY, "from": "off", "to": "on"},
            )
        ]

    def test_a_redundant_source_flag_request_is_a_noop(self, env: Env):
        assert env.service.set_source_analysis("src_one", False, **WHO).changed is False
        env.service.set_source_analysis("src_one", True, **WHO)
        assert env.service.set_source_analysis("src_one", True, **WHO).changed is False
        assert env.kinds() == [EVENT_SOURCE_ANALYSIS]
        env.service.set_source_analysis("src_one", False, **WHO)
        assert env.events()[-1][2]["from"] == "on"
        assert env.events()[-1][2]["to"] == "off"

    @pytest.mark.parametrize("operation", ["flag", "enabled", "effective", "state", "collect"])
    def test_unknown_sources_are_not_found_and_nothing_is_written(self, env: Env, operation: str):
        calls = {
            "flag": lambda: env.service.set_source_analysis("src_nope", True, **WHO),
            "enabled": lambda: env.service.source_analysis_enabled("src_nope"),
            "effective": lambda: env.service.effective_analysis("src_nope"),
            "state": lambda: env.service.get_state("src_nope"),
            "collect": lambda: env.service.start_collection("src_nope", **WHO),
        }
        with pytest.raises(NotFoundError):
            calls[operation]()
        assert env.control.list_control_events() == []
        assert env.control.get_setting(source_analysis_key("src_nope")) is None
        assert env.control.get_source_state("src_nope") is None

    @pytest.mark.parametrize("bad", ["", "ab", "UPPER", "has space", None, 5, "x" * 65])
    def test_source_ids_that_are_not_slugs_are_rejected_with_the_field_name(
        self, env: Env, bad: Any
    ):
        for call in (
            partial(env.service.set_source_analysis, bad, True, **WHO),
            partial(env.service.effective_analysis, bad),
            partial(env.service.start_collection, bad, **WHO),
            partial(env.service.get_state, bad),
        ):
            with pytest.raises(ValidationError) as caught:
                call()
            assert caught.value.fields == ("source_id",)

    def corrupt_setting(self, env: Env, key: str, value: str) -> None:
        """Write `value` behind the repository's back (a tampered file or a buggy older version)."""
        raise NotImplementedError

    def test_a_stored_flag_that_is_neither_true_nor_false_is_a_storage_error(self, env: Env):
        self.corrupt_setting(env, source_analysis_key("src_one"), "maybe")
        with pytest.raises(StorageError):
            env.service.source_analysis_enabled("src_one")
        with pytest.raises(StorageError):
            env.service.effective_analysis("src_one")
        with pytest.raises(StorageError):
            env.service.set_source_analysis("src_one", True, **WHO)

    @pytest.mark.parametrize("bad", [1, "true", None])
    def test_the_source_flag_takes_a_real_bool(self, env: Env, bad: Any):
        with pytest.raises(ValidationError) as caught:
            env.service.set_source_analysis("src_one", bad, **WHO)
        assert caught.value.fields == ("on",)

    @pytest.mark.parametrize(
        ("global_on", "source_on", "expected"),
        [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
    )
    def test_effective_analysis_needs_the_global_switch_and_the_source_flag(
        self, env: Env, global_on: bool, source_on: bool, expected: bool
    ):
        env.add_source("src_two")
        env.enable_source("src_one")
        env.service.set_global_analysis(global_on, **WHO)
        env.service.set_source_analysis("src_one", source_on, **WHO)
        before = env.control.list_control_events()
        assert env.service.effective_analysis("src_one") is expected
        assert env.service.effective_analysis("src_two") is False  # its own flag is off
        assert env.control.list_control_events() == before

    def test_pausing_globally_keeps_the_source_flags_for_when_it_resumes(self, env: Env):
        env.enable_source("src_one")
        env.service.set_source_analysis("src_one", True, **WHO)
        env.service.set_global_analysis(True, **WHO)
        env.service.set_global_analysis(False, **WHO)
        assert env.service.effective_analysis("src_one") is False
        env.service.set_global_analysis(True, **WHO)
        assert env.service.effective_analysis("src_one") is True

    def test_existence_checks_do_not_load_the_whole_source(self, env: Env):
        # The flag reads and writes and `get_state` only need to know the source exists.
        service = env.service_cls(env.control, OnlyExists(env.sources), env.uow, env.clock)
        assert service.source_analysis_enabled("src_one") is False
        assert service.set_source_analysis("src_one", True, **WHO).changed is True
        assert service.get_state("src_one").state is CollectionState.DISABLED
        with pytest.raises(NotFoundError):
            service.source_analysis_enabled("src_nope")

    def test_a_failing_event_append_leaves_the_source_flag_unchanged(self, env: Env):
        flaky = FlakyControl(env.control, "append_control_event")
        with pytest.raises(StorageError):
            env.service_with(control=flaky).set_source_analysis("src_one", True, **WHO)
        assert env.service.source_analysis_enabled("src_one") is False

    # ---- collection state machine ----

    def test_the_public_api_has_no_collection_enable_or_disable(self, env: Env):
        # Enabling and disabling a source is the registry's job (single source of truth).
        assert not hasattr(env.service, "enable_collection")
        assert not hasattr(env.service, "disable_collection")

    @pytest.mark.parametrize("event", [Event.ENABLE, Event.DISABLE, "enable", "disable"], ids=str)
    @pytest.mark.parametrize("state_name", ["none", "disabled", "idle", "failed"])
    def test_enable_and_disable_events_cannot_be_applied_through_the_control_service(
        self, env: Env, event: Any, state_name: str
    ):
        self.put(env, state_name)
        stored_before = env.control.get_source_state("src_one")
        with pytest.raises(InvalidTransition):
            env.service.apply_collection_event("src_one", event, **WHO)
        assert env.control.get_source_state("src_one") == stored_before
        assert env.control.list_control_events() == []

    @pytest.mark.parametrize(("method", "domain_event"), OPERATIONS)
    @pytest.mark.parametrize("state_name", ["idle", "collecting", "backoff_due", "failed"])
    def test_worker_transitions_are_refused_on_a_disabled_source_even_with_a_stale_row(
        self, env: Env, method: str, domain_event: Event, state_name: str
    ):
        # The source is disabled (the registry's flag is the truth) but a state row says
        # otherwise, as a crash between two non-atomic writers could once have left it.
        env.control.save_source_state(STATES[state_name])
        stored_before = env.control.get_source_state("src_one")
        with pytest.raises(InvalidTransition):
            getattr(env.service, method)("src_one", **WHO)
        assert env.control.get_source_state("src_one") == stored_before
        assert env.control.list_control_events() == []

    @pytest.mark.parametrize(
        ("global_on", "flag_on", "source_enabled", "expected"),
        [
            (True, True, True, True),
            (True, True, False, False),
            (True, False, True, False),
            (False, True, True, False),
            (False, False, False, False),
        ],
    )
    def test_effective_analysis_also_needs_the_source_to_be_enabled(
        self, env: Env, global_on: bool, flag_on: bool, source_enabled: bool, expected: bool
    ):
        if source_enabled:
            env.enable_source("src_one")
        env.service.set_global_analysis(global_on, **WHO)
        env.service.set_source_analysis("src_one", flag_on, **WHO)
        assert env.service.effective_analysis("src_one") is expected

    @pytest.mark.parametrize("state_name", list(STATES))
    @pytest.mark.parametrize(("method", "domain_event"), OPERATIONS)
    def test_every_operation_in_every_state_follows_the_state_machine(
        self, env: Env, state_name: str, method: str, domain_event: Event
    ):
        self.put(env, state_name)
        current = STATES[state_name] or SourceState("src_one", CollectionState.DISABLED)
        stored_before = env.control.get_source_state("src_one")
        operation = getattr(env.service, method)
        try:
            expected = apply_event(current, domain_event, START)
        except InvalidTransition:
            with pytest.raises(InvalidTransition):
                operation("src_one", **WHO)
            assert env.control.get_source_state("src_one") == stored_before
            assert env.control.list_control_events() == []
            return
        result = operation("src_one", **WHO)
        assert isinstance(result, CollectionChange)
        assert result.state == expected.state
        assert result.changed is expected.changed
        if not expected.changed:
            assert env.control.get_source_state("src_one") == stored_before
            assert env.control.list_control_events() == []
            return
        assert env.control.get_source_state("src_one") == expected.state
        assert env.events() == [
            (
                COLLECTION_EVENT_KINDS[domain_event],
                "src_one",
                {
                    "action": domain_event.value,
                    "actor": ACTOR,
                    "reason": WHY,
                    "from": current.state.value,
                    "to": expected.state.state.value,
                    "failures": expected.state.consecutive_failures,
                    "next_run_at": (
                        expected.state.next_run_at.isoformat()
                        if expected.state.next_run_at
                        else None
                    ),
                },
            )
        ]
        assert env.control.list_control_events()[0].at == START

    def test_a_normal_collection_cycle_and_its_audit_trail(self, env: Env):
        env.enable_source("src_one")
        env.service.start_collection("src_one", **WHO)
        assert env.service.get_state("src_one").state is CollectionState.COLLECTING
        env.service.finish_collection("src_one", **WHO)
        assert env.service.get_state("src_one") == SourceState("src_one", CollectionState.IDLE)
        assert env.kinds() == ["collection_started", "collection_succeeded"]
        assert_log_reproduces_state(env)
        assert_chains_are_consistent(env)

    def test_failures_back_off_then_retry_then_fail_for_good_and_can_be_reset(self, env: Env):
        env.enable_source("src_one")
        for attempt in range(1, 5):
            env.service.start_collection("src_one", **WHO)
            env.service.fail_collection("src_one", **WHO)
            state = env.service.get_state("src_one")
            assert (state.state, state.consecutive_failures) == (CollectionState.BACKOFF, attempt)
            with pytest.raises(InvalidTransition):
                env.service.retry_collection("src_one", **WHO)  # not due yet
            env.clock.advance(timedelta(hours=2))
            env.service.retry_collection("src_one", **WHO)
        env.service.start_collection("src_one", **WHO)
        env.service.fail_collection("src_one", **WHO)
        assert env.service.get_state("src_one") == SourceState("src_one", CollectionState.FAILED, 5)
        env.service.reset_failed("src_one", **WHO)
        assert env.service.get_state("src_one") == SourceState("src_one", CollectionState.IDLE)
        assert_log_reproduces_state(env)
        assert_chains_are_consistent(env)

    def test_backoff_time_comes_from_the_clock_and_the_jitter(self, env: Env):
        env.enable_source("src_one")
        env.service.start_collection("src_one", **WHO)
        result = env.service.fail_collection("src_one", jitter_s=5.0, **WHO)
        assert result.state.next_run_at == START + timedelta(seconds=65)
        assert env.events()[-1][2]["next_run_at"] == (START + timedelta(seconds=65)).isoformat()

    @pytest.mark.parametrize("jitter", [-1, math.nan, math.inf, "3", None, True])
    def test_an_invalid_jitter_is_rejected_and_writes_nothing(self, env: Env, jitter: Any):
        self.put(env, "collecting")
        before = env.control.list_control_events()
        with pytest.raises(ValidationError):
            env.service.fail_collection("src_one", jitter_s=jitter, **WHO)
        assert env.control.get_source_state("src_one") == STATES["collecting"]
        assert env.control.list_control_events() == before

    def test_reset_failed_works_only_on_failed_sources(self, env: Env):
        for name in ("none", "disabled", "idle", "collecting", "backoff_future"):
            fresh = self.make_env()
            fresh.add_source("src_one")
            self.put(fresh, name)
            with pytest.raises(InvalidTransition):
                fresh.service.reset_failed("src_one", **WHO)
            assert fresh.control.list_control_events() == []

    def test_an_unknown_event_name_is_an_invalid_transition(self, env: Env):
        with pytest.raises(InvalidTransition):
            env.service.apply_collection_event("src_one", "explode", **WHO)  # type: ignore[arg-type]
        assert env.control.list_control_events() == []

    def test_a_failing_event_append_leaves_the_collection_state_unchanged(self, env: Env):
        self.put(env, "idle")
        flaky = FlakyControl(env.control, "append_control_event")
        with pytest.raises(StorageError):
            env.service_with(control=flaky).start_collection("src_one", **WHO)
        assert env.control.get_source_state("src_one") == STATES["idle"]
        assert env.control.list_control_events() == []

    def test_a_failing_state_write_leaves_no_event(self, env: Env):
        env.enable_source("src_one")
        flaky = FlakyControl(env.control, "save_source_state")
        with pytest.raises(StorageError):
            env.service_with(control=flaky).start_collection("src_one", **WHO)
        assert env.control.get_source_state("src_one") == SourceState(
            "src_one", CollectionState.IDLE
        )
        assert env.control.list_control_events() == []

    def test_sources_are_independent(self, env: Env):
        env.add_source("src_two")
        env.enable_source("src_two")
        env.service.start_collection("src_one", **WHO) if False else None
        env.service.start_collection("src_two", **WHO)
        assert env.service.get_state("src_one").state is CollectionState.DISABLED
        assert env.control.get_source_state("src_one") is None
        assert env.service.get_state("src_two").state is CollectionState.COLLECTING

    def test_two_threads_racing_worker_transitions_leave_a_consistent_state_and_log(self, env: Env):
        for _ in range(10):
            fresh = self.make_env()
            fresh.add_source("src_one")
            fresh.enable_source("src_one")
            results = run_all(
                [
                    partial(fresh.service.start_collection, "src_one", **WHO),
                    partial(fresh.service.finish_collection, "src_one", **WHO),
                ]
            )
            for kind, value in results:
                assert kind == "ok" or isinstance(value, InvalidTransition), results
            assert_log_reproduces_state(fresh)
            assert_chains_are_consistent(fresh)

    def test_many_threads_starting_one_source_let_exactly_one_win(self, env: Env):
        self.put(env, "idle")
        results = run_all([partial(env.service.start_collection, "src_one", **WHO)] * 5)
        assert sorted(kind for kind, _ in results) == ["err"] * 4 + ["ok"]
        assert all(isinstance(v, InvalidTransition) for k, v in results if k == "err")
        assert env.kinds() == ["collection_started"]
        assert env.service.get_state("src_one").state is CollectionState.COLLECTING

    # ---- alert threshold ----

    def test_the_default_threshold_is_one_half(self, env: Env):
        assert env.service.alert_min_confidence() == 0.5
        assert env.control.get_setting(SETTING_ALERT_MIN_CONFIDENCE) == "0.5"

    def test_setting_the_threshold_persists_it_and_writes_an_exact_event(self, env: Env):
        result = env.service.set_alert_min_confidence(0.7, **WHO)
        assert result == ThresholdChange(value=0.7, changed=True)
        assert env.service.alert_min_confidence() == 0.7
        assert env.control.get_setting(SETTING_ALERT_MIN_CONFIDENCE) == "0.7"
        assert env.events() == [
            (
                EVENT_ALERT_THRESHOLD,
                None,
                {"action": "set", "actor": ACTOR, "reason": WHY, "from": "0.5", "to": "0.7"},
            )
        ]

    @pytest.mark.parametrize(
        "value",
        [0.0, 1.0, 0.25, math.nextafter(0.0, 1.0), math.nextafter(1.0, 0.0), 0, 1, 0.1, 1 / 3],
    )
    def test_boundary_and_awkward_values_are_stored_exactly(self, env: Env, value: float):
        env.service.set_alert_min_confidence(value, **WHO)
        assert env.service.alert_min_confidence() == value
        assert type(env.service.alert_min_confidence()) is float

    def test_setting_the_default_value_explicitly_is_a_noop(self, env: Env):
        result = env.service.set_alert_min_confidence(0.5, **WHO)
        assert result == ThresholdChange(value=0.5, changed=False)
        assert env.control.list_control_events() == []

    def test_setting_the_same_override_twice_is_a_noop(self, env: Env):
        env.service.set_alert_min_confidence(0.9, **WHO)
        assert env.service.set_alert_min_confidence(0.9, **WHO).changed is False
        assert env.kinds() == [EVENT_ALERT_THRESHOLD]

    def test_negative_zero_is_stored_as_zero(self, env: Env):
        env.service.set_alert_min_confidence(-0.0, **WHO)
        assert math.copysign(1.0, env.service.alert_min_confidence()) == 1.0
        assert env.events()[-1][2]["to"] == "0.0"

    @pytest.mark.parametrize(
        "bad",
        [
            math.nan,
            math.inf,
            -math.inf,
            -0.1,
            1.01,
            "abc",
            "0.5",
            True,
            False,
            None,
            [],
            2,
            10**400,
        ],
        ids=repr,
    )
    def test_invalid_thresholds_are_rejected_leaving_the_old_value_and_the_log(
        self, env: Env, bad: Any
    ):
        env.service.set_alert_min_confidence(0.8, **WHO)
        before = env.control.list_control_events()
        with pytest.raises(ValidationError) as caught:
            env.service.set_alert_min_confidence(bad, **WHO)
        assert caught.value.fields == (SETTING_ALERT_MIN_CONFIDENCE,)
        assert env.service.alert_min_confidence() == 0.8
        assert env.control.list_control_events() == before

    def test_a_failing_event_append_leaves_the_threshold_unchanged(self, env: Env):
        flaky = FlakyControl(env.control, "append_control_event")
        with pytest.raises(StorageError):
            env.service_with(control=flaky).set_alert_min_confidence(0.9, **WHO)
        assert env.service.alert_min_confidence() == 0.5
        assert env.control.list_control_events() == []

    def test_one_hundred_threshold_updates_replay_to_the_final_value(self, env: Env):
        for i in range(100):
            env.service.set_alert_min_confidence((i % 10) / 10, **WHO)
        assert_log_reproduces_state(env)
        assert_chains_are_consistent(env)

    # ---- everything together ----

    def test_a_mixed_session_replays_to_the_persisted_state(self, env: Env):
        env.add_source("src_two")
        env.service.set_global_analysis(True, **WHO)
        env.service.set_source_analysis("src_two", True, **WHO)
        env.enable_source("src_one")
        env.service.start_collection("src_one", **WHO)
        env.service.fail_collection("src_one", **WHO)
        env.enable_source("src_two")
        env.service.set_alert_min_confidence(0.65, **WHO)
        assert_log_reproduces_state(env)
        assert_chains_are_consistent(env)

    def test_a_bool_is_never_taken_for_a_threshold_even_when_it_equals_the_current_value(
        self, env: Env
    ):
        env.service.set_alert_min_confidence(1.0, **WHO)
        with pytest.raises(ValidationError):
            env.service.set_alert_min_confidence(True, **WHO)
        env.service.set_alert_min_confidence(0.0, **WHO)
        with pytest.raises(ValidationError):
            env.service.set_alert_min_confidence(False, **WHO)
        assert env.kinds() == [EVENT_ALERT_THRESHOLD] * 2

    def test_the_reported_threshold_is_always_a_positive_float(self, env: Env):
        assert type(env.service.set_alert_min_confidence(1, **WHO).value) is float
        assert type(env.service.set_alert_min_confidence(0, **WHO).value) is float
        noop = env.service.set_alert_min_confidence(-0.0, **WHO)
        assert noop.changed is False
        assert math.copysign(1.0, noop.value) == 1.0

    def test_astral_characters_in_actor_and_reason_still_fit_in_an_event(self, env: Env):
        emoji = chr(0x1F600)
        env.service.set_global_analysis(True, actor=emoji * MAX_ACTOR, reason=emoji * MAX_REASON)
        _, _, detail = env.events()[0]
        assert (detail["actor"], detail["reason"]) == (emoji * MAX_ACTOR, emoji * MAX_REASON)

    # ---- event kinds are a stable, literal vocabulary ----

    @pytest.mark.parametrize(
        ("method", "state_name", "kind"),
        [
            ("start_collection", "idle", "collection_started"),
            ("finish_collection", "collecting", "collection_succeeded"),
            ("fail_collection", "collecting", "collection_failed"),
            ("retry_collection", "backoff_due", "collection_retried"),
            ("reset_failed", "failed", "collection_reset"),
        ],
    )
    def test_each_collection_event_is_written_with_its_exact_kind_string(
        self, env: Env, method: str, state_name: str, kind: str
    ):
        self.put(env, state_name)
        getattr(env.service, method)("src_one", **WHO)
        assert env.kinds() == [kind]

    def test_the_collection_event_vocabulary_is_exactly_the_five_worker_transitions(self):
        assert COLLECTION_EVENT_KINDS == {
            Event.START: "collection_started",
            Event.SUCCEED: "collection_succeeded",
            Event.FAIL: "collection_failed",
            Event.RETRY: "collection_retried",
            Event.RESET: "collection_reset",
        }
        assert {m for m, _ in OPERATIONS} == {
            "start_collection",
            "finish_collection",
            "fail_collection",
            "retry_collection",
            "reset_failed",
        }

    def test_the_other_control_event_kinds_are_pinned_too(self):
        assert EVENT_GLOBAL_ANALYSIS == "global_analysis_changed"
        assert EVENT_SOURCE_ANALYSIS == "source_analysis_changed"
        assert EVENT_ALERT_THRESHOLD == "alert_threshold_changed"

    # ---- overview (one consistent read model for the CLI listing and status) ----

    def test_the_overview_of_a_fresh_system_is_paused_with_no_sources(self):
        env = self.make_env()
        overview = env.service.overview()
        assert overview.global_analysis is False
        assert overview.alert_min_confidence == 0.5
        assert overview.sources == ()

    def test_the_overview_reports_state_flag_and_effective_analysis_per_source(self, env: Env):
        env.add_source("src_two")
        env.add_source("src_three")
        env.enable_source("src_one")  # idle, analysis off
        env.enable_source("src_two")
        env.control.save_source_state(SourceState("src_two", CollectionState.COLLECTING))
        env.service.set_source_analysis("src_two", True, **WHO)
        env.service.set_source_analysis("src_three", True, **WHO)  # disabled source, flag on
        env.service.set_global_analysis(True, **WHO)

        overview = env.service.overview()

        assert overview.global_analysis is True
        assert [(o.source.id, o.state, o.analysis, o.effective) for o in overview.sources] == [
            ("src_one", CollectionState.IDLE, False, False),
            ("src_three", CollectionState.DISABLED, True, False),
            ("src_two", CollectionState.COLLECTING, True, True),
        ]
        assert all(isinstance(o, SourceOverview) for o in overview.sources)

    def test_the_overview_follows_the_global_switch_and_the_threshold(self, env: Env):
        env.enable_source("src_one")
        env.service.set_source_analysis("src_one", True, **WHO)
        env.service.set_alert_min_confidence(0.8, **WHO)
        assert env.service.overview().sources[0].effective is False  # system paused
        assert env.service.overview().alert_min_confidence == 0.8
        env.service.set_global_analysis(True, **WHO)
        assert env.service.overview().sources[0].effective is True

    def test_the_overview_reads_states_and_flags_in_bulk(self, env: Env):
        for source_id in ("src_two", "src_three"):
            env.add_source(source_id)
            env.enable_source(source_id)
            env.service.set_source_analysis(source_id, True, **WHO)
        bulk = env.service_with(control=NoPerSourceLookups(env.control))
        assert [o.source.id for o in bulk.overview().sources] == ["src_one", "src_three", "src_two"]
        assert [o.analysis for o in bulk.overview().sources] == [False, True, True]

    def test_the_overview_of_a_corrupt_stored_flag_is_a_storage_error(self, env: Env):
        self.corrupt_setting(env, source_analysis_key("src_one"), "maybe")
        with pytest.raises(StorageError):
            env.service.overview()
