"""Pure control state machine: transitions, no-ops, backoff, failure counting, gates."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.control.domain import (
    BACKOFF_CAP_S,
    MAX_CONSECUTIVE_FAILURES,
    CollectionState,
    Event,
    GlobalState,
    SourceState,
    analysis_effective,
    apply_event,
    apply_global,
    backoff_delay_s,
)
from social_tracker.shared.errors import InvalidTransition, ValidationError

NOW = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
S = CollectionState
E = Event


def state(
    s: CollectionState, failures: int | None = None, next_run_at: datetime | None = None
) -> SourceState:
    """A coherent state: FAILED always carries the maximum failure count."""
    if failures is None:
        failures = MAX_CONSECUTIVE_FAILURES if s is S.FAILED else 0
    if s is S.BACKOFF and next_run_at is None:
        next_run_at = NOW + timedelta(seconds=60)
    return SourceState("src_one", s, failures, next_run_at)


# ---- full transition table -------------------------------------------------------------

OK, NOOP, BAD = "ok", "noop", "invalid"

TABLE = {
    # (state, event): (outcome, resulting state)
    (S.DISABLED, E.ENABLE): (OK, S.IDLE),
    (S.IDLE, E.ENABLE): (NOOP, S.IDLE),
    (S.COLLECTING, E.ENABLE): (NOOP, S.COLLECTING),
    (S.BACKOFF, E.ENABLE): (NOOP, S.BACKOFF),
    (S.FAILED, E.ENABLE): (NOOP, S.FAILED),
    (S.DISABLED, E.DISABLE): (NOOP, S.DISABLED),
    (S.IDLE, E.DISABLE): (OK, S.DISABLED),
    (S.COLLECTING, E.DISABLE): (OK, S.DISABLED),
    (S.BACKOFF, E.DISABLE): (OK, S.DISABLED),
    (S.FAILED, E.DISABLE): (OK, S.DISABLED),
    (S.IDLE, E.START): (OK, S.COLLECTING),
    (S.DISABLED, E.START): (BAD, None),
    (S.COLLECTING, E.START): (BAD, None),
    (S.BACKOFF, E.START): (BAD, None),
    (S.FAILED, E.START): (BAD, None),
    (S.COLLECTING, E.SUCCEED): (OK, S.IDLE),
    (S.DISABLED, E.SUCCEED): (BAD, None),
    (S.IDLE, E.SUCCEED): (BAD, None),
    (S.BACKOFF, E.SUCCEED): (BAD, None),
    (S.FAILED, E.SUCCEED): (BAD, None),
    (S.COLLECTING, E.FAIL): (OK, S.BACKOFF),
    (S.DISABLED, E.FAIL): (BAD, None),
    (S.IDLE, E.FAIL): (BAD, None),
    (S.BACKOFF, E.FAIL): (BAD, None),
    (S.FAILED, E.FAIL): (BAD, None),
    (S.BACKOFF, E.RETRY): (OK, S.IDLE),  # now is far past next_run_at in this table test
    (S.DISABLED, E.RETRY): (BAD, None),
    (S.IDLE, E.RETRY): (BAD, None),
    (S.COLLECTING, E.RETRY): (BAD, None),
    (S.FAILED, E.RETRY): (BAD, None),
    (S.FAILED, E.RESET): (OK, S.IDLE),
    (S.DISABLED, E.RESET): (BAD, None),
    (S.IDLE, E.RESET): (BAD, None),
    (S.COLLECTING, E.RESET): (BAD, None),
    (S.BACKOFF, E.RESET): (BAD, None),
}


def test_table_covers_every_state_event_pair():
    assert set(TABLE) == {(s, e) for s in S for e in E}


@pytest.mark.parametrize(("start", "event"), list(TABLE))
def test_transition_table(start: CollectionState, event: Event):
    outcome, expected = TABLE[(start, event)]
    current = state(start)
    now = NOW + timedelta(days=1)
    if outcome == BAD:
        with pytest.raises(InvalidTransition) as info:
            apply_event(current, event, now)
        assert info.value.code == "invalid_transition"
        return
    result = apply_event(current, event, now)
    assert result.state.state is expected
    assert result.changed is (outcome == OK)
    assert result.state.source_id == "src_one"
    if outcome == NOOP:
        assert result.state == current


def test_string_events_are_accepted():
    assert apply_event(state(S.IDLE), "start", NOW).state.state is S.COLLECTING


@pytest.mark.parametrize("bad", ["bogus", "", "START ", None, 3])
def test_unknown_event_is_invalid_transition(bad):
    with pytest.raises(InvalidTransition):
        apply_event(state(S.IDLE), bad, NOW)


@pytest.mark.parametrize("bad", ["bogus", "", "IDLE ", "Idle", None, 3])
def test_unknown_state_is_invalid_transition(bad):
    with pytest.raises(InvalidTransition):
        SourceState("src_one", bad)  # type: ignore[arg-type]


def test_state_strings_are_coerced():
    assert SourceState("src_one", "idle").state is S.IDLE


# ---- specific rules --------------------------------------------------------------------


def test_enable_starts_from_a_clean_slate():
    result = apply_event(state(S.DISABLED), E.ENABLE, NOW)
    assert result.state == SourceState("src_one", S.IDLE, 0, None)


@pytest.mark.parametrize("start", [S.IDLE, S.COLLECTING, S.BACKOFF, S.FAILED])
def test_disable_zeroes_failures_so_a_reenabled_source_starts_fresh(start: CollectionState):
    # Documented decision: DISABLED always carries 0 failures; disable and reset both zero them.
    failures = None if start is S.FAILED else 3
    disabled = apply_event(state(start, failures=failures), E.DISABLE, NOW)
    assert disabled.state.consecutive_failures == 0
    assert apply_event(disabled.state, E.ENABLE, NOW).state.consecutive_failures == 0


def test_disable_clears_backoff_schedule_and_failures():
    result = apply_event(state(S.BACKOFF, failures=2), E.DISABLE, NOW)
    assert result.state == SourceState("src_one", S.DISABLED, 0, None)


def test_start_keeps_failure_count():
    result = apply_event(state(S.IDLE, failures=2), E.START, NOW)
    assert result.state.consecutive_failures == 2


def test_success_resets_failure_count():
    result = apply_event(state(S.COLLECTING, failures=4), E.SUCCEED, NOW)
    assert result.state == SourceState("src_one", S.IDLE, 0, None)


def test_failure_increments_and_schedules_backoff():
    result = apply_event(state(S.COLLECTING, failures=0), E.FAIL, NOW)
    assert result.state.state is S.BACKOFF
    assert result.state.consecutive_failures == 1
    assert result.state.next_run_at == NOW + timedelta(seconds=60)


def test_second_failure_doubles_the_delay_and_jitter_is_added():
    result = apply_event(state(S.COLLECTING, failures=1), E.FAIL, NOW, jitter_s=2.5)
    assert result.state.next_run_at == NOW + timedelta(seconds=122.5)


@pytest.mark.parametrize("prior", [0, 1, 2, 3])
def test_failures_below_the_limit_go_to_backoff(prior: int):
    result = apply_event(state(S.COLLECTING, failures=prior), E.FAIL, NOW)
    assert result.state.state is S.BACKOFF
    assert result.state.consecutive_failures == prior + 1


def test_fifth_consecutive_failure_goes_to_failed():
    assert MAX_CONSECUTIVE_FAILURES == 5
    result = apply_event(state(S.COLLECTING, failures=4), E.FAIL, NOW)
    assert result.state.state is S.FAILED
    assert result.state.consecutive_failures == 5
    assert result.state.next_run_at is None


def test_five_failures_in_a_row_through_retries_end_in_failed():
    current = state(S.IDLE)
    clock = NOW
    for i in range(MAX_CONSECUTIVE_FAILURES):
        current = apply_event(current, E.START, clock).state
        current = apply_event(current, E.FAIL, clock).state
        if i < MAX_CONSECUTIVE_FAILURES - 1:
            assert current.state is S.BACKOFF
            clock = current.next_run_at  # type: ignore[assignment]
            current = apply_event(current, E.RETRY, clock).state
    assert current.state is S.FAILED


def test_a_success_in_between_restarts_the_count():
    current = state(S.COLLECTING, failures=4)
    current = apply_event(current, E.SUCCEED, NOW).state
    current = apply_event(current, E.START, NOW).state
    assert apply_event(current, E.FAIL, NOW).state.state is S.BACKOFF


def test_operator_reset_clears_failures_so_failed_is_recoverable():
    failed = apply_event(state(S.COLLECTING, failures=4), E.FAIL, NOW).state
    idle = apply_event(failed, E.RESET, NOW).state
    assert idle == SourceState("src_one", S.IDLE, 0, None)
    assert apply_event(idle, E.START, NOW).state.state is S.COLLECTING


def test_retry_at_exactly_next_run_at_counts_as_elapsed():
    backoff = state(S.BACKOFF, failures=1, next_run_at=NOW)
    result = apply_event(backoff, E.RETRY, NOW)
    assert result.state.state is S.IDLE
    assert result.state.consecutive_failures == 1  # only success resets the count


def test_retry_one_millisecond_early_is_rejected():
    backoff = state(S.BACKOFF, failures=1, next_run_at=NOW)
    with pytest.raises(InvalidTransition, match="backoff"):
        apply_event(backoff, E.RETRY, NOW - timedelta(milliseconds=1))


def test_start_during_backoff_is_rejected_before_and_after_the_deadline():
    backoff = state(S.BACKOFF, next_run_at=NOW)
    with pytest.raises(InvalidTransition):
        apply_event(backoff, E.START, NOW - timedelta(milliseconds=1))
    with pytest.raises(InvalidTransition):
        apply_event(backoff, E.START, NOW)


def test_naive_now_is_rejected_even_for_noops():
    with pytest.raises(ValidationError, match="timezone-aware"):
        apply_event(state(S.IDLE), E.ENABLE, datetime(2026, 3, 1))


def test_non_datetime_now_is_rejected():
    with pytest.raises(ValidationError):
        apply_event(state(S.IDLE), E.START, "2026-03-01")  # type: ignore[arg-type]


@pytest.mark.parametrize("jitter", [-0.001, -1, float("nan"), float("inf"), True, "1", None])
@pytest.mark.parametrize("prior", [0, 3, MAX_CONSECUTIVE_FAILURES - 1])
def test_invalid_jitter_is_rejected_on_every_failure_outcome(prior: int, jitter):
    # prior == MAX-1 goes to FAILED, where jitter is unused: it must still be validated.
    with pytest.raises(ValidationError, match="jitter"):
        apply_event(state(S.COLLECTING, failures=prior), E.FAIL, NOW, jitter_s=jitter)


def test_rejected_failure_leaves_the_input_state_untouched():
    current = state(S.COLLECTING, failures=MAX_CONSECUTIVE_FAILURES - 1)
    with pytest.raises(ValidationError):
        apply_event(current, E.FAIL, NOW, jitter_s=-1)
    assert current.consecutive_failures == MAX_CONSECUTIVE_FAILURES - 1


def test_valid_jitter_is_accepted_when_the_failure_ends_in_failed():
    result = apply_event(
        state(S.COLLECTING, failures=MAX_CONSECUTIVE_FAILURES - 1), E.FAIL, NOW, jitter_s=12.5
    )
    assert result.state.state is S.FAILED


def test_failure_delay_never_exceeds_the_cap_even_with_jitter():
    result = apply_event(state(S.COLLECTING, failures=3), E.FAIL, NOW, jitter_s=10_000)
    assert result.state.next_run_at == NOW + timedelta(seconds=BACKOFF_CAP_S)


# ---- SourceState invariants ------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, True, 1.0, "1", None])
def test_negative_or_non_int_failure_counts_are_rejected(bad):
    with pytest.raises(ValidationError, match="consecutive_failures"):
        SourceState("src_one", S.IDLE, bad)  # type: ignore[arg-type]


def test_naive_next_run_at_is_rejected():
    with pytest.raises(ValidationError, match="timezone-aware"):
        SourceState("src_one", S.BACKOFF, 1, datetime(2026, 3, 1))


def test_backoff_requires_next_run_at():
    with pytest.raises(ValidationError, match="next_run_at"):
        SourceState("src_one", S.BACKOFF, 1, None)


@pytest.mark.parametrize("other", [S.DISABLED, S.IDLE, S.COLLECTING, S.FAILED])
def test_only_backoff_may_carry_next_run_at(other: CollectionState):
    failures = MAX_CONSECUTIVE_FAILURES if other is S.FAILED else 0
    with pytest.raises(ValidationError, match="next_run_at"):
        SourceState("src_one", other, failures, NOW)


@pytest.mark.parametrize("bad", ["", "Bad Id", "ab", None])
def test_source_id_must_be_a_slug(bad):
    with pytest.raises(ValidationError):
        SourceState(bad, S.IDLE)  # type: ignore[arg-type]


# Coherent combinations: DISABLED has no failures, FAILED has exactly the maximum,
# every other state has strictly fewer than the maximum.
VALID_COMBOS = [
    (S.DISABLED, 0),
    (S.IDLE, 0),
    (S.IDLE, MAX_CONSECUTIVE_FAILURES - 1),
    (S.COLLECTING, 0),
    (S.COLLECTING, MAX_CONSECUTIVE_FAILURES - 1),
    (S.BACKOFF, 1),
    (S.BACKOFF, MAX_CONSECUTIVE_FAILURES - 1),
    (S.FAILED, MAX_CONSECUTIVE_FAILURES),
]
INVALID_COMBOS = [
    (S.DISABLED, 3),
    (S.DISABLED, 1),
    (S.IDLE, 99),
    (S.IDLE, MAX_CONSECUTIVE_FAILURES),
    (S.COLLECTING, MAX_CONSECUTIVE_FAILURES),
    (S.COLLECTING, MAX_CONSECUTIVE_FAILURES + 1),
    (S.BACKOFF, MAX_CONSECUTIVE_FAILURES),
    (S.FAILED, 0),
    (S.FAILED, MAX_CONSECUTIVE_FAILURES - 1),
    (S.FAILED, MAX_CONSECUTIVE_FAILURES + 1),
]


@pytest.mark.parametrize(("which", "failures"), VALID_COMBOS)
def test_coherent_state_and_failure_combinations_are_accepted(which, failures):
    assert state(which, failures=failures).consecutive_failures == failures


@pytest.mark.parametrize(("which", "failures"), INVALID_COMBOS)
def test_incoherent_state_and_failure_combinations_are_rejected(which, failures):
    with pytest.raises(ValidationError, match="consecutive_failures"):
        state(which, failures=failures)


def test_source_state_is_immutable():
    with pytest.raises(AttributeError):
        state(S.IDLE).state = S.DISABLED  # type: ignore[misc]


# ---- backoff delay ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, 60), (1, 120), (2, 240), (3, 480), (4, 960), (5, 1920), (6, 3600), (7, 3600), (100, 3600)],
)
def test_backoff_delay_doubles_and_caps(n: int, expected: float):
    assert backoff_delay_s(n) == expected


def test_backoff_cap_constant():
    assert BACKOFF_CAP_S == 3600


def test_huge_exponent_does_not_blow_up():
    # 2**(10**12) would need ~125 GB: only a clamped exponent returns at all.
    assert backoff_delay_s(10**12) == 3600


def test_cap_applies_after_jitter_so_the_final_delay_never_exceeds_it():
    # Decision: BACKOFF_CAP_S bounds the FINAL delay (spec: capped at 3600 s), jitter included.
    assert backoff_delay_s(100, jitter_s=7.5) == BACKOFF_CAP_S
    assert backoff_delay_s(6, jitter_s=1e9) == BACKOFF_CAP_S


def test_jitter_below_the_cap_is_added_as_is():
    assert backoff_delay_s(0, jitter_s=0.5) == 60.5
    assert backoff_delay_s(5, jitter_s=100) == 2020  # 1920 + 100


@pytest.mark.parametrize(
    ("jitter", "expected"), [(1679.0, 3599.0), (1680.0, 3600.0), (1681.0, 3600.0)]
)
def test_cap_boundary_with_jitter(jitter: float, expected: float):
    assert backoff_delay_s(5, jitter_s=jitter) == expected  # base delay 1920


@pytest.mark.parametrize("n", [-1, -100, True, 1.5, "2", None])
def test_invalid_exponent_is_rejected(n):
    with pytest.raises(ValidationError, match="failure count"):
        backoff_delay_s(n)


@pytest.mark.parametrize("jitter", [-1, float("nan"), float("-inf"), float("inf"), False, "0"])
def test_invalid_jitter_value_is_rejected(jitter):
    with pytest.raises(ValidationError, match="jitter"):
        backoff_delay_s(1, jitter_s=jitter)


# ---- results are values -------------------------------------------------------------------


def test_transition_result_is_frozen():
    result = apply_event(state(S.IDLE), E.START, NOW)
    with pytest.raises(FrozenInstanceError):
        result.changed = False  # type: ignore[misc]


def test_global_result_is_frozen():
    result = apply_global(GlobalState.ACTIVE, "paused")
    with pytest.raises(FrozenInstanceError):
        result.state = GlobalState.ACTIVE  # type: ignore[misc]


# ---- persisted wire strings (unit 1c stores these verbatim) --------------------------------


def test_collection_state_wire_strings_are_pinned():
    assert {s.name: s.value for s in S} == {
        "DISABLED": "disabled",
        "IDLE": "idle",
        "COLLECTING": "collecting",
        "BACKOFF": "backoff",
        "FAILED": "failed",
    }


def test_global_state_wire_strings_are_pinned():
    assert {s.name: s.value for s in GlobalState} == {"ACTIVE": "active", "PAUSED": "paused"}


def test_event_wire_strings_are_pinned():
    assert {e.name: e.value for e in E} == {
        "ENABLE": "enable",
        "DISABLE": "disable",
        "START": "start",
        "SUCCEED": "succeed",
        "FAIL": "fail",
        "RETRY": "retry",
        "RESET": "reset",
    }


@pytest.mark.parametrize("member", [*S, *GlobalState])
def test_enum_members_round_trip_through_their_string_value(member):
    assert type(member)(member.value) is member
    assert str(member) == member.value


# ---- global state and analysis gate ----------------------------------------------------


@pytest.mark.parametrize(
    ("current", "target", "expected", "changed"),
    [
        (GlobalState.ACTIVE, "paused", GlobalState.PAUSED, True),
        (GlobalState.PAUSED, "active", GlobalState.ACTIVE, True),
        (GlobalState.ACTIVE, "active", GlobalState.ACTIVE, False),
        (GlobalState.PAUSED, "paused", GlobalState.PAUSED, False),
    ],
)
def test_global_state_transitions_and_noops(current, target, expected, changed):
    result = apply_global(current, target)
    assert (result.state, result.changed) == (expected, changed)


@pytest.mark.parametrize("bad", ["bogus", "", "ACTIVE", None, 1])
def test_global_unknown_target_or_current_is_invalid(bad):
    with pytest.raises(InvalidTransition):
        apply_global(GlobalState.ACTIVE, bad)
    with pytest.raises(InvalidTransition):
        apply_global(bad, "active")


@pytest.mark.parametrize(
    ("global_state", "source_flag", "source_enabled", "expected"),
    [
        (GlobalState.ACTIVE, True, True, True),
        (GlobalState.ACTIVE, True, False, False),
        (GlobalState.ACTIVE, False, True, False),
        (GlobalState.ACTIVE, False, False, False),
        (GlobalState.PAUSED, True, True, False),
        (GlobalState.PAUSED, True, False, False),
        (GlobalState.PAUSED, False, True, False),
        (GlobalState.PAUSED, False, False, False),
    ],
)
def test_analysis_gate_truth_table(global_state, source_flag, source_enabled, expected):
    assert analysis_effective(global_state, source_flag, source_enabled=source_enabled) is expected


def test_analysis_gate_accepts_state_strings():
    assert analysis_effective("active", True, source_enabled=True) is True


@pytest.mark.parametrize("bad", [1, 0, None, "true"])
def test_analysis_gate_requires_a_real_bool_flag(bad):
    with pytest.raises(ValidationError, match="source_analysis_enabled"):
        analysis_effective(GlobalState.ACTIVE, bad, source_enabled=True)


@pytest.mark.parametrize("bad", [1, 0, None, "true"])
def test_analysis_gate_requires_a_real_bool_for_the_source_enabled_flag(bad):
    with pytest.raises(ValidationError, match="source_enabled"):
        analysis_effective(GlobalState.ACTIVE, True, source_enabled=bad)


def test_analysis_gate_rejects_unknown_global_state():
    with pytest.raises(InvalidTransition):
        analysis_effective("bogus", True, source_enabled=True)
