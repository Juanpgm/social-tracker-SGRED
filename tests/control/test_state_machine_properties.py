"""Property tests: no event sequence can reach an undefined or inconsistent state."""

from datetime import UTC, datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from social_tracker.control.domain import (
    MAX_CONSECUTIVE_FAILURES,
    CollectionState,
    Event,
    SourceState,
    apply_event,
)
from social_tracker.shared.errors import InvalidTransition

T0 = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)

steps = st.lists(
    st.tuples(
        st.sampled_from(list(Event) + ["not_an_event"]),
        st.integers(min_value=0, max_value=7200),  # seconds the clock advances
        st.floats(min_value=0, max_value=30, allow_nan=False),  # jitter
    ),
    max_size=60,
)
initial_states = st.sampled_from(list(CollectionState))


def _initial(s: CollectionState) -> SourceState:
    next_run = T0 + timedelta(seconds=60) if s is CollectionState.BACKOFF else None
    failures = MAX_CONSECUTIVE_FAILURES if s is CollectionState.FAILED else 0
    return SourceState("src_one", s, failures, next_run)


def _run(start: SourceState, sequence):
    current, now = start, T0
    for event, advance_s, jitter in sequence:
        now = now + timedelta(seconds=advance_s)
        try:
            result = apply_event(current, event, now, jitter_s=jitter)
        except InvalidTransition:
            yield current, now, None
            continue
        current = result.state
        yield current, now, result


@settings(max_examples=300, deadline=None)
@given(start=initial_states, sequence=steps)
def test_every_reachable_state_is_defined_and_consistent(start, sequence):
    for current, now, result in _run(_initial(start), sequence):
        assert isinstance(current.state, CollectionState)
        assert 0 <= current.consecutive_failures <= MAX_CONSECUTIVE_FAILURES
        assert (current.next_run_at is not None) == (current.state is CollectionState.BACKOFF)
        if current.state is CollectionState.FAILED:
            assert current.consecutive_failures == MAX_CONSECUTIVE_FAILURES
        if result is not None and result.changed and current.state is CollectionState.BACKOFF:
            assert timedelta(seconds=60) <= current.next_run_at - now <= timedelta(seconds=3631)


@settings(max_examples=300, deadline=None)
@given(start=initial_states, sequence=steps)
def test_disable_is_always_reachable_and_idempotent(start, sequence):
    trace = list(_run(_initial(start), sequence))
    current, now, _ = trace[-1] if trace else (_initial(start), T0, None)
    first = apply_event(current, Event.DISABLE, now)
    assert first.state.state is CollectionState.DISABLED
    second = apply_event(first.state, Event.DISABLE, now)
    assert second.changed is False
    assert second.state == first.state


@settings(max_examples=300, deadline=None)
@given(start=initial_states, sequence=steps)
def test_rejected_events_never_change_state_and_noops_are_flagged(start, sequence):
    current = _initial(start)
    now = T0
    for event, advance_s, jitter in sequence:
        now = now + timedelta(seconds=advance_s)
        try:
            result = apply_event(current, event, now, jitter_s=jitter)
        except InvalidTransition:
            continue  # the caller still holds `current`; nothing was mutated
        assert result.changed == (result.state != current)
        current = result.state


@settings(max_examples=200, deadline=None)
@given(jitter=st.floats(min_value=0, max_value=30, allow_nan=False))
def test_consecutive_failures_always_end_in_failed_without_success(jitter):
    current = SourceState("src_one", CollectionState.IDLE)
    now = T0
    for _ in range(MAX_CONSECUTIVE_FAILURES):
        current = apply_event(current, Event.START, now).state
        current = apply_event(current, Event.FAIL, now, jitter_s=jitter).state
        if current.state is CollectionState.BACKOFF:
            now = current.next_run_at
            current = apply_event(current, Event.RETRY, now).state
    assert current.state is CollectionState.FAILED
