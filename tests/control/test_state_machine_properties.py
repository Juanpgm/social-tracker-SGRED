"""Stateful property tests: arbitrary event interleavings keep every invariant, and the
machine provably reaches the interesting states (BACKOFF, FAILED, recovery) instead of
passing vacuously.

`SourceLifecycle` drives `apply_event` through randomly interleaved events. An independent
oracle (the transition TABLE from the example tests plus a failure-count model) judges every
step, so a wrong transition cannot hide behind a self-consistent implementation.
"""

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from hypothesis import event as hypothesis_event
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from social_tracker.control.domain import (
    BACKOFF_BASE_S,
    BACKOFF_CAP_S,
    MAX_CONSECUTIVE_FAILURES,
    CollectionState,
    Event,
    SourceState,
    apply_event,
    backoff_delay_s,
)
from social_tracker.shared.errors import InvalidTransition
from tests.control.test_state_machine import BAD, NOOP, TABLE

T0 = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
S = CollectionState

# Every coherent starting state (mirrors the SourceState invariants).
COHERENT_STARTS = [
    SourceState("src_one", S.DISABLED),
    SourceState("src_one", S.IDLE),
    *[SourceState("src_one", S.IDLE, n) for n in range(1, MAX_CONSECUTIVE_FAILURES)],
    *[SourceState("src_one", S.COLLECTING, n) for n in range(MAX_CONSECUTIVE_FAILURES)],
    *[
        SourceState("src_one", S.BACKOFF, n, T0 + timedelta(seconds=60))
        for n in range(1, MAX_CONSECUTIVE_FAILURES)
    ],
    SourceState("src_one", S.FAILED, MAX_CONSECUTIVE_FAILURES),
]

seconds = st.integers(min_value=0, max_value=7200)
jitters = st.floats(min_value=0, max_value=30, allow_nan=False)
junk_events = st.sampled_from([*Event, "not_an_event", "", None])


class SourceLifecycle(RuleBasedStateMachine):
    # Class-level coverage, reset by the driving test; asserted after the run.
    visits: ClassVar[Counter[str]] = Counter()

    def __init__(self) -> None:
        super().__init__()
        self.current = SourceState("src_one", S.IDLE)
        self.now = T0
        self.model_failures = 0  # independent count of failures since the last reset

    @initialize(start=st.sampled_from(COHERENT_STARTS))
    def choose_start(self, start: SourceState) -> None:
        self.current = start
        self.model_failures = start.consecutive_failures

    # ---- the single place events are applied and judged -----------------------------

    def _apply(self, raw_event: object, at: datetime, jitter: float = 0.0) -> None:
        before = self.current
        try:
            event = Event(raw_event)  # type: ignore[arg-type]
        except ValueError:
            event = None

        try:
            result = apply_event(before, raw_event, at, jitter_s=jitter)  # type: ignore[arg-type]
        except InvalidTransition:
            self.visits["rejected"] += 1
            assert self.current == before  # a rejected event leaves the state alone
            assert event is None or self._oracle_says_invalid(before, event, at)
            return

        assert event is not None, "an unknown event must never be accepted"
        outcome, expected = TABLE[(before.state, event)]
        assert outcome != BAD, f"{event} should be illegal in {before.state}"
        if event is Event.RETRY:  # accepted only once the backoff deadline has been reached
            assert before.next_run_at is not None
            assert at >= before.next_run_at
        assert result.changed is (outcome != NOOP)
        if outcome == NOOP:
            assert result.state == before
        elif event is Event.FAIL:
            self._check_failure(before, result.state, at)
        else:
            assert result.state.state is expected

        self._update_model(event, before, result.state, at)
        self.current = result.state

    @staticmethod
    def _oracle_says_invalid(before: SourceState, event: Event, at: datetime) -> bool:
        outcome, _ = TABLE[(before.state, event)]
        if (before.state, event) == (S.BACKOFF, Event.RETRY):
            return before.next_run_at is not None and at < before.next_run_at
        return outcome == BAD

    def _check_failure(self, before: SourceState, after: SourceState, at: datetime) -> None:
        failures = before.consecutive_failures + 1
        if failures >= MAX_CONSECUTIVE_FAILURES:
            assert after.state is S.FAILED
            self.visits["into_failed"] += 1
            hypothesis_event("into FAILED")
            return
        assert after.state is S.BACKOFF
        assert after.next_run_at is not None
        delay = (after.next_run_at - at).total_seconds()
        floor = min(BACKOFF_BASE_S * 2**before.consecutive_failures, BACKOFF_CAP_S)
        assert floor <= delay <= BACKOFF_CAP_S
        self.visits["into_backoff"] += 1

    def _update_model(
        self, event: Event, before: SourceState, after: SourceState, at: datetime
    ) -> None:
        if event is Event.FAIL:
            self.model_failures += 1
        elif event in {Event.SUCCEED, Event.DISABLE, Event.RESET} or (
            event is Event.ENABLE and before.state is S.DISABLED
        ):
            self.model_failures = 0
        if event is Event.RESET:
            self.visits["reset_from_failed"] += 1
        if event is Event.RETRY:
            self.visits["retry_when_due"] += 1
        self.now = max(self.now, at)

    # ---- rules ----------------------------------------------------------------------

    @precondition(lambda self: self.current.state is S.IDLE)
    @rule(advance=seconds)
    def start(self, advance: int) -> None:
        self._apply(Event.START, self.now + timedelta(seconds=advance))

    @precondition(lambda self: self.current.state is S.COLLECTING)
    @rule(advance=seconds)
    def succeed(self, advance: int) -> None:
        self._apply(Event.SUCCEED, self.now + timedelta(seconds=advance))

    @precondition(lambda self: self.current.state is S.COLLECTING)
    @rule(advance=seconds, jitter=jitters)
    def fail(self, advance: int, jitter: float) -> None:
        self._apply(Event.FAIL, self.now + timedelta(seconds=advance), jitter)

    @precondition(lambda self: self.current.state is S.BACKOFF)
    @rule(extra=st.integers(min_value=0, max_value=60))
    def retry_when_due(self, extra: int) -> None:
        assert self.current.next_run_at is not None
        self._apply(Event.RETRY, self.current.next_run_at + timedelta(seconds=extra))

    @precondition(lambda self: self.current.state is S.BACKOFF)
    @rule(early=st.integers(min_value=1, max_value=3600))
    def retry_too_early(self, early: int) -> None:
        assert self.current.next_run_at is not None
        self._apply(Event.RETRY, self.current.next_run_at - timedelta(seconds=early))

    @precondition(lambda self: self.current.state is S.FAILED)
    @rule(advance=seconds)
    def operator_reset(self, advance: int) -> None:
        self._apply(Event.RESET, self.now + timedelta(seconds=advance))

    @rule(raw=junk_events, advance=seconds, jitter=jitters)
    def anything(self, raw: object, advance: int, jitter: float) -> None:
        """Any event in any state, including illegal and unknown ones."""
        self._apply(raw, self.now + timedelta(seconds=advance), jitter)

    # ---- invariants, checked after every step ---------------------------------------

    @invariant()
    def state_is_coherent(self) -> None:
        current = self.current
        assert isinstance(current.state, S)
        assert 0 <= current.consecutive_failures <= MAX_CONSECUTIVE_FAILURES
        assert (current.state is S.FAILED) == (
            current.consecutive_failures == MAX_CONSECUTIVE_FAILURES
        )
        assert (current.next_run_at is not None) == (current.state is S.BACKOFF)
        if current.state is S.DISABLED:
            assert current.consecutive_failures == 0

    @invariant()
    def failure_count_matches_the_independent_model(self) -> None:
        assert self.current.consecutive_failures == self.model_failures

    @invariant()
    def disable_is_always_reachable_and_idempotent(self) -> None:
        first = apply_event(self.current, Event.DISABLE, self.now)
        assert first.state.state is S.DISABLED
        second = apply_event(first.state, Event.DISABLE, self.now)
        assert second.changed is False
        assert second.state == first.state


def test_event_interleavings_keep_every_invariant_and_cover_the_interesting_states():
    SourceLifecycle.visits = Counter()
    run_state_machine_as_test(
        SourceLifecycle,
        settings=settings(max_examples=150, stateful_step_count=60, deadline=None),
    )
    # Guard against a vacuous pass: the run must have exercised the paths worth testing.
    visits = SourceLifecycle.visits
    assert visits["into_backoff"] > 0, visits
    assert visits["into_failed"] > 0, visits
    assert visits["retry_when_due"] > 0, visits
    assert visits["reset_from_failed"] > 0, visits
    assert visits["rejected"] > 0, visits


# ---- backoff delay is a pure function with simple bounds -------------------------------


@settings(max_examples=300, deadline=None)
@given(
    n=st.integers(min_value=0, max_value=10**6),
    jitter=st.floats(min_value=0, max_value=1e6, allow_nan=False),
)
def test_backoff_delay_is_bounded_and_monotonic_in_failures(n: int, jitter: float):
    delay = backoff_delay_s(n, jitter)
    assert BACKOFF_BASE_S <= delay <= BACKOFF_CAP_S
    assert backoff_delay_s(n + 1, jitter) >= delay
