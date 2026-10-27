"""Stop signals (SIGINT, SIGTERM) are turned into a graceful stop request, then restored."""

import signal
import threading

import pytest

from social_tracker.entrypoints.signals import stop_on_signals

SIGNALS = [signal.SIGINT, signal.SIGTERM]


@pytest.mark.parametrize("signum", SIGNALS, ids=lambda s: s.name)
def test_a_stop_signal_requests_a_stop_instead_of_killing_the_process(signum):
    requested = []
    with stop_on_signals(lambda: requested.append(signum)):
        signal.raise_signal(signum)
    assert requested == [signum]


@pytest.mark.parametrize(
    ("first", "second", "code"),
    [
        (signal.SIGINT, signal.SIGINT, 130),
        (signal.SIGTERM, signal.SIGTERM, 143),
        (signal.SIGINT, signal.SIGTERM, 143),
        (signal.SIGTERM, signal.SIGINT, 130),
    ],
    ids=["int-int", "term-term", "int-term", "term-int"],
)
def test_a_second_signal_escalates_to_an_immediate_exit_with_the_conventional_code(
    first, second, code
):
    requested = []

    def impatient():
        with stop_on_signals(lambda: requested.append(1)):
            signal.raise_signal(first)
            assert requested == [1]  # the first one only asked for a stop
            signal.raise_signal(second)
            pytest.fail("the second signal must not return")

    with pytest.raises(SystemExit) as caught:
        impatient()
    assert caught.value.code == code
    assert requested == [1]  # and the stop was requested once, not again


def test_the_escalation_unwinds_the_stack_so_cleanup_runs():
    cleaned = []

    def impatient():
        with stop_on_signals(lambda: None):
            try:
                signal.raise_signal(signal.SIGTERM)
                signal.raise_signal(signal.SIGTERM)
            finally:
                cleaned.append("closed")

    with pytest.raises(SystemExit):
        impatient()
    assert cleaned == ["closed"]


def test_signals_during_the_cleanup_after_an_escalation_are_ignored():
    def impatient():
        with stop_on_signals(lambda: None):
            try:
                signal.raise_signal(signal.SIGTERM)
                signal.raise_signal(signal.SIGTERM)
            finally:
                signal.raise_signal(signal.SIGINT)  # the impatient third one: no new exception
                signal.raise_signal(signal.SIGTERM)

    with pytest.raises(SystemExit):
        impatient()


def test_a_new_context_starts_counting_from_zero_again():
    def twice():
        with stop_on_signals(lambda: None):
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGTERM)

    with stop_on_signals(lambda: None):
        signal.raise_signal(signal.SIGTERM)
    with pytest.raises(SystemExit):
        twice()


def test_the_previous_handlers_are_restored_afterwards_even_after_an_error():
    before = {s: signal.getsignal(s) for s in SIGNALS}

    def fail_inside():
        with stop_on_signals(lambda: None):
            assert all(signal.getsignal(s) != before[s] for s in SIGNALS)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fail_inside()
    assert {s: signal.getsignal(s) for s in SIGNALS} == before


def test_outside_the_main_thread_handlers_cannot_be_installed_and_nothing_breaks():
    outcome = []

    def attempt():
        with stop_on_signals(lambda: None):
            outcome.append("ran")

    thread = threading.Thread(target=attempt)
    thread.start()
    thread.join(timeout=5)
    assert outcome == ["ran"]
