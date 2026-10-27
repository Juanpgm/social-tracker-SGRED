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


def test_repeated_signals_request_the_stop_each_time_without_failing():
    requested = []
    with stop_on_signals(lambda: requested.append(1)):
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGTERM)
    assert len(requested) == 3


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
