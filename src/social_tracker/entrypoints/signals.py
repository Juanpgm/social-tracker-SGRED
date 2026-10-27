"""Turn SIGINT and SIGTERM into a graceful stop request for the worker (PC-6).

The FIRST signal only calls `request_stop`: it never raises into the middle of a collection, so
the unit in flight finishes and is committed before the worker exits. A SECOND signal means the
operator will not wait: it raises `SystemExit` (130 for SIGINT, 143 for SIGTERM, the shell
convention 128 + signal number) from the main thread, which unwinds the stack so `finally` blocks
and context managers (the database among them) still run; the collection in flight is abandoned
and the next worker start counts it as interrupted. Signals that arrive during that cleanup are
ignored. The previous handlers are put back on the way out. Signal handlers can only be installed
from the main thread; elsewhere the context manager does nothing (the caller still has
`request_stop`).
"""

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType

_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@contextmanager
def stop_on_signals(request_stop: Callable[[], object]) -> Iterator[None]:
    received = 0

    def handler(signum: int, frame: FrameType | None) -> None:
        nonlocal received
        received += 1
        if received == 1:
            request_stop()
        elif received == 2:
            raise SystemExit(128 + signum)
        # later ones: the exit is already under way, let the cleanup finish

    previous: dict[int, object] = {}
    try:
        for signum in _STOP_SIGNALS:
            previous[signum] = signal.signal(signum, handler)
    except ValueError:  # not the main thread
        pass
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)  # type: ignore[arg-type]
