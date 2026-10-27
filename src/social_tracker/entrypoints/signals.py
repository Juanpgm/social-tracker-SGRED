"""Turn SIGINT and SIGTERM into a graceful stop request for the worker (PC-6).

The handler only calls `request_stop`: it never raises into the middle of a collection, so the
unit in flight finishes and is committed before the worker exits. The previous handlers are put
back on the way out. Signal handlers can only be installed from the main thread; elsewhere the
context manager does nothing (the caller still has `request_stop`).
"""

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType

_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@contextmanager
def stop_on_signals(request_stop: Callable[[], object]) -> Iterator[None]:
    def handler(signum: int, frame: FrameType | None) -> None:
        request_stop()

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
