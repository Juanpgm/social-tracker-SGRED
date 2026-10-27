"""In-memory `UnitOfWork` for service tests.

It snapshots the state of the given in-memory repositories when a block starts and restores
it if the block fails, so rollback behaves like the SQLite implementation: an outermost block
is all-or-nothing, a nested block is a savepoint. One re-entrant lock serialises threads.
"""

import copy
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

_LOCK_TYPES = (type(threading.Lock()), type(threading.RLock()))


def _capture(participant: object) -> dict[str, Any]:
    return {
        name: copy.deepcopy(value)
        for name, value in vars(participant).items()
        if not isinstance(value, _LOCK_TYPES)
    }


class InMemoryUnitOfWork:
    def __init__(self, participants: Sequence[object]) -> None:
        self._participants = tuple(participants)
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            snapshots = [(p, _capture(p)) for p in self._participants]
            try:
                yield
            except BaseException:
                for participant, state in snapshots:
                    for name, value in state.items():
                        setattr(participant, name, value)
                raise

    @contextmanager
    def snapshot(self) -> Iterator[None]:
        """Holding the lock is a consistent view: no other thread's block can run meanwhile."""
        with self._lock:
            yield
