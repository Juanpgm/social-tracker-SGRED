"""A control repository wrapper that fails one chosen call, to test atomicity."""

from typing import Any

from social_tracker.control.ports import ControlRepository
from social_tracker.shared.errors import StorageError


class FlakyControl:
    """Delegates to a real control repository but fails the n-th call of one method."""

    def __init__(self, inner: ControlRepository, method: str, on_call: int = 1) -> None:
        self._inner = inner
        self._method = method
        self._on_call = on_call
        self._calls = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != self._method:
            return attr

        def flaky(*args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            if self._calls == self._on_call:
                raise StorageError("simulated storage failure")
            return attr(*args, **kwargs)

        return flaky
