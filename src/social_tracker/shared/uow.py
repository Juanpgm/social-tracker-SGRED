"""Transaction-context port: lets services group repository writes without knowing the store."""

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable


@runtime_checkable
class UnitOfWork(Protocol):
    """Everything done inside `with uow.transaction():` commits together or not at all.

    Contract (pinned by tests that run against every implementation):
    - an exception leaving the block rolls back every write made inside it and propagates
      unchanged (same object); nothing, `BaseException` included, is swallowed;
    - nested blocks on the same thread JOIN the outer one: a nested failure undoes only the
      nested writes (so the outer block may catch it and carry on), and an outer failure undoes
      the nested writes too;
    - other threads wait until the owning thread's outermost block ends.
    """

    def transaction(self) -> AbstractContextManager[None]: ...
