"""Ports of the gold set."""

from typing import Protocol, runtime_checkable

from social_tracker.classification.domain import Label
from social_tracker.gold.domain import GoldExample


@runtime_checkable
class GoldRepository(Protocol):
    """Persistence port of the labeled texts. An example is keyed by the id its text makes, so
    `add` never overwrites: the same text again leaves the stored example as it is. Examples come
    back in the order they were added. Every row read back is rebuilt through the domain, so a
    tampered row is a `StorageError`."""

    def add(self, example: GoldExample) -> bool:
        """Store the example unless its text is already there: True when stored."""
        ...

    def get(self, example_id: str) -> GoldExample | None: ...

    def count(self) -> int: ...

    def remove(self, example_id: str) -> bool:
        """Forget an example; False if there is none."""
        ...

    # Defined last: the name shadows the builtin inside the class body.
    def list(self, *, label: Label | None = None, limit: int, offset: int = 0) -> list[GoldExample]:
        """Up to `limit` (1..5000) examples after skipping `offset`, in the order added; only
        those with `label` when it is given."""
        ...
