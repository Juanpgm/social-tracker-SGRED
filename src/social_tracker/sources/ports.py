"""Ports of the source registry. Implementations live in adapters/persistence layers."""

from typing import Protocol, runtime_checkable

from social_tracker.sources.domain import SourceBase


@runtime_checkable
class SourceRepository(Protocol):
    """Persistence port for sources.

    Errors (all `DomainError` subclasses): `ConflictError` on duplicate id or on a name equal
    to another source's name after `normalize_name`; `NotFoundError` for get/update/delete of
    an unknown id. Writes are atomic: a failed call leaves the store unchanged.
    """

    def add(self, source: SourceBase) -> None: ...

    def update(self, source: SourceBase) -> None: ...

    def get(self, source_id: str) -> SourceBase: ...

    def find_by_name_ci(self, name: str) -> SourceBase | None: ...

    def delete(self, source_id: str) -> None: ...

    # Defined last: the name shadows the builtin inside the class body.
    def list(self) -> "list[SourceBase]":
        """All sources, sorted by id."""
        ...
