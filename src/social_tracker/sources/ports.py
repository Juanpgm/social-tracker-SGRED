"""Ports of the source registry. Implementations live in adapters/persistence layers."""

from typing import Protocol, runtime_checkable

from social_tracker.sources.domain import Source


@runtime_checkable
class SourceRepository(Protocol):
    """Persistence port for sources.

    Errors (all `DomainError` subclasses): `ConflictError` on duplicate id or on a name equal
    to another source's name after `normalize_name`; `NotFoundError` for get/update/delete of
    an unknown id. Writes are atomic: a failed call leaves the store unchanged.

    The store owns its data: objects handed in are copied, and every read returns a fresh,
    independent object, so tampering with one never alters what is stored.
    """

    def add(self, source: Source) -> None: ...

    def update(self, source: Source) -> None: ...

    def get(self, source_id: str) -> Source: ...

    def find_by_name_ci(self, name: str) -> Source | None: ...

    def delete(self, source_id: str) -> None: ...

    # Defined last: the name shadows the builtin inside the class body.
    def list(self) -> "list[Source]":
        """All sources, sorted by id."""
        ...
