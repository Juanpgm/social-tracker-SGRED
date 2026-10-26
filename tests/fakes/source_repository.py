"""In-memory `SourceRepository` used by service tests and as the contract reference."""

import threading

from social_tracker.shared.errors import ConflictError, NotFoundError
from social_tracker.sources.domain import Source, normalize_name


def _snapshot(source: Source) -> Source:
    """The store owns its data: what goes in and what comes out are independent copies."""
    return source.model_copy(deep=True)


class InMemorySourceRepository:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, Source] = {}
        self._name_owner: dict[str, str] = {}  # normalized name -> source id

    def add(self, source: Source) -> None:
        with self._lock:
            if source.id in self._by_id:
                raise ConflictError(f"source id already exists: {source.id}")
            if normalize_name(source.name) in self._name_owner:
                raise ConflictError(f"source name already exists: {source.name}")
            self._by_id[source.id] = _snapshot(source)
            self._name_owner[normalize_name(source.name)] = source.id

    def update(self, source: Source) -> None:
        with self._lock:
            current = self._by_id.get(source.id)
            if current is None:
                raise NotFoundError(f"source not found: {source.id}")
            new_key = normalize_name(source.name)
            owner = self._name_owner.get(new_key)
            if owner is not None and owner != source.id:
                raise ConflictError(f"source name already exists: {source.name}")
            del self._name_owner[normalize_name(current.name)]
            self._name_owner[new_key] = source.id
            self._by_id[source.id] = _snapshot(source)

    def get(self, source_id: str) -> Source:
        with self._lock:
            try:
                return _snapshot(self._by_id[source_id])
            except KeyError:
                raise NotFoundError(f"source not found: {source_id}") from None

    def find_by_name_ci(self, name: str) -> Source | None:
        with self._lock:
            owner = self._name_owner.get(normalize_name(name))
            return _snapshot(self._by_id[owner]) if owner is not None else None

    def delete(self, source_id: str) -> None:
        with self._lock:
            current = self._by_id.pop(source_id, None)
            if current is None:
                raise NotFoundError(f"source not found: {source_id}")
            del self._name_owner[normalize_name(current.name)]

    def list(self) -> list[Source]:
        with self._lock:
            return [_snapshot(self._by_id[key]) for key in sorted(self._by_id)]
