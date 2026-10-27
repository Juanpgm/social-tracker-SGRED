"""Fixtures of the collector tests."""

import pytest

from tests.collector.harness import FakeFetcher, memory_world, sqlite_world


@pytest.fixture(params=["memory", "sqlite"])
def world(request, tmp_path):
    """The same collector world on the in-memory fakes and on a real SQLite file."""
    fetcher = FakeFetcher()
    built = (
        memory_world(fetcher)
        if request.param == "memory"
        else sqlite_world(tmp_path / "w.db", fetcher)
    )
    yield built
    built.close()
