"""Every service write reads the clock INSIDE its transaction, as the first thing it does.

Why: a timestamp taken before the transaction starts can be older than the timestamp of a
transaction that began later but committed first, so event `at` values would go backwards in
append order. Reading the clock after the write lock is held makes them monotonic.
"""

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from social_tracker.control.domain import CollectionState, SourceState
from social_tracker.control.service import ControlService
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import Clock
from social_tracker.sources.domain import parse_source
from social_tracker.sources.service import SourceRegistry
from social_tracker.sources.yaml_io import dump_sources
from tests.actors import WHO
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.sources.factories import payload

START = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


# ---- a journal of everything that happens, in order ----------------------------------------


class SpyUoW:
    def __init__(self, inner: InMemoryUnitOfWork, journal: list[str]) -> None:
        self._inner, self._journal = inner, journal

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._inner.transaction():
            self._journal.append("begin")
            try:
                yield
            finally:
                self._journal.append("end")


class SpyClock:
    def __init__(self, journal: list[str]) -> None:
        self._journal = journal

    def now(self) -> datetime:
        self._journal.append("now")
        return START


class SpyRepo:
    """Forwards every call and writes its name in the journal."""

    def __init__(self, inner: Any, journal: list[str], label: str) -> None:
        self._inner, self._journal, self._label = inner, journal, label

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def spied(*args: Any, **kwargs: Any) -> Any:
            self._journal.append(f"{self._label}.{name}")
            return attr(*args, **kwargs)

        return spied


class Spied:
    def __init__(self) -> None:
        self.journal: list[str] = []
        sources, control = InMemorySourceRepository(), InMemoryControlRepository()
        self.sources, self.control = sources, control
        uow = SpyUoW(InMemoryUnitOfWork([sources, control]), self.journal)
        spy_sources = SpyRepo(sources, self.journal, "sources")
        spy_control = SpyRepo(control, self.journal, "control")
        clock = SpyClock(self.journal)
        self.registry = SourceRegistry(spy_sources, spy_control, uow, clock)  # type: ignore[arg-type]
        self.service = ControlService(spy_control, spy_sources, uow, clock)  # type: ignore[arg-type]

    def seed_running_source(self, source_id: str = "src_run") -> None:
        """A source that is enabled and idle, written straight into the stores (no journal)."""
        source = parse_source(payload(id=source_id, name=f"Name {source_id}", enabled=True))
        self.sources.add(source)
        self.control.save_source_state(SourceState(source_id, CollectionState.IDLE))


def _registry_add(w: Spied) -> None:
    w.registry.add(payload(id="src_new", name="New"), **WHO)  # supplied timestamps are ignored


def _registry_edit(w: Spied) -> None:
    w.seed_running_source()
    w.registry.edit("src_run", {"name": "Renamed"}, **WHO)


def _registry_edit_noop(w: Spied) -> None:
    w.seed_running_source()
    w.registry.edit("src_run", {}, **WHO)


def _registry_enable(w: Spied) -> None:
    w.sources.add(parse_source(payload(id="src_off", name="Off")))
    w.registry.enable("src_off", **WHO)


def _registry_disable(w: Spied) -> None:
    w.seed_running_source()
    w.registry.disable("src_run", **WHO)


def _registry_delete(w: Spied) -> None:
    w.seed_running_source()
    w.registry.delete("src_run", **WHO)


def _registry_import(w: Spied) -> None:
    text = dump_sources([parse_source(payload(id="src_imp", name="Imported"))])
    w.registry.import_sources(text, **WHO)


def _registry_import_dry_run(w: Spied) -> None:
    text = dump_sources([parse_source(payload(id="src_imp", name="Imported"))])
    w.registry.import_sources(text, dry_run=True)


def _control_global(w: Spied) -> None:
    w.service.set_global_analysis(True, **WHO)


def _control_global_noop(w: Spied) -> None:
    w.service.set_global_analysis(False, **WHO)


def _control_source_flag(w: Spied) -> None:
    w.seed_running_source()
    w.service.set_source_analysis("src_run", True, **WHO)


def _control_collection(w: Spied) -> None:
    w.seed_running_source()
    w.service.start_collection("src_run", **WHO)


def _control_threshold(w: Spied) -> None:
    w.service.set_alert_min_confidence(0.9, **WHO)


def _control_threshold_noop(w: Spied) -> None:
    w.service.set_alert_min_confidence(0.5, **WHO)


WRITES: dict[str, Callable[[Spied], None]] = {
    "registry.add": _registry_add,
    "registry.edit": _registry_edit,
    "registry.edit-noop": _registry_edit_noop,
    "registry.enable": _registry_enable,
    "registry.disable": _registry_disable,
    "registry.delete": _registry_delete,
    "registry.import": _registry_import,
    "registry.import-dry-run": _registry_import_dry_run,
    "control.global": _control_global,
    "control.global-noop": _control_global_noop,
    "control.source-flag": _control_source_flag,
    "control.collection": _control_collection,
    "control.threshold": _control_threshold,
    "control.threshold-noop": _control_threshold_noop,
}


@pytest.mark.parametrize("name", list(WRITES))
def test_every_write_reads_the_clock_once_and_first_inside_its_transaction(name: str):
    spied = Spied()  # seeds go straight to the stores, so the journal holds only the operation
    WRITES[name](spied)
    journal = spied.journal
    assert journal.count("now") == 1, journal
    assert journal[:2] == ["begin", "now"], journal
    assert journal[-1] == "end", journal


@pytest.mark.parametrize("name", list(WRITES))
def test_no_clock_read_happens_outside_a_transaction(name: str):
    spied = Spied()
    WRITES[name](spied)
    depth = 0
    for entry in spied.journal:
        depth += entry == "begin"
        depth -= entry == "end"
        if entry == "now":
            assert depth >= 1, spied.journal


# ---- event timestamps stay monotonic under contention --------------------------------------


class GateClock:
    """Strictly increasing times. The FIRST reader lingers (bounded) until a second thread has
    read too, which is exactly the interleaving that reorders timestamps when the clock is read
    before the write lock is taken. Inside a transaction no second reader can arrive, so the wait
    simply times out."""

    def __init__(self, wait_s: float = 0.3) -> None:
        self._lock = threading.Lock()
        self._reads = 0
        self._second_arrived = threading.Event()
        self._wait_s = wait_s

    def now(self) -> datetime:
        with self._lock:
            self._reads += 1
            number = self._reads
        stamp = START + timedelta(seconds=number)
        if number == 1:
            self._second_arrived.wait(self._wait_s)
            time.sleep(0.05)
        else:
            self._second_arrived.set()
        return stamp


class Stack:
    def __init__(self, kind: str, tmp_path: Any, clock: Clock) -> None:
        self.db: Database | None = None
        if kind == "sqlite":
            self.db = Database(tmp_path / "clock.db")
            self.sources = SqliteSourceRepository(self.db)
            self.control = SqliteControlRepository(self.db)
            self.uow: Any = SqliteUnitOfWork(self.db)
        else:
            self.sources, self.control = InMemorySourceRepository(), InMemoryControlRepository()
            self.uow = InMemoryUnitOfWork([self.sources, self.control])
        self.registry = SourceRegistry(self.sources, self.control, self.uow, clock)
        self.service = ControlService(self.control, self.sources, self.uow, clock)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()

    def event_times(self) -> list[datetime]:
        return [event.at for event in self.control.list_control_events()]


def _two_threads(first: Callable[[], Any], second: Callable[[], Any]) -> list[BaseException]:
    errors: list[BaseException] = []

    def guarded(call: Callable[[], Any]) -> Callable[[], None]:
        def run() -> None:
            try:
                call()
            except BaseException as exc:  # noqa: BLE001 - reported to the test
                errors.append(exc)

        return run

    threads = [threading.Thread(target=guarded(c)) for c in (first, second)]
    threads[0].start()
    time.sleep(0.02)  # the first thread reaches the clock first
    threads[1].start()
    for thread in threads:
        thread.join(30)
    return errors


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_registry_event_times_never_go_backwards_in_append_order(kind: str, tmp_path):
    stack = Stack(kind, tmp_path, GateClock())
    try:
        first = payload(id="src_a", name="A")
        second = payload(id="src_b", name="B")
        for item in (first, second):
            del item["created_at"], item["updated_at"]
        errors = _two_threads(
            lambda: stack.registry.add(first, **WHO), lambda: stack.registry.add(second, **WHO)
        )
        assert errors == []
        times = stack.event_times()
        assert len(times) == 2
        assert times == sorted(times)
    finally:
        stack.close()


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_control_event_times_never_go_backwards_in_append_order(kind: str, tmp_path):
    stack = Stack(kind, tmp_path, GateClock())
    try:
        for source_id in ("src_a", "src_b"):
            stack.sources.add(parse_source(payload(id=source_id, name=f"Name {source_id}")))
        errors = _two_threads(
            lambda: stack.service.set_source_analysis("src_a", True, **WHO),
            lambda: stack.service.set_source_analysis("src_b", True, **WHO),
        )
        assert errors == []
        times = stack.event_times()
        assert len(times) == 2
        assert times == sorted(times)
    finally:
        stack.close()
