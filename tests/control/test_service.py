"""`ControlService` contract on the in-memory fakes and on SQLite, plus SQLite-only checks."""

import threading

import pytest

from social_tracker.control.domain import (
    SETTING_ALERT_MIN_CONFIDENCE,
    CollectionState,
)
from social_tracker.control.service import ControlService
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import StorageError
from social_tracker.sources.domain import parse_source
from tests.control.service_contract import (
    START,
    WHO,
    ControlServiceContract,
    Env,
    assert_chains_are_consistent,
    assert_log_reproduces_state,
)
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.sources.factories import payload


class TestInMemoryControlService(ControlServiceContract):
    def corrupt_setting(self, env: Env, key: str, value: str) -> None:
        env.control._settings[key] = value  # type: ignore[attr-defined]

    def make_env(self, service_cls: type[ControlService] = ControlService) -> Env:
        sources, control = InMemorySourceRepository(), InMemoryControlRepository()
        uow, clock = InMemoryUnitOfWork([sources, control]), FixedClock(START)
        service = service_cls(control, sources, uow, clock)
        return Env(service, control, sources, uow, clock, service_cls)


def sqlite_env(db: Database, service_cls: type[ControlService] = ControlService) -> Env:
    sources, control = SqliteSourceRepository(db), SqliteControlRepository(db)
    uow, clock = SqliteUnitOfWork(db), FixedClock(START)
    return Env(service_cls(control, sources, uow, clock), control, sources, uow, clock, service_cls)


class TestSqliteControlService(ControlServiceContract):
    @pytest.fixture(autouse=True)
    def _databases(self, tmp_path):
        self._root, self._opened = tmp_path, []
        yield
        for db in self._opened:
            db.close()

    def corrupt_setting(self, env: Env, key: str, value: str) -> None:
        with self._opened[-1].transaction() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def make_env(self, service_cls: type[ControlService] = ControlService) -> Env:
        db = Database(self._root / f"control-{len(self._opened)}.db")
        self._opened.append(db)
        return sqlite_env(db, service_cls)


# ---- SQLite-only: durability and corruption ------------------------------------------


def test_everything_the_service_decided_survives_closing_and_reopening(tmp_path):
    path = tmp_path / "durable.db"
    with Database(path) as first:
        env = sqlite_env(first)
        env.add_source("src_one")
        env.service.set_global_analysis(True, **WHO)
        env.service.set_source_analysis("src_one", True, **WHO)
        env.enable_source("src_one")
        env.service.start_collection("src_one", **WHO)
        env.service.set_alert_min_confidence(0.35, **WHO)
        kinds_before = env.kinds()

    with Database(path) as second:
        env = sqlite_env(second)
        assert env.service.global_analysis_enabled() is True
        assert env.service.effective_analysis("src_one") is True
        assert env.service.alert_min_confidence() == 0.35
        assert env.service.get_state("src_one").state is CollectionState.COLLECTING
        assert env.kinds() == kinds_before


def test_a_fresh_database_defaults_to_paused_and_half(tmp_path):
    with Database(tmp_path / "fresh.db") as db:
        env = sqlite_env(db)
        assert env.service.global_analysis_enabled() is False
        assert env.service.alert_min_confidence() == 0.5


def test_a_corrupted_stored_threshold_is_a_storage_error_not_a_wrong_number(tmp_path):
    with Database(tmp_path / "corrupt.db") as db:
        env = sqlite_env(db)
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)",
                (SETTING_ALERT_MIN_CONFIDENCE, "abc"),
            )
        with pytest.raises(StorageError):
            env.service.alert_min_confidence()


def test_concurrent_writers_never_leak_a_database_is_locked_error(tmp_path):
    with Database(tmp_path / "busy.db") as db:
        env = sqlite_env(db)
        for sid in ("src_a", "src_b"):
            env.sources.add(parse_source(payload(id=sid, name=sid)))
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def worker(sid: str, flip: bool) -> None:
            barrier.wait(10)
            try:
                for i in range(25):
                    env.service.set_source_analysis(sid, (i % 2 == 0) == flip, **WHO)
                    env.service.set_alert_min_confidence(((i + flip) % 10) / 10, **WHO)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(sid, flip))
            for sid in ("src_a", "src_b")
            for flip in (True, False)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        assert errors == []
        assert_log_reproduces_state(env)
        assert_chains_are_consistent(env)
        assert env.control.get_source_state("src_a") is None


class _CorruptThreshold:
    """A control repository whose stored threshold is garbage (as a tampered file could be)."""

    def get_setting(self, key: str) -> str | None:
        return "not-a-number"


def test_a_garbage_stored_threshold_is_a_storage_error_with_any_repository():
    service = ControlService(_CorruptThreshold(), None, None, FixedClock(START))  # type: ignore[arg-type]
    with pytest.raises(StorageError) as caught:
        service.alert_min_confidence()
    assert "not-a-number" not in caught.value.message


class _NothingStored:
    def get_setting(self, key: str) -> str | None:
        return None


def test_without_any_stored_value_the_threshold_is_the_default_half():
    service = ControlService(_NothingStored(), None, None, FixedClock(START))  # type: ignore[arg-type]
    assert service.alert_min_confidence() == 0.5


# ---- SQLite-only: the overview is one snapshot, in a fixed number of statements -------


class _WriteOnceAfterStates:
    """A control repository that lets another connection commit right after the states are read,
    which is where the old, unsnapshotted overview could be torn."""

    def __init__(self, inner, concurrent_write) -> None:
        self._inner, self._concurrent_write = inner, concurrent_write

    def list_source_states(self):
        states = self._inner.list_source_states()
        self._concurrent_write()
        return states

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_source_committed_between_the_reads_of_an_overview_is_not_half_shown(tmp_path):
    path = tmp_path / "torn.db"
    with Database(path) as reader, Database(path) as writer:
        env = sqlite_env(reader)
        env.add_source("src_one")
        env.enable_source("src_one")
        other = sqlite_env(writer)

        def commit_a_running_source() -> None:
            other.add_source("src_two")
            other.enable_source("src_two")

        service = ControlService(
            _WriteOnceAfterStates(env.control, commit_a_running_source),
            env.sources,
            env.uow,
            env.clock,
        )
        overview = service.overview()

        # one point in time: src_two is wholly absent, never "enabled" with a missing state
        assert [o.source.id for o in overview.sources] == ["src_one"]
        assert all(
            o.state is not CollectionState.DISABLED for o in overview.sources if o.source.enabled
        )
        # the next overview sees it, complete
        later = env.service.overview()
        assert [(o.source.id, o.source.enabled, o.state) for o in later.sources] == [
            ("src_one", True, CollectionState.IDLE),
            ("src_two", True, CollectionState.IDLE),
        ]


def _statements_for_an_overview(tmp_path, count: int) -> list[str]:
    with Database(tmp_path / f"n{count}.db") as db:
        env = sqlite_env(db)
        for n in range(count):
            env.add_source(f"src_{n:02d}")
            env.enable_source(f"src_{n:02d}")
            env.service.set_source_analysis(f"src_{n:02d}", True, **WHO)
        seen: list[str] = []
        db._conn.set_trace_callback(seen.append)  # type: ignore[union-attr]
        assert len(env.service.overview().sources) == count
        db._conn.set_trace_callback(None)  # type: ignore[union-attr]
        return seen


def test_the_overview_issues_the_same_statements_for_one_source_and_for_thirty(tmp_path):
    few = _statements_for_an_overview(tmp_path, 1)
    many = _statements_for_an_overview(tmp_path, 30)
    assert few  # the trace really recorded something
    assert len(many) == len(few)
    assert [sql.split()[0] for sql in many] == [sql.split()[0] for sql in few]
