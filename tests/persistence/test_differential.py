"""Differential tests: the SQLite repositories must be indistinguishable from the in-memory fakes.

Random operation sequences (with deliberately awkward text: control characters, NUL, case and
Unicode-composition variants) run against a fake and a fresh SQLite database side by side. Every
step must give the same outcome (value, or exception type and code), and the final observable
state must match. Any divergence is a bug in one of the two implementations.
"""

import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from social_tracker.control.domain import CollectionState, ControlEvent, SourceState
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.shared.errors import DomainError
from social_tracker.sources.domain import normalize_name, parse_source
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.sources.factories import payload

AT = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
IDS = ["src_a", "src_b", "src_c"]
NAME_POOL = [
    "Alpha",
    "ALPHA",
    " alpha ",
    "Straße",
    "STRASSE",
    "é",
    "é",
    "ẛ̣",
    "ṩ",
    "a\x00b",
    "a\tb",
    "a\x7fb",
    "a\x85b",
    "x" * 100,
    "x" * 101,
]
KEY_POOL = ["global_analysis_enabled", "note", "k" * 100, "k" * 101, "", "  ", "a\x00", "a\x1fb"]
KIND_POOL = ["paused", "k" * 64, "k" * 65, "", "a\x00", "a\nb", "a\x85b"]

_TEXT = st.text(max_size=12)
names = st.one_of(st.sampled_from(NAME_POOL), _TEXT)
ids = st.sampled_from(IDS)

SETTLE = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    try:
        value = call()
    except DomainError as exc:
        return ("error", type(exc).__name__, exc.code)
    return ("ok", value)


def dump(value: Any) -> Any:
    if isinstance(value, list):
        return [dump(v) for v in value]
    return value.model_dump() if hasattr(value, "model_dump") else value


# ---- sources ----

source_ops = st.one_of(
    st.tuples(st.just("add"), ids, names, st.booleans()),
    st.tuples(st.just("update"), ids, names, st.booleans()),
    st.tuples(st.just("get"), ids),
    st.tuples(st.just("find"), names),
    st.tuples(st.just("delete"), ids),
    st.tuples(st.just("list")),
)


def run_source_op(repo: Any, op: tuple) -> tuple[str, Any]:
    kind = op[0]
    if kind in ("add", "update"):
        _, source_id, name, enabled = op

        def call() -> None:
            source = parse_source(payload(id=source_id, name=name, enabled=enabled))
            getattr(repo, kind)(source)

        return outcome(call)
    if kind == "get":
        return outcome(lambda: dump(repo.get(op[1])))
    if kind == "find":
        return outcome(lambda: dump(repo.find_by_name_ci(op[1])))
    if kind == "delete":
        return outcome(lambda: repo.delete(op[1]))
    return outcome(lambda: dump(repo.list()))


@SETTLE
@given(ops=st.lists(source_ops, max_size=14))
def test_source_repositories_never_diverge(ops):
    fake = InMemorySourceRepository()
    with tempfile.TemporaryDirectory() as root:
        db = Database(Path(root) / "diff.db")
        try:
            real = SqliteSourceRepository(db)
            for op in ops:
                assert run_source_op(real, op) == run_source_op(fake, op), op
            assert dump(real.list()) == dump(fake.list())
            for source in fake.list():  # the stored normalised name must find the source itself
                assert real.find_by_name_ci(source.normalized_name) == source
                assert fake.find_by_name_ci(source.normalized_name) == source
                assert normalize_name(source.normalized_name) == source.normalized_name
        finally:
            db.close()


# ---- control ----

states = st.one_of(
    st.tuples(st.just(CollectionState.IDLE), st.integers(0, 4), st.just(False)),
    st.tuples(st.just(CollectionState.COLLECTING), st.integers(0, 4), st.just(False)),
    st.tuples(st.just(CollectionState.BACKOFF), st.integers(1, 4), st.just(True)),
    st.tuples(st.just(CollectionState.FAILED), st.just(5), st.just(False)),
    st.tuples(st.just(CollectionState.DISABLED), st.just(0), st.just(False)),
    st.tuples(st.just(CollectionState.IDLE), st.integers(-1, 7), st.booleans()),  # often invalid
)
setting_keys = st.one_of(st.sampled_from(KEY_POOL), _TEXT)
setting_values = st.one_of(st.sampled_from(["true", "false", "", "TRUE", "v" * 1001]), _TEXT)
event_kinds = st.one_of(st.sampled_from(KIND_POOL), _TEXT)
event_sources = st.one_of(st.none(), ids, st.just(""), st.just("BAD id"))

control_ops = st.one_of(
    st.tuples(st.just("set"), setting_keys, setting_values),
    st.tuples(st.just("get_setting"), setting_keys),
    st.tuples(st.just("save_state"), ids, states),
    st.tuples(st.just("get_state"), ids),
    st.tuples(st.just("append"), event_kinds, event_sources, _TEXT),
    st.tuples(st.just("events"), event_sources),
)


def run_control_op(repo: Any, op: tuple) -> tuple[str, Any]:
    kind = op[0]
    if kind == "set":
        return outcome(lambda: repo.set_setting(op[1], op[2]))
    if kind == "get_setting":
        return outcome(lambda: repo.get_setting(op[1]))
    if kind == "save_state":
        _, source_id, (state, failures, with_next_run) = op
        next_run = AT + timedelta(minutes=5) if with_next_run else None
        return outcome(
            lambda: repo.save_source_state(SourceState(source_id, state, failures, next_run))
        )
    if kind == "get_state":
        return outcome(lambda: repo.get_source_state(op[1]))
    if kind == "append":
        _, event_kind, source_id, detail = op
        return outcome(
            lambda: repo.append_control_event(ControlEvent(AT, event_kind, source_id, detail))
        )
    return outcome(lambda: repo.list_control_events(op[1]))


@SETTLE
@given(ops=st.lists(control_ops, max_size=16))
def test_control_repositories_never_diverge(ops):
    fake = InMemoryControlRepository()
    with tempfile.TemporaryDirectory() as root:
        db = Database(Path(root) / "diff.db")
        try:
            real = SqliteControlRepository(db)
            for op in ops:
                assert run_control_op(real, op) == run_control_op(fake, op), op
            assert real.list_control_events() == fake.list_control_events()
            for source_id in IDS:
                assert real.get_source_state(source_id) == fake.get_source_state(source_id)
        finally:
            db.close()
