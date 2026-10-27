"""`Source.enabled` is the only switch; the collection state follows it (and never the reverse).

Invariant: for every source, `source.enabled == (collection state is not disabled)`, and a missing
state row means disabled. These are the end-to-end scenarios that cross the registry and the
control service; the random walk over both lives in tests/control/test_service_stateful.py.
"""

import threading
from collections.abc import Iterator
from functools import partial

import pytest

from social_tracker.control.domain import CollectionState, source_analysis_key
from social_tracker.persistence.db import Database
from social_tracker.shared.errors import InvalidTransition
from social_tracker.sources.domain import parse_source
from social_tracker.sources.service import SourceRegistry
from social_tracker.sources.yaml_io import dump_sources
from tests.actors import WHO
from tests.control import test_service as service_suites
from tests.control.service_contract import (
    Env,
    assert_chains_are_consistent,
    assert_enabled_matches_state,
    assert_log_reproduces_state,
    replay,
)
from tests.sources.factories import payload
from tests.sources.registry_contract import data, race


class Stack:
    def __init__(self, env: Env) -> None:
        self.env = env
        self.registry = SourceRegistry(env.sources, env.control, env.uow, env.clock)
        self.service = env.service

    def state(self, source_id: str = "src_one") -> CollectionState:
        row = self.env.control.get_source_state(source_id)
        return row.state if row else CollectionState.DISABLED

    def check(self) -> None:
        assert_enabled_matches_state(self.env)
        assert_log_reproduces_state(self.env)
        assert_chains_are_consistent(self.env)


@pytest.fixture(params=["memory", "sqlite"])
def stack(request, tmp_path) -> Iterator[Stack]:
    if request.param == "memory":
        yield Stack(service_suites.TestInMemoryControlService().make_env())
        return
    db = Database(tmp_path / "invariant.db")
    yield Stack(service_suites.sqlite_env(db))
    db.close()


def test_disabling_a_collecting_source_stops_it_and_blocks_the_worker(stack: Stack):
    stack.registry.add(data(enabled=True), **WHO)
    stack.service.start_collection("src_one", **WHO)
    assert stack.state() is CollectionState.COLLECTING
    stack.registry.disable("src_one", **WHO)
    assert stack.state() is CollectionState.DISABLED
    for finish in (stack.service.finish_collection, stack.service.start_collection):
        with pytest.raises(InvalidTransition):
            finish("src_one", **WHO)
    stack.check()


def test_a_disabled_source_is_never_effectively_analysed_whatever_the_flags_say(stack: Stack):
    stack.registry.add(data(enabled=True), **WHO)
    stack.service.set_global_analysis(True, **WHO)
    stack.service.set_source_analysis("src_one", True, **WHO)
    assert stack.service.effective_analysis("src_one") is True
    stack.registry.disable("src_one", **WHO)
    assert stack.service.effective_analysis("src_one") is False
    assert stack.service.source_analysis_enabled("src_one") is True  # the flag itself is kept
    stack.registry.enable("src_one", **WHO)
    assert stack.service.effective_analysis("src_one") is True
    stack.check()


def test_importing_an_enabled_entry_makes_the_source_runnable_at_once(stack: Stack):
    entry = parse_source(payload(id="src_one", name="One", enabled=True))
    stack.registry.import_sources(dump_sources([entry]), **WHO)
    assert stack.state() is CollectionState.IDLE
    stack.service.start_collection("src_one", **WHO)
    stack.check()


def test_importing_an_entry_without_enabled_stops_a_running_source(stack: Stack):
    stack.registry.add(data(enabled=True), **WHO)
    stack.service.start_collection("src_one", **WHO)
    hand_written = (
        "sources:\n- id: src_one\n  type: news_rss\n  name: One\n"
        "  feed_url: https://example.org/rss.xml\n"
    )
    stack.registry.import_sources(hand_written, **WHO)
    assert stack.env.sources.get("src_one").enabled is False
    assert stack.state() is CollectionState.DISABLED
    stack.check()


def test_replaying_the_log_across_a_delete_drops_the_state_and_flag_the_delete_recorded(
    stack: Stack,
):
    # The exact rule: replaying reproduces the state of every source that EXISTS. A delete removes
    # the state row and the flag row without a collection event, so `source_deleted` carries the
    # previous state and analysis value and the replay drops both.
    stack.registry.add(data(enabled=True), **WHO)
    stack.service.start_collection("src_one", **WHO)
    stack.service.set_source_analysis("src_one", True, **WHO)
    assert replay(stack.env).states["src_one"][0] == "collecting"
    stack.registry.delete("src_one", **WHO)
    detail = stack.env.events()[-1][2]
    assert (detail["state"], detail["analysis"]) == ("collecting", "true")
    replayed = replay(stack.env)
    assert "src_one" not in replayed.states
    assert "src_one" not in replayed.source_flags
    assert stack.env.control.get_source_state("src_one") is None
    assert stack.env.control.get_setting(source_analysis_key("src_one")) is None
    stack.check()


def test_deleting_and_re_adding_a_source_starts_from_a_clean_disabled_slate(stack: Stack):
    stack.registry.add(data(enabled=True), **WHO)
    stack.service.start_collection("src_one", **WHO)
    stack.service.set_source_analysis("src_one", True, **WHO)
    stack.registry.delete("src_one", **WHO)
    stack.registry.add(data(), **WHO)
    assert stack.state() is CollectionState.DISABLED
    assert stack.service.source_analysis_enabled("src_one") is False
    stack.check()


@pytest.mark.parametrize("trial", range(8))
def test_disabling_while_a_worker_starts_ends_consistent_whoever_wins(stack: Stack, trial: int):
    stack.registry.add(data(enabled=True), **WHO)
    results = race(
        [
            partial(stack.registry.disable, "src_one", **WHO),
            partial(stack.service.start_collection, "src_one", **WHO),
        ]
    )
    for kind, value in results:
        assert kind == "ok" or isinstance(value, InvalidTransition), results
    assert stack.state() is CollectionState.DISABLED
    assert stack.env.sources.get("src_one").enabled is False
    stack.check()


@pytest.mark.parametrize("trial", range(8))
def test_enabling_while_a_worker_starts_ends_consistent_whoever_wins(stack: Stack, trial: int):
    stack.registry.add(data(), **WHO)
    results = race(
        [
            partial(stack.registry.enable, "src_one", **WHO),
            partial(stack.service.start_collection, "src_one", **WHO),
        ]
    )
    for kind, value in results:
        assert kind == "ok" or isinstance(value, InvalidTransition), results
    assert stack.env.sources.get("src_one").enabled is True
    assert stack.state() in (CollectionState.IDLE, CollectionState.COLLECTING)
    stack.check()


def test_many_threads_flipping_the_flag_never_leave_the_two_stores_apart(stack: Stack):
    stack.registry.add(data(), **WHO)
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def flip(turn: int) -> None:
        barrier.wait(10)
        try:
            for i in range(15):
                if (i + turn) % 2:
                    stack.registry.enable("src_one", **WHO)
                else:
                    stack.registry.disable("src_one", **WHO)
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            errors.append(exc)

    threads = [threading.Thread(target=flip, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert errors == []
    stack.check()
