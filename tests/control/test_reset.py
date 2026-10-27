"""Resetting a failed source: audited, idempotent, and it clears the stale error shown for it.

A source that failed five times is left alone until the operator acts. `reset_if_failed` is the
one entry point of the CLI and the web UI: it moves a failed source to idle, forgets the failure
count AND the stale `last_error`, and is a harmless no-op for any other state.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.control.domain import (
    MAX_CONSECUTIVE_FAILURES,
    CollectionState,
    FetchStatus,
    SourceState,
)
from social_tracker.control.service import ControlService
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import NotFoundError, ValidationError
from social_tracker.sources.service import SourceRegistry
from tests.actors import WHO
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.sources.factories import payload

NOW = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
FETCHED = NOW - timedelta(hours=2)


class Rig:
    def __init__(self) -> None:
        self.sources, self.repo = InMemorySourceRepository(), InMemoryControlRepository()
        uow, self.clock = InMemoryUnitOfWork([self.sources, self.repo]), FixedClock(NOW)
        self.registry = SourceRegistry(self.sources, self.repo, uow, self.clock)
        self.service = ControlService(self.repo, self.sources, uow, self.clock)
        self.registry.add(payload(id="src_one"), **WHO)
        self.registry.enable("src_one", **WHO)

    def fail_completely(self) -> None:
        self.repo.save_source_state(
            SourceState("src_one", CollectionState.FAILED, MAX_CONSECUTIVE_FAILURES, None)
        )
        self.repo.save_fetch_status(
            FetchStatus("src_one", FETCHED, "http 503", etag='"v1"', last_modified="Sat, 01 Mar")
        )

    def kinds(self) -> list[str]:
        return [e.kind for e in self.repo.list_control_events("src_one")]


@pytest.fixture
def rig() -> Rig:
    return Rig()


def test_a_failed_source_is_reset_to_idle_and_audited(rig):
    rig.fail_completely()
    change = rig.service.reset_if_failed("src_one", actor="cli:ana", reason="fixed the feed")
    assert change.changed is True
    state = rig.service.get_state("src_one")
    assert (state.state, state.consecutive_failures) == (CollectionState.IDLE, 0)
    last = rig.repo.list_control_events("src_one")[-1]
    assert last.kind == "collection_reset"
    detail = json.loads(last.detail)
    assert (detail["actor"], detail["reason"]) == ("cli:ana", "fixed the feed")
    assert (detail["from"], detail["to"]) == ("failed", "idle")


def test_the_reset_clears_the_stale_error_but_keeps_the_rest_of_the_fetch_status(rig):
    rig.fail_completely()
    rig.service.reset_if_failed("src_one", **WHO)
    assert rig.repo.get_fetch_status("src_one") == FetchStatus(
        "src_one", FETCHED, None, etag='"v1"', last_modified="Sat, 01 Mar"
    )


def test_a_failed_source_without_any_fetch_status_is_reset_too(rig):
    rig.repo.save_source_state(
        SourceState("src_one", CollectionState.FAILED, MAX_CONSECUTIVE_FAILURES, None)
    )
    assert rig.service.reset_if_failed("src_one", **WHO).changed is True
    assert rig.repo.get_fetch_status("src_one") is None


def test_resetting_twice_is_an_idempotent_no_op_the_second_time(rig):
    rig.fail_completely()
    rig.service.reset_if_failed("src_one", **WHO)
    events = rig.kinds()
    again = rig.service.reset_if_failed("src_one", **WHO)
    assert again.changed is False
    assert again.state.state is CollectionState.IDLE
    assert rig.kinds() == events


@pytest.mark.parametrize(
    "state",
    [
        SourceState("src_one", CollectionState.IDLE),
        SourceState("src_one", CollectionState.COLLECTING, 2),
        SourceState("src_one", CollectionState.BACKOFF, 2, NOW + timedelta(minutes=5)),
    ],
    ids=["idle", "collecting", "backoff"],
)
def test_any_other_state_is_left_untouched_with_its_error(rig, state):
    rig.repo.save_source_state(state)
    rig.repo.save_fetch_status(FetchStatus("src_one", FETCHED, "timeout"))
    before = rig.kinds()
    change = rig.service.reset_if_failed("src_one", **WHO)
    assert change.changed is False
    assert rig.service.get_state("src_one") == state
    assert rig.repo.get_fetch_status("src_one").last_error == "timeout"
    assert rig.kinds() == before


def test_a_disabled_source_is_not_reset(rig):
    rig.registry.disable("src_one", **WHO)
    assert rig.service.reset_if_failed("src_one", **WHO).changed is False


def test_an_unknown_or_malformed_source_is_refused(rig):
    with pytest.raises(NotFoundError):
        rig.service.reset_if_failed("src_missing", **WHO)
    with pytest.raises(ValidationError):
        rig.service.reset_if_failed("Not A Slug", **WHO)


def test_a_reset_needs_an_actor_and_a_reason(rig):
    rig.fail_completely()
    with pytest.raises(ValidationError):
        rig.service.reset_if_failed("src_one", actor="", reason="x")
    with pytest.raises(ValidationError):
        rig.service.reset_if_failed("src_one", actor="a", reason="   ")
    assert rig.service.get_state("src_one").state is CollectionState.FAILED


@pytest.mark.parametrize("state", ["idle", "disabled"])
def test_the_actor_and_reason_are_checked_even_when_there_is_nothing_to_reset(rig, state):
    if state == "disabled":
        rig.registry.disable("src_one", **WHO)
    with pytest.raises(ValidationError):
        rig.service.reset_if_failed("src_one", actor="a", reason="   ")
    with pytest.raises(ValidationError):
        rig.service.reset_if_failed("src_one", actor="", reason="x")


def test_reset_failed_itself_clears_the_error_and_still_refuses_other_states(rig):
    rig.fail_completely()
    rig.service.reset_failed("src_one", **WHO)
    assert rig.repo.get_fetch_status("src_one").last_error is None
    from social_tracker.shared.errors import InvalidTransition

    with pytest.raises(InvalidTransition):
        rig.service.reset_failed("src_one", **WHO)


# ---- enabling a source forgets what went wrong before ----


def test_enabling_a_source_clears_its_stale_error(rig):
    rig.registry.disable("src_one", **WHO)
    rig.repo.save_fetch_status(FetchStatus("src_one", FETCHED, "http 503", etag='"v1"'))
    rig.registry.enable("src_one", **WHO)
    status = rig.repo.get_fetch_status("src_one")
    assert (status.last_error, status.last_fetch_at, status.etag) == (None, FETCHED, '"v1"')


def test_disabling_a_source_keeps_its_last_error_for_the_operator_to_read(rig):
    rig.repo.save_fetch_status(FetchStatus("src_one", FETCHED, "http 503"))
    rig.registry.disable("src_one", **WHO)
    assert rig.repo.get_fetch_status("src_one").last_error == "http 503"


def test_enabling_an_enabled_source_changes_nothing_including_its_error(rig):
    rig.repo.save_fetch_status(FetchStatus("src_one", FETCHED, "timeout"))
    rig.registry.enable("src_one", **WHO)  # already enabled: a no-op
    assert rig.repo.get_fetch_status("src_one").last_error == "timeout"


def test_enabling_a_source_that_never_fetched_creates_no_fetch_status(rig):
    rig.registry.disable("src_one", **WHO)
    rig.registry.enable("src_one", **WHO)
    assert rig.repo.get_fetch_status("src_one") is None


def test_the_error_of_a_failed_source_disappears_after_disable_and_enable(rig):
    rig.fail_completely()
    rig.registry.disable("src_one", **WHO)
    rig.registry.enable("src_one", **WHO)
    assert rig.repo.get_fetch_status("src_one").last_error is None
    assert rig.service.get_state("src_one").state is CollectionState.IDLE
