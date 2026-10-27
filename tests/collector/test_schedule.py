"""Which sources are due for a collection (CI-4): the pure scheduling rule."""

from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.collector.domain import is_due
from social_tracker.control.domain import CollectionState, FetchStatus, SourceState
from social_tracker.sources.domain import parse_source
from tests.sources.factories import payload

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


def source(type_: str = "news_rss", **overrides):
    return parse_source(payload(type_, enabled=True, **overrides))


def idle() -> SourceState:
    return SourceState("src_one", CollectionState.IDLE)


def fetched(ago: timedelta | None) -> FetchStatus:
    return FetchStatus("src_one", None if ago is None else NOW - ago)


def test_an_idle_source_that_was_never_fetched_is_due():
    assert is_due(source(), idle(), None, NOW) is True
    assert is_due(source(), idle(), fetched(None), NOW) is True


@pytest.mark.parametrize(
    ("ago_s", "due"),
    [(0, False), (599, False), (599.999, False), (600, True), (601, True), (86_400, True)],
)
def test_an_idle_source_is_due_once_its_poll_interval_has_passed_default_ten_minutes(ago_s, due):
    assert is_due(source(), idle(), fetched(timedelta(seconds=ago_s)), NOW) is due


@pytest.mark.parametrize(("ago_s", "due"), [(1799, False), (1800, True)])
def test_the_poll_interval_comes_from_the_source_configuration(ago_s, due):
    configured = source(poll_interval_s=1800)
    assert is_due(configured, idle(), fetched(timedelta(seconds=ago_s)), NOW) is due


def test_a_fetch_in_the_future_does_not_make_a_source_due():
    status = FetchStatus("src_one", NOW + timedelta(hours=1))
    assert is_due(source(), idle(), status, NOW) is False


@pytest.mark.parametrize(("offset_s", "due"), [(-1, False), (0, True), (1, True)])
def test_a_source_in_backoff_is_due_when_its_retry_time_has_come(offset_s, due):
    retry = NOW - timedelta(seconds=offset_s)
    state = SourceState("src_one", CollectionState.BACKOFF, 2, retry)
    assert is_due(source(), state, fetched(timedelta(hours=5)), NOW) is due


@pytest.mark.parametrize(
    "state",
    [
        SourceState("src_one", CollectionState.COLLECTING),
        SourceState("src_one", CollectionState.FAILED, 5),
        SourceState("src_one", CollectionState.DISABLED),
        None,
    ],
    ids=["collecting", "failed", "disabled", "no-state"],
)
def test_sources_that_are_busy_stopped_or_without_state_are_never_due(state):
    assert is_due(source(), state, None, NOW) is False


def test_a_disabled_source_is_never_due_whatever_its_state_says():
    off = parse_source(payload(enabled=False))
    assert is_due(off, idle(), None, NOW) is False


@pytest.mark.parametrize("type_", ["instagram", "live_stream", "recorded_audio"])
def test_only_rss_sources_are_collected_by_this_collector(type_):
    assert is_due(source(type_), idle(), None, NOW) is False
