"""A5/A6: what the operator pages need to know about the worker and the alert step (when the worker
last passed, when the alerts were last refreshed, with which threshold) is kept in the settings
store: telemetry, not an operator action, so it never writes a control event."""

from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.control.domain import (
    SETTING_ALERTS_LAST_REFRESH,
    SETTING_ALERTS_SYNCED_THRESHOLD,
    SETTING_WORKER_LAST_PASS,
    validate_setting,
)
from social_tracker.control.service import ControlService
from social_tracker.shared.clock import FixedClock
from social_tracker.shared.errors import StorageError, ValidationError
from tests.actors import WHO
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork

NOW = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


class Rig:
    def __init__(self) -> None:
        self.sources, self.repo = InMemorySourceRepository(), InMemoryControlRepository()
        self.clock = FixedClock(NOW)
        self.service = ControlService(
            self.repo, self.sources, InMemoryUnitOfWork([self.sources, self.repo]), self.clock
        )


@pytest.fixture
def rig() -> Rig:
    return Rig()


def test_a_fresh_system_has_no_pass_no_refresh_and_no_synced_threshold(rig):
    live = rig.service.liveness()
    assert (live.last_pass_at, live.last_alert_refresh_at, live.synced_threshold) == (
        None,
        None,
        None,
    )
    assert live.global_analysis is False
    assert live.alert_min_confidence == 0.5


def test_the_pass_and_the_refresh_are_recorded_with_the_time_of_the_clock(rig):
    rig.service.record_worker_pass()
    rig.clock.advance(timedelta(minutes=2))
    rig.service.record_alert_refresh()
    live = rig.service.liveness()
    assert live.last_pass_at == NOW
    assert live.last_alert_refresh_at == NOW + timedelta(minutes=2)


def test_the_last_value_wins(rig):
    rig.service.record_worker_pass()
    rig.clock.advance(timedelta(hours=1))
    rig.service.record_worker_pass()
    assert rig.service.liveness().last_pass_at == NOW + timedelta(hours=1)


def test_the_synced_threshold_round_trips_exactly(rig):
    rig.service.record_alerts_synced(0.35)
    assert rig.service.alerts_synced_threshold() == 0.35
    rig.service.record_alerts_synced(0.0)
    assert rig.service.alerts_synced_threshold() == 0.0


def test_telemetry_is_not_audited(rig):
    rig.service.record_worker_pass()
    rig.service.record_alert_refresh()
    rig.service.record_alerts_synced(0.5)
    assert rig.repo.list_control_events() == []


def test_global_analysis_shows_in_the_liveness(rig):
    rig.service.set_global_analysis(True, **WHO)
    assert rig.service.liveness().global_analysis is True


@pytest.mark.parametrize(
    "bad",
    ["", "yesterday", "2026-03-01", "2026-03-01T08:00:00", "2026-03-01T08:00:00+05:00"],
)
def test_a_stored_time_that_is_not_utc_iso_is_refused_on_write(bad):
    with pytest.raises(ValidationError):
        validate_setting(SETTING_WORKER_LAST_PASS, bad)
    with pytest.raises(ValidationError):
        validate_setting(SETTING_ALERTS_LAST_REFRESH, bad)


@pytest.mark.parametrize("bad", ["", "high", "1.5", "-0.1", "nan", "0,5"])
def test_a_synced_threshold_that_is_not_a_probability_is_refused_on_write(bad):
    with pytest.raises(ValidationError):
        validate_setting(SETTING_ALERTS_SYNCED_THRESHOLD, bad)


def test_a_damaged_stored_value_is_a_storage_error_not_a_crash_of_the_page(rig):
    rig.repo._settings[SETTING_WORKER_LAST_PASS] = "garbage"  # noqa: SLF001
    with pytest.raises(StorageError):
        rig.service.liveness()
    rig.repo._settings[SETTING_WORKER_LAST_PASS] = NOW.isoformat()  # noqa: SLF001
    rig.repo._settings[SETTING_ALERTS_SYNCED_THRESHOLD] = "garbage"  # noqa: SLF001
    with pytest.raises(StorageError):
        rig.service.alerts_synced_threshold()
