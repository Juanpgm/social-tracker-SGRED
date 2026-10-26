"""Clock port: tz-aware UTC only, deterministic fixed clock, no silent time travel."""

from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest

from social_tracker.shared.clock import Clock, FixedClock, SystemClock, ensure_utc
from social_tracker.shared.errors import ValidationError

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_system_clock_returns_aware_utc_and_satisfies_protocol():
    clock: Clock = SystemClock()
    now = clock.now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)
    assert abs((datetime.now(UTC) - now).total_seconds()) < 5


def test_ensure_utc_rejects_naive():
    with pytest.raises(ValidationError, match="timezone-aware"):
        ensure_utc(datetime(2026, 1, 1))


def test_ensure_utc_converts_other_offsets_to_utc():
    plus3 = timezone(timedelta(hours=3))
    out = ensure_utc(datetime(2026, 1, 1, 3, 0, tzinfo=plus3))
    assert out == datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    assert out.utcoffset() == timedelta(0)


@pytest.mark.parametrize("bad", [None, "2026-01-01", 0, 1.5])
def test_ensure_utc_rejects_non_datetime(bad):
    with pytest.raises(ValidationError, match="datetime"):
        ensure_utc(bad)


def test_fixed_clock_is_deterministic():
    clock = FixedClock(T0)
    assert clock.now() == T0
    assert clock.now() == T0


def test_fixed_clock_rejects_naive_start():
    with pytest.raises(ValidationError):
        FixedClock(datetime(2026, 1, 1))


def test_fixed_clock_normalizes_start_to_utc():
    plus3 = timezone(timedelta(hours=3))
    clock = FixedClock(datetime(2026, 1, 1, 15, 0, tzinfo=plus3))
    assert clock.now() == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert clock.now().utcoffset() == timedelta(0)


def test_advance_moves_forward_and_zero_is_allowed():
    clock = FixedClock(T0)
    clock.advance(timedelta(seconds=90))
    assert clock.now() == T0 + timedelta(seconds=90)
    clock.advance(timedelta(0))
    assert clock.now() == T0 + timedelta(seconds=90)


def test_advance_rejects_negative_delta_and_keeps_time():
    clock = FixedClock(T0)
    with pytest.raises(ValidationError, match="backwards"):
        clock.advance(timedelta(microseconds=-1))
    assert clock.now() == T0


def test_advance_overflow_is_a_domain_error_and_keeps_time():
    clock = FixedClock(T0)
    with pytest.raises(ValidationError, match="overflow"):
        clock.advance(timedelta.max)
    assert clock.now() == T0


class _OffsetlessTz(tzinfo):
    """An aware-looking tzinfo whose utcoffset() is None, which Python treats as naive."""

    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None

    def tzname(self, dt):
        return "offsetless"


def test_ensure_utc_rejects_tzinfo_without_an_offset():
    value = datetime(2026, 1, 1, 12, 0, tzinfo=_OffsetlessTz())
    assert value.tzinfo is not None
    with pytest.raises(ValidationError, match="timezone-aware"):
        ensure_utc(value)
