"""How long ago something happened, in words: what the operator pages say next to the last pass of
the worker and the last refresh of the alerts."""

from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.entrypoints.web.timefmt import ago
from social_tracker.shared.errors import ValidationError

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "less than a minute ago"),
        (59, "less than a minute ago"),
        (60, "1 minute ago"),
        (119, "1 minute ago"),
        (120, "2 minutes ago"),
        (3599, "59 minutes ago"),
        (3600, "1 hour ago"),
        (7199, "1 hour ago"),
        (7200, "2 hours ago"),
        (86399, "23 hours ago"),
        (86400, "1 day ago"),
        (172800, "2 days ago"),
        (86400 * 400, "400 days ago"),
    ],
)
def test_the_unit_grows_with_the_distance(seconds, text):
    assert ago(NOW - timedelta(seconds=seconds), NOW) == text


def test_nothing_recorded_is_never():
    assert ago(None, NOW) == "never"


@pytest.mark.parametrize("seconds", [1, 59, 3600, 10**7])
def test_a_time_in_the_future_is_a_skewed_clock_not_a_crash(seconds):
    assert ago(NOW + timedelta(seconds=seconds), NOW) == "less than a minute ago"


def test_a_naive_time_is_refused_not_guessed():
    with pytest.raises(ValidationError):
        ago(datetime(2026, 3, 1, 11, 0), NOW)
