"""A4/A7/A11 in the filter: dates read in the operator's zone (or with an explicit offset), the
timestamp a range applies to, hiding what still waits for the model, and showing withdrawn
alerts next to the live ones."""

import hashlib
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from social_tracker.alerts.domain import (
    Alert,
    AlertFilter,
    AlertStatus,
    parse_moment,
    project,
)
from social_tracker.persistence.alerts_repo import SqliteAlertRepository
from social_tracker.persistence.db import Database
from social_tracker.shared.errors import ValidationError
from tests.alerts.support import T0, make_evidence, make_member

BOGOTA = ZoneInfo("America/Bogota")
MADRID = ZoneInfo("Europe/Madrid")


# ---- dates read in a zone ----


def test_a_day_is_the_day_of_the_zone_not_of_utc():
    assert parse_moment("2026-03-01", end=False, zone=BOGOTA) == datetime(
        2026, 3, 1, 5, 0, tzinfo=UTC
    )
    assert parse_moment("2026-03-01", end=True, zone=BOGOTA) == datetime(
        2026, 3, 2, 4, 59, 59, 999999, tzinfo=UTC
    )


def test_utc_remains_the_default_so_old_callers_do_not_change():
    assert parse_moment("2026-03-01", end=False) == datetime(2026, 3, 1, tzinfo=UTC)


def test_a_moment_without_an_offset_is_in_the_zone():
    assert parse_moment("2026-03-01T10:30:00", end=False, zone=BOGOTA) == datetime(
        2026, 3, 1, 15, 30, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-03-01T10:30:00-05:00", datetime(2026, 3, 1, 15, 30, tzinfo=UTC)),
        ("2026-03-01T10:30:00+01:00", datetime(2026, 3, 1, 9, 30, tzinfo=UTC)),
        ("2026-03-01T10:30:00Z", datetime(2026, 3, 1, 10, 30, tzinfo=UTC)),
        ("2026-03-01T10:30:00+00:00", datetime(2026, 3, 1, 10, 30, tzinfo=UTC)),
    ],
)
def test_an_explicit_offset_wins_over_the_zone(text, expected):
    assert parse_moment(text, end=False, zone=BOGOTA) == expected


def test_the_day_of_a_spring_forward_is_23_hours_long():
    start = parse_moment("2026-03-29", end=False, zone=MADRID)
    end = parse_moment("2026-03-29", end=True, zone=MADRID)
    assert end - start == timedelta(hours=23) - timedelta(microseconds=1)


@pytest.mark.parametrize(
    "bad", ["", "  ", "yesterday", "2026-13-01", "2026-02-30", "2026-03-01T25:00:00", "1" * 80]
)
def test_a_bad_date_is_refused_in_any_zone(bad):
    with pytest.raises(ValidationError):
        parse_moment(bad, end=False, zone=BOGOTA)


# ---- the filter ----


def test_the_new_fields_default_to_the_old_behaviour():
    flt = AlertFilter()
    assert (flt.hide_provisional, flt.include_withdrawn, flt.when) == (False, False, "last_seen")


@pytest.mark.parametrize("when", ["first_seen", "last_seen"])
def test_both_timestamps_can_be_chosen(when):
    assert AlertFilter(when=when).when == when


@pytest.mark.parametrize("when", ["", "created_at", "FIRST_SEEN", None, 3])
def test_any_other_timestamp_is_refused(when):
    with pytest.raises(ValidationError):
        AlertFilter(when=when)


@pytest.mark.parametrize("field", ["hide_provisional", "include_withdrawn"])
@pytest.mark.parametrize("value", ["yes", 1, None])
def test_the_flags_must_be_booleans(field, value):
    with pytest.raises(ValidationError):
        AlertFilter(**{field: value})


# ---- in the store ----


@pytest.fixture
def repo(tmp_path):
    db = Database(tmp_path / "f.db")
    yield SqliteAlertRepository(db)
    db.close()


def store(
    repo, item, *, at=T0, withdrawn=False, provisional=False, first=None, **overrides
) -> Alert:
    member = make_member(
        item,
        occurred_at=first or at,
        evidence=make_evidence(provisional=provisional),
        content_hash=hashlib.sha256(item.encode()).hexdigest(),
        **overrides,
    )
    alert = project(member.alert_id, [member], previous=None, now=at)
    changes = {"last_seen_at": at}
    if withdrawn:
        changes |= {"withdrawn_at": at, "withdrawn_reason": "the verdict is now no_risk"}
    alert = Alert(**{**alert.__dict__, **changes})
    repo.save(alert)
    repo.save_member(member)
    return alert


def ids(repo, **kw):
    return {a.item_id for a in repo.list(AlertFilter(**kw), limit=100, offset=0)}


def test_hide_provisional_leaves_out_what_waits_for_the_model(repo):
    store(repo, "itm_wait", provisional=True)
    store(repo, "itm_done")
    assert ids(repo) == {"itm_wait", "itm_done"}
    assert ids(repo, hide_provisional=True) == {"itm_done"}
    assert repo.count(AlertFilter(hide_provisional=True)) == 1


def test_include_withdrawn_adds_them_to_the_live_ones(repo):
    store(repo, "itm_live")
    store(repo, "itm_gone", withdrawn=True, active=False)
    assert ids(repo) == {"itm_live"}
    assert ids(repo, include_withdrawn=True) == {"itm_live", "itm_gone"}
    assert ids(repo, status="withdrawn") == {"itm_gone"}
    assert ids(repo, status="withdrawn", include_withdrawn=True) == {"itm_gone"}
    assert repo.count(AlertFilter(include_withdrawn=True)) == 2


def test_include_withdrawn_still_honours_a_status(repo):
    store(repo, "itm_new")
    store(repo, "itm_seen_gone", withdrawn=True, active=False)
    flt = AlertFilter(status=AlertStatus.SEEN, include_withdrawn=True)
    assert repo.list(flt, limit=10, offset=0) == []


def test_a_range_applies_to_the_last_sighting_by_default_and_to_the_first_on_request(repo):
    store(repo, "itm_a", first=T0, at=T0 + timedelta(days=5))
    store(repo, "itm_b", first=T0 + timedelta(days=4), at=T0 + timedelta(days=4))
    window = {"since": T0 + timedelta(days=3), "until": T0 + timedelta(days=4, hours=1)}
    assert ids(repo, **window) == {"itm_b"}  # last seen: a ended at day 5
    assert ids(repo, when="last_seen", **window) == {"itm_b"}
    assert ids(repo, when="first_seen", **window) == {"itm_b"}
    early = {"since": T0 - timedelta(hours=1), "until": T0 + timedelta(hours=1)}
    assert ids(repo, when="first_seen", **early) == {"itm_a"}
    assert ids(repo, when="last_seen", **early) == set()


def test_the_boundaries_of_a_range_are_inclusive_on_both_timestamps(repo):
    store(repo, "itm_a", first=T0, at=T0)
    for when in ("first_seen", "last_seen"):
        assert ids(repo, when=when, since=T0, until=T0) == {"itm_a"}
        assert ids(repo, when=when, since=T0 + timedelta(microseconds=1)) == set()
        assert ids(repo, when=when, until=T0 - timedelta(microseconds=1)) == set()
