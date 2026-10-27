"""AL-5: alert statuses, their legal moves and the audit trail of every change."""

import itertools
import threading
from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertEventKind, AlertStatus
from social_tracker.shared.errors import InvalidTransition, NotFoundError, ValidationError
from tests.alerts.world import alert_world, attach
from tests.llm.harness import sqlite_llm

NEW, SEEN, DISMISSED = "new", "seen", "dismissed"
WHO = {"actor": "operator", "reason": "reviewed on screen"}


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    world.put("a", confidence=0.9)
    world.refresh()
    yield world
    world.close()


def current(w):
    return w.alerts()[0]


def path_to(w, alert_id, status):
    """Walk an alert to a status through legal moves only."""
    if status != NEW:
        w.service.set_status(alert_id, status, **WHO)


LEGAL = [(NEW, SEEN), (NEW, DISMISSED), (SEEN, DISMISSED), (DISMISSED, SEEN)]


@pytest.mark.parametrize(("start", "target"), LEGAL)
def test_every_legal_move_changes_the_status_and_leaves_one_audit_entry(w, start, target):
    alert = current(w)
    path_to(w, alert.id, start)
    before = len(w.repo.events(alert.id))
    w.clock.advance(timedelta(minutes=5))
    result = w.service.set_status(alert.id, target, actor="ana", reason="checked with the desk")
    assert (result.changed, result.alert.status.value) == (True, target)
    stored = w.repo.get(alert.id)
    assert stored.status.value == target
    assert stored.status_changed_at == w.clock.now()
    events = w.repo.events(alert.id)
    assert len(events) == before + 1
    last = events[-1]
    assert last.kind is AlertEventKind.STATUS_CHANGED
    assert (last.actor, last.reason) == ("ana", "checked with the desk")
    assert (last.before, last.after) == (start, target)
    assert last.at == w.clock.now()


ALL = [NEW, SEEN, DISMISSED]
ILLEGAL = [(a, b) for a, b in itertools.product(ALL, ALL) if a != b and (a, b) not in LEGAL]


@pytest.mark.parametrize(("start", "target"), ILLEGAL)
def test_every_illegal_move_is_refused_and_writes_nothing(w, start, target):
    alert = current(w)
    path_to(w, alert.id, start)
    events = w.repo.events(alert.id)
    stored = w.repo.get(alert.id)
    with pytest.raises(InvalidTransition):
        w.service.set_status(alert.id, target, **WHO)
    assert w.repo.get(alert.id) == stored
    assert w.repo.events(alert.id) == events


@pytest.mark.parametrize("status", ALL)
def test_asking_for_the_status_an_alert_has_is_a_no_op_without_audit(w, status):
    alert = current(w)
    path_to(w, alert.id, status)
    events = w.repo.events(alert.id)
    stored = w.repo.get(alert.id)
    result = w.service.set_status(alert.id, status, **WHO)
    assert result.changed is False
    assert w.repo.get(alert.id) == stored
    assert w.repo.events(alert.id) == events


def test_a_double_click_is_harmless(w):
    alert = current(w)
    first = w.service.set_status(alert.id, SEEN, **WHO)
    second = w.service.set_status(alert.id, SEEN, **WHO)
    assert (first.changed, second.changed) == (True, False)
    kinds = [e.kind for e in w.repo.events(alert.id)]
    assert kinds.count(AlertEventKind.STATUS_CHANGED) == 1


def test_a_dismissed_alert_can_be_reopened_as_seen_and_then_dismissed_again(w):
    alert = current(w)
    for target in (DISMISSED, SEEN, DISMISSED):
        assert w.service.set_status(alert.id, target, **WHO).changed
    assert w.repo.get(alert.id).status is AlertStatus.DISMISSED


@pytest.mark.parametrize(
    "bad_id", ["alt_" + "0" * 32, "itm_nope", "x", "' OR 1=1 --", "", "A" * 80]
)
def test_an_unknown_or_malformed_alert_id_is_not_found(w, bad_id):
    with pytest.raises(NotFoundError):
        w.service.set_status(bad_id, SEEN, **WHO)


@pytest.mark.parametrize(
    "bad", ["", "SEEN", "Seen ", "withdrawn", "archived", "' OR '1'='1", None, 7]
)
def test_an_invalid_status_is_a_validation_error(w, bad):
    alert = current(w)
    with pytest.raises(ValidationError):
        w.service.set_status(alert.id, bad, **WHO)
    assert w.repo.get(alert.id).status is AlertStatus.NEW


@pytest.mark.parametrize(
    "who",
    [
        {"actor": "", "reason": "why"},
        {"actor": "  ", "reason": "why"},
        {"actor": "ana", "reason": ""},
        {"actor": "ana", "reason": "bell\x07"},
        {"actor": "ana\x00", "reason": "why"},
        {"actor": "a" * 101, "reason": "why"},
        {"actor": "ana", "reason": "r" * 501},
        {"actor": "ana", "reason": "hidden" + chr(0x202E) + "text"},
    ],
)
def test_actor_and_reason_are_required_and_must_be_plain(w, who):
    alert = current(w)
    with pytest.raises(ValidationError):
        w.service.set_status(alert.id, SEEN, **who)
    assert w.repo.get(alert.id).status is AlertStatus.NEW


def test_a_status_change_does_not_touch_what_the_alert_shows(w):
    alert = current(w)
    w.service.set_status(alert.id, SEEN, **WHO)
    after = w.repo.get(alert.id)
    for field in ("label", "severity", "confidence", "excerpt", "occurrences", "tags", "evidence"):
        assert getattr(after, field) == getattr(alert, field)


def test_a_new_copy_joining_a_seen_alert_keeps_what_the_operator_decided(w):
    alert = current(w)
    w.service.set_status(alert.id, DISMISSED, **WHO)
    text = w.llm.items.get(alert.item_id).text
    w.put("b", text=text, confidence=0.6, source="src_other")
    w.refresh()
    again = w.repo.get(alert.id)
    assert (again.status, again.occurrences) == (AlertStatus.DISMISSED, 2)


def test_a_withdrawn_alert_can_still_be_marked(w):
    alert = current(w)
    w.service.set_status(alert.id, SEEN, **WHO)
    with w.db.transaction() as conn:
        conn.execute("UPDATE classifications SET label = 'no_risk'")
    w.refresh()
    assert w.alerts() == []
    assert w.service.set_status(alert.id, DISMISSED, **WHO).changed


# ---- concurrency ----


def two_workers(w):
    """A second service over the same database file, with its own connection."""
    other = attach(sqlite_llm(w.db.path.parent / "alerts.db"), w.settings)
    return other


def test_a_status_requested_by_many_workers_at_once_changes_once(w):
    alert = current(w)
    others = [two_workers(w) for _ in range(5)]
    barrier = threading.Barrier(len(others))
    results = []

    def go(world):
        barrier.wait()
        results.append(world.service.set_status(alert.id, SEEN, **WHO).changed)

    threads = [threading.Thread(target=go, args=(o,)) for o in others]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert sorted(results) == [False] * 4 + [True]
        assert [e.kind for e in w.repo.events(alert.id)].count(AlertEventKind.STATUS_CHANGED) == 1
    finally:
        for o in others:
            o.close()


def test_two_opposite_requests_at_once_leave_a_consistent_trail(w):
    alert = current(w)
    others = [two_workers(w) for _ in range(6)]
    barrier = threading.Barrier(len(others))
    outcomes = []

    def go(world, target):
        barrier.wait()
        try:
            outcomes.append(world.service.set_status(alert.id, target, **WHO).changed)
        except InvalidTransition:
            outcomes.append("refused")

    threads = [
        threading.Thread(target=go, args=(o, SEEN if n % 2 else DISMISSED))
        for n, o in enumerate(others)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        trail = [e for e in w.repo.events(alert.id) if e.kind is AlertEventKind.STATUS_CHANGED]
        assert 1 <= len(trail) <= len(others)
        assert trail[0].before == NEW
        for earlier, later in itertools.pairwise(trail):
            assert earlier.after == later.before  # every entry starts where the last one ended
        assert trail[-1].after == w.repo.get(alert.id).status.value
        assert len(outcomes) == len(others)
    finally:
        for o in others:
            o.close()
