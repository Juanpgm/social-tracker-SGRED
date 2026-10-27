"""A8: one alert cannot grow, or become costlier to merge into, without bound. Past `max_members`
a copy is counted (not stored), a stronger one takes the place of the weakest, the first and the
latest members stay, and the trail is capped too. Merging a copy reads no other member."""

from datetime import timedelta

import pytest

from social_tracker.alerts.domain import (
    AlertEventKind,
    AlertSettings,
    Severity,
)
from social_tracker.classification.domain import Label
from social_tracker.shared.errors import ValidationError
from tests.alerts.world import alert_world
from tests.classification.harness import PUBLISHED

MINUTE = timedelta(minutes=1)
START = PUBLISHED + MINUTE


@pytest.fixture
def small(tmp_path):
    world = alert_world(tmp_path, AlertSettings(max_members=5, max_events=10))
    world.llm.ready()
    world.llm.ready("src_other")
    yield world
    world.close()


def flood(w, count, *, first=None, label=Label.UNINTENTIONAL_RUMOR, confidence=0.6, start=1):
    """`count` near copies of one original, one minute apart, each with its own words."""
    first = first or w.put("c0", text="Mismo rumor repetido sobre temblor 0", fetched=START)
    copies = [
        w.put(
            f"c{n}",
            text=f"Mismo rumor repetido sobre temblor {n}",
            near_of=first.id,
            label=label,
            confidence=confidence,
            fetched=START + n * MINUTE,
        )
        for n in range(start, start + count)
    ]
    return first, copies


def test_the_cap_has_sane_bounds():
    assert AlertSettings().max_members == 500
    assert AlertSettings().max_events == 200
    for bad in (0, -1, 5001, True, 2.5):
        with pytest.raises(ValidationError):
            AlertSettings(max_members=bad)
    for bad in (9, 2001, True):
        with pytest.raises(ValidationError):
            AlertSettings(max_events=bad)
    assert AlertSettings(max_members=1, max_events=10).max_members == 1


def test_a_flood_is_one_alert_with_a_capped_member_list_and_the_true_count(small):
    first, _ = flood(small, 14)
    report = small.refresh()
    assert (report.created, report.merged, report.errors) == (1, 14, 0)
    alert = small.only()
    assert alert.occurrences == 15
    assert alert.overflow == 10
    assert len(small.repo.members(alert.id)) == 5
    assert first.id in {m.item_id for m in small.repo.members(alert.id)}  # the first stays


def test_the_latest_copy_is_always_a_member(small):
    _, copies = flood(small, 14)
    small.refresh()
    members = {m.item_id for m in small.repo.members(small.only().id)}
    assert copies[-1].id in members
    assert small.only().last_seen_at == START + 14 * MINUTE


def test_looking_again_does_not_count_a_copy_twice(small):
    flood(small, 14)
    small.refresh()
    before = small.only()
    assert small.refresh().created == 0
    assert small.service.rebuild().errors == 0
    assert small.service.rebuild().errors == 0
    assert small.only().occurrences == before.occurrences == 15
    assert small.only().overflow == 10


def test_a_stronger_copy_at_the_cap_takes_a_place_and_leads(small):
    first, _ = flood(small, 8)
    small.refresh()
    strong = small.put(
        "boss",
        text="Mismo rumor repetido sobre temblor boss",
        near_of=first.id,
        label=Label.DELIBERATE_DISINFORMATION,
        confidence=0.95,
        fetched=START + 20 * MINUTE,
    )
    small.refresh()
    alert = small.only()
    assert (alert.severity, alert.item_id) == (Severity.CRITICAL, strong.id)
    assert alert.occurrences == 10
    members = small.repo.members(alert.id)
    assert len(members) == 5
    assert strong.id in {m.item_id for m in members}


def test_the_weakest_copy_is_the_one_given_up(small):
    first, _ = flood(small, 4, confidence=0.9)  # 5 members, all high
    weak = small.put(
        "weak",
        text="Mismo rumor repetido sobre temblor weak",
        near_of=first.id,
        confidence=0.5,
        fetched=START + 30 * MINUTE,
    )
    late_strong = small.put(
        "late",
        text="Mismo rumor repetido sobre temblor late",
        near_of=first.id,
        confidence=0.9,
        fetched=START + 40 * MINUTE,
    )
    small.refresh()
    members = {m.item_id for m in small.repo.members(small.only().id)}
    assert len(members) == 5
    assert late_strong.id in members  # the latest stays
    assert weak.id not in members  # the weakest went when the next one came


def test_the_original_can_be_given_up_and_its_later_copies_still_find_the_alert(tmp_path):
    """The lead and the first member are kept, and when everything is fetched together that is
    not necessarily the original: its copies must still find the alert it was counted in."""
    w = alert_world(tmp_path, AlertSettings(max_members=5, max_events=10))
    w.llm.ready()
    try:
        original = w.put("orig", text="Mismo rumor repetido sobre temblor 0", fetched=START)
        for n in range(1, 61):
            w.put(
                f"c{n}",
                text=f"Mismo rumor repetido sobre temblor {n}",
                near_of=original.id,
                fetched=START,
            )
        w.refresh()
        assert len(w.alerts()) == 1
        assert w.only().occurrences == 61
    finally:
        w.close()


def test_merging_a_copy_reads_no_other_member(small, monkeypatch):
    first, _ = flood(small, 3)
    small.refresh()
    reads = []
    real = small.repo.members
    monkeypatch.setattr(
        small.repo, "members", lambda alert_id: reads.append(alert_id) or real(alert_id)
    )
    flood(small, 30, first=first, start=10)
    small.refresh()
    assert reads == []  # the projection was kept up to date, not rebuilt from every member
    assert small.only().occurrences == 34


def test_a_big_flood_with_the_default_cap_stays_incremental(tmp_path, monkeypatch):
    w = alert_world(tmp_path)
    w.llm.ready()
    try:
        first, _ = flood(w, 1)
        w.refresh()
        reads = []
        real = w.repo.members
        monkeypatch.setattr(w.repo, "members", lambda a: reads.append(a) or real(a))
        flood(w, 400, first=first, start=2)
        report = w.refresh()
        assert (report.merged, report.errors) == (400, 0)
        assert reads == []
        assert w.only().occurrences == 402
    finally:
        w.close()


# ---- the trail ----


def test_the_trail_is_capped_but_says_how_much_was_left_out(small):
    flood(small, 25)
    small.refresh()
    alert = small.only()
    stored = small.repo.events(alert.id)
    assert len(stored) <= 10
    assert alert.events_dropped >= 16
    detail = small.service.detail(alert.id)
    assert len(detail.events) <= 10
    assert detail.events_hidden == alert.events_dropped + (
        small.repo.count_events(alert.id) - len(detail.events)
    )
    assert stored[0].kind is AlertEventKind.CREATED  # the creation is never lost


def test_what_a_person_decides_is_always_on_the_trail_even_when_the_trail_is_full(small):
    flood(small, 25)
    small.refresh()
    small.service.set_status(small.only().id, "dismissed", actor="op", reason="noise")
    last = small.repo.events(small.only().id)[-1]
    assert (last.kind, last.actor, last.after) == (AlertEventKind.STATUS_CHANGED, "op", "dismissed")


def test_the_detail_shows_only_the_latest_events_when_there_are_more_than_the_cap(small):
    flood(small, 25)
    small.refresh()
    small.service.set_status(small.only().id, "seen", actor="op", reason="one")
    detail = small.service.detail(small.only().id)
    assert detail.events[-1].kind is AlertEventKind.STATUS_CHANGED
    assert len(detail.events) <= 10
    assert detail.events_hidden >= 1


def test_a_short_trail_has_nothing_hidden(small):
    small.put("a", text="un rumor solo", fetched=START)
    small.refresh()
    detail = small.service.detail(small.only().id)
    assert detail.events_hidden == 0
    assert len(detail.events) == 1
