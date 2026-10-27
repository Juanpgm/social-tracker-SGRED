"""AL-6: the same content, or a near copy, seen within 24 hours is one alert with an occurrence
count and its linked sources and items; the severity of a merged alert can rise, never fall."""

from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertEventKind, Severity
from social_tracker.classification.domain import Label, Stage
from tests.alerts.world import alert_world
from tests.classification.harness import PUBLISHED
from tests.llm.support import response

RUMOR, DELIBERATE, RECYCLED = (
    Label.UNINTENTIONAL_RUMOR,
    Label.DELIBERATE_DISINFORMATION,
    Label.RECYCLED_CONTENT,
)
SAME = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
OTHER_SOURCE = "src_other"
DAY = timedelta(hours=24)
FIRST_SEEN = PUBLISHED + timedelta(minutes=1)


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    world.llm.ready(OTHER_SOURCE)
    world.llm.ready("src_third")
    yield world
    world.close()


def trail(w, alert_id):
    return [e.kind for e in w.repo.events(alert_id)]


def test_the_same_content_from_two_sources_is_one_alert_with_two_occurrences(w):
    a = w.put("a", text=SAME, confidence=0.6)
    b = w.put("b", text=SAME, confidence=0.6, source=OTHER_SOURCE, source_name="Other News")
    report = w.refresh()
    assert (report.created, report.merged) == (1, 1)
    alert = w.only()
    assert alert.occurrences == 2
    assert {s for s, _ in alert.linked_sources} == {"src_news", OTHER_SOURCE}
    assert {m.item_id for m in w.repo.members(alert.id)} == {a.id, b.id}
    assert trail(w, alert.id) == [AlertEventKind.CREATED, AlertEventKind.MERGED]


@pytest.mark.parametrize(
    ("gap", "merged"),
    [
        (timedelta(0), True),
        (timedelta(hours=23, minutes=59, seconds=59), True),
        (DAY, True),
        (DAY + timedelta(seconds=1), False),
        (timedelta(days=3), False),
    ],
)
def test_the_window_is_24_hours_inclusive(w, gap, merged):
    w.put("a", text=SAME, confidence=0.6, fetched=FIRST_SEEN)
    w.put("b", text=SAME, confidence=0.6, source=OTHER_SOURCE, fetched=FIRST_SEEN + gap)
    w.refresh()
    assert len(w.alerts()) == (1 if merged else 2)


def test_the_window_counts_both_ways_whichever_was_seen_first(w):
    w.put("a", text=SAME, confidence=0.6, fetched=FIRST_SEEN + DAY)
    w.put("b", text=SAME, confidence=0.6, source=OTHER_SOURCE, fetched=FIRST_SEEN)
    w.refresh()
    assert len(w.alerts()) == 1
    w.put(
        "c",
        text=SAME,
        confidence=0.6,
        source="src_third",
        fetched=FIRST_SEEN - timedelta(seconds=1),
    )
    w.refresh()
    # c is 24:00:01 before a but only 1 second before b: it joins through b
    assert w.only().occurrences == 3


def test_a_chain_of_copies_each_within_a_day_of_the_last_is_one_alert(w):
    for n, source in enumerate(("src_news", OTHER_SOURCE, "src_third")):
        w.put(
            f"n{n}",
            text=SAME,
            confidence=0.6,
            source=source,
            fetched=FIRST_SEEN + n * (DAY - timedelta(hours=1)),
        )
    w.refresh()
    assert w.only().occurrences == 3


def test_different_content_is_a_different_alert(w):
    w.put("a", text=SAME, confidence=0.6)
    w.put("b", text="Dicen que habrá una réplica enorme esta noche en Cali", confidence=0.6)
    w.refresh()
    assert len(w.alerts()) == 2


@pytest.mark.parametrize("link", ["near_of", "duplicate_of"])
def test_a_copy_that_points_to_an_original_joins_its_alert_whatever_its_text(w, link):
    original = w.put("a", text=SAME, confidence=0.6)
    w.put(
        "b",
        text=SAME + " por favor",
        confidence=0.6,
        source=OTHER_SOURCE,
        **{link: original.id},
    )
    w.refresh()
    assert w.only().occurrences == 2


def test_a_near_copy_outside_the_window_is_another_alert(w):
    original = w.put("a", text=SAME, confidence=0.6, fetched=FIRST_SEEN)
    w.put(
        "b", text=SAME + " otra vez", confidence=0.6, source=OTHER_SOURCE,
        near_of=original.id, fetched=FIRST_SEEN + DAY + timedelta(seconds=1),
    )  # fmt: skip
    w.refresh()
    assert len(w.alerts()) == 2


def test_recycled_content_a_week_later_is_its_own_alert(w):
    original = w.put("a", text=SAME, confidence=0.6, fetched=FIRST_SEEN)
    w.put(
        "b", text=SAME, label=RECYCLED, confidence=0.9, source=OTHER_SOURCE,
        recycled_of=original.id, fetched=FIRST_SEEN + timedelta(days=8),
    )  # fmt: skip
    w.refresh()
    assert len(w.alerts()) == 2


# ---- severity: up, never down ----


def test_merging_a_less_severe_copy_never_lowers_the_severity(w):
    w.put("a", text=SAME, label=DELIBERATE, confidence=0.9)
    w.put("b", text=SAME, confidence=0.5, source=OTHER_SOURCE)
    w.refresh()
    alert = w.only()
    assert alert.severity is Severity.CRITICAL
    assert (alert.label, alert.confidence) == (DELIBERATE, 0.9)
    assert alert.occurrences == 2


def test_merging_a_more_severe_copy_raises_the_severity_and_takes_the_lead(w):
    first = w.put("a", text=SAME, confidence=0.5, fetched=FIRST_SEEN)
    w.refresh()
    assert w.only().severity is Severity.MEDIUM
    second = w.put(
        "b", text=SAME, label=DELIBERATE, confidence=0.9, source=OTHER_SOURCE,
        fetched=FIRST_SEEN + timedelta(hours=2),
    )  # fmt: skip
    report = w.refresh()
    assert (report.merged, report.created) == (1, 0)
    alert = w.only()
    assert alert.severity is Severity.CRITICAL
    assert alert.item_id == second.id
    assert alert.occurrences == 2
    assert alert.first_seen_at == first.fetched_at
    assert alert.last_seen_at == second.fetched_at


def test_whichever_order_the_copies_are_processed_in_the_result_is_the_same(tmp_path):
    outcomes = []
    for order in (("a", "b", "c"), ("c", "b", "a")):
        world = alert_world(tmp_path, name=f"order-{order[0]}.db")
        world.llm.ready()
        world.llm.ready(OTHER_SOURCE)
        world.llm.ready("src_third")
        specs = {
            "a": dict(confidence=0.5),
            "b": dict(confidence=0.75, source=OTHER_SOURCE),
            "c": dict(confidence=0.9, label=DELIBERATE, source="src_third"),
        }
        items = {n: world.put(n, text=SAME, **specs[n]) for n in "abc"}
        for name in order:  # processed one at a time, in this order
            world.service.sync_item(items[name].id)
        alert = world.only()
        outcomes.append(
            (alert.severity, alert.occurrences, alert.label, alert.confidence, alert.item_id)
        )
        world.close()
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][:4] == (Severity.CRITICAL, 3, DELIBERATE, 0.9)


# ---- members that stop qualifying ----


def test_one_member_that_stops_qualifying_leaves_the_alert_to_the_others(w):
    w.llm.seed(1, text=SAME, confidence=0.55)
    w.llm.seed(2, text=SAME, source=OTHER_SOURCE, confidence=0.55, priority=0.1)
    w.refresh()
    assert w.only().occurrences == 2
    w.llm.judge.script.append(response(label=Label.NO_RISK, confidence=0.95))
    w.llm.judge.script.append(response(label=DELIBERATE, confidence=0.9))
    w.llm.escalate()
    w.refresh()
    alert = w.only()
    assert alert.withdrawn is False
    assert alert.occurrences == 1
    assert alert.label is DELIBERATE


def test_a_copy_that_arrives_after_the_alert_was_withdrawn_brings_it_back(w):
    stored = w.llm.seed(1, text=SAME, confidence=0.55)
    w.refresh()
    with w.db.transaction() as conn:  # a reclassification by the cascade (not the model)
        conn.execute(
            "UPDATE classifications SET label = 'no_risk', confidence = 0.95, stage = 'rules' "
            "WHERE item_id = ?",
            (stored.item_id,),
        )
    w.refresh()
    assert w.alerts() == []
    w.put(
        "b", text=SAME, confidence=0.9, source=OTHER_SOURCE, fetched=FIRST_SEEN + timedelta(hours=1)
    )
    report = w.refresh()
    assert report.reactivated == 1
    alert = w.only()
    assert (alert.occurrences, alert.withdrawn) == (1, False)
    assert alert.confidence == 0.9
    assert AlertEventKind.REACTIVATED in trail(w, alert.id)


def test_the_copies_a_model_judgment_settles_stay_one_alert_and_follow_the_verdict(w):
    original = w.llm.seed(1, text=SAME, confidence=0.55)
    copy = w.llm.seed(
        2,
        text=SAME,
        source=OTHER_SOURCE,
        needs_llm=False,
        priority=0.0,
        duplicate_of=original.item_id,
        awaiting_original=True,
        stage=Stage.DEDUP,
        label=original.label,
        confidence=original.confidence,
        rationale=f"duplicate of {original.item_id}: reused its {original.label.value} verdict",
    )
    w.refresh()
    first = w.only()
    assert first.occurrences == 2
    assert first.evidence.provisional is True
    w.llm.judge.default = response(label=DELIBERATE, confidence=0.9)
    w.llm.escalate()
    settled = w.llm.verdict(copy)
    assert (settled.settle_reason, settled.label) == ("duplicate", DELIBERATE)
    report = w.refresh()
    assert report.created == 0
    alert = w.only()
    assert alert.id == first.id
    assert (alert.severity, alert.occurrences) == (Severity.CRITICAL, 2)
    assert w.repo.member(copy.item_id).evidence.settle_reason == "duplicate"


def test_the_items_behind_a_merged_alert_are_all_kept_even_when_one_is_purged(w):
    a = w.put("a", text=SAME, confidence=0.6)
    w.put("b", text=SAME, confidence=0.6, source=OTHER_SOURCE)
    w.refresh()
    alert = w.only()
    w.llm.items.purge_fetched_before(a.fetched_at + timedelta(seconds=1))
    assert w.llm.items.get(a.id) is None
    assert w.refresh().errors == 0
    assert {m.item_id for m in w.repo.members(alert.id)} >= {a.id}
    assert w.only().occurrences == 2


# ---- an item that bridges two alerts makes them one ----


def bridge_world(w):
    """Three copies: `early` and `late` are 46 hours apart, `middle` is within a day of both."""
    early = w.put("early", text=SAME, confidence=0.6, fetched=FIRST_SEEN)
    late = w.put(
        "late",
        text=SAME,
        confidence=0.7,
        source=OTHER_SOURCE,
        fetched=FIRST_SEEN + timedelta(hours=46),
    )
    middle = w.put(
        "middle", text=SAME, confidence=0.9, label=DELIBERATE, source="src_third",
        fetched=FIRST_SEEN + timedelta(hours=23),
    )  # fmt: skip
    return early, late, middle


def test_an_item_between_two_alerts_makes_them_one_whatever_the_order(tmp_path):
    results = []
    for order in (
        "early late middle",
        "late early middle",
        "middle late early",
        "late middle early",
    ):
        world = alert_world(tmp_path, name=f"bridge-{order.replace(' ', '-')}.db")
        world.llm.ready()
        world.llm.ready(OTHER_SOURCE)
        world.llm.ready("src_third")
        items = dict(zip(("early", "late", "middle"), bridge_world(world), strict=True))
        for name in order.split():
            world.service.sync_item(items[name].id)
            world.clock.advance(timedelta(minutes=1))
        alert = world.only()
        results.append((alert.occurrences, alert.severity, alert.label, alert.item_id))
        assert {m.item_id for m in world.repo.members(alert.id)} == {i.id for i in items.values()}
        world.close()
    assert len(set(results)) == 1
    assert results[0][:3] == (3, Severity.CRITICAL, DELIBERATE)


def test_the_absorbed_alert_leaves_a_note_in_the_trail_of_the_one_that_stays(w):
    early, late, middle = bridge_world(w)
    w.service.sync_item(early.id)
    w.clock.advance(timedelta(minutes=1))
    w.service.sync_item(late.id)
    assert len(w.alerts()) == 2
    w.clock.advance(timedelta(minutes=1))
    w.service.sync_item(middle.id)
    alert = w.only()
    assert alert.id == w.repo.member(early.id).alert_id  # the oldest alert is the one that stays
    merged = [e for e in w.repo.events(alert.id) if e.kind is AlertEventKind.MERGED]
    assert len(merged) == 2  # the item that bridged, and the alert it absorbed
    assert any("absorbed" in e.reason for e in merged)


def test_when_alerts_become_one_the_one_that_needs_attention_most_decides_the_status(w):
    early, late, middle = bridge_world(w)
    w.service.sync_item(early.id)
    w.clock.advance(timedelta(minutes=1))
    w.service.sync_item(late.id)
    first = w.repo.member(early.id).alert_id
    w.service.set_status(first, "dismissed", actor="tester", reason="not relevant")
    w.service.sync_item(middle.id)
    assert w.only().status.value == "new"  # the absorbed one had not been looked at
