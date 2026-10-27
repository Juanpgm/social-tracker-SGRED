"""A12: merging keeps the audit trail of the alerts it absorbs, finds an original that is seen
AFTER its copies, and does not depend on the order the items arrived in."""

from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertEventKind, AlertStatus
from social_tracker.classification.domain import Label
from tests.alerts.world import alert_world
from tests.classification.harness import PUBLISHED

MINUTE = timedelta(minutes=1)
START = PUBLISHED + MINUTE
OTHER = "src_other"


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    world.llm.ready(OTHER)
    yield world
    world.close()


# ---- the trail of an absorbed alert is not lost ----


def bridge(w):
    a = w.put("a", text="Rumor uno sobre temblor en Cali", fetched=START)
    b = w.put("b", text="Otro tema distinto de inundacion en Cali", fetched=START + MINUTE)
    first_pass = w.clock.now()
    w.refresh()
    by_item = {x.item_id: x for x in w.alerts()}
    w.service.set_status(by_item[b.id].id, "seen", actor="op", reason="checked the flood claim")
    w.clock.advance(MINUTE)
    w.put(
        "c",
        text="Rumor uno sobre temblor en Cali",
        source=OTHER,
        near_of=b.id,
        fetched=START + 2 * MINUTE,
    )
    w.refresh()
    return a, b, by_item, first_pass


def test_the_trail_of_the_absorbed_alert_moves_to_the_alert_that_stays(w):
    bridge(w)
    kept = w.only()
    entries = [e for e in w.repo.events(kept.id) if e.from_absorbed]
    kinds = [(e.kind, e.actor, e.reason) for e in entries]
    assert (AlertEventKind.CREATED, "system:alerts") in [(k, a_) for k, a_, _ in kinds]
    assert (
        AlertEventKind.STATUS_CHANGED,
        "op",
        "checked the flood claim",
    ) in kinds


def test_absorbed_entries_keep_their_original_time_and_are_marked(w):
    _, _, _, first_pass = bridge(w)
    kept = w.only()
    entries = w.repo.events(kept.id)
    absorbed_creation = [e for e in entries if e.from_absorbed and e.kind is AlertEventKind.CREATED]
    assert [e.at for e in absorbed_creation] == [first_pass]
    own = [e for e in entries if not e.from_absorbed]
    assert own[0].kind is AlertEventKind.CREATED  # the kept alert's own creation is not marked


def test_the_absorbed_alert_is_gone_but_its_history_is_not(w):
    _, b, by_item, _ = bridge(w)
    assert w.repo.get(by_item[b.id].id) is None
    assert any(e.from_absorbed for e in w.repo.events(w.only().id))


def test_the_status_the_operator_gave_one_of_the_alerts_still_counts(w):
    bridge(w)
    assert w.only().status is AlertStatus.NEW  # A was never looked at: new wins over seen


# ---- an original that is seen after its copies ----


def chain(w, n):
    """item 0 is the original; item k says it is a near copy of item k-1."""
    items = []
    for k in range(n):
        items.append(
            w.put(
                f"k{k}",
                text=f"Rumor encadenado numero {k} sobre el temblor",
                near_of=items[-1].id if items else None,
                fetched=START + k * MINUTE,
            )
        )
    return items


def test_copies_synced_before_their_original_still_end_in_one_alert(w):
    items = chain(w, 6)
    for item in reversed(items):  # the worst order: the newest copy first
        w.service.sync_item(item.id)
    alert = w.only()
    assert alert.occurrences == 6
    assert {m.item_id for m in w.repo.members(alert.id)} == {i.id for i in items}


def test_the_order_of_arrival_does_not_change_the_result(tmp_path):
    results = []
    for order in ("forward", "backward", "interleaved"):
        world = alert_world(tmp_path / order)
        world.llm.ready()
        try:
            items = chain(world, 6)
            sequence = {
                "forward": items,
                "backward": items[::-1],
                "interleaved": [items[3], items[0], items[5], items[1], items[4], items[2]],
            }[order]
            for item in sequence:
                world.service.sync_item(item.id)
            alert = world.only()
            results.append(
                (
                    alert.occurrences,
                    frozenset(m.item_id for m in world.repo.members(alert.id)),
                    alert.first_seen_at,
                    alert.last_seen_at,
                )
            )
        finally:
            world.close()
    assert len(set(results)) == 1


def test_many_near_copies_are_one_alert_whatever_the_order_of_their_ids(w):
    original = w.put("orig", text="Mismo rumor repetido sobre temblor 0", fetched=START)
    for n in range(1, 61):
        w.put(
            f"c{n}",
            text=f"Mismo rumor repetido sobre temblor {n}",
            near_of=original.id,
            fetched=START,  # all fetched together: the order is the order of their ids
        )
    report = w.refresh()
    assert report.errors == 0
    assert len(w.alerts()) == 1
    assert w.only().occurrences == 61


def test_the_sync_goes_through_the_items_in_the_order_they_were_fetched(w):
    late = w.put("late", text="rumor tardio sobre temblor", fetched=START + 5 * MINUTE)
    early = w.put("early", text="rumor temprano sobre temblor", fetched=START)
    seen = []
    real = w.service.sync_item

    def spy(item_id, min_confidence=None):
        seen.append(item_id)
        return real(item_id, min_confidence)

    w.service.sync_item = spy  # type: ignore[method-assign]
    real_sync = w.service._sync  # noqa: SLF001
    w.service._sync = lambda item_id, threshold: (  # type: ignore[method-assign]  # noqa: SLF001
        seen.append(item_id) or real_sync(item_id, threshold)
    )
    w.refresh()
    assert seen == [early.id, late.id]


def test_a_reverse_link_to_an_alert_outside_the_window_is_not_followed(w):
    original = w.put("orig", text="El rumor original", fetched=START + timedelta(hours=30))
    w.put(
        "copy",
        text="Otra version del rumor",
        near_of=original.id,
        label=Label.UNINTENTIONAL_RUMOR,
        fetched=START,
    )
    w.refresh()
    assert len(w.alerts()) == 2
