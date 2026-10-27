"""Two workers (two connections to one database file) must never give an item two alerts nor lose
an occurrence, and a rebuild running next to the worker's step must land where a quiet one would."""

import threading
from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertFilter
from tests.alerts.world import alert_world, attach
from tests.llm.harness import sqlite_llm

TEXT = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"


@pytest.fixture
def pair(tmp_path):
    """Two alert services over the same file, each with its own connection."""
    first = alert_world(tmp_path, name="shared.db")
    first.llm.ready()
    first.llm.ready("src_other")
    second = attach(sqlite_llm(tmp_path / "shared.db"), first.settings)
    yield first, second
    second.close()
    first.close()


def run_together(*calls):
    barrier = threading.Barrier(len(calls))
    errors: list[BaseException] = []

    def wrap(call):
        def go():
            barrier.wait()
            try:
                call()
            except BaseException as exc:  # noqa: BLE001 - reported by the caller's assertion
                errors.append(exc)

        return go

    threads = [threading.Thread(target=wrap(c)) for c in calls]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return errors


def test_two_workers_stepping_over_the_same_items_make_one_alert_each(pair):
    first, second = pair
    for n in range(12):
        first.put(f"n{n}", confidence=0.7)
    errors = run_together(first.service.refresh, second.service.refresh)
    assert errors == []
    assert first.repo.count(AlertFilter()) == 12
    assert all(a.occurrences == 1 for a in first.alerts())
    assert first.refresh().created == 0


def test_two_workers_each_handling_one_of_two_copies_make_one_alert_of_two(pair):
    first, second = pair
    for attempt in range(8):
        text = f"{TEXT} variante {attempt}"
        a = first.put(f"a{attempt}", text=text, confidence=0.6)
        b = first.put(f"b{attempt}", text=text, confidence=0.6, source="src_other")
        errors = run_together(
            lambda a=a: first.service.sync_item(a.id), lambda b=b: second.service.sync_item(b.id)
        )
        assert errors == []
        merged = first.alerts(source_id="src_other")
        mine = [x for x in merged if {m.item_id for m in first.repo.members(x.id)} >= {a.id, b.id}]
        assert len(mine) == 1, attempt
        assert mine[0].occurrences == 2
    assert first.repo.count(AlertFilter()) == 8


def test_many_workers_on_one_item_leave_one_alert(pair):
    first, second = pair
    item = first.put("solo", confidence=0.9)
    errors = run_together(
        *[lambda: first.service.sync_item(item.id) for _ in range(3)],
        *[lambda: second.service.sync_item(item.id) for _ in range(3)],
    )
    assert errors == []
    assert first.repo.count(AlertFilter()) == 1
    assert first.only().occurrences == 1
    assert [e.kind.value for e in first.repo.events(first.only().id)] == ["created"]


def test_a_rebuild_running_next_to_the_workers_step_ends_where_a_quiet_one_would(pair, tmp_path):
    first, second = pair
    for n in range(10):
        first.put(f"n{n}", confidence=0.5 + (n % 5) / 10)
        first.put(f"c{n}", text=f"{TEXT} {n}", confidence=0.6)
        first.put(f"d{n}", text=f"{TEXT} {n}", confidence=0.8, source="src_other")
    errors = run_together(first.service.refresh, second.service.rebuild, first.service.refresh)
    assert errors == []
    quiet = alert_world(tmp_path, name="quiet.db")
    quiet.llm.ready()
    quiet.llm.ready("src_other")
    try:
        for n in range(10):
            quiet.put(f"n{n}", confidence=0.5 + (n % 5) / 10)
            quiet.put(f"c{n}", text=f"{TEXT} {n}", confidence=0.6)
            quiet.put(f"d{n}", text=f"{TEXT} {n}", confidence=0.8, source="src_other")
        quiet.service.rebuild()

        def shape(world):
            return sorted(
                (a.occurrences, a.severity.value, round(a.confidence, 4)) for a in world.alerts()
            )

        assert shape(first) == shape(quiet)
        assert first.refresh().created == 0
    finally:
        quiet.close()


def test_a_worker_and_a_person_changing_the_status_while_the_step_runs_agree(pair):
    first, second = pair
    for n in range(6):
        first.put(f"n{n}", confidence=0.7)
    first.refresh()
    alert = first.alerts()[0]

    def mark():
        for target in ("seen", "dismissed"):
            second.service.set_status(alert.id, target, actor="ana", reason="reviewed")

    errors = run_together(first.service.refresh, mark, first.service.rebuild)
    assert errors == []
    stored = first.repo.get(alert.id)
    assert stored.status.value == "dismissed"
    assert stored.occurrences == 1
    first.clock.advance(timedelta(seconds=1))
