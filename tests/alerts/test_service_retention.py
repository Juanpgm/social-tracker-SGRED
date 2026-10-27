"""AL-8 and the lifetime of an alert next to its item and its source: retention by the age of the
alert, an alert that outlives its item, a source that was removed, and restarts."""

from datetime import timedelta

import pytest

from social_tracker.alerts.domain import AlertSettings
from social_tracker.items.domain import ItemStatus
from tests.actors import WHO
from tests.alerts.world import alert_world

DAY = timedelta(days=1)


def make(tmp_path, retention=180, name="retention.db"):
    world = alert_world(tmp_path, AlertSettings(retention_days=retention), name=name)
    world.llm.ready()
    return world


@pytest.fixture
def w(tmp_path):
    world = make(tmp_path)
    yield world
    world.close()


def age_of(w, days):
    """The instant `days` days before the clock."""
    return w.clock.now() - timedelta(days=days)


# ---- the 180-day retention ----


@pytest.mark.parametrize(
    ("age", "kept"),
    [
        (timedelta(days=179), True),
        (timedelta(days=180) - timedelta(microseconds=1), True),
        (timedelta(days=180), True),
        (timedelta(days=180) + timedelta(microseconds=1), False),
        (timedelta(days=181), False),
        (timedelta(days=400), False),
    ],
)
def test_an_alert_is_kept_until_it_is_strictly_older_than_180_days(w, age, kept):
    w.put("a", confidence=0.9, fetched=w.clock.now() - age)
    w.refresh()
    removed = w.service.purge_expired()
    assert (removed, len(w.alerts())) == ((0, 1) if kept else (1, 0))


def test_the_age_of_an_alert_is_that_of_its_latest_occurrence(w):
    text = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    w.put("a", text=text, confidence=0.9, fetched=age_of(w, 180) - timedelta(hours=20))
    w.put(
        "b",
        text=text,
        confidence=0.9,
        source="src_other",
        fetched=age_of(w, 179) - timedelta(hours=22),
    )
    w.refresh()
    assert w.only().occurrences == 2
    assert w.service.purge_expired() == 0  # a is old, b is not: the alert is not expired


def test_a_purge_with_nothing_old_is_a_no_op(w):
    w.put("a", confidence=0.9)
    w.refresh()
    state = w.alerts()
    assert w.service.purge_expired() == 0
    assert w.alerts() == state


def test_a_purged_alert_does_not_come_back_while_the_verdict_stays(w):
    item = w.put("a", confidence=0.9, fetched=age_of(w, 181))
    w.refresh()
    assert w.service.purge_expired() == 1
    assert w.refresh().created == 0
    assert w.service.rebuild().created == 0
    assert w.alerts() == []
    assert w.llm.items.get(item.id) is not None  # the item and its verdict are still there


def test_the_retention_can_be_shorter(tmp_path):
    world = make(tmp_path, retention=7, name="short.db")
    try:
        world.put("a", confidence=0.9, fetched=age_of(world, 7))
        world.put("b", confidence=0.9, fetched=age_of(world, 8))
        world.refresh()
        assert world.service.purge_expired() == 1
        assert len(world.alerts()) == 1
    finally:
        world.close()


def test_a_forgotten_item_is_not_raised_even_when_it_is_asked_for_by_name(w):
    item = w.put("a", confidence=0.9, fetched=age_of(w, 181))
    w.refresh()
    assert w.service.purge_expired() == 1
    assert w.service.sync_item(item.id).value == "skipped"
    assert w.alerts() == []
    assert w.repo.member(item.id) is None


# ---- retention 0: keep only until it has been looked at ----


def test_with_zero_days_an_alert_is_kept_until_it_has_been_displayed_and_handled(tmp_path):
    world = make(tmp_path, retention=0, name="zero.db")
    try:
        world.put("a", confidence=0.9)
        world.put("b", confidence=0.8)
        world.put("c", confidence=0.7)
        world.refresh()
        first, second, third = world.alerts()
        world.service.set_status(first.id, "seen", actor="ana", reason="read it")
        world.service.set_status(second.id, "dismissed", actor="ana", reason="not relevant")
        assert world.service.purge_expired() == 2
        assert [a.id for a in world.alerts()] == [third.id]
    finally:
        world.close()


def test_with_zero_days_a_handled_alert_does_not_come_back(tmp_path):
    world = make(tmp_path, retention=0, name="zero2.db")
    try:
        world.put("a", confidence=0.9)
        world.refresh()
        world.service.set_status(world.alerts()[0].id, "seen", actor="ana", reason="read it")
        world.service.purge_expired()
        assert world.refresh().created == 0
        assert world.service.rebuild().created == 0
        assert world.alerts() == []
    finally:
        world.close()


def test_with_zero_days_a_withdrawn_alert_goes_too(tmp_path):
    world = make(tmp_path, retention=0, name="zero3.db")
    try:
        world.put("a", confidence=0.9)
        world.refresh()
        with world.db.transaction() as conn:
            conn.execute("UPDATE classifications SET label = 'no_risk'")
        world.refresh()
        assert world.service.purge_expired() == 1
        assert world.alerts(status="withdrawn") == []
    finally:
        world.close()


def test_with_zero_days_a_new_alert_is_never_purged_however_old(tmp_path):
    world = make(tmp_path, retention=0, name="zero4.db")
    try:
        world.put("a", confidence=0.9, fetched=age_of(world, 1000))
        world.refresh()
        assert world.service.purge_expired() == 0
        assert len(world.alerts()) == 1
    finally:
        world.close()


@pytest.mark.parametrize("bad", [-1, -180, 3651, 10**9, 1.5, "180", None, True])
def test_a_negative_or_absurd_retention_is_refused(bad):
    from social_tracker.shared.errors import ValidationError

    with pytest.raises(ValidationError):
        AlertSettings(retention_days=bad)


# ---- an alert outlives its item ----


def test_the_alert_keeps_its_excerpt_and_evidence_when_the_item_is_purged(w):
    item = w.put("a", confidence=0.9, text="Mañana habrá un temblor fuerte en Cali", title="Alerta")
    w.refresh()
    before = w.only()
    w.llm.items.purge_fetched_before(item.fetched_at + DAY)
    assert w.llm.items.get(item.id) is None
    report = w.refresh()
    assert (report.errors, report.created, report.updated) == (0, 0, 0)
    after = w.only()
    assert after == before
    assert after.excerpt == "Alerta | Mañana habrá un temblor fuerte en Cali"
    assert after.evidence.stage == "rules"
    detail = w.service.detail(after.id)
    assert [m.item_id for m in detail.members] == [item.id]


def test_a_verdict_that_changes_after_its_item_is_gone_still_updates_the_alert(w):
    item = w.put("a", confidence=0.6)
    w.refresh()
    w.llm.items.purge_fetched_before(item.fetched_at + DAY)
    with w.db.transaction() as conn:
        conn.execute(
            "UPDATE classifications SET confidence = 0.95, label = 'deliberate_disinformation'"
        )
    report = w.refresh()
    assert report.updated == 1
    assert w.only().severity.value == "critical"
    assert w.only().excerpt  # kept from when the text was there


def test_an_item_that_is_purged_before_the_alert_step_ran_raises_no_alert(w):
    item = w.put("a", confidence=0.9)
    w.llm.items.purge_fetched_before(item.fetched_at + DAY)
    assert w.refresh().created == 0
    assert w.alerts() == []


def test_forgetting_old_verdicts_leaves_the_alert_alone(w):
    item = w.put("a", confidence=0.9)
    w.refresh()
    w.llm.items.purge_fetched_before(item.fetched_at + DAY)
    w.clock.advance(timedelta(days=400))
    w.llm.base.service.prune_history()
    assert w.llm.classifications.get(item.id) is None
    assert w.refresh().errors == 0
    assert w.only().occurrences == 1


# ---- an alert outlives its source ----


def test_when_the_source_is_deleted_the_alert_keeps_its_name_and_says_so(w):
    w.put("a", confidence=0.9)
    w.refresh()
    alert = w.only()
    assert w.service.removed_sources([alert.source_id]) == frozenset()
    w.llm.base.registry.delete("src_news", **WHO)
    assert w.service.removed_sources([alert.source_id]) == {"src_news"}
    after = w.only()
    assert (after.source_id, after.source_name) == ("src_news", "City News")
    detail = w.service.detail(alert.id)
    assert detail.removed_sources == {"src_news"}
    assert w.refresh().errors == 0


def test_only_the_deleted_sources_are_reported_removed(w):
    w.llm.ready("src_other")
    w.put("a", confidence=0.9)
    w.refresh()
    w.llm.base.registry.delete("src_other", **WHO)
    assert w.service.removed_sources(["src_news", "src_other", "src_never"]) == {
        "src_other",
        "src_never",
    }


def test_items_of_a_source_that_is_gone_keep_their_alerts_through_a_rebuild(w):
    w.put("a", confidence=0.9)
    w.refresh()
    w.llm.base.registry.delete("src_news", **WHO)
    report = w.service.rebuild()
    assert (report.created, report.withdrawn, report.errors) == (0, 0, 0)
    assert len(w.alerts()) == 1


def test_an_item_reclassified_as_new_does_not_break_its_alert(w):
    item = w.put("a", confidence=0.9)
    w.refresh()
    w.llm.classifications.delete(item.id)
    w.llm.items.set_status(item.id, ItemStatus.NEW)
    report = w.refresh()
    assert (report.errors, report.withdrawn) == (0, 0)
    assert len(w.alerts()) == 1
