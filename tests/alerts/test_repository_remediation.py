"""The alert store after the remediation (migration 0012): chronological and dirty-driven sync,
verdict signatures of forgotten items, the provisional spare in the purge, the poison-item backoff,
the overflow of a capped alert, reverse links, absorbed trails and capped trails."""

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from social_tracker.alerts.domain import (
    Alert,
    AlertEvent,
    AlertEventKind,
    AlertStatus,
    ForgottenSignature,
    Severity,
    alert_id_for,
)
from social_tracker.classification.domain import Label
from social_tracker.persistence.alerts_repo import SqliteAlertRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from tests.alerts.support import HASH, T0, make_evidence, make_member
from tests.alerts.test_repository import classify, store
from tests.items.factories import make_item

DAY = timedelta(hours=24)
RUMOR = Label.UNINTENTIONAL_RUMOR


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "alerts.db")
    yield database
    database.close()


@pytest.fixture
def repo(db: Database) -> SqliteAlertRepository:
    return SqliteAlertRepository(db)


def add_item(db, name: str, fetched=T0):
    item = make_item(name, text=f"Texto distinto {name}", published_at=T0, fetched_at=fetched)
    SqliteItemRepository(db).add_new([item])
    return item


def refs(repo, **kw):
    args = {"min_confidence": 0.5, "full": False, "after": None, "limit": 100}
    return repo.sync_ids(**{**args, **kw})


def ids(found):
    return [r.item_id for r in found]


# ---- the new fields round-trip ----


def test_the_new_alert_fields_come_back_as_they_went_in(repo):
    alert = store(repo)
    changed = Alert(
        **{**alert.__dict__, "downgraded_by_model": True, "overflow": 7, "events_dropped": 3}
    )
    repo.save(changed)
    assert repo.get(alert.id) == changed


def test_an_event_keeps_its_absorbed_mark(repo):
    alert = store(repo)
    repo.add_event(AlertEvent(alert.id, T0, AlertEventKind.MERGED, "a", "b", from_absorbed=True))
    repo.add_event(AlertEvent(alert.id, T0, AlertEventKind.MERGED, "a", "c"))
    assert [e.from_absorbed for e in repo.events(alert.id)] == [True, False]


# ---- chronological order ----


def test_the_sync_lists_items_in_the_order_they_were_fetched_then_by_id(repo, db):
    late = add_item(db, "late", fetched=T0 + timedelta(hours=2))
    early = add_item(db, "early", fetched=T0)
    middle = add_item(db, "middle", fetched=T0 + timedelta(hours=1))
    for item in (late, early, middle):
        classify(db, item.id, RUMOR, 0.7)
    assert ids(refs(repo)) == [early.id, middle.id, late.id]


def test_items_fetched_together_are_ordered_by_id(repo, db):
    items = [add_item(db, f"n{n}") for n in range(4)]
    for item in items:
        classify(db, item.id, RUMOR, 0.7)
    assert ids(refs(repo)) == sorted(i.id for i in items)


def test_paging_continues_from_the_cursor_of_the_last_item(repo, db):
    items = [add_item(db, f"n{n}", fetched=T0 + timedelta(minutes=n)) for n in range(5)]
    for item in items:
        classify(db, item.id, RUMOR, 0.7)
    order = [i.id for i in items]
    first = refs(repo, limit=2)
    assert ids(first) == order[:2]
    second = refs(repo, limit=2, after=first[-1].cursor)
    assert ids(second) == order[2:4]
    assert ids(refs(repo, limit=2, after=second[-1].cursor)) == order[4:]
    assert refs(repo, after=refs(repo)[-1].cursor) == []


def test_members_whose_item_is_gone_are_still_listed_by_their_occurrence(repo, db):
    store(repo, "itm_gone", at=T0 + timedelta(hours=5))
    item = add_item(db, "x", fetched=T0)
    classify(db, item.id, RUMOR, 0.7)
    assert ids(refs(repo, full=True)) == [item.id, "itm_gone"]


# ---- the dirty marks ----


def test_a_new_risk_verdict_is_dirty_until_it_is_cleared(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    assert ids(repo.dirty_ids(after=None, limit=10)) == [item.id]
    repo.clear_dirty(item.id)
    assert repo.dirty_ids(after=None, limit=10) == []
    repo.clear_dirty(item.id)  # clearing twice is harmless


def test_a_changed_verdict_is_dirty_again(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    repo.clear_dirty(item.id)
    with db.transaction() as conn:
        conn.execute("UPDATE classifications SET confidence = 0.8 WHERE item_id = ?", (item.id,))
    assert ids(repo.dirty_ids(after=None, limit=10)) == [item.id]


def test_dirty_items_are_chronological_and_paged(repo, db):
    items = [add_item(db, f"n{n}", fetched=T0 - timedelta(minutes=n)) for n in range(4)]
    for item in items:
        classify(db, item.id, RUMOR, 0.7)
    order = [i.id for i in reversed(items)]
    page = repo.dirty_ids(after=None, limit=3)
    assert ids(page) == order[:3]
    assert ids(repo.dirty_ids(after=page[-1].cursor, limit=3)) == order[3:]


def test_a_dirty_item_without_an_item_row_is_still_listed_so_that_it_can_be_cleared(repo, db):
    classify(db, "itm_gone", RUMOR, 0.7)
    assert ids(repo.dirty_ids(after=None, limit=10)) == ["itm_gone"]


def test_the_limit_of_the_dirty_page_is_bounded(repo):
    with pytest.raises(Exception):  # noqa: B017, PT011
        repo.dirty_ids(after=None, limit=0)
    with pytest.raises(Exception):  # noqa: B017, PT011
        repo.dirty_ids(after=None, limit=1001)


# ---- forgotten items remember the verdict they were forgotten on ----


def forget(repo, db, name="a", label=RUMOR, confidence=0.9, **overrides):
    item = add_item(db, name)
    classify(db, item.id, label, confidence, **overrides)
    store(repo, item.id, label=label, confidence=confidence, stage="rules", settled_at=None)
    repo.purge(before=T0 + DAY, handled=False, now=T0 + 2 * DAY)
    return item


def test_a_forgotten_item_keeps_the_signature_of_its_verdict(repo, db):
    item = forget(repo, db)
    sig = repo.forgotten_signature(item.id)
    assert sig == ForgottenSignature(RUMOR.value, 0.9, None)
    assert sig.matches(RUMOR.value, 0.9, None)


@pytest.mark.parametrize(
    ("label", "confidence", "settled"),
    [
        (Label.DELIBERATE_DISINFORMATION.value, 0.9, None),
        (RUMOR.value, 0.91, None),
        (RUMOR.value, 0.9, T0),
    ],
)
def test_a_verdict_that_differs_in_any_part_of_the_signature_does_not_match(
    repo, db, label, confidence, settled
):
    item = forget(repo, db)
    assert not repo.forgotten_signature(item.id).matches(label, confidence, settled)


def test_an_unchanged_forgotten_verdict_is_not_listed_but_a_changed_one_is(repo, db):
    item = forget(repo, db)
    assert refs(repo) == []
    with db.transaction() as conn:
        conn.execute(
            "UPDATE classifications SET label = 'deliberate_disinformation', confidence = 0.95 "
            "WHERE item_id = ?",
            (item.id,),
        )
    assert ids(refs(repo)) == [item.id]


def test_a_settled_verdict_after_the_forget_is_listed(repo, db):
    item = forget(repo, db)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE classifications SET settled_at = '2026-03-02T00:00:00+00:00', "
            "settle_reason = 'llm' WHERE item_id = ?",
            (item.id,),
        )
    assert ids(refs(repo)) == [item.id]


def test_clearing_the_entry_makes_the_item_eligible_at_once(repo, db):
    item = forget(repo, db)
    repo.clear_forgotten(item.id)
    assert repo.is_forgotten(item.id) is False
    assert repo.forgotten_signature(item.id) is None
    assert ids(refs(repo)) == [item.id]


def test_an_entry_from_before_the_migration_has_no_signature_and_keeps_suppressing(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.9)
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO alert_forgotten (item_id, forgotten_at) VALUES (?, ?)",
            (item.id, "2026-03-01T12:00:00+00:00"),
        )
    sig = repo.forgotten_signature(item.id)
    assert sig == ForgottenSignature(None, None, None)
    assert sig.matches("anything", 0.1, T0)
    assert refs(repo) == []


def test_the_memory_of_a_forget_outlives_the_verdict_of_its_item(repo, db):
    """An old GUID listed again after its item was purged must not bring its alert back: the
    entry stays (with the signature) until it is a year old."""
    item = forget(repo, db)
    with db.transaction() as conn:
        conn.execute("DELETE FROM classifications WHERE item_id = ?", (item.id,))
    repo.purge(before=None, handled=False, now=T0 + 3 * DAY)
    assert repo.is_forgotten(item.id) is True
    classify(db, item.id, RUMOR, 0.9)  # the item is listed again, with the same verdict
    assert refs(repo) == []


def test_an_entry_is_dropped_when_it_is_older_than_a_year(repo, db):
    item = forget(repo, db)
    repo.purge(before=None, handled=False, now=T0 + 2 * DAY + timedelta(days=364))
    assert repo.is_forgotten(item.id) is True
    repo.purge(before=None, handled=False, now=T0 + 2 * DAY + timedelta(days=366))
    assert repo.is_forgotten(item.id) is False


def test_the_overflow_items_of_a_purged_alert_are_forgotten_too(repo, db):
    alert = store(repo, "itm_a", at=T0)
    repo.add_overflow(alert.id, make_member("itm_over", confidence=0.6, content_hash="4" * 64))
    repo.purge(before=T0 + DAY, handled=False, now=T0 + 2 * DAY)
    assert repo.forgotten_signature("itm_over") == ForgottenSignature(RUMOR.value, 0.6, None)
    assert repo.overflow_of("itm_over") is None


# ---- the purge spares what still waits for the model ----


def provisional_alert(repo, status=AlertStatus.DISMISSED):
    return store(
        repo,
        "itm_p",
        status=status,
        evidence=make_evidence(provisional=True),
        content_hash="5" * 64,
    )


def test_a_handled_purge_spares_an_alert_whose_member_waits_for_the_model(repo):
    waiting = provisional_alert(repo)
    done = store(repo, "itm_d", status=AlertStatus.DISMISSED, content_hash="6" * 64)
    assert repo.purge(before=None, handled=True, now=T0) == 1
    assert repo.get(waiting.id) is not None
    assert repo.get(done.id) is None


def test_a_handled_purge_does_not_spare_a_withdrawn_provisional_member(repo):
    gone = store(
        repo,
        "itm_p",
        withdrawn=True,
        active=False,
        evidence=make_evidence(provisional=True),
        content_hash="5" * 64,
    )
    assert repo.purge(before=None, handled=True, now=T0) == 1
    assert repo.get(gone.id) is None


def test_an_old_provisional_alert_still_goes_with_the_age_cutoff(repo):
    waiting = provisional_alert(repo)
    assert repo.purge(before=T0 + DAY, handled=False, now=T0 + 2 * DAY) == 1
    assert repo.get(waiting.id) is None


# ---- poison items ----


BACKOFF = {
    "park_after": 3,
    "backoff_base": timedelta(seconds=30),
    "backoff_cap": timedelta(hours=1),
}


def test_a_failing_item_waits_longer_each_time_and_is_parked_at_the_limit(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    assert repo.record_failure(item.id, T0, **BACKOFF) == 1
    assert refs(repo, now=T0) == []  # backing off
    assert ids(refs(repo, now=T0 + timedelta(seconds=31))) == [item.id]
    assert repo.record_failure(item.id, T0 + timedelta(seconds=31), **BACKOFF) == 2
    assert refs(repo, now=T0 + timedelta(seconds=31 + 59)) == []  # 60 s this time
    assert ids(refs(repo, now=T0 + timedelta(seconds=31 + 61))) == [item.id]
    assert repo.record_failure(item.id, T0 + timedelta(minutes=5), **BACKOFF) == 3
    assert refs(repo, now=T0 + timedelta(days=30)) == []  # parked: never again by itself
    assert repo.failure_counts(T0 + timedelta(days=30)) == (0, 1)


def test_an_item_is_tried_again_exactly_when_its_backoff_ends(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    repo.record_failure(item.id, T0, **BACKOFF)
    ends = T0 + timedelta(seconds=30)
    assert refs(repo, now=ends - timedelta(microseconds=1)) == []
    assert ids(refs(repo, now=ends)) == [item.id]
    assert repo.dirty_ids(after=None, limit=10, now=ends - timedelta(microseconds=1)) == []
    assert ids(repo.dirty_ids(after=None, limit=10, now=ends)) == [item.id]
    assert repo.failure_counts(ends - timedelta(microseconds=1)) == (1, 0)
    assert repo.failure_counts(ends) == (0, 0)


def test_the_backoff_is_capped(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    for _ in range(20):
        repo.record_failure(
            item.id,
            T0,
            park_after=100,
            backoff_base=timedelta(minutes=10),
            backoff_cap=timedelta(hours=1),
        )
    assert ids(refs(repo, now=T0 + timedelta(hours=1, seconds=1))) == [item.id]


def test_a_failure_is_forgotten_when_the_item_works(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    repo.record_failure(item.id, T0, **BACKOFF)
    repo.clear_failure(item.id)
    assert ids(refs(repo, now=T0)) == [item.id]
    assert repo.failure_counts(T0) == (0, 0)
    repo.clear_failure(item.id)  # harmless twice


def test_failure_counts_tell_retrying_from_parked(repo, db):
    a, b = add_item(db, "a"), add_item(db, "b")
    repo.record_failure(a.id, T0, **BACKOFF)
    for _ in range(3):
        repo.record_failure(b.id, T0, **BACKOFF)
    assert repo.failure_counts(T0) == (1, 1)
    assert repo.failure_counts(T0 + DAY) == (0, 1)  # a's backoff is over: it will be tried


def test_releasing_the_parked_gives_them_another_chance(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    for _ in range(3):
        repo.record_failure(item.id, T0, **BACKOFF)
    assert refs(repo, now=T0) == []  # parked
    assert repo.release_parked() == 1
    assert ids(refs(repo, now=T0)) == [item.id]  # at once, not after the old backoff
    assert repo.release_parked() == 0


def test_a_failing_item_is_hidden_from_the_dirty_list_too(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    repo.record_failure(item.id, T0, **BACKOFF)
    assert repo.dirty_ids(after=None, limit=10, now=T0) == []
    assert ids(repo.dirty_ids(after=None, limit=10, now=T0 + DAY)) == [item.id]
    assert ids(repo.dirty_ids(after=None, limit=10)) == [item.id]  # no clock: no backoff


# ---- overflow ----


def test_an_overflow_item_is_remembered_with_its_alert_and_goes_with_it(repo):
    alert = store(repo, "itm_a")
    over = make_member("itm_over", confidence=0.6, content_hash="4" * 64)
    repo.add_overflow(alert.id, over)
    assert repo.overflow_of("itm_over") == alert.id
    repo.add_overflow(alert.id, over)  # twice: one row
    other = store(repo, "itm_b", content_hash="2" * 64)
    repo.move_overflow(alert.id, other.id)
    assert repo.overflow_of("itm_over") == other.id
    repo.delete(other.id)
    assert repo.overflow_of("itm_over") is None


def test_an_overflow_item_is_not_pending(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    alert = store(repo, "itm_other", content_hash="3" * 64)
    repo.add_overflow(alert.id, make_member(item.id, confidence=0.7, content_hash="4" * 64))
    assert item.id not in ids(refs(repo))
    assert item.id not in ids(refs(repo, full=True))


def test_a_member_given_up_is_still_found_by_its_content(repo):
    alert = store(repo, "itm_a", content_hash="2" * 64)
    repo.add_overflow(alert.id, make_member("itm_gone", content_hash="7" * 64, occurred_at=T0))
    found = repo.find_merge_targets(content_hash="7" * 64, related_items=[], around=T0, window=DAY)
    assert found == [alert.id]
    assert (
        repo.find_merge_targets(
            content_hash="7" * 64,
            related_items=[],
            around=T0 + DAY + timedelta(seconds=1),
            window=DAY,
        )
        == []
    )


def test_a_member_given_up_is_still_found_by_the_copies_that_point_at_it(repo):
    alert = store(repo, "itm_a", content_hash="2" * 64)
    repo.add_overflow(alert.id, make_member("itm_gone", content_hash="7" * 64, occurred_at=T0))
    found = repo.find_merge_targets(
        content_hash="8" * 64, related_items=["itm_gone"], around=T0, window=DAY
    )
    assert found == [alert.id]


# ---- members and the cap ----


def test_members_are_counted_and_deleted_one_by_one(repo):
    alert = store(repo, "itm_a")
    repo.save_member(make_member("itm_b", alert_id=alert.id, occurred_at=T0 + timedelta(hours=1)))
    assert repo.count_members(alert.id) == 2
    repo.delete_member("itm_b")
    assert repo.count_members(alert.id) == 1
    repo.delete_member("itm_b")  # harmless twice
    assert repo.count_members(alert.id) == 1


def test_the_weakest_member_is_the_inactive_then_the_least_severe_never_the_protected(repo):
    alert = store(repo, "itm_first", severity=Severity.MEDIUM, confidence=0.6, at=T0)
    rows = [
        ("itm_hi", Severity.HIGH, 0.8, True, 1),
        ("itm_lo", Severity.LOW, 0.5, True, 2),
        ("itm_dead", Severity.CRITICAL, 0.9, False, 3),
    ]
    for item, severity, confidence, active, hour in rows:
        repo.save_member(
            make_member(
                item,
                alert_id=alert.id,
                severity=severity,
                confidence=confidence,
                active=active,
                occurred_at=T0 + timedelta(hours=hour),
            )
        )
    assert repo.weakest_member(alert.id, protect="itm_hi").item_id == "itm_dead"
    repo.delete_member("itm_dead")
    assert repo.weakest_member(alert.id, protect="itm_hi").item_id == "itm_lo"
    repo.delete_member("itm_lo")
    # only the earliest member and the protected lead remain: nothing can be evicted
    assert repo.weakest_member(alert.id, protect="itm_hi") is None


# ---- reverse links ----


def stored_copy(repo, name, *, near_of=None, duplicate_of=None, at=T0, content_hash=HASH):
    return store(
        repo,
        name,
        at=at,
        content_hash=content_hash,
        evidence=make_evidence(near_of=near_of, duplicate_of=duplicate_of),
    )


def test_the_original_finds_the_alert_of_the_copies_that_point_at_it(repo):
    alert = stored_copy(repo, "itm_copy", near_of="itm_orig", content_hash="9" * 64)
    found = repo.find_merge_targets(
        content_hash="8" * 64, related_items=[], around=T0, window=DAY, item_id="itm_orig"
    )
    assert found == [alert.id]


def test_the_original_finds_the_copies_that_say_duplicate_of(repo):
    alert = stored_copy(repo, "itm_copy", duplicate_of="itm_orig", content_hash="9" * 64)
    found = repo.find_merge_targets(
        content_hash="8" * 64, related_items=[], around=T0, window=DAY, item_id="itm_orig"
    )
    assert found == [alert.id]


def test_a_reverse_link_outside_the_window_is_not_a_link(repo):
    stored_copy(repo, "itm_copy", near_of="itm_orig", at=T0, content_hash="9" * 64)
    assert (
        repo.find_merge_targets(
            content_hash="8" * 64,
            related_items=[],
            around=T0 + DAY + timedelta(seconds=1),
            window=DAY,
            item_id="itm_orig",
        )
        == []
    )


def test_without_an_item_id_only_the_forward_links_are_followed(repo):
    stored_copy(repo, "itm_copy", near_of="itm_orig", content_hash="9" * 64)
    assert (
        repo.find_merge_targets(content_hash="8" * 64, related_items=[], around=T0, window=DAY)
        == []
    )


def test_several_alerts_are_found_oldest_first_each_once(repo):
    older = stored_copy(repo, "itm_c1", near_of="itm_orig", at=T0, content_hash="9" * 64)
    newer = stored_copy(
        repo, "itm_c2", near_of="itm_orig", at=T0 + timedelta(hours=1), content_hash="7" * 64
    )
    found = repo.find_merge_targets(
        content_hash="8" * 64, related_items=[], around=T0, window=DAY, item_id="itm_orig"
    )
    assert found == [older.id, newer.id]


# ---- the trail: absorbed, capped, last N ----


def trail(repo, alert, n, kind=AlertEventKind.MERGED, start=0):
    for i in range(n):
        repo.add_event(
            AlertEvent(alert.id, T0 + timedelta(minutes=start + i), kind, "system:alerts", f"e{i}")
        )


def test_the_trail_of_an_absorbed_alert_is_copied_with_its_original_times(repo):
    kept = store(repo, "itm_a")
    gone = store(repo, "itm_b", content_hash="2" * 64)
    trail(repo, gone, 2, start=10)
    repo.add_event(
        AlertEvent(gone.id, T0 + timedelta(minutes=3), AlertEventKind.STATUS_CHANGED, "op", "seen")
    )
    left = repo.copy_events(gone.id, kept.id, limit=10)
    assert left == 0
    copied = repo.events(kept.id)
    assert [e.reason for e in copied] == ["e0", "e1", "seen"]
    assert all(e.from_absorbed for e in copied)
    assert copied[2].at == T0 + timedelta(minutes=3)
    assert copied[2].actor == "op"
    assert len(repo.events(gone.id)) == 3  # the source is untouched until it is deleted


def test_a_copy_of_a_trail_stops_at_the_limit_and_says_how_many_were_left_out(repo):
    kept = store(repo, "itm_a")
    gone = store(repo, "itm_b", content_hash="2" * 64)
    trail(repo, gone, 7)
    assert repo.copy_events(gone.id, kept.id, limit=4) == 3
    assert len(repo.events(kept.id)) == 4
    assert repo.copy_events(gone.id, kept.id, limit=0) == 7


def test_events_can_be_read_last_n_in_the_order_they_happened(repo):
    alert = store(repo)
    trail(repo, alert, 10)
    assert repo.count_events(alert.id) == 10
    assert [e.reason for e in repo.events(alert.id, limit=3)] == ["e7", "e8", "e9"]
    assert len(repo.events(alert.id)) == 10
    assert repo.events(alert.id, limit=0) == []


def test_dropping_events_is_counted_on_the_alert(repo):
    alert = store(repo)
    repo.drop_events(alert.id, 1)
    repo.drop_events(alert.id, 4)
    assert repo.get(alert.id).events_dropped == 5
    repo.drop_events(alert_id_for("itm_none"), 2)  # an alert that is gone: harmless


# ---- boundaries the mutation sweep found unpinned ----


def test_a_limited_copy_of_a_trail_keeps_what_matters_before_the_routine_entries(repo):
    kept = store(repo, "itm_a")
    gone = store(repo, "itm_b", content_hash="2" * 64)
    trail(repo, gone, 5)  # five routine entries first
    for n, reason in enumerate(("seen by ana", "dismissed by ana")):
        repo.add_event(
            AlertEvent(
                gone.id,
                T0 + timedelta(minutes=30 + n),
                AlertEventKind.STATUS_CHANGED,
                "ana",
                reason,
            )
        )
    assert repo.copy_events(gone.id, kept.id, limit=3) == 4
    copied = repo.events(kept.id)
    assert [e.reason for e in copied if e.kind is AlertEventKind.STATUS_CHANGED] == [
        "seen by ana",
        "dismissed by ana",
    ]
    assert len(copied) == 3


def test_the_memory_of_a_forget_is_kept_for_exactly_a_year(repo, db):
    item = forget(repo, db)
    forgotten_at = T0 + 2 * DAY
    exactly = forgotten_at + timedelta(days=365)
    repo.purge(before=None, handled=False, now=exactly)
    assert repo.is_forgotten(item.id) is True  # exactly a year old is still kept
    repo.purge(before=None, handled=False, now=exactly + timedelta(microseconds=1))
    assert repo.is_forgotten(item.id) is False


def test_each_failure_doubles_the_wait(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, RUMOR, 0.7)
    many = {**BACKOFF, "park_after": 10, "backoff_cap": timedelta(days=1)}
    waits = []
    for n in range(1, 6):
        repo.record_failure(item.id, T0, **many)
        waits.append(n)
        expected = timedelta(seconds=30) * 2 ** (n - 1)
        assert refs(repo, now=T0 + expected - timedelta(microseconds=1)) == []
        assert ids(refs(repo, now=T0 + expected)) == [item.id]
    assert waits == [1, 2, 3, 4, 5]
