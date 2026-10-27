"""The SQLite alert store (migration 0010): round trips, the list and its filters, merge lookup,
the sync queries and the retention purge, each at its boundaries."""

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from social_tracker.alerts.domain import (
    Alert,
    AlertEvent,
    AlertEventKind,
    AlertFilter,
    AlertStatus,
    Severity,
    alert_id_for,
    project,
)
from social_tracker.classification.domain import Classification, Label, Stage
from social_tracker.persistence.alerts_repo import SqliteAlertRepository
from social_tracker.persistence.classification_repo import SqliteClassificationRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.items_repo import SqliteItemRepository
from social_tracker.shared.errors import StorageError
from tests.alerts.support import HASH, T0, make_member
from tests.items.factories import make_item

DAY = timedelta(hours=24)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "alerts.db")
    yield database
    database.close()


@pytest.fixture
def repo(db: Database) -> SqliteAlertRepository:
    return SqliteAlertRepository(db)


def store(
    repo: SqliteAlertRepository,
    item: str = "itm_one",
    *,
    at=T0,
    status: AlertStatus = AlertStatus.NEW,
    withdrawn: bool = False,
    **member_overrides,
) -> Alert:
    """Store a one-member alert the way the service does and return it."""
    member = make_member(item, occurred_at=at, **member_overrides)
    alert = project(member.alert_id, [member], previous=None, now=at)
    changes = {"status": status}
    if withdrawn:
        changes |= {"withdrawn_at": at, "withdrawn_reason": "the verdict fell under the threshold"}
    alert = Alert(**{**alert.__dict__, **changes})
    repo.save(alert)
    repo.save_member(member)
    return alert


# ---- round trips ----


def test_an_alert_comes_back_as_it_went_in(repo):
    alert = store(repo, severity=Severity.HIGH, confidence=0.75, tags=("earthquake", "prediction"))
    assert repo.get(alert.id) == alert


def test_an_unknown_alert_is_none(repo):
    assert repo.get(alert_id_for("itm_none")) is None
    assert repo.member("itm_none") is None
    assert repo.members(alert_id_for("itm_none")) == []
    assert repo.events(alert_id_for("itm_none")) == []


def test_saving_twice_leaves_one_row_and_the_latest_values(repo):
    alert = store(repo)
    changed = Alert(**{**alert.__dict__, "status": AlertStatus.SEEN, "occurrences": 2})
    repo.save(changed)
    repo.save(changed)
    assert repo.get(alert.id) == changed
    assert repo.count(AlertFilter()) == 1


def test_a_member_comes_back_as_it_went_in_and_is_replaced_by_its_item(repo):
    alert = store(repo, "itm_a")
    first = repo.member("itm_a")
    assert first == make_member("itm_a", occurred_at=T0)
    newer = make_member("itm_a", confidence=0.9, active=False, label=Label.NO_RISK)
    repo.save_member(newer)
    assert repo.member("itm_a") == newer
    assert repo.members(alert.id) == [newer]


def test_members_are_listed_oldest_first_then_by_item_id(repo):
    alert = store(repo, "itm_b", at=T0)
    for item, at in (("itm_c", T0 + timedelta(hours=1)), ("itm_a", T0)):
        repo.save_member(make_member(item, alert_id=alert.id, occurred_at=at))
    assert [m.item_id for m in repo.members(alert.id)] == ["itm_a", "itm_b", "itm_c"]


def test_a_member_of_an_alert_that_does_not_exist_is_refused(repo):
    with pytest.raises(StorageError):
        repo.save_member(make_member("itm_a", alert_id=alert_id_for("itm_zzz")))


def test_the_trail_keeps_the_order_things_happened_in(repo):
    alert = store(repo)
    kinds = [AlertEventKind.CREATED, AlertEventKind.MERGED, AlertEventKind.STATUS_CHANGED]
    for n, kind in enumerate(kinds):
        repo.add_event(
            AlertEvent(alert.id, T0 - timedelta(hours=n), kind, "tester", "why", "a", "b")
        )
    assert [e.kind for e in repo.events(alert.id)] == kinds
    assert repo.events(alert.id)[0].actor == "tester"


def test_an_event_of_an_alert_that_does_not_exist_is_refused(repo):
    with pytest.raises(StorageError):
        repo.add_event(AlertEvent(alert_id_for("itm_q"), T0, AlertEventKind.CREATED, "x", "y"))


def test_a_stored_row_that_no_longer_builds_an_alert_is_a_storage_error(repo, db):
    alert = store(repo)
    with db.transaction() as conn:
        conn.execute("UPDATE alerts SET tags = '[\"Bad Tag\"]' WHERE id = ?", (alert.id,))
    with pytest.raises(StorageError):
        repo.get(alert.id)
    with pytest.raises(StorageError):
        repo.list(AlertFilter(), limit=10, offset=0)


def test_the_database_itself_refuses_nonsense(db):
    with pytest.raises(StorageError), db.transaction() as conn:
        conn.execute(
            "INSERT INTO alerts (id, label, severity) VALUES ('alt_zzz', 'no_risk', 'low')"
        )


# ---- finding the alert a copy belongs to ----


def merge(repo, **kw):
    args = {"content_hash": HASH, "related_items": (), "around": T0, "window": DAY}
    return repo.find_merge_target(**{**args, **kw})


def test_the_same_content_within_the_window_finds_the_alert(repo):
    alert = store(repo)
    assert merge(repo) == alert.id


@pytest.mark.parametrize(
    ("offset", "found"),
    [
        (timedelta(0), True),
        (timedelta(hours=23, minutes=59, seconds=59), True),
        (DAY, True),
        (DAY + timedelta(seconds=1), False),
        (-timedelta(hours=23, minutes=59, seconds=59), True),
        (-DAY, True),
        (-(DAY + timedelta(seconds=1)), False),
        (DAY + timedelta(microseconds=1), False),
    ],
)
def test_the_window_is_inclusive_at_24_hours_both_ways(repo, offset, found):
    alert = store(repo)
    assert merge(repo, around=T0 + offset) == (alert.id if found else None)


def test_other_content_does_not_merge(repo):
    store(repo)
    assert merge(repo, content_hash="b" * 64) is None


def test_a_copy_that_points_to_an_original_finds_its_alert_whatever_its_own_hash(repo):
    alert = store(repo, "itm_orig")
    assert merge(repo, content_hash="c" * 64, related_items=("itm_orig",)) == alert.id
    assert merge(repo, content_hash="c" * 64, related_items=("itm_other",)) is None
    assert merge(repo, content_hash="c" * 64, related_items=()) is None


def test_the_oldest_alert_wins_when_two_match(repo):
    older = store(repo, "itm_a", at=T0 - timedelta(hours=3))
    store(repo, "itm_b", at=T0 - timedelta(hours=2))
    assert merge(repo) == older.id


def test_a_withdrawn_alert_can_still_take_a_copy(repo):
    alert = store(repo, withdrawn=True, active=False)
    assert merge(repo) == alert.id


# ---- the list ----


def fill(repo) -> dict[str, Alert]:
    """Five alerts of different severity, label, source, status, tag and time."""
    spec = [
        ("itm_a", 0, dict(severity=Severity.LOW, confidence=0.55, tags=("earthquake",))),
        (
            "itm_b",
            1,
            dict(
                severity=Severity.CRITICAL,
                label=Label.DELIBERATE_DISINFORMATION,
                tags=("secretariat_attribution", "earthquake"),
                source_id="src_gov",
                source_name="Gov",
            ),
        ),
        (
            "itm_c",
            2,
            dict(severity=Severity.HIGH, label=Label.RECYCLED_CONTENT, tags=("prediction",)),
        ),
        ("itm_d", 3, dict(severity=Severity.MEDIUM, tags=("earthquake",))),
        ("itm_e", 4, dict(severity=Severity.HIGH, tags=("other_disaster",))),
    ]
    made = {}
    for item, hour, extra in spec:
        made[item] = store(
            repo, item, at=T0 + timedelta(hours=hour), content_hash=item[-1] * 64, **extra
        )
    return made


def ids(repo, flt=None, **page):
    return [
        a.item_id for a in repo.list(flt or AlertFilter(), **{"limit": 50, "offset": 0, **page})
    ]


def test_the_list_is_newest_first(repo):
    fill(repo)
    assert ids(repo) == ["itm_e", "itm_d", "itm_c", "itm_b", "itm_a"]


def test_equal_times_are_ordered_by_id_descending(repo):
    a = store(repo, "itm_a", at=T0)
    b = store(repo, "itm_b", at=T0)
    assert [x.id for x in repo.list(AlertFilter(), limit=10, offset=0)] == sorted(
        [a.id, b.id], reverse=True
    )


@pytest.mark.parametrize(
    ("flt", "expected"),
    [
        ({"severity": "high"}, ["itm_e", "itm_c"]),
        ({"severity": "low"}, ["itm_a"]),
        ({"label": "recycled_content"}, ["itm_c"]),
        ({"label": "deliberate_disinformation"}, ["itm_b"]),
        ({"source_id": "src_gov"}, ["itm_b"]),
        ({"source_id": "src_news"}, ["itm_e", "itm_d", "itm_c", "itm_a"]),
        ({"source_id": "src_unknown"}, []),
        ({"tag": "earthquake"}, ["itm_d", "itm_b", "itm_a"]),
        ({"tag": "secretariat_attribution"}, ["itm_b"]),
        ({"tag": "prediction"}, ["itm_c"]),
        ({"tag": "nothing"}, []),
        ({"status": "new"}, ["itm_e", "itm_d", "itm_c", "itm_b", "itm_a"]),
        ({"status": "seen"}, []),
        ({"severity": "high", "label": "recycled_content"}, ["itm_c"]),
        ({"severity": "high", "tag": "earthquake"}, []),
        ({"severity": "critical", "source_id": "src_gov", "tag": "earthquake"}, ["itm_b"]),
    ],
)
def test_each_filter_and_their_combinations(repo, flt, expected):
    fill(repo)
    assert ids(repo, AlertFilter(**flt)) == expected
    assert repo.count(AlertFilter(**flt)) == len(expected)


def test_the_date_range_is_inclusive_at_both_ends(repo):
    fill(repo)
    flt = AlertFilter(since=T0 + timedelta(hours=1), until=T0 + timedelta(hours=3))
    assert ids(repo, flt) == ["itm_d", "itm_c", "itm_b"]
    just_after = AlertFilter(since=T0 + timedelta(hours=1, microseconds=1))
    assert ids(repo, just_after) == ["itm_e", "itm_d", "itm_c"]
    just_before = AlertFilter(until=T0 + timedelta(hours=3, microseconds=-1))
    assert ids(repo, just_before) == ["itm_c", "itm_b", "itm_a"]
    single = AlertFilter(since=T0 + timedelta(hours=2), until=T0 + timedelta(hours=2))
    assert ids(repo, single) == ["itm_c"]


def test_a_source_filter_matches_any_linked_source(repo):
    alert = store(repo, "itm_a")
    repo.save_member(
        make_member("itm_b", alert_id=alert.id, source_id="src_second", source_name="Second")
    )
    assert ids(repo, AlertFilter(source_id="src_second")) == ["itm_a"]


def test_withdrawn_alerts_are_hidden_unless_asked_for(repo):
    store(repo, "itm_a", at=T0)
    store(repo, "itm_w", at=T0 + timedelta(hours=1), withdrawn=True, active=False)
    assert ids(repo) == ["itm_a"]
    assert ids(repo, AlertFilter(status="withdrawn")) == ["itm_w"]
    assert ids(repo, AlertFilter(status="new")) == ["itm_a"]
    assert repo.count(AlertFilter()) == 1
    assert repo.count(AlertFilter(status="withdrawn")) == 1


def test_status_filters_split_the_alerts(repo):
    store(repo, "itm_a", at=T0, status=AlertStatus.SEEN)
    store(repo, "itm_b", at=T0 + timedelta(hours=1), status=AlertStatus.DISMISSED)
    store(repo, "itm_c", at=T0 + timedelta(hours=2))
    assert ids(repo, AlertFilter(status="seen")) == ["itm_a"]
    assert ids(repo, AlertFilter(status="dismissed")) == ["itm_b"]
    assert ids(repo, AlertFilter(status="new")) == ["itm_c"]


@pytest.mark.parametrize(
    ("limit", "offset", "expected"),
    [
        (2, 0, ["itm_e", "itm_d"]),
        (2, 2, ["itm_c", "itm_b"]),
        (2, 4, ["itm_a"]),
        (2, 5, []),
        (2, 500, []),
        (5, 0, ["itm_e", "itm_d", "itm_c", "itm_b", "itm_a"]),
        (1, 4, ["itm_a"]),
        (200, 0, ["itm_e", "itm_d", "itm_c", "itm_b", "itm_a"]),
    ],
)
def test_pages_cover_the_list_without_gaps_or_overlaps(repo, limit, offset, expected):
    fill(repo)
    assert ids(repo, limit=limit, offset=offset) == expected


def test_an_empty_store_lists_nothing(repo):
    assert ids(repo) == []
    assert repo.count(AlertFilter()) == 0
    counts = repo.counts()
    assert (counts.active, counts.new, counts.withdrawn) == (0, 0, 0)


def test_the_counts_split_by_status_and_severity(repo):
    fill(repo)
    store(
        repo, "itm_s", at=T0, status=AlertStatus.SEEN, severity=Severity.HIGH, content_hash="9" * 64
    )
    store(repo, "itm_w", at=T0, withdrawn=True, active=False, content_hash="8" * 64)
    counts = repo.counts()
    assert (counts.active, counts.new, counts.withdrawn) == (6, 5, 1)
    assert dict(counts.by_status) == {"new": 5, "seen": 1}
    assert dict(counts.by_severity) == {"low": 1, "medium": 1, "high": 3, "critical": 1}
    assert dict(counts.new_by_severity) == {"low": 1, "medium": 1, "high": 2, "critical": 1}


# ---- the sync queries ----


def classify(db, item_id: str, label: Label, confidence: float, **overrides) -> Classification:
    fields = dict(
        item_id=item_id,
        label=label,
        confidence=confidence,
        stage=Stage.RULES,
        rationale="prediction_claim",
        classified_at=T0,
        classifier_version="cascade-2/topics-0",
    )
    fields.update(overrides)
    verdict = Classification(**fields)
    SqliteClassificationRepository(db).add(verdict)
    return verdict


def add_item(db, name: str):
    item = make_item(name, text=f"Texto distinto {name}", published_at=T0)
    SqliteItemRepository(db).add_new([item])
    return item


def sync(repo, **kw):
    args = {"min_confidence": 0.5, "full": False, "after": None, "limit": 100}
    return repo.sync_ids(**{**args, **kw})


def test_a_qualifying_verdict_of_an_existing_item_without_an_alert_is_pending(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, Label.UNINTENTIONAL_RUMOR, 0.5)
    assert sync(repo) == [item.id]


@pytest.mark.parametrize(
    ("label", "confidence", "pending"),
    [
        (Label.UNINTENTIONAL_RUMOR, 0.49, False),
        (Label.UNINTENTIONAL_RUMOR, 0.5, True),
        (Label.DELIBERATE_DISINFORMATION, 0.9, True),
        (Label.RECYCLED_CONTENT, 0.5, True),
        (Label.NO_RISK, 0.99, False),
        (Label.UNCLASSIFIED, 0.0, False),
    ],
)
def test_only_risk_verdicts_at_the_threshold_are_pending(repo, db, label, confidence, pending):
    item = add_item(db, "a")
    classify(db, item.id, label, confidence)
    assert (sync(repo) == [item.id]) is pending


def test_out_of_scope_verdicts_never_are(repo, db):
    from social_tracker.classification.domain import OutOfScopeReason

    item = add_item(db, "a")
    classify(db, item.id, Label.OUT_OF_SCOPE, 1.0, scope_reason=OutOfScopeReason.NO_TOPIC_MATCH)
    assert sync(repo, min_confidence=0.0) == []


def test_the_verdict_of_an_item_that_is_gone_is_not_pending(repo, db):
    classify(db, make_item("gone").id, Label.UNINTENTIONAL_RUMOR, 0.9)
    assert sync(repo) == []


def test_an_item_that_already_has_an_alert_is_not_pending_until_its_verdict_changes(repo, db):
    item = add_item(db, "a")
    verdict = classify(db, item.id, Label.UNINTENTIONAL_RUMOR, 0.6)
    store(
        repo,
        item.id,
        label=verdict.label,
        confidence=verdict.confidence,
        stage=verdict.stage.value,
        settled_at=verdict.settled_at,
    )
    assert sync(repo) == []
    with db.transaction() as conn:
        conn.execute("UPDATE classifications SET confidence = 0.61 WHERE item_id = ?", (item.id,))
    assert sync(repo) == [item.id]


@pytest.mark.parametrize(
    "change",
    [
        "label = 'deliberate_disinformation'",
        "confidence = 0.9",
        "stage = 'llm'",
        "settled_at = '2026-03-01T13:00:00+00:00', settle_reason = 'llm'",
        "label = 'no_risk'",
    ],
)
def test_any_change_of_the_verdict_a_member_was_judged_on_is_noticed(repo, db, change):
    item = add_item(db, "a")
    verdict = classify(db, item.id, Label.UNINTENTIONAL_RUMOR, 0.6)
    store(repo, item.id, label=verdict.label, confidence=0.6, stage="rules", settled_at=None)
    with db.transaction() as conn:
        conn.execute(f"UPDATE classifications SET {change} WHERE item_id = ?", (item.id,))
    assert sync(repo) == [item.id]


def test_a_full_sync_also_lists_the_members_whose_verdict_did_not_change(repo, db):
    item = add_item(db, "a")
    verdict = classify(db, item.id, Label.UNINTENTIONAL_RUMOR, 0.6)
    store(repo, item.id, label=verdict.label, confidence=0.6, stage="rules")
    assert sync(repo) == []
    assert sync(repo, full=True) == [item.id]


def test_sync_ids_are_ordered_paged_and_bounded(repo, db):
    items = [add_item(db, f"n{n}") for n in range(5)]
    for item in items:
        classify(db, item.id, Label.UNINTENTIONAL_RUMOR, 0.7)
    everything = sorted(i.id for i in items)
    assert sync(repo) == everything
    assert sync(repo, limit=2) == everything[:2]
    assert sync(repo, limit=2, after=everything[1]) == everything[2:4]
    assert sync(repo, limit=2, after=everything[4]) == []
    with pytest.raises(Exception):  # noqa: B017, PT011 - any refusal of a limit out of range
        sync(repo, limit=0)
    with pytest.raises(Exception):  # noqa: B017, PT011
        sync(repo, limit=1001)


# ---- retention ----


def test_purge_removes_the_alerts_strictly_older_than_the_cutoff(repo):
    now = T0 + timedelta(days=200)
    a179 = store(repo, "itm_a", at=now - timedelta(days=179), content_hash="1" * 64)
    a180 = store(repo, "itm_b", at=now - timedelta(days=180), content_hash="2" * 64)
    a181 = store(repo, "itm_c", at=now - timedelta(days=181), content_hash="3" * 64)
    a180_1 = store(
        repo, "itm_d", at=now - timedelta(days=180, microseconds=1), content_hash="4" * 64
    )
    removed = repo.purge(before=now - timedelta(days=180), handled=False, now=now)
    assert removed == 2
    assert repo.get(a179.id) is not None
    assert repo.get(a180.id) is not None
    assert repo.get(a181.id) is None
    assert repo.get(a180_1.id) is None


def test_purge_takes_the_members_and_the_trail_with_the_alert(repo):
    alert = store(repo, at=T0)
    repo.add_event(AlertEvent(alert.id, T0, AlertEventKind.CREATED, "x", "y"))
    repo.purge(before=T0 + timedelta(days=1), handled=False, now=T0 + timedelta(days=2))
    assert repo.get(alert.id) is None
    assert repo.member("itm_one") is None
    assert repo.events(alert.id) == []


def test_a_purged_item_is_remembered_so_that_it_does_not_raise_the_alert_again(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, Label.UNINTENTIONAL_RUMOR, 0.9)
    assert sync(repo) == [item.id]
    store(repo, item.id, label=Label.UNINTENTIONAL_RUMOR, confidence=0.9, stage="rules")
    repo.purge(before=T0 + timedelta(days=1), handled=False, now=T0 + timedelta(days=2))
    assert repo.is_forgotten(item.id)
    assert sync(repo) == []
    assert sync(repo, full=True) == []


def test_the_memory_of_a_purge_goes_when_the_verdict_does(repo, db):
    item = add_item(db, "a")
    classify(db, item.id, Label.UNINTENTIONAL_RUMOR, 0.9)
    store(repo, item.id, label=Label.UNINTENTIONAL_RUMOR, confidence=0.9, stage="rules")
    repo.purge(before=T0 + timedelta(days=1), handled=False, now=T0 + timedelta(days=2))
    assert repo.is_forgotten(item.id)
    with db.transaction() as conn:
        conn.execute("DELETE FROM classifications WHERE item_id = ?", (item.id,))
    repo.purge(before=None, handled=False, now=T0 + timedelta(days=3))
    assert not repo.is_forgotten(item.id)


def test_with_nothing_to_remove_a_purge_is_a_no_op(repo):
    alert = store(repo)
    assert repo.purge(before=None, handled=False, now=T0) == 0
    assert repo.purge(before=T0 - timedelta(days=1), handled=False, now=T0) == 0
    assert repo.get(alert.id) is not None


def test_handled_purges_the_alerts_that_have_been_looked_at_or_withdrawn(repo):
    new = store(repo, "itm_n", content_hash="1" * 64)
    seen = store(repo, "itm_s", status=AlertStatus.SEEN, content_hash="2" * 64)
    dismissed = store(repo, "itm_d", status=AlertStatus.DISMISSED, content_hash="3" * 64)
    withdrawn = store(repo, "itm_w", withdrawn=True, active=False, content_hash="4" * 64)
    assert repo.purge(before=None, handled=True, now=T0) == 3
    assert repo.get(new.id) is not None
    assert [repo.get(a.id) for a in (seen, dismissed, withdrawn)] == [None, None, None]


def test_the_foreign_keys_are_on_so_nothing_is_left_orphaned(db, repo):
    alert = store(repo)
    repo.purge(before=T0 + timedelta(days=1), handled=False, now=T0 + timedelta(days=2))
    with db.snapshot() as conn:
        for table in ("alert_items", "alert_events"):
            assert (
                conn.execute(
                    f"SELECT count(*) FROM {table} WHERE alert_id = ?", (alert.id,)
                ).fetchone()[0]
                == 0
            )


def test_saving_a_member_under_another_alert_moves_it(db, repo):
    first = store(repo, "itm_a")
    other = make_member("itm_a", alert_id=alert_id_for("itm_other"))
    repo.save(Alert(**{**first.__dict__, "id": other.alert_id, "item_id": "itm_other"}))
    repo.save_member(other)  # one row per item, whichever alert it is under
    with db.snapshot() as conn:
        rows = conn.execute("SELECT alert_id FROM alert_items WHERE item_id = 'itm_a'").fetchall()
    assert [r[0] for r in rows] == [other.alert_id]
