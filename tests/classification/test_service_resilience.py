"""A poison item, a slow catalog, sending items back, and the history pruning (W5, W6, C5, S10)."""

import threading
from datetime import timedelta

import pytest

from social_tracker.classification import service as service_module
from social_tracker.classification.domain import (
    Classification,
    ClassifierSettings,
    Label,
    OutOfScopeReason,
    Stage,
)
from social_tracker.classification.ports import CatalogResult
from social_tracker.items.domain import ItemStatus
from social_tracker.persistence.db import Database
from social_tracker.shared.errors import StorageError, ValidationError
from tests.actors import WHO
from tests.classification.harness import (
    OFF_TOPIC,
    PREDICTION,
    START,
    days,
    long_text,
    memory_world,
    sqlite_world,
)
from tests.fakes.classification_repositories import InMemoryClassificationRepository


def run(world, limit: int = 100):
    return world.service.classify_pending(limit)


class Poisoned(InMemoryClassificationRepository):
    """Refuses the verdict of the items named in `poison` with `error`. The names live on the
    class so that the rollback of a transaction (which restores instance attributes) keeps them."""

    poison: set[str] = set()
    error: Exception = ValidationError("stored data is invalid")

    def add(self, classification):
        if classification.item_id in type(self).poison:
            raise self.error
        return super().add(classification)


def poisoned(error: Exception | None = None) -> Poisoned:
    attrs = {"poison": set(), **({"error": error} if error else {})}
    return type("Poisoned", (Poisoned,), attrs)()


# ---- a poison item (W6) ----


def test_an_item_that_fails_three_runs_in_a_row_is_parked_and_stops_costing_anything():
    repo = poisoned()
    world = memory_world(classifications=repo)
    world.ready()
    bad = world.add("bad", long_text(1), published_at=START - days(5))  # first in the queue
    good = world.add("good", long_text(2))
    type(repo).poison.add(bad.id)
    reports = [run(world) for _ in range(4)]
    assert [(r.errors, r.failed) for r in reports] == [(1, 0), (1, 0), (1, 1), (0, 0)]
    assert world.status(bad) is ItemStatus.CLASSIFICATION_FAILED
    assert world.status(good) is ItemStatus.CLASSIFIED  # classified in the first run
    assert reports[0].classified == 1  # the item behind the poison one was not held up
    assert reports[3].skipped == 0  # parked: it is not even listed any more
    assert world.service.stats().failed == 1


def test_the_error_count_restarts_after_a_success():
    repo = poisoned()
    world = memory_world(classifications=repo)
    world.ready()
    item = world.add("flaky", long_text(3))
    type(repo).poison.add(item.id)
    run(world)
    run(world)  # two errors in a row
    type(repo).poison.clear()
    assert run(world).classified == 1  # it recovers...
    assert world.status(item) is ItemStatus.CLASSIFIED
    assert world.classifications.note_failure(item.id, START) == 1  # ...and the count started over


def test_the_number_of_consecutive_errors_is_configurable():
    repo = poisoned()
    world = memory_world(classifications=repo, settings=ClassifierSettings(max_item_failures=1))
    world.ready()
    item = world.add("bad", long_text(4))
    type(repo).poison.add(item.id)
    report = run(world)
    assert (report.errors, report.failed) == (1, 1)
    assert world.status(item) is ItemStatus.CLASSIFICATION_FAILED


def test_any_unexpected_error_of_one_item_is_contained_not_just_domain_errors():
    repo = poisoned(RuntimeError("a bug in some detector"))
    world = memory_world(classifications=repo)
    world.ready()
    bad, good = (
        world.add("bad", long_text(5), published_at=START - days(1)),
        world.add("g", long_text(6)),
    )
    type(repo).poison.add(bad.id)
    report = run(world)
    assert (report.errors, report.classified) == (1, 1)
    assert world.status(good) is ItemStatus.CLASSIFIED


def test_a_storage_error_is_not_blamed_on_the_item_and_ends_the_run():
    repo = poisoned(StorageError("database error: SQLITE_BUSY"))
    world = memory_world(classifications=repo)
    world.ready()
    item = world.add("a", long_text(7))
    type(repo).poison.add(item.id)
    for _ in range(5):
        report = run(world)
        assert (report.errors, report.storage_errors, report.failed) == (0, 1, 0)
    assert world.status(item) is ItemStatus.NEW  # never parked for a problem of the database


def test_a_storage_error_while_recording_a_failure_ends_the_run():
    class BrokenCounter(Poisoned):
        def note_failure(self, item_id, at):
            raise StorageError("database error: SQLITE_BUSY")

    repo = type("BrokenCounter", (BrokenCounter,), {"poison": set()})()
    world = memory_world(classifications=repo)
    world.ready()
    items = [world.add(f"n{n}", long_text(n)) for n in range(3)]
    type(repo).poison.add(min(items, key=lambda i: i.id).id)
    report = run(world)
    assert report.storage_errors == 1


def test_parked_items_can_be_retried_by_hand_and_a_fixed_item_is_classified():
    repo = poisoned()
    world = memory_world(classifications=repo, settings=ClassifierSettings(max_item_failures=1))
    world.ready()
    item = world.add("a", long_text(8))
    type(repo).poison.add(item.id)
    run(world)
    assert world.status(item) is ItemStatus.CLASSIFICATION_FAILED
    type(repo).poison.clear()
    assert run(world).classified == 0  # still parked: nothing retries it by itself
    assert world.service.reclassify(ItemStatus.CLASSIFICATION_FAILED) == 1
    assert world.status(item) is ItemStatus.NEW
    assert run(world).classified == 1
    assert world.service.stats().failed == 0


# ---- the catalog is resolved before the write transaction opens (W5) ----


class HoldsNothing:
    """A catalog that, while it is being asked, tries to take the write lock from another thread
    (a second writer in real life) and records whether it could."""

    def __init__(self, try_write) -> None:
        self.try_write = try_write
        self.results: list[bool] = []

    def check(self, text, published_at):
        outcome: list[bool] = []
        worker = threading.Thread(target=lambda: outcome.append(self.try_write()))
        worker.start()
        worker.join(5)
        self.results.append(bool(outcome and outcome[0]))
        return CatalogResult("checked")


def test_a_slow_catalog_does_not_hold_the_write_lock_of_the_memory_unit_of_work():
    holder: dict = {}

    def other_writer_can_enter() -> bool:
        lock = holder["world"].uow._lock
        acquired = lock.acquire(timeout=1)
        if acquired:
            lock.release()
        return acquired

    catalog = HoldsNothing(other_writer_can_enter)
    world = memory_world(catalog=catalog)
    holder["world"] = world
    world.ready()
    item = world.add("a", PREDICTION)
    run(world)
    assert catalog.results == [True]
    assert world.verdict(item).catalog_status == "checked"


def test_a_slow_catalog_does_not_hold_the_sqlite_write_lock(tmp_path):
    path = tmp_path / "slow.db"
    contenders: list[Database] = []

    def other_process_writes() -> bool:
        db = Database(path, busy_timeout_ms=100)
        contenders.append(db)
        try:
            with db.transaction():  # BEGIN IMMEDIATE: fails with a busy error if a writer holds it
                pass
        except StorageError:
            return False
        return True

    catalog = HoldsNothing(other_process_writes)
    world = sqlite_world(path, catalog=catalog)
    try:
        world.ready()
        item = world.add("a", PREDICTION)
        assert run(world).classified == 1
        assert catalog.results == [True]  # the second writer got in while the catalog was asked
        assert world.verdict(item).catalog_status == "checked"
    finally:
        for db in contenders:
            db.close()
        world.close()


def test_the_catalog_is_asked_once_per_earthquake_item_and_never_for_the_rest():
    catalog = HoldsNothing(lambda: True)
    world = memory_world(catalog=catalog)
    world.ready()
    world.add("a", PREDICTION)
    world.add("b", OFF_TOPIC)
    world.add("c", "Las lluvias provocaron un deslizamiento en la vía hacia Yumbo esta semana")
    run(world)
    assert len(catalog.results) == 1


def test_a_catalog_that_changes_its_mind_between_items_is_recorded_per_item():
    answers = iter(["checked", "mismatch"])

    class Changing:
        def check(self, text, published_at):
            return CatalogResult(next(answers))

    world = memory_world(catalog=Changing())
    world.ready()
    first = world.add("a", long_text(1), published_at=START - days(1))
    second = world.add("b", long_text(2))
    run(world)
    assert (world.verdict(first).catalog_status, world.verdict(second).catalog_status) == (
        "checked",
        "mismatch",
    )


def test_an_item_disabled_while_the_catalog_was_slow_is_left_untouched():
    holder: dict = {}

    class Disabling:
        def check(self, text, published_at):
            holder["world"].control.set_source_analysis("src_news", False, **WHO)
            return CatalogResult("checked")

    world = memory_world(catalog=Disabling())
    holder["world"] = world
    world.ready()
    item = world.add("a", PREDICTION)
    report = run(world)
    assert (report.classified, report.skipped) == (0, 1)  # the eligibility is checked again
    assert world.status(item) is ItemStatus.NEW


# ---- sending items back to the queue (C5) ----


def test_out_of_scope_items_are_judged_again_with_the_new_vocabulary():
    world = memory_world()
    world.ready()
    hail = world.add("a", "El granizo cayó sobre Cali esta tarde y dañó varios techos")
    quake = world.add("b", PREDICTION)
    run(world)
    assert world.verdict(hail).label is Label.OUT_OF_SCOPE
    world.topics.set_text("version: 1\nother_disaster: [granizo]\nplaces: [Cali]\n", **WHO)
    assert world.service.reclassify(ItemStatus.OUT_OF_SCOPE) == 1
    assert world.verdict(hail) is None
    assert world.status(hail) is ItemStatus.NEW
    assert world.status(quake) is ItemStatus.CLASSIFIED  # classified items are never touched
    run(world)
    assert world.verdict(hail).label is not Label.OUT_OF_SCOPE
    assert world.verdict(hail).classifier_version == "cascade-2/topics-1"


def test_reclassify_honours_the_limit_and_only_the_named_status():
    world = memory_world()
    world.ready()
    items = [world.add(f"n{n}", f"Hubo una fiesta {'x' * n} en el barrio") for n in range(1, 4)]
    run(world)
    assert world.service.reclassify(ItemStatus.OUT_OF_SCOPE, 2) == 2
    assert sorted(world.status(i).value for i in items) == ["new", "new", "out_of_scope"]
    assert world.service.reclassify(ItemStatus.CLASSIFICATION_FAILED) == 0


@pytest.mark.parametrize("status", [ItemStatus.NEW, ItemStatus.CLASSIFIED, "bogus", None])
def test_only_out_of_scope_and_failed_items_can_be_redone(status):
    world = memory_world()
    with pytest.raises(ValidationError):
        world.service.reclassify(status)


@pytest.mark.parametrize("limit", [0, -1, 1001, True, 1.5])
def test_reclassify_refuses_a_bad_limit(limit):
    with pytest.raises(ValidationError):
        memory_world().service.reclassify(ItemStatus.OUT_OF_SCOPE, limit)


def test_an_item_purged_between_the_listing_and_the_redo_is_skipped():
    world = memory_world()
    world.ready()
    gone = world.add("a", OFF_TOPIC)
    run(world)
    real_list = world.items.list_by_status

    def listing_then_purge(*args, **kwargs):
        found = real_list(*args, **kwargs)
        world.items._items.pop(gone.id)
        return found

    world.items.list_by_status = listing_then_purge
    assert world.service.reclassify(ItemStatus.OUT_OF_SCOPE) == 0


def test_the_english_reason_is_counted_apart_in_the_statistics():
    world = memory_world()
    world.ready()
    world.add("a", "The mayor said that the new bridge will be open for the people soon")
    world.add("b", OFF_TOPIC)
    run(world)
    assert world.service.stats().out_of_scope_by_reason == (
        ("no_topic_match", 1),
        ("non_spanish", 1),
    )


def test_a_non_spanish_item_is_out_of_scope_with_its_own_reason():
    world = memory_world()
    world.ready()
    item = world.add("a", "The mayor said that the new bridge will be open for the people soon")
    run(world)
    verdict = world.verdict(item)
    assert (verdict.label, verdict.scope_reason) == (
        Label.OUT_OF_SCOPE,
        OutOfScopeReason.NON_SPANISH,
    )
    assert "not Spanish" in verdict.rationale


# ---- pruning the verdict history (S10) ----


def old_verdict(item_id: str, when) -> Classification:
    return Classification(
        item_id=item_id,
        label=Label.NO_RISK,
        confidence=0.6,
        stage=Stage.RULES,
        rationale="old",
        classified_at=when,
        classifier_version="v",
    )


def test_old_verdicts_of_vanished_items_are_forgotten_and_the_rest_kept():
    world = memory_world()
    world.ready()
    present = world.add("present", long_text(1))
    run(world)
    world.clock.advance(days(200))  # past the 180 days of retention
    world.classifications.add(old_verdict("itm_vanished", START - days(1)))
    world.classifications.add(old_verdict("itm_young", START + days(100)))
    assert world.service.prune_history() == 1
    assert world.classifications.get("itm_vanished") is None
    assert world.classifications.get("itm_young") is not None  # not old enough yet
    assert world.verdict(present) is not None  # its item still exists


def test_the_boundary_of_the_verdict_retention_is_strict():
    world = memory_world(settings=ClassifierSettings(classification_retention_days=10))
    world.classifications.add(old_verdict("itm_edge", START - days(10)))
    world.classifications.add(old_verdict("itm_over", START - days(10) - timedelta(microseconds=1)))
    assert world.service.prune_history() == 1
    assert world.classifications.get("itm_edge") is not None
    assert world.classifications.get("itm_over") is None


def test_pruning_pages_through_many_old_verdicts_whose_items_still_exist(monkeypatch):
    monkeypatch.setattr(service_module, "PRUNE_PAGE", 3)
    world = memory_world(settings=ClassifierSettings(classification_retention_days=1))
    world.ready()
    kept = [world.add(f"k{n}", long_text(n)) for n in range(7)]  # old verdicts that must stay
    for item in kept:
        world.classifications.add(old_verdict(item.id, START - days(50) - timedelta(seconds=1)))
    for n in range(8):  # and old verdicts of vanished items behind them
        world.classifications.add(old_verdict(f"itm_vanished{n}", START - days(40)))
    assert world.service.prune_history() == 8
    assert all(world.classifications.get(i.id) is not None for i in kept)


def test_pruning_an_empty_history_forgets_nothing():
    assert memory_world().service.prune_history() == 0


def test_pruning_forgets_the_error_counters_of_the_forgotten_verdicts():
    world = memory_world()
    world.classifications.add(old_verdict("itm_vanished", START - days(400)))
    world.classifications.note_failure("itm_vanished", START)
    world.service.prune_history()
    assert world.classifications.note_failure("itm_vanished", START) == 1
