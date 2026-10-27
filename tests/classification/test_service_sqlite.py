"""The classifier on SQLite: concurrency, a locked database, scale and the retention interplay."""

import threading
from dataclasses import replace

from social_tracker.classification.domain import Label
from social_tracker.classification.similarity import band_keys, pack
from social_tracker.classification.similarity import signature as minhash
from social_tracker.classification.text import tokens
from social_tracker.items.domain import ItemStatus
from social_tracker.items.service import ItemService
from social_tracker.persistence.db import Database
from tests.classification.harness import (
    OFF_TOPIC,
    PREDICTION,
    PUBLISHED,
    START,
    days,
    long_text,
    sqlite_world,
)


def test_a_full_run_on_sqlite_classifies_and_advances_the_items(tmp_path):
    world = sqlite_world(tmp_path / "a.db")
    try:
        world.ready()
        quake, off = world.add("a", PREDICTION), world.add("b", OFF_TOPIC)
        report = world.service.classify_pending(10)
        assert (report.classified, report.out_of_scope, report.errors) == (1, 1, 0)
        assert world.status(quake) is ItemStatus.CLASSIFIED
        assert world.status(off) is ItemStatus.OUT_OF_SCOPE
        assert world.verdict(quake).label is Label.UNINTENTIONAL_RUMOR
        assert world.service.classify_pending(10).classified == 0  # idempotent
    finally:
        world.close()


def test_the_full_text_index_is_untouched_by_classification(tmp_path):
    world = sqlite_world(tmp_path / "fts.db")
    try:
        world.ready()
        item = world.add("a", PREDICTION)
        world.service.classify_pending(10)
        assert [i.id for i in world.items.search("temblor")] == [item.id]
        assert world.db.query("INSERT INTO items_fts(items_fts) VALUES ('integrity-check')") == []
    finally:
        world.close()


def test_two_classifiers_racing_over_the_same_items_classify_each_one_once(tmp_path):
    path = tmp_path / "race.db"
    first = sqlite_world(path)
    second = sqlite_world(path)
    try:
        first.ready()
        items = [first.add(f"n{n}", long_text(n)) for n in range(30)]
        reports = {}
        barrier = threading.Barrier(2)

        def work(name, world):
            barrier.wait()
            reports[name] = world.service.classify_pending(100)

        threads = [
            threading.Thread(target=work, args=("first", first)),
            threading.Thread(target=work, args=("second", second)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert all(not t.is_alive() for t in threads)
        assert reports["first"].errors == reports["second"].errors == 0
        assert reports["first"].classified + reports["second"].classified == 30
        assert first.classifications.stats().total == 30
        assert {first.status(i) for i in items} == {ItemStatus.CLASSIFIED}
    finally:
        first.close()
        second.close()


def test_a_locked_database_is_reported_as_an_error_and_nothing_is_lost(tmp_path):
    path = tmp_path / "locked.db"
    world = sqlite_world(path, busy_timeout_ms=50)
    holder = Database(path)
    try:
        world.ready()
        item = world.add("a", PREDICTION)
        with holder.transaction():
            report = world.service.classify_pending(10)
        assert (report.classified, report.errors, report.storage_errors) == (0, 0, 1)
        assert world.verdict(item) is None
        assert world.status(item) is ItemStatus.NEW
        assert world.service.classify_pending(10).classified == 1  # released: it resumes
    finally:
        holder.close()
        world.close()


def _statements(world) -> list[str]:
    seen: list[str] = []
    connection = world.db._connection()
    connection.set_trace_callback(seen.append)
    try:
        world.service.classify_pending(1)
    finally:
        connection.set_trace_callback(None)
    return seen


def _bulk_signatures(world, count: int) -> None:
    with world.db.transaction() as conn:
        for n in range(count):
            sig = minhash(tokens(f"el sismo numero {n} fue distinto " + long_text(n + 1000, 12)))
            conn.execute(
                "INSERT INTO item_signatures (item_id, content_hash, ref_at, minhash) "
                "VALUES (?, ?, ?, ?)",
                (f"itm_bulk{n:06d}", f"{n:064x}", PUBLISHED.isoformat(), pack(sig)),
            )
            conn.executemany(
                "INSERT INTO item_signature_bands (item_id, band, key) VALUES (?, ?, ?)",
                [(f"itm_bulk{n:06d}", band, key) for band, key in enumerate(band_keys(sig))],
            )


def test_classifying_one_item_costs_the_same_with_one_signature_as_with_five_thousand(tmp_path):
    small, big = sqlite_world(tmp_path / "small.db"), sqlite_world(tmp_path / "big.db")
    try:
        for world, count in ((small, 1), (big, 5000)):
            world.ready()
            _bulk_signatures(world, count)
            world.add("a", PREDICTION)
        few, many = _statements(small), _statements(big)
        assert len(many) == len(few)  # no statement per stored signature: the lookup is indexed
        assert big.verdict(big.items.list(limit=1)[0]) is not None
    finally:
        small.close()
        big.close()


def test_retention_zero_purges_classified_and_out_of_scope_items_once_and_keeps_the_verdicts(
    tmp_path,
):
    world = sqlite_world(tmp_path / "purge.db")
    try:
        world.ready()
        quake, off, pending = (
            world.add("a", PREDICTION),
            world.add("b", OFF_TOPIC),
            world.add("c", long_text(9), published_at=START + days(1)),
        )
        world.service.classify_pending(2)  # the two oldest only
        assert world.status(pending) is ItemStatus.NEW
        world.clock.advance(days(2))  # past the first-day grace of an out-of-scope item
        purge = ItemService(world.items, world.uow, world.clock, retention_days=0).purge_expired()
        assert purge.deleted == 2
        assert world.items.get(quake.id) is None
        assert world.items.get(off.id) is None
        assert world.items.get(pending.id) is not None  # unclassified items stay
        assert world.verdict(quake) is not None  # the verdicts and signatures outlive the items
        assert world.signatures.get(quake.id) is not None
        # The feed lists the purged entries again: they are not ingested a second time.
        again = world.items.add_new([quake, off])
        assert (again.new, again.duplicate) == (0, 2)
        assert world.items.count() == 1
        assert world.db.query("INSERT INTO items_fts(items_fts) VALUES ('integrity-check')") == []
    finally:
        world.close()


def test_a_copy_of_a_purged_original_is_still_recognised_by_its_signature(tmp_path):
    world = sqlite_world(tmp_path / "purged-original.db")
    try:
        world.ready("src_a")
        world.add_source("src_b")
        original = world.add("a", long_text(5), source_id="src_a")
        world.service.classify_pending(10)
        queued = world.verdict(original)
        if queued.needs_llm:  # an item still waiting for the language model is not purged
            assert world.classifications.settle(replace(queued, needs_llm=False, priority=0.0))
        ItemService(world.items, world.uow, world.clock, retention_days=0).purge_expired()
        assert world.items.get(original.id) is None
        copy = world.add("b", long_text(5), source_id="src_b", published_at=PUBLISHED + days(8))
        world.service.classify_pending(10)
        assert world.verdict(copy).recycled_of == original.id
    finally:
        world.close()
