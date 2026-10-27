"""The cached check inside the real cascade (SQLite): what the verdict records and changes."""

from datetime import timedelta

import pytest

from social_tracker.catalog.domain import CatalogEvent, CatalogSettings, CatalogSource
from social_tracker.catalog.service import CachedCatalogCheck
from social_tracker.classification.domain import Label
from social_tracker.classification.ports import CatalogResult
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.shared.errors import StorageError
from tests.classification.harness import PUBLISHED, START, sqlite_world

REPORT = "El SGC reportó un sismo de magnitud 4.2 en Cali hace un momento"
PREDICTION = "Mañana habrá un temblor de magnitud 7.5 en Cali, compartan este mensaje"
ATTRIBUTION = "Según la Secretaría de Gestión del Riesgo, evacuen la zona ya: sismo de magnitud 4.2"


class Lazy:
    """The check is built before the database exists: forward to the real one once it does."""

    def __init__(self) -> None:
        self.real = None
        self.depths: list[int] = []
        self.db = None

    def check(self, text, published_at):
        self.depths.append(self.db._tx_depth)
        return self.real.check(text, published_at)


@pytest.fixture
def setup(tmp_path):
    lazy = Lazy()
    world = sqlite_world(tmp_path / "w.db", catalog=lazy)
    repo = SqliteCatalogRepository(world.db)
    lazy.db = world.db
    lazy.real = CachedCatalogCheck(repo, world.clock, CatalogSettings())
    world.ready()
    yield world, repo, lazy
    world.close()


def fresh_cache(repo, *, magnitude=4.2, minutes=0):
    event = CatalogEvent(
        CatalogSource.USGS,
        "us1",
        PUBLISHED + timedelta(minutes=minutes),
        3.5,
        -76.5,
        magnitude,
        "Cali",
    )
    repo.upsert_events([event], START)
    for source in CatalogSource:
        repo.record_success(source, START)


def classify(world, text):
    item = world.add("ext-1", text)
    report = world.service.classify_pending()
    assert report.classified == 1, report
    return world.verdict(item)


def test_a_report_that_agrees_with_the_catalog_is_consistent_and_settled(setup):
    world, repo, _ = setup
    fresh_cache(repo, magnitude=4.2)
    verdict = classify(world, REPORT)
    assert verdict.catalog_status == "consistent"
    assert verdict.label is Label.NO_RISK
    assert not verdict.needs_llm


def test_a_report_the_catalog_contradicts_is_flagged_and_sent_to_review(setup):
    world, repo, _ = setup
    fresh_cache(repo, magnitude=6.0)
    verdict = classify(world, REPORT)
    assert verdict.catalog_status == "mismatch"
    assert "catalog_mismatch" in {s.name for s in verdict.signals}
    assert verdict.label is Label.UNCLASSIFIED
    assert verdict.needs_llm


def test_a_prediction_with_a_claimed_magnitude_nobody_recorded_gains_evidence(setup):
    world, repo, _ = setup
    plain = classify(world, PREDICTION.replace("magnitud 7.5", "fuerza"))
    fresh_cache(repo, magnitude=3.0)
    flagged = classify_second(world, PREDICTION)
    assert flagged.catalog_status == "mismatch"
    assert flagged.confidence > plain.confidence


def classify_second(world, text):
    item = world.add("ext-2", text)
    world.service.classify_pending()
    return world.verdict(item)


def test_a_consistent_attribution_is_less_suspect_than_an_unchecked_one(setup):
    world, repo, _ = setup
    unchecked = classify(world, ATTRIBUTION)
    assert unchecked.catalog_status == "unavailable"
    fresh_cache(repo, magnitude=4.2)
    checked = classify_second(
        world, ATTRIBUTION.replace("evacuen la zona ya:", "ordena evacuar el barrio de inmediato:")
    )
    assert checked.catalog_status == "consistent"
    assert checked.confidence < unchecked.confidence
    assert checked.priority < unchecked.priority
    assert unchecked.label is Label.DELIBERATE_DISINFORMATION
    assert checked.label is Label.UNCLASSIFIED  # still sent to review, never settled
    assert checked.needs_llm


def test_a_stale_cache_changes_nothing_and_says_unavailable(setup):
    world, repo, _ = setup
    fresh_cache(repo, magnitude=6.0)
    world.clock.advance(timedelta(hours=3))  # past max_age
    verdict = classify(world, REPORT)
    assert verdict.catalog_status == "unavailable"
    assert "catalog_mismatch" not in {s.name for s in verdict.signals}
    assert verdict.label is Label.NO_RISK


def test_a_text_with_no_claim_records_no_claim_and_changes_nothing(setup):
    world, repo, _ = setup
    fresh_cache(repo)
    verdict = classify(world, "Se sintió un temblor leve esta mañana en varios barrios de Cali")
    assert verdict.catalog_status == "no_claim"
    assert verdict.label is Label.NO_RISK


def test_the_check_runs_before_the_write_transaction_opens(setup):
    world, repo, lazy = setup
    fresh_cache(repo)
    classify(world, REPORT)
    assert lazy.depths == [0]


def test_a_cache_that_fails_to_read_never_blocks_the_classification(setup):
    world, repo, lazy = setup

    class Broken:
        def check(self, text, published_at):
            raise StorageError("database error: SQLITE_BUSY")

    world.service._catalog = Broken()
    verdict = classify(world, REPORT)
    assert verdict.catalog_status == "unavailable"
    assert verdict.label is Label.NO_RISK


def test_an_answer_that_is_not_a_status_name_is_unavailable(setup):
    world, _, _ = setup

    class Odd:
        def check(self, text, published_at):
            return CatalogResult("Mismatch!")

    world.service._catalog = Odd()
    assert classify(world, REPORT).catalog_status == "unavailable"
