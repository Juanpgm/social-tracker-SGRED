"""Composition of the classifier: settings from the environment, wired services, the worker hook."""

from pathlib import Path

import pytest

from social_tracker.bootstrap import (
    CLASSIFICATION_RETENTION_DAYS_ENV,
    DB_PATH_ENV,
    NEAR_DUPLICATE_CANDIDATES_ENV,
    NEAR_DUPLICATE_THRESHOLD_ENV,
    RECYCLE_DAYS_ENV,
    build_container,
    build_worker,
    parse_classifier_settings,
)
from social_tracker.classification.domain import ClassifierSettings, Label
from social_tracker.items.domain import ItemStatus
from social_tracker.shared.errors import ValidationError
from tests.actors import WHO
from tests.collector.harness import FEED, FakeFetcher, dated_now
from tests.sources.factories import payload


def env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {DB_PATH_ENV: str(tmp_path / "c.db"), **extra}


def test_settings_default_when_the_variables_are_blank_or_absent():
    assert parse_classifier_settings({}) == ClassifierSettings()
    blank = {
        RECYCLE_DAYS_ENV: "  ",
        NEAR_DUPLICATE_THRESHOLD_ENV: "",
        NEAR_DUPLICATE_CANDIDATES_ENV: " ",
        CLASSIFICATION_RETENTION_DAYS_ENV: "",
    }
    assert parse_classifier_settings(blank) == ClassifierSettings()


@pytest.mark.parametrize(
    ("name", "field", "raw", "value"),
    [
        (NEAR_DUPLICATE_CANDIDATES_ENV, "candidate_limit", "1", 1),
        (NEAR_DUPLICATE_CANDIDATES_ENV, "candidate_limit", " 1000 ", 1000),
        (CLASSIFICATION_RETENTION_DAYS_ENV, "classification_retention_days", "1", 1),
        (CLASSIFICATION_RETENTION_DAYS_ENV, "classification_retention_days", "3650", 3650),
    ],
)
def test_the_candidate_limit_and_the_verdict_retention_come_from_the_environment(
    name, field, raw, value
):
    assert getattr(parse_classifier_settings({name: raw}), field) == value


@pytest.mark.parametrize("name", [NEAR_DUPLICATE_CANDIDATES_ENV, CLASSIFICATION_RETENTION_DAYS_ENV])
@pytest.mark.parametrize("raw", ["0", "-1", "3651", "1.5", "abc", "1e2", "7 days", "+7", "٣"])
def test_an_invalid_candidate_limit_or_retention_is_refused_with_the_variable_name(name, raw):
    with pytest.raises(ValidationError) as caught:
        parse_classifier_settings({name: raw})
    assert name in caught.value.message


def test_the_candidate_limit_stops_at_one_thousand_and_the_retention_does_not():
    with pytest.raises(ValidationError):
        parse_classifier_settings({NEAR_DUPLICATE_CANDIDATES_ENV: "1001"})
    assert (
        parse_classifier_settings(
            {CLASSIFICATION_RETENTION_DAYS_ENV: "1001"}
        ).classification_retention_days
        == 1001
    )


@pytest.mark.parametrize(("raw", "days"), [("1", 1), ("7", 7), (" 14 ", 14), ("3650", 3650)])
def test_the_recycled_content_window_comes_from_the_environment(raw, days):
    assert parse_classifier_settings({RECYCLE_DAYS_ENV: raw}).recycle_days == days


@pytest.mark.parametrize("raw", ["0", "-1", "3651", "1.5", "abc", "1e2", "٣", "7 days", "+7"])
def test_an_invalid_window_is_refused_with_the_variable_name(raw):
    with pytest.raises(ValidationError) as caught:
        parse_classifier_settings({RECYCLE_DAYS_ENV: raw})
    assert RECYCLE_DAYS_ENV in caught.value.message


@pytest.mark.parametrize(
    ("raw", "threshold"),
    [("0.8", 0.8), ("1", 1.0), ("1.0", 1.0), ("0.0001", 0.0001), (" 0.5 ", 0.5)],
)
def test_the_similarity_threshold_comes_from_the_environment(raw, threshold):
    settings = parse_classifier_settings({NEAR_DUPLICATE_THRESHOLD_ENV: raw})
    assert settings.near_duplicate_threshold == threshold


@pytest.mark.parametrize(
    "raw", ["0", "0.0", "1.01", "2", "-0.5", ".5", "abc", "0,8", "1e-1", "0.12345", "nan", "80%"]
)
def test_an_invalid_threshold_is_refused_with_the_variable_name(raw):
    with pytest.raises(ValidationError) as caught:
        parse_classifier_settings({NEAR_DUPLICATE_THRESHOLD_ENV: raw})
    assert NEAR_DUPLICATE_THRESHOLD_ENV in caught.value.message


def test_an_invalid_setting_stops_the_start_before_the_database_is_opened(tmp_path):
    with pytest.raises(ValidationError):
        build_container(env(tmp_path, **{RECYCLE_DAYS_ENV: "0"}))
    assert not (tmp_path / "c.db").exists()


def test_the_container_exposes_the_classifier_and_the_topic_configuration(tmp_path):
    container = build_container(env(tmp_path, **{RECYCLE_DAYS_ENV: "3"}))
    try:
        assert container.topics.current().version == 0
        assert container.classifier.classify_pending().eligible_sources == 0
        container.topics.set_text("version: 1\nearthquake: [sismo]\n", **WHO)
        assert container.topics.current().version == 1
    finally:
        container.close()


def test_the_configured_window_reaches_the_classifier(tmp_path):
    container = build_container(env(tmp_path, **{RECYCLE_DAYS_ENV: "3"}))
    try:
        assert container.classifier_settings.recycle_days == 3
    finally:
        container.close()


def test_a_worker_pass_collects_then_classifies(tmp_path):
    built = build_worker(env(tmp_path), fetcher=FakeFetcher(dated_now(FEED)))
    try:
        built.sources.add(payload("news_rss", id="src_news", name="News"), **WHO)
        built.sources.enable("src_news", **WHO)
        built.control.set_source_analysis("src_news", True, **WHO)
        built.control.set_global_analysis(True, **WHO)
        summary = built.worker.run(once=True, reset_analysis=False)
        assert summary is not None
        assert summary.classification.classified == 1  # the quake entry
        assert summary.classification.out_of_scope == 1  # the other one
        items = {i.title: i for i in built.items.list_items()}
        assert items["Sismo en Cali"].status is ItemStatus.CLASSIFIED
        assert items["Otra noticia"].status is ItemStatus.OUT_OF_SCOPE
        verdict = built.container.classifier.get(items["Sismo en Cali"].id)
        assert verdict.label in (Label.UNCLASSIFIED, Label.NO_RISK)
    finally:
        built.close()


def test_the_hourly_purge_also_forgets_old_verdicts_of_items_that_are_gone(tmp_path):
    from datetime import UTC, datetime, timedelta

    from social_tracker.bootstrap import _purge_everything
    from social_tracker.classification.domain import Classification, Stage

    container = build_container(env(tmp_path))
    try:
        repo = container.classifier._classifications
        old = datetime.now(UTC) - timedelta(days=400)
        recent = datetime.now(UTC) - timedelta(days=10)
        for name, when in (("itm_oldverdict", old), ("itm_newverdict", recent)):
            repo.add(
                Classification(
                    item_id=name,
                    label=Label.NO_RISK,
                    confidence=0.6,
                    stage=Stage.RULES,
                    rationale="r",
                    classified_at=when,
                    classifier_version="v",
                )
            )
        _purge_everything(container)()
        assert repo.get("itm_oldverdict") is None  # older than 180 days and no item behind it
        assert repo.get("itm_newverdict") is not None
    finally:
        container.close()


def test_the_verdict_history_is_pruned_even_when_the_item_purge_fails(tmp_path):
    from social_tracker.bootstrap import _purge_everything

    container = build_container(env(tmp_path))
    calls = []

    class Boom:
        def purge_expired(self):
            raise RuntimeError("boom")

    class Spy:
        def prune_history(self):
            calls.append("pruned")

    try:
        fake = type("C", (), {"items": Boom(), "classifier": Spy()})()
        with pytest.raises(RuntimeError):
            _purge_everything(fake)()
        assert calls == ["pruned"]
    finally:
        container.close()
