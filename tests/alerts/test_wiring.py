"""The alert step in the running system: its settings from the environment, its place in the
worker's pass, the hourly housekeeping and the whole path from a feed to an alert."""

import logging
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from social_tracker.alerts.domain import AlertFilter, AlertSettings, Severity, SyncReport
from social_tracker.bootstrap import (
    ALERT_CRITICAL_CATALOG_ENV,
    ALERT_CRITICAL_DELIBERATE_ENV,
    ALERT_CRITICAL_WINDOW_ENV,
    ALERT_DEDUP_WINDOW_ENV,
    ALERT_HIGH_ENV,
    ALERT_MEDIUM_ENV,
    ALERT_RETENTION_DAYS_ENV,
    DB_PATH_ENV,
    build_container,
    build_worker,
    parse_alert_settings,
)
from social_tracker.collector.worker import Worker
from social_tracker.shared.errors import StorageError, ValidationError
from tests.actors import WHO
from tests.collector.harness import FakeFetcher, World, dated_now, memory_world, rss
from tests.sources.factories import payload

RUMOR_TEXT = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"


# ---- settings from the environment ----


def test_without_variables_the_settings_are_the_products():
    assert parse_alert_settings({}) == AlertSettings()
    blank = {
        name: "  "
        for name in (
            ALERT_RETENTION_DAYS_ENV,
            ALERT_DEDUP_WINDOW_ENV,
            ALERT_CRITICAL_WINDOW_ENV,
            ALERT_HIGH_ENV,
            ALERT_MEDIUM_ENV,
            ALERT_CRITICAL_DELIBERATE_ENV,
            ALERT_CRITICAL_CATALOG_ENV,
        )
    }
    assert parse_alert_settings(blank) == AlertSettings()


def test_every_number_can_be_set():
    settings = parse_alert_settings(
        {
            ALERT_RETENTION_DAYS_ENV: "90",
            ALERT_DEDUP_WINDOW_ENV: "12",
            ALERT_CRITICAL_WINDOW_ENV: "6",
            ALERT_CRITICAL_DELIBERATE_ENV: "0.9",
            ALERT_CRITICAL_CATALOG_ENV: "0.75",
            ALERT_HIGH_ENV: "0.65",
            ALERT_MEDIUM_ENV: "0.4",
        }
    )
    assert settings == AlertSettings(
        critical_disinformation=0.9,
        critical_catalog=0.75,
        high=0.65,
        medium=0.4,
        critical_window=timedelta(hours=6),
        dedup_window=timedelta(hours=12),
        retention_days=90,
    )


@pytest.mark.parametrize("days", ["0", "1", "180", "3650", " 7 "])
def test_retention_accepts_zero_up_to_ten_years(days):
    assert parse_alert_settings({ALERT_RETENTION_DAYS_ENV: days}).retention_days == int(days)


@pytest.mark.parametrize(
    "days", ["-1", "3651", "1.5", "abc", "1e3", "+5", "٣٠", "30 days", "9" * 9]
)
def test_a_bad_retention_is_refused_naming_the_variable(days):
    with pytest.raises(ValidationError) as caught:
        parse_alert_settings({ALERT_RETENTION_DAYS_ENV: days})
    assert ALERT_RETENTION_DAYS_ENV in caught.value.message


@pytest.mark.parametrize("name", [ALERT_DEDUP_WINDOW_ENV, ALERT_CRITICAL_WINDOW_ENV])
@pytest.mark.parametrize("hours", ["0", "-1", "721", "1.5", "abc", "24h"])
def test_a_bad_window_is_refused_naming_the_variable(name, hours):
    with pytest.raises(ValidationError) as caught:
        parse_alert_settings({name: hours})
    assert name in caught.value.message


@pytest.mark.parametrize(
    "name",
    [
        ALERT_CRITICAL_DELIBERATE_ENV,
        ALERT_CRITICAL_CATALOG_ENV,
        ALERT_HIGH_ENV,
        ALERT_MEDIUM_ENV,
    ],
)
@pytest.mark.parametrize("value", ["-0.1", "1.1", "2", "abc", "0.12345", ".5", "1e-1", "50%"])
def test_a_bad_threshold_is_refused_naming_the_variable(name, value):
    with pytest.raises(ValidationError) as caught:
        parse_alert_settings({name: value})
    assert name in caught.value.message


def test_thresholds_that_contradict_each_other_are_refused():
    with pytest.raises(ValidationError) as caught:
        parse_alert_settings({ALERT_MEDIUM_ENV: "0.9", ALERT_HIGH_ENV: "0.7"})
    assert "alert settings" in caught.value.message


def test_a_critical_threshold_of_zero_is_refused():
    with pytest.raises(ValidationError):
        parse_alert_settings({ALERT_CRITICAL_DELIBERATE_ENV: "0"})


def test_a_bad_setting_stops_the_start_before_the_database_is_opened(tmp_path: Path):
    with pytest.raises(ValidationError):
        build_container({DB_PATH_ENV: str(tmp_path / "x.db"), ALERT_RETENTION_DAYS_ENV: "-5"})
    assert not (tmp_path / "x.db").exists()


def test_the_container_carries_the_alert_service_with_its_settings(tmp_path: Path):
    container = build_container(
        {DB_PATH_ENV: str(tmp_path / "c.db"), ALERT_RETENTION_DAYS_ENV: "45"}
    )
    try:
        assert container.alert_settings.retention_days == 45
        assert container.alerts.settings.retention_days == 45
        assert container.alerts.list().total == 0
        assert container.alerts.refresh() == SyncReport()
    finally:
        container.close()


# ---- the worker's pass ----


class Recorder:
    def __init__(self, result=None, error=None) -> None:
        self.calls = 0
        self.result, self.error = result, error

    def __call__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def make(world: World, alerts, *, escalate=None, classify=None, stop=None) -> Worker:
    return Worker(
        world.collector,
        world.control,
        world.clock,
        stop or threading.Event(),
        sleeper=lambda seconds: None,
        classify=classify,
        escalate=escalate,
        alerts=alerts,
    )


@pytest.fixture
def world() -> World:
    return memory_world(FakeFetcher())


def test_the_alert_step_comes_last_after_the_model_has_settled_what_it_could(world):
    world.ready()
    order = []
    worker = make(
        world,
        lambda: (order.append("alerts"), "alerted")[1],
        classify=lambda: (order.append("classify"), "report")[1],
        escalate=lambda: (order.append("escalate"), "judged")[1],
    )
    summary = worker.run_pass()
    assert order == ["classify", "escalate", "alerts"]
    assert summary.alerts == "alerted"


def test_no_alert_step_while_global_analysis_is_off(world):
    world.add_source()
    recorder = Recorder("x")
    summary = make(world, recorder).run_pass()
    assert (summary.global_off, recorder.calls, summary.alerts) == (True, 0, None)


def test_without_the_hook_a_pass_is_what_it_was(world):
    world.ready()
    assert make(world, None).run_pass().alerts is None


def test_the_hook_runs_on_every_pass(world):
    world.analysis_on()
    recorder = Recorder("x")
    worker = make(world, recorder)
    worker.run_pass()
    worker.run_pass()
    assert recorder.calls == 2


@pytest.mark.parametrize(
    "error", [StorageError("database error: SQLITE_BUSY"), RuntimeError("boom"), ValueError("x")]
)
def test_a_failing_alert_step_costs_that_step_only(world, error, caplog):
    world.ready()
    with caplog.at_level(logging.DEBUG):
        summary = make(
            world,
            Recorder(error=error),
            classify=Recorder("report"),
            escalate=Recorder("judged"),
        ).run_pass()
    assert (summary.alerts, summary.classification, summary.escalation) == (
        None,
        "report",
        "judged",
    )
    assert summary.errors == 1
    assert "boom" not in caplog.text


def test_a_failing_escalation_does_not_stop_the_alerts_of_what_the_cascade_settled(world):
    world.ready()
    alerts = Recorder("alerted")
    summary = make(world, alerts, escalate=Recorder(error=RuntimeError("boom"))).run_pass()
    assert (alerts.calls, summary.alerts) == (1, "alerted")


def test_a_stop_requested_before_the_step_skips_it(world):
    world.ready()
    stop = threading.Event()
    recorder = Recorder("x")

    def escalate():
        stop.set()
        return "judged"

    summary = make(world, recorder, escalate=escalate, stop=stop).run_pass()
    assert (recorder.calls, summary.alerts, summary.escalation) == (0, None, "judged")


def test_the_log_line_names_counts_and_never_text(world, caplog):
    world.ready()
    with caplog.at_level(logging.INFO):
        make(world, lambda: SyncReport(created=3, merged=1, withdrawn=2)).run_pass()
    assert "created 3" in caplog.text
    assert "merged 1" in caplog.text
    assert "withdrawn 2" in caplog.text


# ---- from a feed to an alert ----


def env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {DB_PATH_ENV: str(tmp_path / "w.db"), **extra}


def test_a_rumor_in_a_feed_becomes_an_alert_in_one_pass_and_only_one_in_the_next(tmp_path):
    feed = dated_now(
        rss(("a", "Alerta", RUMOR_TEXT), ("b", "Partido", "El equipo ganó el partido de ayer"))
    )
    built = build_worker(env(tmp_path), fetcher=FakeFetcher(feed))
    try:
        built.sources.add(payload(id="src_news", enabled=True), **WHO)
        built.control.set_source_analysis("src_news", True, **WHO)
        built.worker.start()
        built.control.set_global_analysis(True, **WHO)
        first = built.worker.run_pass()
        alerts = built.container.alerts
        found = alerts.list(AlertFilter()).alerts
        assert len(found) == 1
        alert = found[0]
        assert alert.label.value == "unintentional_rumor"
        assert alert.source_id == "src_news"
        assert alert.severity in (Severity.MEDIUM, Severity.HIGH)
        assert alert.excerpt.startswith("Alerta | Mañana habrá")
        assert first.alerts.created == 1
        second = built.worker.run_pass()
        assert alerts.list().total == 1
        assert second.alerts.created == 0
    finally:
        built.close()


def test_the_hourly_purge_also_forgets_alerts_past_their_retention(tmp_path):
    from social_tracker.bootstrap import _purge_everything

    calls = []

    class Boom:
        def purge_expired(self):
            raise RuntimeError("boom")

    class Spy:
        def __init__(self, name):
            self.name = name

        def prune_history(self):
            calls.append(self.name)

        def purge_expired(self):
            calls.append(self.name)

    fake = type(
        "C",
        (),
        {
            "items": Boom(),
            "classifier": Spy("pruned"),
            "catalog": Spy("catalog purged"),
            "alerts": Spy("alerts purged"),
        },
    )()
    with pytest.raises(RuntimeError):
        _purge_everything(fake)()
    # the alerts come last: they clean up after the verdicts the earlier steps forgot
    assert calls == ["pruned", "catalog purged", "alerts purged"]
