"""What the README and the how-to say about the alerts and the gold set is what the code does."""

from pathlib import Path

import pytest
import yaml

from social_tracker.alerts.domain import (
    DEFAULT_RETENTION_DAYS,
    KNOWN_TAGS,
    MAX_EXCERPT_CHARS,
    AlertSettings,
)
from social_tracker.bootstrap import (
    ALERT_CRITICAL_CATALOG_ENV,
    ALERT_CRITICAL_DELIBERATE_ENV,
    ALERT_CRITICAL_WINDOW_ENV,
    ALERT_DEDUP_WINDOW_ENV,
    ALERT_HIGH_ENV,
    ALERT_MEDIUM_ENV,
    ALERT_RETENTION_DAYS_ENV,
)
from social_tracker.gold.domain import MAX_GOLD_EXAMPLES, MAX_GOLD_TEXT_CHARS

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
HOWTO = (ROOT / "docs" / "howto.md").read_text(encoding="utf-8")


def section(title: str) -> str:
    return " ".join(README.split(f"## {title}", 1)[1].split("\n## ", 1)[0].split())


def row_of(variable: str) -> str:
    return next(line for line in README.splitlines() if line.startswith(f"| `{variable}`"))


ENV = [
    ALERT_RETENTION_DAYS_ENV,
    ALERT_DEDUP_WINDOW_ENV,
    ALERT_CRITICAL_WINDOW_ENV,
    ALERT_CRITICAL_DELIBERATE_ENV,
    ALERT_CRITICAL_CATALOG_ENV,
    ALERT_HIGH_ENV,
    ALERT_MEDIUM_ENV,
]


@pytest.mark.parametrize("name", ENV)
def test_every_alert_variable_is_in_the_configuration_table(name: str):
    assert f"`{name}`" in README
    assert row_of(name).count("|") == 4  # a name, a meaning and a default


def test_the_defaults_in_the_table_are_the_enforced_ones():
    s = AlertSettings()
    expected = {
        ALERT_RETENTION_DAYS_ENV: str(s.retention_days),
        ALERT_DEDUP_WINDOW_ENV: str(int(s.dedup_window.total_seconds() // 3600)),
        ALERT_CRITICAL_WINDOW_ENV: str(int(s.critical_window.total_seconds() // 3600)),
        ALERT_CRITICAL_DELIBERATE_ENV: str(s.critical_disinformation),
        ALERT_CRITICAL_CATALOG_ENV: str(s.critical_catalog),
        ALERT_HIGH_ENV: str(s.high),
        ALERT_MEDIUM_ENV: str(s.medium),
    }
    for name, default in expected.items():
        assert row_of(name).rstrip().endswith(f"| `{default}` |"), name
    assert s.retention_days == DEFAULT_RETENTION_DAYS == 180


def test_the_alerts_section_states_the_policy_numbers_and_the_limits():
    text = section("Alerts")
    s = AlertSettings()
    for fact in (
        "**at least** the alert threshold (default `0.5`, inclusive",
        f"{s.critical_disinformation} or more",
        f"{s.critical_catalog} or more",
        "at most 24 hours old (inclusive)",
        f"{MAX_EXCERPT_CHARS} characters",
        "`new`, `seen`, `dismissed`",
        "within 24 hours (inclusive",
        "strictly older",
        "`SOCIAL_TRACKER_ALERT_RETENTION_DAYS` (180)",
        "withdrawn",
        "alerts rebuild",
    ):
        assert fact in text, fact


def test_the_alerts_section_names_every_tag_an_alert_can_carry():
    text = section("Alerts")
    for tag in KNOWN_TAGS:
        assert f"`{tag}`" in text, tag


def test_nothing_leaves_the_machine_and_the_documents_say_so():
    assert "stored and shown on this machine only" in section("Alerts")
    assert "no external notifications" in HOWTO.lower()
    for text in (section("Alerts"), HOWTO):
        lowered = text.lower()
        assert "no publication, webhook, e-mail" in lowered or "no webhooks, no e-mail" in lowered


def test_the_gold_section_states_the_limits_and_that_it_never_applies_anything():
    text = section("Gold set and calibration")
    assert "never in the repository" in text
    assert f"{MAX_GOLD_EXAMPLES:,}" in text
    assert f"{MAX_GOLD_TEXT_CHARS:,}" in text
    assert "nothing is ever applied" in text
    assert "`--with-llm` (never the default)" in text
    assert "0.30 to 0.95 in steps of 0.05" in text
    seed = yaml.safe_load((ROOT / "tests" / "data" / "gold_examples.yaml").read_text("utf-8"))
    assert f"the same {len(seed['examples'])} examples" in text


def test_the_web_section_describes_the_alerts_page_and_no_longer_denies_it():
    web = README.split("## Web UI", 1)[1].split("\n## ", 1)[0]
    flat = " ".join(web.split())
    assert "there is no alerts view yet" not in flat
    assert "**Alerts** page" in flat
    assert "never asks the model server anything" in flat


def test_the_exit_codes_mention_the_new_commands():
    assert "alerts refresh" in README.split("| `1` |", 1)[1].split("| `2` |", 1)[0]
    assert "classify calibrate" in README.split("| `3` |", 1)[1].split("| `70` |", 1)[0]
