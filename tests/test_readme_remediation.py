"""What the README and the how-to say about the second round of the alerts is what the code does:
severity wording, the age of an event, statuses that come back, the memory of a forgotten alert,
the caps, the redaction, the time zone, the operator pages and the gold export."""

from pathlib import Path

import pytest

from social_tracker.alerts.domain import AlertSettings
from social_tracker.bootstrap import (
    ALERT_MAX_EVENTS_ENV,
    ALERT_MAX_MEMBERS_ENV,
    TIMEZONE_ENV,
)
from social_tracker.shared.timezone import DEFAULT_TIMEZONE

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
HOWTO = (ROOT / "docs" / "howto.md").read_text(encoding="utf-8")


def flat(text: str) -> str:
    return " ".join(text.split())


def section(title: str) -> str:
    return flat(README.split(f"## {title}", 1)[1].split("\n## ", 1)[0])


def row_of(variable: str) -> str:
    return next(line for line in README.splitlines() if line.startswith(f"| `{variable}`"))


# ---- the table of variables ----


def test_the_new_variables_are_in_the_table_with_the_enforced_defaults():
    s = AlertSettings()
    for name, default in (
        (TIMEZONE_ENV, DEFAULT_TIMEZONE),
        (ALERT_MAX_MEMBERS_ENV, str(s.max_members)),
        (ALERT_MAX_EVENTS_ENV, str(s.max_events)),
    ):
        row = row_of(name)
        assert row.count("|") == 4, name
        assert row.rstrip().endswith(f"| `{default}` |"), name


# ---- severity ----


@pytest.mark.parametrize("document", ["README", "howto"])
def test_the_severity_wording_says_what_a_merge_and_the_model_do(document):
    text = flat(README if document == "README" else HOWTO)
    assert "never lowers on a merge" in text
    assert "follows the model's final verdict" in text
    assert "can raise it and never lower it" not in text
    assert "can rise and never falls" not in text


def test_a_rise_puts_a_handled_alert_back_to_new():
    text = section("Alerts")
    assert "goes back to `new`" in text
    assert "severity rises" in text
    assert "system:alerts" in text
    assert "same or a lower severity keeps" in text


def test_the_age_of_an_event_is_not_oversold():
    text = section("Alerts")
    assert "the one clock a feed cannot forge" not in text
    assert "the first sighting is the application's own clock" in text
    assert "the publication date is not" in text
    assert "unknown age is never fresh" in text
    assert "backdated" in text
    assert "lowers a critical contradiction of the catalogs to high" in text
    assert "a catalog that is stale or incomplete is `unavailable`" in text


def test_the_model_downgrade_is_documented():
    text = section("Alerts")
    assert "downgraded by the model" in text
    assert "never hides" in text
    assert "Show withdrawn" in text


def test_the_memory_of_a_forgotten_alert_has_an_exact_rule():
    text = section("Alerts")
    assert "stays forgotten while its verdict is the same" in text
    assert "label, confidence and the time the model settled it" in text
    assert "a year" in text
    assert "waits for the model is never purged" in text


def test_the_caps_and_the_incremental_merge_are_documented():
    text = section("Alerts")
    s = AlertSettings()
    assert f"{s.max_members} items" in text
    assert f"{s.max_events} entries" in text
    assert "only counted" in text
    assert "events_dropped" not in text  # a column name is not documentation
    assert "more entries not shown" in text


def test_the_trail_of_an_absorbed_alert_is_kept():
    text = section("Alerts")
    assert "trail of the absorbed alert" in text
    assert "from an absorbed alert" in text


def test_the_step_looks_only_at_what_changed_and_parks_poison_items():
    text = section("Alerts")
    assert "only at the verdicts written since" in text
    assert "parked" in text
    assert "alerts rebuild" in text
    assert "a changed threshold is applied by the next refresh" in text


def test_the_redaction_is_described_with_what_it_leaves_alone():
    text = section("Alerts")
    for fact in (
        "vendor keys",
        "any URL scheme",
        "ordinary Spanish prose",
        "rationale of the model",
    ):
        assert fact in text, fact


# ---- the operator pages ----


def test_the_web_section_describes_the_header_the_filters_and_the_zone():
    web = flat(README.split("## Web UI", 1)[1].split("\n## ", 1)[0])
    assert "n/a - no worker running" not in web
    assert "last worker pass" in web
    assert "read-only" in web
    assert "hide provisional" in web
    assert "Show withdrawn" in web
    assert TIMEZONE_ENV in web
    assert "first or last sighting" in web


def test_the_cli_table_documents_the_alert_changes():
    assert "--hide-provisional" in README
    assert "--include-withdrawn" in README
    assert "--when first_seen" in README
    assert "threshold set` applies it at once" in flat(README)


# ---- the gold set ----


def test_the_gold_export_refuses_a_work_tree_and_says_so():
    text = section("Gold set and calibration")
    assert "--allow-repo-path" in text
    assert "git work tree" in text
    assert "gold_export*.yaml" in text


def test_the_calibration_caveat_is_printed_not_only_documented():
    text = section("Gold set and calibration")
    assert "no history, no catalog; CPU stages only" in text
