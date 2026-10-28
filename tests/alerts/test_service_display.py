"""A4/A7: what the list needs next to the fetch time (the publication time of the item that leads
the alert, when the feed's date can be believed) and the sources to choose from by name."""

import json
from datetime import timedelta

import pytest

from social_tracker.alerts.domain import UNKNOWN_AGE
from social_tracker.classification.domain import Label
from tests.alerts.world import alert_world
from tests.classification.harness import PUBLISHED

HOUR = timedelta(hours=1)


@pytest.fixture
def w(tmp_path):
    world = alert_world(tmp_path)
    world.llm.ready()
    world.llm.ready("src_other")
    yield world
    world.close()


def flag(w, item, *flags):
    with w.db.transaction() as conn:
        conn.execute("UPDATE items SET flags = ? WHERE id = ?", (json.dumps(list(flags)), item.id))


def test_the_alert_says_when_its_item_was_published_and_when_it_was_fetched(w):
    w.put("a", published=PUBLISHED, fetched=PUBLISHED + 2 * HOUR)
    w.refresh()
    alert = w.only()
    assert alert.published_at == PUBLISHED
    assert alert.first_seen_at == PUBLISHED + 2 * HOUR


@pytest.mark.parametrize("name", ["published_at_missing", "published_at_future"])
def test_a_date_the_feed_did_not_give_is_not_shown_as_a_publication_time(w, name):
    item = w.put("a", published=PUBLISHED, fetched=PUBLISHED + HOUR)
    flag(w, item, name)
    w.refresh()
    assert w.only().published_at is None


def test_a_publication_time_survives_the_store(w):
    w.put("a", published=PUBLISHED, fetched=PUBLISHED + HOUR)
    w.refresh()
    assert w.repo.get(w.only().id).published_at == PUBLISHED


def test_the_publication_time_is_that_of_the_leading_item(w):
    text = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    w.put("a", text=text, confidence=0.6, published=PUBLISHED, fetched=PUBLISHED + HOUR)
    w.put(
        "b",
        text=text,
        source="src_other",
        label=Label.DELIBERATE_DISINFORMATION,
        confidence=0.95,
        published=PUBLISHED + 3 * HOUR,
        fetched=PUBLISHED + 4 * HOUR,
    )
    w.refresh()
    assert w.only().published_at == PUBLISHED + 3 * HOUR


def test_an_old_row_without_the_column_value_reads_as_unknown(w):
    w.put("a")
    w.refresh()
    with w.db.transaction() as conn:
        conn.execute("UPDATE alerts SET published_at = NULL")
    assert w.only().published_at is None


def test_an_unknown_age_is_never_a_publication_time():
    assert UNKNOWN_AGE.total_seconds() == 10**10


# ---- the sources to choose from ----


def test_the_choices_are_names_with_ids_sorted_by_name(w):
    w.llm.base.add_source("src_zeta")
    w.llm.base.add_source("src_alpha")
    choices = w.service.source_choices()
    names = [c.name for c in choices]
    assert names == sorted(names, key=str.casefold)
    by_id = {c.id: c for c in choices}
    assert by_id["src_alpha"].name == "Name of src_alpha"
    assert by_id["src_alpha"].removed is False


def test_a_source_that_was_removed_is_still_offered_with_the_name_it_had(w):
    w.llm.base.add_source("src_gone")
    w.put("a", source="src_gone", source_name="Gone Gazette")
    w.refresh()
    w.llm.base.registry.delete("src_gone", actor="tester", reason="removed")
    by_id = {c.id: c for c in w.service.source_choices()}
    assert (by_id["src_gone"].name, by_id["src_gone"].removed) == ("Gone Gazette", True)


def test_a_registered_source_with_no_alert_is_offered_too(w):
    assert "src_other" in {c.id for c in w.service.source_choices()}


def test_a_hostile_name_is_cleaned(w):
    w.put("a", source_name="Bad\x1b[31m Name‮")
    w.refresh()
    for choice in w.service.source_choices():
        assert "\x1b" not in choice.name
        assert "‮" not in choice.name
