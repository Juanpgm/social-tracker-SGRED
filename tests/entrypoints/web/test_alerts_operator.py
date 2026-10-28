"""What an operator needs from the alerts page and the front page: whether the system is alive,
which verdicts still wait for the model, where each alert comes from, and dates they can type in
their own time zone. Real templates, real database, nothing mocked."""

import re
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV, TIMEZONE_ENV, Container
from social_tracker.bootstrap import build_container as build
from social_tracker.classification.domain import Signal, Topic
from social_tracker.control.domain import (
    SETTING_ALERTS_LAST_REFRESH,
    SETTING_WORKER_LAST_PASS,
)
from social_tracker.entrypoints.web.app import create_app
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from tests.entrypoints.web.alerts_support import (
    DELIBERATE,
    plant,
    raise_alerts,
)
from tests.entrypoints.web.conftest import BASE, Web, visible_text

ROW = re.compile(r'<tr id="alert-(alt_[0-9a-f]{32})"')


def make_web(tmp_path: Path, zone: str) -> Web:
    env = {DB_PATH_ENV: str(tmp_path / "w.db"), BUSY_TIMEOUT_ENV: "2000", TIMEZONE_ENV: zone}

    def services() -> Container:
        return build(env)

    return Web(TestClient(create_app(services, host="127.0.0.1"), base_url=BASE), env, services)


@pytest.fixture
def bogota(tmp_path: Path) -> Web:
    web = make_web(tmp_path, "America/Bogota")
    web.seed("src_one")
    return web


def row_html(page: str, alert_id: str) -> str:
    found = re.search(rf'<tr id="alert-{alert_id}".*?</tr>', page, re.DOTALL)
    assert found
    return found.group()


def record(web: Web, key: str, moment: datetime) -> None:
    with Database(Path(web.env[DB_PATH_ENV])) as db:
        SqliteControlRepository(db).set_setting(key, moment.isoformat())


def list_ids(web: Web, **query: str) -> list[str]:
    response = web.client.get("/alerts", params=query)
    assert response.status_code == 200, response.text[:300]
    return ROW.findall(response.text)


# ---- A4: what each row says ----


def test_a_verdict_that_waits_for_the_model_is_badged_in_the_list(web: Web):
    web.seed("src_one")
    plant(web, "wait", needs_llm=True, confidence=0.7)
    plant(web, "done", confidence=0.7)
    alerts = raise_alerts(web)
    page = web.client.get("/alerts").text
    waiting = next(a for a in alerts if a.evidence.provisional)
    final = next(a for a in alerts if not a.evidence.provisional)
    assert "pending model" in visible_text(row_html(page, waiting.id))
    assert "pending model" not in visible_text(row_html(page, final.id))


def test_the_tags_of_an_alert_are_in_its_row(web: Web):
    web.seed("src_one")
    plant(
        web,
        "gov",
        topics=(Topic.SECRETARIAT, Topic.EARTHQUAKE),
        catalog="mismatch",
        confidence=0.9,
        signals=(
            Signal("official_attribution_unlinked", DELIBERATE, 0.4),
            Signal("prediction_claim", DELIBERATE, 0.45),
        ),
    )
    (alert,) = raise_alerts(web)
    text = visible_text(row_html(web.client.get("/alerts").text, alert.id))
    for tag in ("earthquake", "secretariat", "secretariat_attribution", "catalog_contradiction"):
        assert tag in text, tag


def test_a_tag_in_a_row_leads_to_that_filter(web: Web):
    web.seed("src_one")
    plant(web, "a")
    (alert,) = raise_alerts(web)
    assert 'href="/alerts?tag=earthquake"' in row_html(web.client.get("/alerts").text, alert.id)


def test_the_number_of_linked_sources_is_in_the_row(web: Web):
    web.seed("src_one")
    web.seed("src_gov", name="Gov Office")
    same = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    plant(web, "a", text=same)
    plant(web, "b", text=same, source_id="src_gov", source_name="Gov Office")
    plant(web, "c", text="otro rumor distinto sobre el rio")
    alerts = raise_alerts(web)
    page = web.client.get("/alerts").text
    texts = {a.occurrences: visible_text(row_html(page, a.id)) for a in alerts}
    assert "2 sources" in texts[2]
    assert "1 source" in texts[1]
    assert "1 sources" not in texts[1]


def test_published_and_fetched_times_are_side_by_side(web: Web):
    web.seed("src_one")
    plant(web, "a", hours=0)
    (alert,) = raise_alerts(web)
    text = visible_text(row_html(web.client.get("/alerts").text, alert.id))
    assert "Published 2026-03-01 08:00 UTC" in text
    assert "Fetched 2026-03-01 08:01 UTC" in text


def test_an_unknown_publication_date_says_so(web: Web):
    web.seed("src_one")
    plant(web, "a", flags=("published_at_missing",))
    (alert,) = raise_alerts(web)
    text = visible_text(row_html(web.client.get("/alerts").text, alert.id))
    assert "Published unknown" in text


def test_hide_provisional_is_a_filter_that_is_off_by_default(web: Web):
    web.seed("src_one")
    plant(web, "wait", needs_llm=True, confidence=0.7)
    plant(web, "done", confidence=0.7)
    raise_alerts(web)
    page = web.client.get("/alerts").text
    box = re.search(r'<input[^>]*name="hide_provisional"[^>]*>', page)
    assert box
    assert "checked" not in box.group()
    assert len(list_ids(web)) == 2
    assert len(list_ids(web, hide_provisional="1")) == 1
    checked = web.client.get("/alerts", params={"hide_provisional": "1"}).text
    assert "checked" in re.search(r'<input[^>]*name="hide_provisional"[^>]*>', checked).group()


def test_an_alert_the_model_downgraded_is_in_the_default_list_and_badged(web: Web):
    web.seed("src_one")
    plant(web, "p", needs_llm=True, confidence=0.75)
    (alert,) = raise_alerts(web)
    with Database(Path(web.env[DB_PATH_ENV])) as db, db.transaction() as conn:
        conn.execute(
            "UPDATE classifications SET label = 'no_risk', confidence = 0.9, stage = 'llm', "
            "needs_llm = 0, priority = 0, settled_at = '2026-03-01T09:00:00+00:00', "
            "settle_reason = 'llm' WHERE item_id = ?",
            (alert.item_id,),
        )
    with closing(web.services()) as services:
        services.alerts.refresh()
    page = web.client.get("/alerts").text
    assert alert.id in ROW.findall(page)
    assert "downgraded by the model" in visible_text(row_html(page, alert.id))


# ---- A11: withdrawn alerts can be shown, and are counted ----


def withdrawn_world(web: Web):
    web.seed("src_one")
    plant(web, "live", confidence=0.8, text="rumor vivo sobre el temblor")
    plant(web, "gone", confidence=0.8, text="rumor que se retira por el temblor")
    raise_alerts(web)
    with Database(Path(web.env[DB_PATH_ENV])) as db, db.transaction() as conn:
        conn.execute(
            "UPDATE classifications SET label = 'no_risk' WHERE item_id IN "
            "(SELECT id FROM items WHERE external_id = 'gone')"
        )
    with closing(web.services()) as services:
        services.alerts.refresh()


def test_the_header_counts_the_withdrawn_alerts_and_links_to_them(web: Web):
    withdrawn_world(web)
    text = visible_text(web.client.get("/alerts").text)
    assert "Withdrawn: 1" in text
    assert 'href="/alerts?status=withdrawn"' in web.client.get("/alerts").text


def test_show_withdrawn_adds_them_to_the_list(web: Web):
    withdrawn_world(web)
    assert len(list_ids(web)) == 1
    assert len(list_ids(web, withdrawn="1")) == 2
    assert len(list_ids(web, status="withdrawn")) == 1
    page = web.client.get("/alerts", params={"withdrawn": "1"}).text
    box = re.search(r'<input[^>]*name="withdrawn"[^>]*>', page)
    assert box
    assert "checked" in box.group()


# ---- A5: the header says whether anything is being analysed ----


def test_a_fresh_system_says_nothing_has_run_and_nothing_is_analysed(web: Web):
    text = visible_text(web.client.get("/alerts").text)
    assert "Analysis: off" in text
    assert "last worker pass: never" in text
    assert "last alert refresh: never" in text
    assert "nothing is being analysed" in text


def test_the_empty_state_when_analysis_is_on_is_the_usual_one(web: Web):
    with closing(web.services()) as services:
        services.control.set_global_analysis(True, actor="tester", reason="go")
    text = visible_text(web.client.get("/alerts").text)
    assert "Analysis: on" in text
    assert "nothing is being analysed" not in text
    assert "No alerts yet" in text


def test_the_header_shows_when_the_worker_and_the_alerts_last_did_something(web: Web):
    now = datetime.now(UTC)
    record(web, SETTING_WORKER_LAST_PASS, now - timedelta(minutes=2, seconds=5))
    record(web, SETTING_ALERTS_LAST_REFRESH, now - timedelta(hours=3, minutes=1))
    text = visible_text(web.client.get("/alerts").text)
    assert "last worker pass: 2 minutes ago" in text
    assert "last alert refresh: 3 hours ago" in text


def test_a_worker_that_stopped_long_ago_is_visible(web: Web):
    record(web, SETTING_WORKER_LAST_PASS, datetime.now(UTC) - timedelta(days=3))
    assert "last worker pass: 3 days ago" in visible_text(web.client.get("/alerts").text)


def test_rows_with_analysis_off_still_say_nothing_is_being_analysed(web: Web):
    web.seed("src_one")
    plant(web, "a")
    raise_alerts(web)
    text = visible_text(web.client.get("/alerts").text)
    assert "Analysis: off" in text
    assert "nothing is being analysed" in text


# ---- A6: the threshold is shown, read-only ----


def test_the_threshold_is_shown_with_how_to_change_it_and_cannot_be_set_here(web: Web):
    page = web.client.get("/alerts").text
    text = visible_text(page)
    assert "alert threshold: 0.50" in text
    assert "social-tracker alerts threshold set" in text
    assert "threshold" not in re.findall(r'<(?:input|select)[^>]*name="([^"]+)"', page)
    assert web.post("/alerts/threshold", {"value": "0.9"}).status_code in (404, 405)


def test_a_threshold_that_was_changed_but_not_applied_yet_says_so(web: Web):
    with closing(web.services()) as services:
        services.alerts.refresh()
        services.control.set_alert_min_confidence(0.8, actor="tester", reason="stricter")
    text = visible_text(web.client.get("/alerts").text)
    assert "alert threshold: 0.80" in text
    assert "not applied yet" in text
    with closing(web.services()) as services:
        services.alerts.refresh()
    assert "not applied yet" not in visible_text(web.client.get("/alerts").text)


def test_the_alerts_that_failed_to_evaluate_are_counted_in_the_header(web: Web):
    web.seed("src_one")
    item_id = plant(web, "a")
    with Database(Path(web.env[DB_PATH_ENV])) as db:
        from social_tracker.persistence.alerts_repo import SqliteAlertRepository

        repo = SqliteAlertRepository(db)
        for _ in range(5):
            repo.record_failure(
                item_id,
                datetime.now(UTC),
                park_after=5,
                backoff_base=timedelta(seconds=30),
                backoff_cap=timedelta(hours=1),
            )
    assert "1 parked" in visible_text(web.client.get("/alerts").text)


# ---- A5: the front page ----


def test_the_front_page_shows_real_numbers_instead_of_the_placeholder(web: Web):
    text = visible_text(web.client.get("/").text)
    assert "n/a - no worker running" not in text
    assert "Last worker pass: never" in text
    assert "Last alert refresh: never" in text


def test_the_front_page_shows_when_the_worker_last_passed(web: Web):
    record(web, SETTING_WORKER_LAST_PASS, datetime.now(UTC) - timedelta(minutes=7, seconds=3))
    assert "Last worker pass: 7 minutes ago" in visible_text(web.client.get("/").text)


def test_the_front_page_says_how_many_verdicts_wait_for_the_model(web: Web):
    web.seed("src_one")
    plant(web, "wait", needs_llm=True)
    text = visible_text(web.client.get("/").text)
    assert "verdict" in text
    assert "waiting for the model" in text


# ---- A15: the badge is on every page ----


@pytest.mark.parametrize(
    "path", ["/", "/alerts", "/howto", "/sources/new", "/sources/src_one/edit"]
)
def test_every_page_shows_how_many_alerts_are_new(web: Web, path):
    web.seed("src_one")
    plant(web, "a")
    raise_alerts(web)
    page = web.client.get(path).text
    assert re.search(r'<span id="alerts-badge"[^>]*>1</span>', page), path


def test_the_error_pages_show_the_badge_too(web: Web):
    web.seed("src_one")
    plant(web, "a")
    raise_alerts(web)
    response = web.client.get("/alerts/alt_" + "0" * 32)
    assert response.status_code == 404
    assert re.search(r'<span id="alerts-badge"[^>]*>1</span>', response.text)


def test_a_page_without_a_database_still_renders_with_no_badge(web: Web, monkeypatch):
    from social_tracker.alerts.service import AlertService
    from social_tracker.shared.errors import StorageError

    def broken(self):
        raise StorageError("down")

    monkeypatch.setattr(AlertService, "counts", broken)
    response = web.client.get("/howto")
    assert response.status_code == 200
    assert re.search(r'<span id="alerts-badge"[^>]* hidden', response.text)


# ---- A15: a page past the last one is a friendly empty page ----


@pytest.mark.parametrize("page", ["40001", "9999999", "1000000"])
def test_a_page_beyond_what_can_exist_is_the_friendly_empty_page_not_a_422(web: Web, page):
    web.seed("src_one")
    plant(web, "a")
    raise_alerts(web)
    response = web.client.get("/alerts", params={"page": page})
    assert response.status_code == 200
    assert "No alerts on this page" in visible_text(response.text)
    assert 'href="/alerts"' in response.text


# ---- A7: dates in the operator's own zone ----


def plant_at(web: Web, name: str, fetched: datetime, **kw):
    return plant(web, name, fetched_at=fetched, published_at=fetched - timedelta(minutes=1), **kw)


@pytest.fixture
def night(bogota: Web) -> Web:
    """One alert fetched at 2026-03-01 04:00 UTC: 23:00 of Feb 28 in Bogota."""
    plant_at(bogota, "night", datetime(2026, 3, 1, 4, 0, tzinfo=UTC))
    raise_alerts(bogota)
    return bogota


def test_a_day_typed_in_the_form_is_a_day_of_the_local_zone(night: Web):
    assert len(list_ids(night, since="2026-03-01")) == 0  # starts at 05:00 UTC
    assert len(list_ids(night, since="2026-02-28")) == 1
    assert len(list_ids(night, until="2026-02-28")) == 1  # ends at 04:59:59 UTC the next day
    assert len(list_ids(night, until="2026-02-27")) == 0


def test_an_explicit_offset_is_honoured(night: Web):
    assert len(list_ids(night, since="2026-03-01T04:00:00+00:00")) == 1
    assert len(list_ids(night, since="2026-03-01T04:00:01+00:00")) == 0
    assert len(list_ids(night, since="2026-02-28T23:00:00-05:00")) == 1
    assert len(list_ids(night, until="2026-03-01T03:59:59Z")) == 0


def test_times_are_drawn_in_the_local_zone_with_their_offset(night: Web):
    text = visible_text(night.client.get("/alerts").text)
    assert "Fetched 2026-02-28 23:00 -05" in text
    assert "Published 2026-02-28 22:59 -05" in text


def test_the_form_says_which_zone_its_dates_are_in_and_which_timestamp_they_filter(night: Web):
    page = night.client.get("/alerts").text
    text = visible_text(page)
    assert "America/Bogota" in text
    select = re.search(r'<select id="f-when" name="when">.*?</select>', page, re.DOTALL)
    assert select
    assert 'value="last_seen" selected' in select.group()
    assert "Last seen" in select.group()
    assert "First seen" in select.group()


def test_a_range_can_apply_to_the_first_sighting(bogota: Web):
    same = "Mañana habrá un temblor fuerte en Cali, compartan este mensaje"
    plant_at(bogota, "a", datetime(2026, 3, 1, 12, 0, tzinfo=UTC), text=same)
    bogota.seed("src_gov", name="Gov Office")
    plant_at(
        bogota,
        "b",
        datetime(2026, 3, 2, 11, 0, tzinfo=UTC),
        text=same,
        source_id="src_gov",
        source_name="Gov Office",
    )
    raise_alerts(bogota)
    assert len(list_ids(bogota, since="2026-03-02")) == 1  # last seen on the 2nd (06:00 local)
    assert len(list_ids(bogota, since="2026-03-02", when="first_seen")) == 0  # first seen the 1st
    assert len(list_ids(bogota, until="2026-03-01", when="first_seen")) == 1
    assert len(list_ids(bogota, until="2026-03-01")) == 0


def test_an_unknown_timestamp_to_filter_by_is_refused(web: Web):
    assert web.client.get("/alerts", params={"when": "created_at"}).status_code == 400


# ---- A7: the source is chosen by name ----


def test_the_source_filter_is_a_choice_of_names_with_ids_as_values(web: Web):
    web.seed("src_one")
    web.seed("src_gov", name="Gov Office")
    page = web.client.get("/alerts").text
    select = re.search(r'<select id="f-source" name="source">.*?</select>', page, re.DOTALL)
    assert select
    assert re.search(r'<option value="src_gov">[^<]*</option>', select.group())
    assert 'value="src_one"' in select.group()


def test_a_removed_source_is_still_offered_by_the_name_its_alerts_kept(web: Web):
    web.seed("src_one")
    plant(web, "a", source_id="src_one", source_name="City News")
    raise_alerts(web)
    with closing(web.services()) as services:
        services.sources.delete("src_one", actor="tester", reason="gone")
    page = web.client.get("/alerts").text
    select = re.search(r'<select id="f-source" name="source">.*?</select>', page, re.DOTALL).group()
    assert re.search(r'<option value="src_one">City News[^<]*\(removed\)</option>', select)
    assert len(list_ids(web, source="src_one")) == 1


def test_the_chosen_source_stays_selected(web: Web):
    web.seed("src_gov", name="Gov Office")
    page = web.client.get("/alerts", params={"source": "src_gov"}).text
    assert re.search(r'<option value="src_gov" selected>', page)


# ---- the detail page ----


def test_the_detail_shows_the_publication_time_and_the_flags(web: Web):
    web.seed("src_one")
    plant(web, "p", needs_llm=True, confidence=0.7)
    (alert,) = raise_alerts(web)
    text = visible_text(web.client.get(f"/alerts/{alert.id}").text)
    assert "Published" in text
    assert "pending model" in text


def test_a_long_trail_ends_with_a_row_saying_how_many_entries_are_not_shown(web: Web):
    web.seed("src_one")
    plant(web, "orig", text="El rumor original sobre el temblor")
    (alert,) = raise_alerts(web)
    with Database(Path(web.env[DB_PATH_ENV])) as db, db.transaction() as conn:
        conn.execute("UPDATE alerts SET events_dropped = 41 WHERE id = ?", (alert.id,))
    text = visible_text(web.client.get(f"/alerts/{alert.id}").text)
    assert "41 more entries not shown" in text
