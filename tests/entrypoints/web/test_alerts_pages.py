"""The alerts page (UI-7): the list, its filters and pages, the detail, the counter and the panel
about the model. Real templates, real database, nothing mocked."""

import re
from contextlib import closing
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode

import pytest

from social_tracker.bootstrap import DB_PATH_ENV
from social_tracker.persistence.db import Database
from tests.entrypoints.web.alerts_support import (
    DELIBERATE,
    RECYCLED,
    alert_for,
    plant,
    raise_alerts,
)
from tests.entrypoints.web.conftest import Web, visible_text
from tests.entrypoints.web.test_resources import Counting, shapes

ROW = re.compile(r'<tr id="alert-(alt_[0-9a-f]{32})"')


def rows_of(page: str) -> list[str]:
    return ROW.findall(page)


def list_ids(web: Web, **query: str) -> list[str]:
    response = web.client.get("/alerts", params=query)
    assert response.status_code == 200, response.text[:300]
    return rows_of(response.text)


@pytest.fixture
def filled(web: Web) -> Web:
    """Four alerts: different severity, label, source, tag and time."""
    web.seed("src_one")
    web.seed("src_gov", name="Gov Office")
    plant(web, "a", confidence=0.55, hours=0)
    plant(
        web,
        "b",
        label=DELIBERATE,
        confidence=0.9,
        hours=1,
        source_id="src_gov",
        source_name="Gov Office",
    )
    plant(web, "c", label=RECYCLED, confidence=0.75, hours=2)
    plant(web, "d", confidence=0.65, hours=3, topics=())
    raise_alerts(web)
    return web


# ---- the page ----


def test_the_page_is_a_real_document_with_the_security_headers(web: Web):
    response = web.client.get("/alerts")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["x-frame-options"] == "DENY"
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "<h1>Alerts</h1>" in response.text


def test_with_no_alerts_the_page_explains_where_they_come_from(web: Web):
    text = visible_text(web.client.get("/alerts").text)
    assert "No alerts yet" in text
    assert "stored and shown here, on this machine only" in text
    assert "<table" not in web.client.get("/alerts").text


def test_the_list_is_newest_first_one_row_per_alert(filled: Web):
    page = filled.client.get("/alerts").text
    ids = rows_of(page)
    assert len(ids) == 4
    order = re.findall(r'class="fetched">Fetched (\d\d\d\d-\d\d-\d\d \d\d:\d\d) UTC<', page)
    assert len(order) == 4
    assert order == sorted(order, reverse=True)


def test_each_row_shows_what_a_person_needs_to_decide(filled: Web):
    page = filled.client.get("/alerts", params={"label": "deliberate_disinformation"}).text
    text = visible_text(page)
    assert "critical" in text
    assert "deliberate_disinformation" in text
    assert "0.90" in text
    assert "Gov Office" in text or "City News" in text
    assert "Texto b" in text
    assert "new" in text
    assert ">Details<" in page


def test_the_nav_links_to_the_alerts_and_shows_how_many_are_new(filled: Web):
    page = filled.client.get("/alerts").text
    assert '<a href="/alerts">Alerts' in page
    assert re.search(r'<span id="alerts-badge"[^>]*>4</span>', page)
    assert 'aria-label="4 new alerts"' in page
    assert re.search(r'id="alerts-badge"', filled.client.get("/").text)


def test_the_badge_is_hidden_when_nothing_is_new(web: Web):
    assert re.search(r'<span id="alerts-badge"[^>]* hidden', web.client.get("/alerts").text)


# ---- filters ----


@pytest.mark.parametrize(
    ("query", "count"),
    [
        ({}, 4),
        ({"severity": "critical"}, 1),
        ({"severity": "high"}, 1),
        ({"severity": "medium"}, 2),
        ({"severity": "low"}, 0),
        ({"label": "recycled_content"}, 1),
        ({"label": "unintentional_rumor"}, 2),
        ({"source": "src_gov"}, 1),
        ({"source": "src_one"}, 3),
        ({"source": "src_never"}, 0),
        ({"status": "new"}, 4),
        ({"status": "seen"}, 0),
        ({"status": "withdrawn"}, 0),
        ({"tag": "earthquake"}, 3),
        ({"tag": "prediction"}, 4),
        ({"severity": "medium", "label": "unintentional_rumor"}, 2),
        ({"severity": "medium", "source": "src_gov"}, 0),
        ({"severity": "", "label": "", "status": "", "source": "", "tag": ""}, 4),
        ({"since": "2026-03-01", "until": "2026-03-01"}, 4),
        ({"since": "2026-03-02"}, 0),
        ({"until": "2026-02-28"}, 0),
        ({"unknown": "x", "page": "1"}, 4),
    ],
)
def test_each_filter_and_their_combinations(filled: Web, query, count):
    assert len(list_ids(filled, **query)) == count


def test_the_date_range_is_inclusive_of_both_days(filled: Web):
    assert len(list_ids(filled, since="2026-03-01", until="2026-03-01")) == 4
    assert len(list_ids(filled, since="2026-03-01T09:30:00", until="2026-03-01T10:30:00")) == 1


@pytest.mark.parametrize(
    "query",
    [
        {"severity": "urgent"},
        {"label": "no_risk"},
        {"status": "archived"},
        {"source": "' OR 1=1 --"},
        {"tag": "x'; DROP TABLE alerts; --"},
        {"since": "yesterday"},
        {"until": "2026-02-30"},
        {"since": "2026-03-02", "until": "2026-03-01"},
        {"severity": "x" * 5000},
    ],
)
def test_a_malformed_filter_is_a_clear_error_page_that_never_echoes_it(filled: Web, query):
    response = filled.client.get("/alerts", params=query)
    assert response.status_code == 400
    page = response.text
    assert 'role="alert"' in page
    for value in query.values():
        if len(value) > 3 and "'" not in value:
            assert value not in page
    assert "DROP TABLE" not in page
    assert len(list_ids(filled)) == 4  # the store is intact


def test_the_filter_form_keeps_what_was_chosen_and_escapes_it(filled: Web):
    page = filled.client.get(
        "/alerts", params={"severity": "high", "source": "src_one", "since": "2026-03-01"}
    ).text
    assert re.search(r'<option value="high" selected', page)
    assert 'value="src_one"' in page
    assert 'name="since" value="2026-03-01"' in page
    assert re.search(r'<label for="f-severity">', page)


def test_every_filter_control_has_a_label_and_the_form_is_a_get(filled: Web):
    page = filled.client.get("/alerts").text
    form = re.search(r'<form[^>]*class="filters"[^>]*>', page)
    assert form
    assert 'method="get"' in form.group()
    for name in ("severity", "label", "status", "source", "tag", "since", "until"):
        assert f'for="f-{name}"' in page
        assert f'id="f-{name}"' in page


# ---- pages ----


@pytest.fixture
def many(web: Web) -> Web:
    web.seed("src_one")
    for n in range(60):
        plant(web, f"n{n:02d}", confidence=0.5 + (n % 5) / 100, hours=n)
    raise_alerts(web)
    return web


def test_sixty_alerts_make_three_pages_of_25_25_and_10(many: Web):
    assert [len(list_ids(many, page=str(p))) for p in (1, 2, 3)] == [25, 25, 10]
    everything = [i for p in (1, 2, 3) for i in list_ids(many, page=str(p))]
    assert len(set(everything)) == 60


def test_the_first_page_is_the_default(many: Web):
    assert list_ids(many) == list_ids(many, page="1")


def test_the_links_between_pages_keep_the_filters(many: Web):
    page = many.client.get("/alerts", params={"page": "2", "severity": "medium"}).text
    previous = re.search(r'<a href="([^"]+)" rel="prev"', page)
    following = re.search(r'<a href="([^"]+)" rel="next"', page)
    assert previous
    assert "severity=medium" in previous.group(1)
    assert "page=1" in previous.group(1) or "page" not in previous.group(1)
    if following:
        assert "severity=medium" in following.group(1)
        assert "page=3" in following.group(1)


def test_there_is_no_previous_on_the_first_page_and_no_next_on_the_last(many: Web):
    first = many.client.get("/alerts").text
    last = many.client.get("/alerts", params={"page": "3"}).text
    assert 'rel="prev"' not in first
    assert 'rel="next"' in first
    assert 'rel="next"' not in last
    assert 'rel="prev"' in last
    assert "Page 3 of 3" in visible_text(last)


def test_a_page_past_the_end_is_an_empty_state_with_a_way_back(many: Web):
    page = many.client.get("/alerts", params={"page": "4"}).text
    assert rows_of(page) == []
    assert "No alerts on this page" in visible_text(page)
    assert re.search(r'<a href="/alerts[^"]*">First page</a>', page)


@pytest.mark.parametrize("page", ["0", "-1", "abc", "1.5", "", "99999999999", "٣", "1 "])
def test_a_page_that_is_not_a_page_is_refused_or_ignored_safely(many: Web, page):
    response = many.client.get("/alerts", params={"page": page})
    assert response.status_code in (200, 400)
    if response.status_code == 200:
        assert page in ("", "1 ") or rows_of(response.text) == []


def test_the_last_page_boundary_is_exact(web: Web):
    web.seed("src_one")
    for n in range(25):
        plant(web, f"e{n:02d}", hours=n)
    raise_alerts(web)
    assert len(list_ids(web, page="1")) == 25
    assert 'rel="next"' not in web.client.get("/alerts").text
    assert list_ids(web, page="2") == []
    plant(web, "extra", hours=99)
    raise_alerts(web)
    assert len(list_ids(web, page="2")) == 1


# ---- the empty state with filters ----


def test_filters_that_match_nothing_say_so_and_offer_to_clear_them(filled: Web):
    page = filled.client.get("/alerts", params={"severity": "low"}).text
    assert "No alerts match these filters" in visible_text(page)
    assert '<a href="/alerts">Clear the filters</a>' in page


# ---- the detail ----


def test_the_detail_shows_the_alert_its_evidence_its_items_and_its_trail(filled: Web):
    item_id = plant(filled, "z", label=DELIBERATE, confidence=0.9, catalog="mismatch")
    raise_alerts(filled)
    alert = alert_for(filled, item_id)
    page = filled.client.get(f"/alerts/{alert.id}").text
    text = visible_text(page)
    for needle in (
        "critical",
        "deliberate_disinformation",
        "0.900",
        "Deciding stage",
        "rules",
        "Catalog",
        "mismatch",
        "prediction_claim",
        "Texto z",
        "created",
        "Occurrences",
    ):
        assert needle in text, needle
    assert 'id="alert-status"' in page
    assert re.search(r'id="item-itm_[0-9a-f]{32}"', page)


def test_the_detail_of_a_removed_source_keeps_the_name_and_says_so(filled: Web):
    alert_id = rows_of(filled.client.get("/alerts", params={"source": "src_gov"}).text)[0]
    with closing(filled.services()) as services:
        services.sources.delete("src_gov", actor="op", reason="gone")
    text = visible_text(filled.client.get(f"/alerts/{alert_id}").text)
    assert "Gov Office" in text
    assert "source removed" in text
    assert "source removed" in visible_text(
        filled.client.get("/alerts", params={"source": "src_gov"}).text
    )


@pytest.mark.parametrize("bad", ["alt_" + "0" * 32, "nope", "x", "A" * 70, "alt_%27%20OR%201=1"])
def test_an_unknown_alert_is_a_404_page_about_alerts(filled: Web, bad):
    response = filled.client.get(f"/alerts/{bad}")
    assert response.status_code == 404
    assert "That alert does not exist" in visible_text(response.text)


def test_a_withdrawn_alert_is_found_by_its_status_and_says_why(filled: Web):
    alert = raise_alerts(filled)[0]
    with Database(Path(filled.env[DB_PATH_ENV])) as db, db.transaction() as conn:
        conn.execute("UPDATE classifications SET label = 'no_risk'")
    with closing(filled.services()) as services:
        services.alerts.refresh()
    assert len(list_ids(filled)) == 0
    assert alert.id in list_ids(filled, status="withdrawn")
    detail = visible_text(filled.client.get(f"/alerts/{alert.id}").text)
    assert "Withdrawn" in detail
    assert "the verdict is now no_risk" in detail


# ---- the cost does not grow with the rows ----


def test_the_list_runs_the_same_statements_for_1_and_25_sources(web: Web):
    counting = Counting(web.env)
    client = counting.client()
    web.seed("src_one")
    plant(web, "first")
    raise_alerts(web)
    counting.reset()
    assert client.get("/alerts").text.count('<tr id="alert-') == 1
    few = (counting.opened, shapes(counting.statements))
    for n in range(24):
        web.seed(f"src_{n:02d}", name=f"Source {n:02d}")
        plant(web, f"m{n:02d}", source_id=f"src_{n:02d}", hours=n + 1)
    raise_alerts(web)
    counting.reset()
    page = client.get("/alerts").text
    many = (counting.opened, shapes(counting.statements))
    assert page.count('<tr id="alert-') == 25
    assert few[0] == many[0] == 1
    assert many[1] == few[1]


# ---- the model panel on the front page (UI-9) ----


def test_the_front_page_has_a_read_only_panel_about_the_model(web: Web):
    page = web.client.get("/").text
    text = visible_text(page)
    assert "Language model" in text
    assert "st-gemma4-e4b" in text
    assert "waiting" in text
    assert "this hour: 0 of 200" in text
    assert "not asked from this page" in text
    assert "llm status" in text
    section = page.split('aria-labelledby="model-heading"')[1].split("</section>")[0]
    assert "<form" not in section
    assert "<button" not in section


def test_the_panel_never_calls_the_model(web: Web):
    from social_tracker.llm.service import EscalationService

    calls = []

    def forbidden(self, *args, **kwargs):
        calls.append(1)
        raise AssertionError("the model server was asked")

    original = (EscalationService.status, EscalationService.ping, EscalationService.escalate)
    EscalationService.status = EscalationService.ping = EscalationService.escalate = forbidden
    try:
        assert web.client.get("/").status_code == 200
    finally:
        EscalationService.status, EscalationService.ping, EscalationService.escalate = original
    assert calls == []


def test_the_front_page_has_no_placeholder_for_the_worker_any_more(web: Web):
    text = visible_text(web.client.get("/").text)
    assert "n/a - no worker running" not in text
    assert "Last worker pass:" in text


def test_the_query_of_a_link_is_encoded_not_spliced(many: Web):
    odd = {"source": "src_one", "tag": "earthquake", "page": "2"}
    page = many.client.get("/alerts?" + urlencode(odd)).text
    for href in re.findall(r'href="(/alerts\?[^"]*)"', page):
        assert "&amp;" in href or "&" not in href.replace("&amp;", "")


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags.append(tag)


def test_the_page_is_well_formed_enough_to_parse(filled: Web):
    parser = _Tags()
    parser.feed(filled.client.get("/alerts").text)
    assert parser.tags.count("table") == 1
    assert parser.tags.count("main") == 1


# ---- accessibility basics, like every other page ----


@pytest.mark.parametrize("path", ["/alerts", "/alerts?severity=low", "/alerts?page=9"])
def test_the_list_has_a_language_landmarks_a_status_region_and_labelled_controls(
    filled: Web, path: str
):
    page = filled.client.get(path).text
    assert '<html lang="en">' in page
    assert 'class="skip" href="#main"' in page
    assert '<main id="main">' in page
    assert '<div id="live-region" role="status" aria-live="polite">' in page
    for field_id in re.findall(r'<(?:input|select|textarea)\b[^>]*\bid="([^"]+)"', page):
        assert f'for="{field_id}"' in page, field_id
    assert not re.search(r"\son[a-z]+\s*=", page)
    for script in re.findall(r"<script\b[^>]*>", page):
        assert "src=" in script


def test_the_table_has_column_headers_and_every_button_names_its_alert(filled: Web):
    page = filled.client.get("/alerts").text
    assert page.count('<th scope="col">') == 10  # the tags got a column of their own
    for label in re.findall(r'<button[^>]*aria-label="([^"]+)"', page):
        assert "alert" in label
    assert 'role="button"' not in page
    assert "tabindex" not in page


def test_the_detail_page_has_its_own_landmarks_and_headings(filled: Web):
    alert_id = rows_of(filled.client.get("/alerts").text)[0]
    page = filled.client.get(f"/alerts/{alert_id}").text
    assert '<main id="main">' in page
    assert page.count("<h1>") == 1
    for heading in ("Status", "Evidence", "Items behind this alert", "Trail"):
        assert f">{heading}</h2>" in page
