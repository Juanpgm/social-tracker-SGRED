"""The Last fetch and Last error columns show what the collector recorded (UI-2)."""

import re
from datetime import UTC, datetime
from pathlib import Path

from social_tracker.control.domain import FetchStatus
from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from tests.entrypoints.web.conftest import Web, row_of

FETCHED = datetime(2026, 3, 1, 8, 30, 15, tzinfo=UTC)


def record(web: Web, status: FetchStatus) -> None:
    with Database(Path(web.env["SOCIAL_TRACKER_DB_PATH"])) as db:
        SqliteControlRepository(db).save_fetch_status(status)


def cells(web: Web, source_id: str) -> list[str]:
    """The text of the two placeholder-or-value cells that follow the health cell."""
    page_row = row_of(web.client.get("/").text, source_id)
    found = re.findall(
        r'<td class="(?:health-\w+)">.*?</td>\s*<td[^>]*>(.*?)</td>\s*<td[^>]*>(.*?)</td>',
        page_row,
        re.DOTALL,
    )
    assert found, page_row
    return list(found[0])


def test_a_source_that_was_never_fetched_shows_explicit_placeholders(web: Web):
    web.seed("src_new")
    assert cells(web, "src_new") == ["n/a", "n/a"]


def test_the_last_successful_fetch_is_shown_in_utc_and_no_error(web: Web):
    web.seed("src_ok")
    record(web, FetchStatus("src_ok", FETCHED, None))
    assert cells(web, "src_ok") == ["2026-03-01 08:30:15 UTC", "n/a"]


def test_the_last_error_is_shown_next_to_the_last_good_fetch(web: Web):
    web.seed("src_bad")
    record(web, FetchStatus("src_bad", FETCHED, "http 503"))
    assert cells(web, "src_bad") == ["2026-03-01 08:30:15 UTC", "http 503"]


def test_a_source_that_never_succeeded_shows_its_error_and_no_time(web: Web):
    web.seed("src_dns")
    record(web, FetchStatus("src_dns", None, "name lookup failed"))
    assert cells(web, "src_dns") == ["n/a", "name lookup failed"]


def test_an_error_text_is_escaped_never_interpreted_as_markup(web: Web):
    web.seed("src_xss")
    record(web, FetchStatus("src_xss", None, '<script>alert("x")</script> & <b>'))
    page = web.client.get("/").text
    assert "<script>alert" not in page
    assert "&lt;script&gt;alert(&#34;x&#34;)&lt;/script&gt; &amp; &lt;b&gt;" in row_of(
        page, "src_xss"
    )


def test_the_row_fragment_after_a_toggle_shows_the_same_values(web: Web):
    web.seed("src_tog")
    record(web, FetchStatus("src_tog", FETCHED, "timeout"))
    response = web.post_htmx("/sources/src_tog/analysis", {"on": "1"})
    assert "2026-03-01 08:30:15 UTC" in response.text
    assert "timeout" in response.text


def test_enabling_a_source_drops_its_stale_error_but_keeps_the_last_fetch(web: Web):
    web.seed("src_tog")
    record(web, FetchStatus("src_tog", FETCHED, "timeout"))
    response = web.post_htmx("/sources/src_tog/enable")
    assert "2026-03-01 08:30:15 UTC" in response.text
    assert "timeout" not in response.text
