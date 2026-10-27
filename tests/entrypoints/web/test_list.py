"""The sources list (UI-2), the global switch panel and the worker-status placeholders (UI-9)."""

from contextlib import closing
from pathlib import Path

from tests.actors import WHO
from tests.entrypoints.web.conftest import Web, row_of, visible_text


def test_an_empty_registry_shows_an_empty_state_with_how_to_links(web: Web):
    page = web.client.get("/").text
    assert "No sources yet" in visible_text(page)
    for source_type in ("news_rss", "instagram", "live_stream", "recorded_audio"):
        assert f'href="/howto#{source_type}"' in page
    assert 'href="/sources/new"' in page


def test_a_source_row_shows_what_slice_one_knows_and_n_a_for_the_rest(web: Web):
    web.seed("city_news", name="City News", credibility=4)
    row = visible_text(row_of(web.client.get("/").text, "city_news"))
    assert "City News" in row
    assert "news_rss" in row
    assert "recorded" in row  # mode
    assert "4" in row
    assert "disabled" in row  # collection state of a fresh source
    assert row.count("n/a") == 2  # last fetch and last error do not exist yet


def test_an_enabled_source_shows_idle_and_a_failed_one_shows_failures_and_retry(web: Web):
    web.seed("src_ok")
    web.seed("src_bad", name="Broken Feed")
    with closing(web.services()) as services:
        for source_id in ("src_ok", "src_bad"):
            services.sources.enable(source_id, **WHO)
        services.control.start_collection("src_bad", **WHO)
        services.control.fail_collection("src_bad", **WHO)
    page = web.client.get("/").text
    assert "idle" in visible_text(row_of(page, "src_ok"))
    bad = visible_text(row_of(page, "src_bad"))
    assert "backoff" in bad
    assert "1 failure" in bad
    assert "retry after" in bad


def test_the_page_lists_every_source_in_id_order(web: Web):
    web.seed("src_b", name="Bravo")
    web.seed("src_a", name="Alpha")
    page = web.client.get("/").text
    assert page.index('id="row-src_a"') < page.index('id="row-src_b"')


def test_the_global_switch_reflects_the_stored_state(web: Web):
    assert "Analysis is off" in visible_text(web.client.get("/").text)
    with closing(web.services()) as services:
        services.control.set_global_analysis(True, **WHO)
    assert "Analysis is on" in visible_text(web.client.get("/").text)


def test_worker_status_is_a_read_only_placeholder(web: Web):
    text = visible_text(web.client.get("/").text)
    assert "GPU and model: n/a - no worker running" in text
    assert "Stream lag: n/a - no worker running" in text


def test_names_are_escaped_so_markup_in_them_is_inert(web: Web):
    web.seed("src_xss", name="<script>alert(1)</script><img src=x onerror=alert(2)>")
    page = web.client.get("/").text
    assert "<script>alert(1)" not in page
    assert "<img src=x" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_a_known_notice_code_is_shown_and_an_unknown_one_is_ignored(web: Web):
    assert "no longer exists" in visible_text(web.client.get("/?notice=not_found").text)
    page = web.client.get("/?notice=%3Cb%3Ehacked").text
    assert "hacked" not in page


def test_a_failed_source_shows_its_failures_and_no_retry_time(web: Web):
    from social_tracker.control.domain import MAX_CONSECUTIVE_FAILURES, CollectionState, SourceState
    from social_tracker.persistence.control_repo import SqliteControlRepository
    from social_tracker.persistence.db import Database

    web.seed("src_dead", name="Dead Feed")
    web.post("/sources/src_dead/enable")
    failed = SourceState("src_dead", CollectionState.FAILED, MAX_CONSECUTIVE_FAILURES)
    with Database(Path(web.env["SOCIAL_TRACKER_DB_PATH"])) as db:
        SqliteControlRepository(db).save_source_state(failed)
    row = visible_text(row_of(web.client.get("/").text, "src_dead"))
    assert "failed" in row
    assert f"{MAX_CONSECUTIVE_FAILURES} failures" in row
    assert "retry after" not in row
