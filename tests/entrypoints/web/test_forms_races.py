"""Edits that race (R11), add forms submitted twice (R4) and the one-click guard of the forms."""

import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Any

from fastapi.testclient import TestClient

from tests.entrypoints.web.conftest import BASE, Web, visible_text
from tests.entrypoints.web.test_forms_add import BASE_FORM, VALID, add, error_of, stored
from tests.entrypoints.web.test_forms_edit import EDIT, current, edit, news_form
from tests.entrypoints.web.test_resilience import token_of

VERSION = re.compile(r'name="updated_at" value="([^"]*)"')


def version_in(page: str) -> str:
    found = VERSION.search(page)
    assert found, "the form carries no version"
    return found.group(1)


# ---- R11: the edit form carries the version it was opened at ----


def test_the_edit_form_carries_the_stored_version_in_a_hidden_field(web: Web):
    web.seed()
    page = web.client.get(EDIT).text
    assert version_in(page) == current(web).updated_at.isoformat()
    assert re.search(r'<input type="hidden" name="updated_at"', page)


def test_the_add_form_has_no_version_field(web: Web):
    assert "updated_at" not in web.client.get("/sources/new/news_rss").text


def test_an_edit_made_after_someone_else_saved_is_a_409_with_the_current_values(web: Web):
    web.seed(name="Original")
    opened_at = version_in(web.client.get(EDIT).text)
    assert edit(web, news_form(name="Theirs", credibility="2")).status_code == 303
    mine = web.post(EDIT, {**news_form(name="Mine", credibility="5"), "updated_at": opened_at})
    assert mine.status_code == 409
    assert "changed by someone else since you opened it" in visible_text(mine.text)
    assert 'name="name" value="Theirs"' in mine.text  # refreshed: what is stored now
    assert 'name="credibility" value="2"' in mine.text
    assert version_in(mine.text) == current(web).updated_at.isoformat()
    source = current(web)
    assert (source.name, source.credibility) == ("Theirs", 2)  # nothing of "Mine" was written
    assert len(web.events("source_edited")) == 1


def test_a_rejected_edit_keeps_the_version_it_started_from(web: Web):
    web.seed()
    opened_at = version_in(web.client.get(EDIT).text)
    rejected = web.post(EDIT, {**news_form(credibility="9"), "updated_at": opened_at})
    assert rejected.status_code == 422
    assert version_in(rejected.text) == opened_at
    fixed = web.post(EDIT, {**news_form(credibility="4"), "updated_at": version_in(rejected.text)})
    assert fixed.status_code == 303


def test_the_refreshed_form_can_be_saved(web: Web):
    web.seed()
    opened_at = version_in(web.client.get(EDIT).text)
    edit(web, news_form(name="Theirs"))
    refused = web.post(EDIT, {**news_form(name="Mine"), "updated_at": opened_at})
    retry = web.post(EDIT, {**news_form(name="Mine"), "updated_at": version_in(refused.text)})
    assert retry.status_code == 303
    assert current(web).name == "Mine"


def test_toggling_the_source_after_the_form_was_opened_also_makes_it_stale(web: Web):
    web.seed()
    opened_at = version_in(web.client.get(EDIT).text)
    web.post("/sources/src_one/enable")
    stale = web.post(EDIT, {**news_form(name="Mine"), "updated_at": opened_at})
    assert stale.status_code == 409
    assert current(web).name != "Mine"


def test_a_missing_or_malformed_version_is_refused_like_a_stale_one(web: Web):
    web.seed()
    for version in (None, "", "yesterday", "2026-13-45T00:00:00+00:00", "x" * 5000):
        form = news_form(name="Mine")
        if version is not None:
            form["updated_at"] = version
        response = web.post(EDIT, form)
        assert response.status_code == 409, version
        assert version_in(response.text) == current(web).updated_at.isoformat()
    assert current(web).name != "Mine"


def test_a_naive_timestamp_never_matches(web: Web):
    web.seed()
    naive = current(web).updated_at.replace(tzinfo=None).isoformat()
    assert web.post(EDIT, {**news_form(name="Mine"), "updated_at": naive}).status_code == 409


def test_a_stale_form_for_a_deleted_source_is_still_not_found(web: Web):
    web.seed()
    opened_at = version_in(web.client.get(EDIT).text)
    with closing(web.services()) as services:
        services.sources.delete("src_one", actor="other", reason="gone")
    response = web.post(EDIT, {**news_form(), "updated_at": opened_at})
    assert response.status_code == 404


def test_simultaneous_saves_from_the_same_version_let_exactly_one_win(web: Web):
    web.seed(name="Original")
    opened_at = version_in(web.client.get(EDIT).text)

    def save(n: int) -> int:
        client = TestClient(web.client.app, base_url=BASE)
        form = {
            **news_form(name=f"Writer {n}"),
            "updated_at": opened_at,
            "csrf_token": token_of(client),
        }
        return client.post(EDIT, data=form, follow_redirects=False).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = sorted(pool.map(save, range(6)))
    assert codes == [303, 409, 409, 409, 409, 409]
    assert len(web.events("source_edited")) == 1


# ---- R4: a double submit of the add form ----


def test_submitting_the_same_add_form_twice_is_one_source_and_two_successes(web: Web):
    first, second = add(web), add(web)
    assert (first.status_code, second.status_code) == (303, 303)
    assert second.headers["location"] == "/?notice=added"
    assert len(stored(web)) == 1
    assert len(web.events("source_added")) == 1


def test_an_identical_repeat_succeeds_even_after_the_source_was_enabled_meanwhile(web: Web):
    add(web)
    web.post("/sources/city_news/enable")
    again = add(web)
    assert again.status_code == 303
    assert len(web.events("source_added")) == 1


def test_the_same_id_with_other_values_is_still_a_conflict_that_names_the_id(web: Web):
    add(web)
    clash = add(web, name="Another Name")
    assert clash.status_code == 409
    assert "id already exists" in error_of(clash.text, "id")
    assert len(stored(web)) == 1


def test_every_differing_field_breaks_the_identity_of_a_repeat(web: Web):
    add(web)
    for override in (
        {"credibility": "4"},
        {"feed_url": "https://example.org/other.xml"},
        {"poll_interval_s": "900"},
        {"keywords": "flood"},
    ):
        assert add(web, **override).status_code == 409, override


def test_a_repeat_of_every_source_type_is_idempotent(web: Web):
    for source_type in VALID:
        form = {**BASE_FORM, **VALID[source_type], "id": f"id_{source_type}", "name": source_type}
        assert web.post(f"/sources/new/{source_type}", form).status_code == 303
        assert web.post(f"/sources/new/{source_type}", form).status_code == 303
    assert len(stored(web)) == len(VALID)
    assert len(web.events("source_added")) == len(VALID)


def test_the_same_name_under_another_id_is_a_conflict_not_a_repeat(web: Web):
    add(web)
    clash = add(web, id="other_id")
    assert clash.status_code == 409
    assert error_of(clash.text, "name")


def forms_of(page: str) -> list[str]:
    return re.findall(r"<form[^>]*>", page)


def test_the_forms_that_post_once_ask_the_browser_to_disable_their_button(web: Web):
    web.seed()
    once: dict[str, Any] = {
        "add": web.client.get("/sources/new/news_rss").text,
        "edit": web.client.get(EDIT).text,
        "delete": web.client.get("/sources/src_one/delete").text,
    }
    for name, page in once.items():
        (form,) = forms_of(page)
        assert "data-once" in form, name
    home = forms_of(web.client.get("/").text)
    assert any("/analysis" in f and "data-once" in f for f in home)
    assert not any("hx-post" in f and "data-once" in f for f in home)  # htmx guards its own rows
