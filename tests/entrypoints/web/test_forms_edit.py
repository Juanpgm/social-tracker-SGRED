"""Editing a source: id and type are fixed, secrets are never shown, blanks keep or clear."""

import json
from contextlib import closing
from typing import Any

from tests.entrypoints.web.conftest import Web, visible_text
from tests.entrypoints.web.test_forms_add import CANARY, error_of

EDIT = "/sources/src_one/edit"


def current(web: Web, source_id: str = "src_one") -> Any:
    with closing(web.services()) as services:
        return services.sources.get(source_id)


def news_form(**overrides: str) -> dict[str, str]:
    return {
        "name": "Renamed Feed",
        "credibility": "4",
        "feed_url": "https://example.org/rss.xml",
        "poll_interval_s": "600",
        "keywords": "",
        **overrides,
    }


def test_the_edit_form_is_prefilled_and_shows_the_id_and_type_as_plain_text(web: Web):
    web.seed(name="City News", credibility=2, keywords=["flood", "fire"])
    page = web.client.get(EDIT).text
    assert 'name="name" value="City News"' in page
    assert 'name="credibility" value="2"' in page
    assert 'name="feed_url" value="https://example.org/rss.xml"' in page
    assert "flood\nfire" in page
    text = visible_text(page)
    assert "src_one" in text
    assert "news_rss" in text
    assert 'name="id"' not in page
    assert 'name="type"' not in page
    assert 'href="/howto#news_rss"' in page


def test_the_stored_secret_reference_is_never_shown_in_the_edit_form(web: Web):
    web.seed(type_="instagram")
    page = web.client.get(EDIT).text
    assert "ig_main" not in page.lower()
    assert 'name="secret_ref" value=""' in page  # the box is empty
    assert "reference is set" in visible_text(page)


def test_editing_saves_the_changes_and_audits_them(web: Web):
    web.seed(name="Old")
    response = web.post(EDIT, news_form(reason="renamed after merger"))
    assert response.status_code == 303
    assert response.headers["location"] == "/?notice=updated"
    source = current(web)
    assert (source.name, source.credibility) == ("Renamed Feed", 4)
    (event,) = web.events("source_edited")
    detail = json.loads(event.detail)
    assert detail["changed"] == ["credibility", "name"]
    assert (detail["actor"], detail["reason"]) == ("web:local", "renamed after merger")


def test_saving_without_changes_is_a_no_op_that_says_so(web: Web):
    web.seed(name="Renamed Feed", credibility=4)
    response = web.post(EDIT, news_form())
    assert response.status_code == 303
    assert response.headers["location"] == "/?notice=unchanged"
    assert web.events("source_edited") == []


def test_the_type_the_id_and_the_enabled_flag_cannot_be_changed_by_a_forged_post(web: Web):
    web.seed()
    forged = news_form(type="instagram", id="other_id", enabled="true", mode="streaming")
    assert web.post(EDIT, forged).status_code == 303
    source = current(web)
    assert (source.type, source.id, source.enabled, source.mode) == (
        "news_rss",
        "src_one",
        False,
        "recorded",
    )


def test_an_invalid_edit_shows_the_errors_and_keeps_the_stored_source(web: Web):
    web.seed(name="Keep Me")
    response = web.post(EDIT, news_form(credibility="7", feed_url="ftp://example.org"))
    assert response.status_code == 422
    assert error_of(response.text, "credibility")
    assert error_of(response.text, "feed_url")
    assert current(web).name == "Keep Me"
    assert web.events("source_edited") == []


def test_renaming_to_another_sources_name_is_a_conflict_but_its_own_name_is_fine(web: Web):
    web.seed("src_one", name="One")
    web.seed("src_two", name="Two")
    clash = web.post(EDIT, news_form(name="  two "))
    assert clash.status_code == 409
    assert "already exists" in error_of(clash.text, "name")
    own = web.post(EDIT, news_form(name="ONE", credibility="3"))
    assert own.status_code == 303
    assert current(web).name == "ONE"


def test_an_edit_of_an_unknown_or_malformed_id_is_a_not_found_page(web: Web):
    for path in ("/sources/src_gone/edit", "/sources/UPPER/edit", "/sources/x/edit"):
        assert web.client.get(path).status_code == 404, path
        response = web.post(path, news_form())
        assert response.status_code == 404, path
        assert "does not exist" in visible_text(response.text)


def test_a_source_deleted_in_another_tab_while_editing_is_a_not_found_page(web: Web):
    web.seed()
    assert web.client.get(EDIT).status_code == 200
    with closing(web.services()) as services:
        services.sources.delete("src_one", actor="other-tab", reason="deleted elsewhere")
    response = web.post(EDIT, news_form())
    assert response.status_code == 404
    assert "does not exist" in visible_text(response.text)


def test_a_blank_reference_keeps_the_stored_one_and_a_new_one_replaces_it(web: Web):
    web.seed(type_="instagram")
    base = {"name": "Insta", "credibility": "3", "handle": "some.handle", "poll_interval_s": ""}
    assert web.post(EDIT, {**base, "secret_ref": "", "reason": "keep"}).status_code == 303
    assert current(web).secret_ref == "env:ig_main"
    changed = web.post(EDIT, {**base, "secret_ref": "env:IG_NEW"})
    assert changed.status_code == 303
    assert current(web).secret_ref == "env:IG_NEW"
    kept, replaced = (json.loads(e.detail) for e in web.events("source_edited"))
    assert "secret_ref" not in kept["changed"]
    assert "secret_ref" in replaced["changed"]
    assert all("IG_NEW" not in e.detail for e in web.events())


def test_a_secret_pasted_as_a_new_reference_is_rejected_and_not_echoed(web: Web):
    web.seed(type_="instagram")
    form = {"name": "Insta", "credibility": "3", "handle": "some.handle", "secret_ref": CANARY}
    response = web.post(EDIT, form)
    assert response.status_code == 422
    assert error_of(response.text, "secret_ref")
    assert CANARY not in response.text
    assert current(web).secret_ref == "env:ig_main"


def test_blank_keywords_clear_them_and_a_blank_interval_keeps_the_stored_one(web: Web):
    web.seed(keywords=["flood"], poll_interval_s=300)
    assert web.post(EDIT, news_form(poll_interval_s="", name="Same Name")).status_code == 303
    source = current(web)
    assert source.keywords is None
    assert source.poll_interval_s == 300


def test_a_recorded_audio_source_can_move_from_a_feed_to_a_directory(web: Web):
    web.seed(type_="recorded_audio")
    form = {"name": "Audio", "credibility": "3", "feed_url": "", "watch_dir": "/srv/in"}
    assert web.post(EDIT, form).status_code == 303
    source = current(web)
    assert (source.feed_url, source.watch_dir) == (None, "/srv/in")


def test_unchecking_a_box_on_edit_turns_the_option_off(web: Web):
    web.seed(type_="instagram")
    form = {"name": "Insta", "credibility": "3", "handle": "some.handle", "include_posts": "on"}
    assert web.post(EDIT, form).status_code == 303
    source = current(web)
    assert (source.include_posts, source.include_stories) == (True, False)


def test_the_edit_starting_values_never_include_the_secret_reference(web: Web):
    from social_tracker.entrypoints.web.forms import values_from_source

    web.seed(type_="instagram")
    values, checked = values_from_source(current(web))
    assert "secret_ref" not in values
    assert values["handle"] == "some.handle"
    assert checked == {"include_stories": True, "include_posts": True}
