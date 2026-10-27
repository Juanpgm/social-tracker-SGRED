"""A locked database on add, edit and delete: the form comes back with what was typed (R8), and
stored addresses never show their query string on the edit form (S-S2)."""

import json
import re
from contextlib import closing

import pytest

from tests.entrypoints.web.conftest import Web, visible_text
from tests.entrypoints.web.test_forms_add import BASE_FORM, CANARY, VALID, stored
from tests.entrypoints.web.test_forms_edit import EDIT, current, edit, news_form
from tests.entrypoints.web.test_resilience import impatient, token_of, write_lock

BUSY = "database is busy or unavailable"


def input_value(page: str, name: str) -> str | None:
    found = re.search(rf'name="{name}" value="([^"]*)"', page)
    return found.group(1) if found else None


def add_form(**overrides: str) -> dict[str, str]:
    return {**BASE_FORM, **VALID["news_rss"], "reason": "first feed", **overrides}


def test_a_busy_add_returns_the_form_with_the_typed_values_and_a_busy_alert(web: Web):
    client = impatient(web)
    form = {**add_form(keywords="flood\nfire"), "csrf_token": token_of(client)}
    with write_lock(web):
        response = client.post("/sources/new/news_rss", data=form)
    assert response.status_code == 503
    page = response.text
    assert BUSY in visible_text(page)
    assert 'role="alert"' in page.split("<form")[0].split("<main")[1]
    assert input_value(page, "id") == "city_news"
    assert input_value(page, "name") == "City News"
    assert input_value(page, "credibility") == "3"
    assert input_value(page, "feed_url") == "https://example.org/rss.xml"
    assert input_value(page, "reason") == "first feed"
    assert "flood\nfire" in page
    assert stored(web) == []
    assert "SQLITE" not in page
    assert 'action="/sources/new/news_rss"' in page


def test_the_busy_form_can_be_submitted_again_once_the_lock_is_gone(web: Web):
    client = impatient(web)
    form = {**add_form(), "csrf_token": token_of(client)}
    with write_lock(web):
        assert client.post("/sources/new/news_rss", data=form).status_code == 503
    again = client.post("/sources/new/news_rss", data=form, follow_redirects=False)
    assert again.status_code == 303
    assert len(stored(web)) == 1


@pytest.mark.parametrize(
    ("url", "echoed"),
    [
        ("https://example.org/rss.xml", True),
        ("https://" + "user:pw" + "@example.org/rss.xml", False),
        ("https://example.org/rss.xml?token=abc", False),
        ("https://example.org/rss.xml#frag", False),
        ("not a url", False),
    ],
)
def test_a_busy_form_echoes_an_address_only_when_it_carries_nothing_secret(
    web: Web, url: str, echoed: bool
):
    client = impatient(web)
    form = {**add_form(feed_url=url), "csrf_token": token_of(client)}
    with write_lock(web):
        response = client.post("/sources/new/news_rss", data=form)
    assert response.status_code == 503
    assert (input_value(response.text, "feed_url") == url) is echoed
    for fragment in ("pw@", "token=abc", "frag"):
        if not echoed:
            assert fragment not in response.text


def test_a_busy_instagram_add_never_echoes_the_secret_reference(web: Web):
    client = impatient(web)
    form = {
        **BASE_FORM,
        **VALID["instagram"],
        "secret_ref": CANARY,
        "csrf_token": token_of(client),
    }
    with write_lock(web):
        response = client.post("/sources/new/instagram", data=form)
    assert response.status_code == 503
    assert CANARY not in response.text
    assert "IG_MAIN" not in response.text
    assert input_value(response.text, "handle") == "some.handle"
    assert 'name="include_stories" checked' in response.text.replace("\n", " ") or (
        "checked" in response.text
    )


def test_a_busy_edit_returns_the_typed_values_and_changes_nothing(web: Web):
    web.seed(name="Original")
    client = impatient(web)
    version = current(web).updated_at.isoformat()
    form = {
        **news_form(name="Mine", credibility="5", reason="why"),
        "updated_at": version,
        "csrf_token": token_of(client),
    }
    with write_lock(web):
        response = client.post(EDIT, data=form)
    assert response.status_code == 503
    assert BUSY in visible_text(response.text)
    assert input_value(response.text, "name") == "Mine"
    assert input_value(response.text, "credibility") == "5"
    assert input_value(response.text, "updated_at") == version
    assert current(web).name == "Original"
    assert web.events("source_edited") == []


def test_a_busy_delete_shows_the_confirmation_again_with_the_reason_and_keeps_the_source(
    web: Web,
):
    web.seed(name="City News")
    client = impatient(web)
    with write_lock(web):
        response = client.post(
            "/sources/src_one/delete", data={"reason": "retired", "csrf_token": token_of(client)}
        )
    assert response.status_code == 503
    assert BUSY in visible_text(response.text)
    assert input_value(response.text, "reason") == "retired"
    assert 'action="/sources/src_one/delete"' in response.text
    assert current(web).name == "City News"
    assert web.events("source_deleted") == []


def test_the_reason_typed_on_a_rejected_delete_comes_back_too(web: Web):
    web.seed()
    response = web.post("/sources/src_one/delete", {"reason": "x" * 1000})
    assert response.status_code == 422
    assert input_value(response.text, "reason") == "x" * 1000


# ---- S-S2: the query string of a stored address stays out of the edit form ----

STREAM = "https://example.org/live.mp3?station=7"
MASKED = "https://example.org/live.mp3?***"


def seed_stream(web: Web) -> None:
    web.seed("src_one", "live_stream", name="Radio", stream_url=STREAM)


def stream_form(**overrides: str) -> dict[str, str]:
    return {
        "name": "Radio",
        "credibility": "3",
        "stream_url": MASKED,
        "chunk_seconds": "30",
        "language": "es",
        **overrides,
    }


def test_the_edit_form_masks_the_query_string_of_the_stored_address(web: Web):
    seed_stream(web)
    page = web.client.get(EDIT).text
    assert "station=7" not in page
    assert input_value(page, "stream_url") == MASKED
    assert "query string" in visible_text(page)


def test_an_address_without_a_query_is_shown_whole(web: Web):
    web.seed("src_one", "live_stream", stream_url="https://example.org/live.mp3")
    assert input_value(web.client.get(EDIT).text, "stream_url") == "https://example.org/live.mp3"


def test_saving_with_the_masked_address_keeps_the_stored_one(web: Web):
    seed_stream(web)
    response = edit(web, stream_form(name="Radio Two"))
    assert response.status_code == 303
    source = current(web)
    assert (source.name, source.stream_url) == ("Radio Two", STREAM)
    (event,) = web.events("source_edited")
    assert json.loads(event.detail)["changed"] == ["name"]


def test_saving_only_the_masked_address_is_no_change(web: Web):
    seed_stream(web)
    response = edit(web, stream_form(language="es"))
    assert response.headers["location"] == "/?notice=unchanged"
    assert web.events("source_edited") == []


def test_a_new_address_replaces_the_stored_one(web: Web):
    seed_stream(web)
    response = edit(web, stream_form(stream_url="https://example.org/other.mp3"))
    assert response.status_code == 303
    assert current(web).stream_url == "https://example.org/other.mp3"


def test_the_mask_is_only_a_mask_for_the_address_it_was_made_from(web: Web):
    web.seed("src_one", "live_stream", stream_url="https://example.org/live.mp3")
    response = edit(web, stream_form(stream_url=MASKED))  # nothing to keep: this is a new value
    assert response.status_code == 303
    assert current(web).stream_url == MASKED


def test_a_rejected_edit_never_shows_the_stored_query_either(web: Web):
    seed_stream(web)
    response = edit(web, stream_form(credibility="9"))
    assert response.status_code == 422
    assert "station=7" not in response.text
    assert input_value(response.text, "stream_url") == MASKED


def test_a_feed_address_with_a_query_is_masked_for_recorded_audio_too(web: Web):
    web.seed("src_one", "recorded_audio", feed_url="https://example.org/pod.xml?show=3")
    page = web.client.get(EDIT).text
    assert "show=3" not in page
    assert input_value(page, "feed_url") == "https://example.org/pod.xml?***"
    with closing(web.services()) as services:
        assert services.sources.get("src_one").feed_url == "https://example.org/pod.xml?show=3"
