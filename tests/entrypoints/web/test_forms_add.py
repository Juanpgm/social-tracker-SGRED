"""Adding a source (UI-3): type-adaptive form, server-side validation shown inline."""

import json
import logging
import re
from contextlib import closing
from typing import Any

import pytest

from tests.entrypoints.web.conftest import Web, visible_text
from tests.sources.factories import EMBEDDED_CREDENTIALS_URL

# Built by concatenation so the repository hygiene scan does not flag these fixtures.
CANARY = "ghp_" + "canary0123456789abcdefghij"
KEY_PARAM = "api" + "_key"

BASE_FORM = {"id": "city_news", "name": "City News", "credibility": "3"}
VALID: dict[str, dict[str, str]] = {
    "news_rss": {"feed_url": "https://example.org/rss.xml", "poll_interval_s": "600"},
    "instagram": {
        "handle": "some.handle",
        "secret_ref": "env:IG_MAIN",
        "include_stories": "on",
        "include_posts": "on",
    },
    "live_stream": {"stream_url": "https://example.org/live.m3u8", "chunk_seconds": "30"},
    "recorded_audio": {"feed_url": "https://example.org/podcast.xml"},
}


def add(web: Web, source_type: str = "news_rss", **overrides: str) -> Any:
    form = {**BASE_FORM, **VALID[source_type], **overrides}
    return web.post(f"/sources/new/{source_type}", form)


def stored(web: Web) -> list[Any]:
    with closing(web.services()) as services:
        return services.sources.list()


def error_of(page: str, field: str) -> str:
    """The inline error shown for one field, as text (empty when there is none)."""
    found = re.search(rf'<p class="error" id="err-{field}">(.*?)</p>', page, re.DOTALL)
    return visible_text(found.group(1)) if found else ""


# ---- the type chooser and the forms ----


def test_the_chooser_offers_every_type_with_its_form_and_its_how_to(web: Web):
    page = web.client.get("/sources/new").text
    for source_type in VALID:
        assert f'href="/sources/new/{source_type}"' in page
        assert f'href="/howto#{source_type}"' in page


@pytest.mark.parametrize(
    ("source_type", "fields"),
    [
        ("news_rss", ["feed_url", "poll_interval_s", "keywords"]),
        ("instagram", ["handle", "secret_ref", "include_stories", "include_posts"]),
        ("live_stream", ["stream_url", "chunk_seconds", "language"]),
        ("recorded_audio", ["feed_url", "watch_dir", "poll_interval_s"]),
    ],
)
def test_each_type_gets_its_own_fields_and_a_how_to_link(
    web: Web, source_type: str, fields: list[str]
):
    page = web.client.get(f"/sources/new/{source_type}").text
    for name in ["id", "name", "credibility", "reason", *fields]:
        assert f'name="{name}"' in page, (source_type, name)
        assert f'for="f-{name}"' in page
    assert f'href="/howto#{source_type}"' in page
    assert 'name="csrf_token"' in page
    assert visible_text(page).count("Reason") >= 1


def test_the_fields_of_one_type_are_not_offered_for_another(web: Web):
    page = web.client.get("/sources/new/news_rss").text
    for foreign in ("handle", "stream_url", "watch_dir", "secret_ref"):
        assert f'name="{foreign}"' not in page


def test_the_form_defaults_are_sensible(web: Web):
    page = web.client.get("/sources/new/instagram").text
    assert 'name="credibility" value="3"' in page
    assert 'name="reason" value="via web UI"' in page
    assert 'name="include_stories" checked' in page


@pytest.mark.parametrize("method", ["get", "post"])
def test_an_unknown_type_is_a_not_found_page(web: Web, method: str):
    path = "/sources/new/%3Cb%3Emarker"
    response = web.client.get(path) if method == "get" else web.post(path, BASE_FORM)
    assert response.status_code == 404
    assert "marker" not in response.text


# ---- happy paths ----


@pytest.mark.parametrize("source_type", list(VALID))
def test_a_valid_form_creates_a_disabled_source_and_lands_on_the_list(web: Web, source_type: str):
    response = add(web, source_type)
    assert response.status_code == 303
    assert response.headers["location"] == "/?notice=added"
    (source,) = stored(web)
    assert (source.id, source.type, source.name, source.credibility, source.enabled) == (
        "city_news",
        source_type,
        "City News",
        3,
        False,
    )
    assert "City News" in visible_text(web.client.get("/").text)
    (event,) = web.events("source_added")
    detail = json.loads(event.detail)
    assert (detail["actor"], detail["reason"]) == ("web:local", "via web UI")


def test_recorded_audio_accepts_a_watched_directory_instead_of_a_feed(web: Web):
    form = {**BASE_FORM, "watch_dir": "/srv/audio/incoming", "feed_url": ""}
    assert web.post("/sources/new/recorded_audio", form).status_code == 303
    assert stored(web)[0].watch_dir == "/srv/audio/incoming"


def test_a_custom_reason_is_recorded(web: Web):
    add(web, reason="new local feed")
    assert json.loads(web.events("source_added")[0].detail)["reason"] == "new local feed"


def test_optional_blank_fields_fall_back_to_the_defaults_and_keywords_are_split(web: Web):
    form = {
        **BASE_FORM,
        **VALID["news_rss"],
        "poll_interval_s": "",
        "keywords": "flood\n fire ,x\n",
    }
    assert web.post("/sources/new/news_rss", form).status_code == 303
    source = stored(web)[0]
    assert source.poll_interval_s == 600
    assert source.keywords == ("flood", "fire ,x")


def test_unchecked_boxes_mean_false(web: Web):
    form = {**BASE_FORM, "handle": "some.handle", "secret_ref": "env:IG_MAIN"}
    assert web.post("/sources/new/instagram", form).status_code == 303
    source = stored(web)[0]
    assert (source.include_stories, source.include_posts) == (False, False)


def test_fields_the_form_does_not_offer_cannot_be_smuggled_in(web: Web):
    smuggled = {"enabled": "true", "type": "instagram", "mode": "streaming", "password": "x"}
    assert add(web, **smuggled).status_code == 303
    source = stored(web)[0]
    assert (source.enabled, source.type, source.mode) == (False, "news_rss", "recorded")


# ---- credibility ----


@pytest.mark.parametrize("bad", ["0", "6", "3.5", "high", "", "-1", "1e1", "٣", " ", "null"])
def test_an_invalid_credibility_is_an_inline_error_and_nothing_is_stored(web: Web, bad: str):
    response = add(web, credibility=bad)
    assert response.status_code == 422
    assert error_of(response.text, "credibility")
    assert 'aria-invalid="true"' in response.text
    assert 'name="name" value="City News"' in response.text  # what was typed is kept
    assert stored(web) == []


@pytest.mark.parametrize("good", ["1", "2", "3", "4", "5", " 4 "])
def test_every_credibility_from_one_to_five_is_accepted(web: Web, good: str):
    assert add(web, credibility=good).status_code == 303
    assert stored(web)[0].credibility == int(good)


def test_credibility_errors_do_not_hide_other_errors(web: Web):
    response = add(web, credibility="9", name="")
    assert error_of(response.text, "credibility")
    assert error_of(response.text, "name")


# ---- name and id ----


@pytest.mark.parametrize("bad", ["", "   ", "\t", "x" * 101])
def test_a_blank_or_too_long_name_is_an_inline_error(web: Web, bad: str):
    response = add(web, name=bad)
    assert response.status_code == 422
    assert error_of(response.text, "name")
    assert stored(web) == []


@pytest.mark.parametrize("good", ["x" * 100, "  padded  ", "Ñandú — Radio"])
def test_a_name_up_to_one_hundred_characters_is_accepted(web: Web, good: str):
    assert add(web, name=good).status_code == 303
    assert stored(web)[0].name == good.strip()


@pytest.mark.parametrize("duplicate", ["City News", "city news", "  CITY NEWS  ", "CiTy NeWs"])
def test_a_duplicate_name_is_a_conflict_on_the_name_field(web: Web, duplicate: str):
    web.seed("other_one", name="City News")
    response = add(web, name=duplicate)
    assert response.status_code == 409
    assert "already exists" in error_of(response.text, "name")
    assert len(stored(web)) == 1


def test_a_duplicate_id_is_a_conflict_on_the_id_field(web: Web):
    web.seed("city_news", name="Another Name")
    response = add(web)
    assert response.status_code == 409
    assert "already exists" in error_of(response.text, "id")
    assert len(stored(web)) == 1


@pytest.mark.parametrize("bad", ["", "ab", "UPPER_case", "with space", "a" * 65, "x-y", "../etc"])
def test_an_invalid_id_is_an_inline_error(web: Web, bad: str):
    response = add(web, id=bad)
    assert response.status_code == 422
    assert error_of(response.text, "id")
    assert stored(web) == []


# ---- type specific validation ----


@pytest.mark.parametrize(
    ("source_type", "field", "bad"),
    [
        ("news_rss", "feed_url", "file:///etc/passwd"),
        ("news_rss", "feed_url", "ftp://example.org/rss"),
        ("news_rss", "feed_url", "not a url"),
        ("news_rss", "feed_url", ""),
        ("news_rss", "feed_url", EMBEDDED_CREDENTIALS_URL),
        ("news_rss", "feed_url", f"https://example.org/rss?{KEY_PARAM}=abc"),
        ("news_rss", "poll_interval_s", "5"),
        ("news_rss", "poll_interval_s", "often"),
        ("instagram", "handle", "has space"),
        ("instagram", "handle", "<b>x</b>"),
        ("instagram", "handle", ""),
        ("live_stream", "stream_url", "javascript:alert(1)"),
        ("live_stream", "chunk_seconds", "5"),
        ("live_stream", "chunk_seconds", "61"),
        ("live_stream", "language", "not a tag"),
    ],
)
def test_per_type_validation_errors_appear_next_to_the_field(
    web: Web, source_type: str, field: str, bad: str
):
    response = add(web, source_type, **{field: bad})
    assert response.status_code == 422, (field, bad)
    assert error_of(response.text, field), (field, bad)
    assert stored(web) == []


def test_recorded_audio_needs_exactly_one_origin(web: Web):
    both = add(web, "recorded_audio", watch_dir="/srv/in")
    neither = add(web, "recorded_audio", feed_url="")
    for response in (both, neither):
        assert response.status_code == 422
        assert "exactly one of feed_url or watch_dir" in visible_text(response.text)
    assert stored(web) == []


def test_instagram_needs_a_secret_reference(web: Web):
    response = add(web, "instagram", secret_ref="")
    assert response.status_code == 422
    assert error_of(response.text, "secret_ref")


# ---- credentials are never accepted, shown or logged ----


@pytest.mark.parametrize(
    "pasted", [CANARY, "env:" + "x" * 80, "Bearer " + CANARY, "token=" + CANARY]
)
def test_a_secret_pasted_into_the_reference_field_is_rejected_and_never_echoed(
    web: Web, caplog: pytest.LogCaptureFixture, pasted: str
):
    caplog.set_level(logging.DEBUG)
    response = add(web, "instagram", secret_ref=pasted)
    assert response.status_code == 422
    assert error_of(response.text, "secret_ref")
    assert pasted not in response.text
    assert CANARY not in response.text
    assert CANARY not in caplog.text
    assert stored(web) == []
    assert all(CANARY not in e.detail for e in web.events())


def test_a_valid_reference_is_not_echoed_back_either_when_another_field_fails(web: Web):
    response = add(web, "instagram", secret_ref="env:IG_MAIN", handle="bad handle")
    assert response.status_code == 422
    assert "IG_MAIN" not in response.text


def test_a_url_with_a_secret_in_it_is_rejected_and_not_echoed(web: Web):
    leaky = f"https://example.org/rss?{KEY_PARAM}={CANARY}"
    response = add(web, feed_url=leaky)
    assert response.status_code == 422
    assert CANARY not in response.text
    embedded = add(web, feed_url="https://user:" + CANARY + "@example.org/rss")
    assert embedded.status_code == 422
    assert CANARY not in embedded.text


def test_a_credential_field_that_is_not_part_of_the_form_is_ignored_and_not_echoed(web: Web):
    response = add(web, credibility="9", **{"pass" + "word": CANARY, "cookie": CANARY})
    assert response.status_code == 422
    assert CANARY not in response.text


# ---- markup in user text is inert ----

XSS = '"><script>alert(1)</script><img src=x onerror=alert(2)>'


def test_markup_in_a_name_is_escaped_when_the_form_is_shown_again(web: Web):
    response = add(web, name=XSS, credibility="9")
    assert response.status_code == 422
    assert "<script>alert(1)" not in response.text
    assert "<img src=x" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_markup_in_a_handle_and_in_errors_stays_inert(web: Web):
    response = add(web, "instagram", handle=XSS)
    assert response.status_code == 422
    assert "<script>alert(1)" not in response.text
    assert "<img src=x" not in response.text


def test_stored_markup_in_a_name_and_a_url_is_inert_in_the_list_and_the_edit_form(web: Web):
    url = 'https://example.org/rss?q="><script>alert(1)</script>'
    assert add(web, name=XSS, feed_url=url).status_code == 303
    for page in (web.client.get("/").text, web.client.get("/sources/city_news/edit").text):
        assert "<script>alert(1)" not in page
        assert "<img src=x" not in page


def test_the_reason_is_validated_and_markup_in_it_is_inert(web: Web):
    blank = add(web, reason="  ")
    too_long = add(web, reason="r" * 501)
    for response in (blank, too_long):
        assert response.status_code == 422
        assert error_of(response.text, "reason")
    assert stored(web) == []
    echoed = add(web, reason=XSS, name="")
    assert "<script>alert(1)" not in echoed.text


def test_markup_in_keywords_is_inert_in_the_edit_form_textarea(web: Web):
    payload = "</textarea><script>alert(1)</script>"
    assert add(web, keywords=payload).status_code == 303
    page = web.client.get("/sources/city_news/edit").text
    assert "<script>alert(1)" not in page
    assert "&lt;/textarea&gt;&lt;script&gt;" in page
    again = add(web, id="second_one", name="", keywords=payload)  # and after a failed submit
    assert again.status_code == 422
    assert "<script>alert(1)" not in again.text


def test_error_messages_are_escaped_even_if_a_message_ever_carried_markup(web: Web):
    """The domain never echoes input today; the template must not rely on that."""
    from social_tracker.shared.errors import ConflictError, ValidationError

    markup = "<img src=x onerror=alert(1)>"

    for error in (
        ValidationError(f"name: {markup}; <b>general</b>", ("name",)),
        ConflictError(f"a source with this name {markup}"),
    ):
        response = _with_failing_add(web, error)
        assert response.status_code in (409, 422)
        assert markup not in response.text
        assert "&lt;img src=x" in response.text


def _with_failing_add(web: Web, error: Exception) -> Any:
    import dataclasses

    from fastapi.testclient import TestClient

    from social_tracker.bootstrap import Container
    from social_tracker.entrypoints.web.app import create_app
    from tests.entrypoints.web.conftest import BASE, TOKEN_FIELD

    class Raises:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        def add(self, *args: Any, **kwargs: Any) -> Any:
            raise error

    def factory() -> Container:
        real = web.services()
        return dataclasses.replace(real, sources=Raises(real.sources))  # type: ignore[arg-type]

    client = TestClient(create_app(factory, host="127.0.0.1"), base_url=BASE)
    found = TOKEN_FIELD.search(client.get("/").text)
    assert found
    form = {**BASE_FORM, **VALID["news_rss"], "csrf_token": found.group(1)}
    return client.post("/sources/new/news_rss", data=form)
