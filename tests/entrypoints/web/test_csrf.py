"""Every state-changing request needs the CSRF token of its own session, and its body is bounded."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from social_tracker.entrypoints.web.app import create_app
from social_tracker.entrypoints.web.formdata import MAX_BODY_BYTES, MAX_FIELDS
from social_tracker.entrypoints.web.routes import router
from social_tracker.entrypoints.web.security import CSRF_COOKIE_NAME, CSRF_FIELD, CsrfTokens
from tests.entrypoints.web.conftest import BASE, TOKEN_FIELD, Web

SOURCE_ID = "src_one"
FORM = {"content-type": "application/x-www-form-urlencoded"}


def post_routes(web: Web) -> list[str]:
    """Every POST path the app serves, found from the router so a new route cannot be missed."""
    paths = sorted({r.path for r in router.routes if "POST" in getattr(r, "methods", ())})
    return [p.replace("{source_id}", SOURCE_ID).replace("{source_type}", "news_rss") for p in paths]


def test_the_discovery_helper_finds_the_state_changing_routes(web: Web):
    paths = post_routes(web)
    assert "/analysis" in paths
    assert f"/sources/{SOURCE_ID}/enable" in paths
    assert len(paths) >= 4


def other_session(web: Web) -> tuple[TestClient, str]:
    """A second browser: its own cookie and a token that is valid only there."""
    client = TestClient(web.client.app, base_url=BASE)
    found = TOKEN_FIELD.search(client.get("/").text)
    assert found
    return client, found.group(1)


def test_every_post_route_refuses_a_missing_token(web: Web):
    web.seed()
    web.csrf()  # the session cookie exists, only the form field is absent
    for path in post_routes(web):
        response = web.client.post(path, data={"on": "1", "name": "x"})
        assert response.status_code == 403, path
        assert "refused" in response.text
    assert [e.kind for e in web.events()] == ["source_added"]


def test_every_post_route_refuses_a_wrong_token(web: Web):
    web.seed()
    for path in post_routes(web):
        assert web.post(path, {"on": "1"}, token="0" * 64).status_code == 403, path
        assert web.post(path, {"on": "1"}, token="").status_code == 403, path
    assert [e.kind for e in web.events()] == ["source_added"]


def test_every_post_route_refuses_a_token_replayed_from_another_session(web: Web):
    web.seed()
    stolen = other_session(web)[1]
    for path in post_routes(web):
        assert web.post(path, {"on": "1"}, token=stolen).status_code == 403, path
    assert [e.kind for e in web.events()] == ["source_added"]


def test_every_post_route_refuses_a_valid_token_without_the_cookie(web: Web):
    web.seed()
    client, token = other_session(web)
    client.cookies.clear()
    for path in post_routes(web):
        response = client.post(path, data={CSRF_FIELD: token, "on": "1"})
        assert response.status_code == 403, path
    assert [e.kind for e in web.events()] == ["source_added"]


def test_a_token_from_before_a_restart_is_refused(web: Web):
    token = web.csrf()
    restarted = TestClient(create_app(web.services, host="127.0.0.1"), base_url=BASE)
    restarted.cookies.update(web.client.cookies)
    response = restarted.post("/analysis", data={CSRF_FIELD: token, "on": "1"})
    assert response.status_code == 403


def test_a_forged_or_malformed_cookie_is_replaced_and_gets_no_access(web: Web):
    web.client.cookies.set(CSRF_COOKIE_NAME, "short")
    token = web.csrf()  # the page issued a fresh cookie, so this token matches it
    assert web.post("/analysis", {"on": "1"}, token=token).status_code == 303
    attacker = TestClient(web.client.app, base_url=BASE)
    attacker.cookies.set(CSRF_COOKIE_NAME, "A" * 43)
    assert attacker.post("/analysis", data={CSRF_FIELD: token}).status_code == 403


def test_the_token_works_more_than_once_in_a_session_for_several_tabs(web: Web):
    token = web.csrf()
    assert web.post("/analysis", {"on": "1"}, token=token).status_code == 303
    assert web.post("/analysis", {"on": "0"}, token=token).status_code == 303


def test_the_cookie_is_http_only_and_same_site_strict(web: Web):
    cookie = web.client.get("/").headers["set-cookie"].lower()
    assert CSRF_COOKIE_NAME in cookie
    assert "httponly" in cookie
    assert "samesite=strict" in cookie


def test_a_returning_browser_keeps_its_cookie(web: Web):
    web.client.get("/")
    assert "set-cookie" not in web.client.get("/").headers


@pytest.mark.parametrize(
    "content_type", ["application/json", "text/plain", "multipart/form-data; boundary=x", ""]
)
def test_only_url_encoded_forms_are_accepted(web: Web, content_type: str):
    token = web.csrf()
    body = json.dumps({CSRF_FIELD: token, "on": "1"})
    response = web.client.post("/analysis", content=body, headers={"content-type": content_type})
    assert response.status_code == 415
    assert web.events() == []


def test_a_charset_parameter_on_the_content_type_is_fine(web: Web):
    token = web.csrf()
    response = web.client.post(
        "/analysis",
        content=f"{CSRF_FIELD}={token}&on=1",
        headers={"content-type": "application/x-www-form-urlencoded; charset=UTF-8"},
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_a_declared_oversize_body_is_refused_before_it_is_read(web: Web):
    token = web.csrf()
    body = f"{CSRF_FIELD}={token}&reason=" + "a" * MAX_BODY_BYTES
    response = web.client.post("/analysis", content=body, headers=FORM)
    assert response.status_code == 413
    assert web.events() == []


def test_a_streamed_oversize_body_without_a_length_is_refused(web: Web):
    token = web.csrf()

    def chunks() -> Iterator[bytes]:
        yield f"{CSRF_FIELD}={token}&reason=".encode()
        for _ in range(4):
            yield b"a" * (MAX_BODY_BYTES // 2)

    response = web.client.post("/analysis", content=chunks(), headers=FORM)
    assert response.status_code == 413
    assert web.events() == []


def test_a_body_exactly_at_the_limit_is_accepted_and_one_byte_more_is_not(web: Web):
    token = web.csrf()
    prefix = f"{CSRF_FIELD}={token}&on=1&pad="
    exact = prefix + "a" * (MAX_BODY_BYTES - len(prefix))
    assert len(exact) == MAX_BODY_BYTES
    ok = web.client.post("/analysis", content=exact, headers=FORM, follow_redirects=False)
    assert ok.status_code == 303
    over = web.client.post("/analysis", content=exact + "a", headers=FORM)
    assert over.status_code == 413


@pytest.mark.parametrize(
    "raw",
    [
        f"{CSRF_FIELD}=%FF%FE&on=1",  # percent-encoded bytes that are not UTF-8
        CSRF_FIELD,  # a field without a value
        "&".join(f"f{i}=1" for i in range(MAX_FIELDS + 1)),  # too many fields
    ],
)
def test_malformed_form_encodings_are_a_400_not_a_crash(web: Web, raw: str):
    web.csrf()
    response = web.client.post("/analysis", content=raw, headers=FORM)
    assert response.status_code == 400
    assert "Traceback" not in response.text
    assert web.events() == []


def test_raw_non_utf8_bytes_in_the_body_are_a_400(web: Web):
    web.csrf()
    response = web.client.post("/analysis", content=b"csrf_token=\xff\xfe&on=1", headers=FORM)
    assert response.status_code == 400


def test_the_first_value_of_a_repeated_field_is_the_one_that_counts(web: Web):
    token = web.csrf()
    good_first = f"{CSRF_FIELD}={token}&{CSRF_FIELD}=forged&on=1"
    assert (
        web.client.post(
            "/analysis", content=good_first, headers=FORM, follow_redirects=False
        ).status_code
        == 303
    )
    bad_first = f"{CSRF_FIELD}=forged&{CSRF_FIELD}={token}&on=1"
    assert web.client.post("/analysis", content=bad_first, headers=FORM).status_code == 403


def test_every_form_on_every_page_carries_the_token(web: Web):
    web.seed("src_one")
    web.seed("src_two", name="Two")
    for path in ("/", "/sources/new/news_rss", "/sources/new/instagram", "/sources/src_one/edit"):
        page = web.client.get(path).text
        assert page.count("<form") >= 1
        assert page.count("<form") == page.count('name="csrf_token"'), path
    confirm = web.client.get("/sources/src_one/delete").text
    assert confirm.count("<form") == confirm.count('name="csrf_token"') == 1


def test_a_malformed_cookie_never_counts_even_with_the_token_computed_for_it(web: Web):
    secret = b"k" * 32
    app = create_app(web.services, host="127.0.0.1", secret=secret)
    client = TestClient(app, base_url=BASE)
    client.cookies.set(CSRF_COOKIE_NAME, "short")
    forged = CsrfTokens(secret).token_for("short")
    response = client.post("/analysis", data={CSRF_FIELD: forged, "on": "1"})
    assert response.status_code == 403
    assert web.events() == []
    assert "short" not in response.headers["set-cookie"]  # the bad cookie is replaced, not kept


@pytest.mark.parametrize("length", ["abc", "-5", "1.5", ""])
def test_a_garbled_content_length_is_a_400(web: Web, length: str):
    token = web.csrf()
    response = web.client.post(
        "/analysis",
        content=f"{CSRF_FIELD}={token}&on=1".encode(),
        headers={**FORM, "content-length": length},
    )
    assert response.status_code in (400, 413)
    assert web.events() == []
