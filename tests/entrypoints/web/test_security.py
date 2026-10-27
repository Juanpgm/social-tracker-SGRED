"""Host check, security headers, error pages and offline assets of the local web UI."""

import re

import pytest
from fastapi.testclient import TestClient

from social_tracker.entrypoints.web.app import create_app
from social_tracker.shared.errors import StorageError
from tests.entrypoints.web.conftest import BASE, Web

HEADERS = {
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "x-content-type-options": "nosniff",
    "cache-control": "no-store",
}
ASSET_URL = re.compile(
    r"""(?:src|href|action|data|poster|srcset)\s*=\s*["']\s*(?:https?:)?//""", re.I
)


@pytest.mark.parametrize("path", ["/", "/nope", "/static/app.css", "/sources/new/news_rss"])
def test_every_response_carries_the_security_headers(web: Web, path: str):
    response = web.client.get(path)
    for name, value in HEADERS.items():
        assert response.headers[name] == value, (path, name)
    csp = response.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "form-action 'self'" in csp


def test_a_json_error_response_is_html_not_the_framework_default(web: Web):
    response = web.client.get("/nope")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert "detail" not in response.text


@pytest.mark.parametrize(
    "host",
    ["evil.example", "127.0.0.1.evil.example", "localhost.evil.example", "", "0.0.0.0:8765"],
)
def test_a_foreign_host_header_is_rejected_on_reads_and_writes(web: Web, host: str):
    for response in (
        web.client.get("/", headers={"host": host}),
        web.client.post("/analysis", data={"on": "1"}, headers={"host": host}),
    ):
        assert response.status_code == 400
        assert "Invalid host" in response.text
        assert response.headers["x-frame-options"] == "DENY"


@pytest.mark.parametrize(
    "host", ["127.0.0.1:8765", "localhost:8765", "LOCALHOST", "[::1]:8765", "127.0.0.1"]
)
def test_the_loopback_names_are_accepted(web: Web, host: str):
    assert web.client.get("/", headers={"host": host}).status_code == 200


def test_the_bound_host_is_accepted_and_nothing_else_is(tmp_path):
    from social_tracker.bootstrap import build_container

    env = {"SOCIAL_TRACKER_DB_PATH": str(tmp_path / "h.db")}
    app = create_app(lambda: build_container(env), host="192.168.1.20")
    client = TestClient(app, base_url="http://192.168.1.20:8765")
    assert client.get("/").status_code == 200
    assert client.get("/", headers={"host": "192.168.1.21"}).status_code == 400


def test_an_unexpected_failure_is_a_plain_error_page_without_a_traceback(web: Web):
    def broken():
        raise RuntimeError("boom-" + "internal-detail")

    app = create_app(broken, host="127.0.0.1")
    client = TestClient(app, base_url=BASE, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 500
    assert "Something went wrong" in response.text
    assert "boom" not in response.text
    assert "Traceback" not in response.text
    assert response.headers["x-frame-options"] == "DENY"


def test_a_busy_database_is_a_friendly_page_not_a_traceback():
    def locked():
        raise StorageError("database error: SQLITE_BUSY")

    client = TestClient(create_app(locked, host="127.0.0.1"), base_url=BASE)
    response = client.get("/")
    assert response.status_code == 503
    assert "database is busy or unavailable" in response.text
    assert "SQLITE" not in response.text


@pytest.mark.parametrize("path", ["/", "/sources/new", "/sources/new/instagram", "/howto"])
def test_no_page_loads_an_asset_from_the_network(web: Web, path: str):
    web.seed()
    page = web.client.get(path).text
    assert "/static/htmx.min.js" in page
    assert not ASSET_URL.search(page), ASSET_URL.search(page)
    targets = re.findall(r'(?:src|href|action)="([^"]*)"', page)
    assert targets
    assert all(t.startswith(("/", "#")) for t in targets), targets


def test_static_assets_are_served_locally(web: Web):
    assert web.client.get("/static/htmx.min.js").text.startswith("var htmx=")
    assert web.client.get("/static/app.css").status_code == 200


def test_the_framework_documentation_pages_are_not_served(web: Web):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert web.client.get(path).status_code == 404, path


def test_the_htmx_configuration_swaps_error_answers_and_forbids_script_evaluation(web: Web):
    import json

    page = web.client.get("/").text
    found = re.search(r"<meta name=\"htmx-config\" content='([^']+)'>", page)
    assert found
    config = json.loads(found.group(1))
    assert config["responseHandling"] == [{"code": ".*", "swap": True}]  # 4xx and 5xx show too
    assert config["allowEval"] is False
    assert config["allowScriptTags"] is False
    assert config["includeIndicatorStyles"] is False  # no injected <style>: CSP forbids it
