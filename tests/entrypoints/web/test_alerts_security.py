"""The alert pages treat every stored string as hostile (XSS inert, real templates), offer no way to
publish, share, mail or approve anything (G-1, UI-11) and load nothing from outside (offline)."""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from social_tracker.entrypoints.web.app import create_app
from tests.entrypoints.web.alerts_support import plant, raise_alerts
from tests.entrypoints.web.conftest import Web, visible_text

PAYLOADS = [
    "<script>alert(1)</script>",
    '"><img src=x onerror=alert(1)>',
    "'><svg/onload=alert(1)>",
    "</td></tr></table><iframe src=//evil.example></iframe>",
    "javascript:alert(1)",
    "{{ 7*7 }}{% raw %}",
    "&lt;b&gt;already escaped&lt;/b&gt;",
    "<a href='http://evil.example'>click</a>",
]


class Parsed(HTMLParser):
    """The elements, attributes and links a browser would act on."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attributes: list[tuple[str, str, str | None]] = []
        self.scripts_without_src = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags.append(tag)
        for name, value in attrs:
            self.attributes.append((tag, name, value))
        if tag == "script" and not any(n == "src" for n, _ in attrs):
            self.scripts_without_src += 1


def parse(page: str) -> Parsed:
    parser = Parsed()
    parser.feed(page)
    return parser


def assert_inert(page: str) -> None:
    parsed = parse(page)
    assert parsed.scripts_without_src == 0
    for forbidden in ("img", "iframe", "svg", "object", "embed", "style", "link"):
        if forbidden == "link":
            assert all(
                (a != "href") or v == "/static/app.css"
                for t, a, v in parsed.attributes
                if t == "link"
            )
            continue
        assert forbidden not in parsed.tags, forbidden
    assert not [a for _, a, _ in parsed.attributes if a.startswith("on")]
    for tag, name, value in parsed.attributes:
        if name in ("href", "src", "action", "hx-post", "hx-get") and value:
            assert not value.lower().startswith(("javascript:", "data:", "//", "http:", "https:"))
            assert "evil" not in value, (tag, name, value)


@pytest.fixture
def hostile(web: Web) -> Web:
    web.seed("src_one")
    for n, payload in enumerate(PAYLOADS):
        plant(
            web,
            f"x{n}",
            title=payload,
            text=f"{payload} manana habra un temblor",
            source_name=payload[:100],
            rationale=payload[:300],
            hours=n,
        )
    raise_alerts(web)
    return web


def test_hostile_strings_are_inert_in_the_list(hostile: Web):
    page = hostile.client.get("/alerts").text
    assert_inert(page)
    assert "&lt;script&gt;" in page
    assert "<script>alert(1)" not in page


def test_hostile_strings_are_inert_on_every_detail_page(hostile: Web):
    ids = re.findall(r'<tr id="alert-(alt_[0-9a-f]{32})"', hostile.client.get("/alerts").text)
    assert len(ids) == len(PAYLOADS)
    for alert_id in ids:
        assert_inert(hostile.client.get(f"/alerts/{alert_id}").text)


def test_the_text_is_shown_as_typed_not_interpreted(hostile: Web):
    page = hostile.client.get("/alerts").text
    text = visible_text(page)
    assert "<script>alert(1)</script>" in text  # as plain text (entities decoded for reading)
    assert "{{ 7*7 }}" in text  # a template expression stays text
    assert "49" not in text.replace("0.49", "")  # nothing was evaluated
    assert "&amp;lt;b&amp;gt;" in page  # a string that already looks escaped is escaped again


def test_hostile_strings_are_inert_in_a_status_fragment(hostile: Web):
    ids = re.findall(r'<tr id="alert-(alt_[0-9a-f]{32})"', hostile.client.get("/alerts").text)
    fragment = hostile.post_htmx(f"/alerts/{ids[0]}/seen", {"view": "list"}).text
    assert_inert(fragment)
    detail = hostile.post_htmx(f"/alerts/{ids[1]}/seen", {"view": "detail"}).text
    assert_inert(detail)


def test_a_hostile_filter_value_is_not_reflected_unescaped(hostile: Web):
    response = hostile.client.get("/alerts", params={"tag": '"><script>alert(1)</script>'})
    assert response.status_code == 400
    assert "<script>alert(1)" not in response.text
    assert_inert(response.text)


def test_the_excerpt_of_an_alert_cannot_close_its_cell(hostile: Web):
    page = hostile.client.get("/alerts").text
    assert page.count("<table") == 1
    assert page.count("</table>") == 1
    assert "<iframe" not in page


# ---- no way out (G-1, UI-11) ----

ALLOWED_ROUTES = {
    ("GET", "/alerts"),
    ("GET", "/alerts/{alert_id}"),
    ("POST", "/alerts/{alert_id}/seen"),
    ("POST", "/alerts/{alert_id}/dismiss"),
    ("POST", "/alerts/{alert_id}/reopen"),
}
FORBIDDEN = re.compile(
    r"publish|share|tweet|post to|send|e-?mail|mailto|notify|notification|approve|approval|draft"
    r"|debunk|broadcast|webhook|forward|export|download|copy link|whatsapp|telegram",
    re.IGNORECASE,
)
TEMPLATES = (
    Path(__file__).parents[3] / "src" / "social_tracker" / "entrypoints" / "web" / "templates"
)


def test_the_only_alert_routes_are_the_list_the_detail_and_the_three_marks(web: Web):
    app = create_app(web.services, host="127.0.0.1")
    found = set()
    for included in app.routes:
        router = getattr(included, "original_router", None)
        for route in getattr(router, "routes", []):
            path = getattr(route, "path", "")
            if "alert" in path:
                found |= {(m, path) for m in getattr(route, "methods", set()) - {"HEAD"}}
    assert found == ALLOWED_ROUTES


def test_no_route_of_the_whole_app_is_about_publishing_sharing_or_mailing(web: Web):
    app = create_app(web.services, host="127.0.0.1")
    paths = []
    for included in app.routes:
        router = getattr(included, "original_router", None)
        paths += [getattr(r, "path", "") for r in getattr(router, "routes", [])]
    assert paths
    forbidden = re.compile(r"publish|share|tweet|send|mail|notify|approve|draft|debunk|webhook")
    assert [p for p in paths if forbidden.search(p)] == []


def test_no_template_of_the_alerts_offers_to_publish_share_mail_or_approve():
    names = sorted(p.name for p in TEMPLATES.glob("*alert*"))
    assert len(names) >= 4  # the list, the row, the status panel and the detail
    for name in names:
        source = (TEMPLATES / name).read_text(encoding="utf-8")
        parsed = parse(source)
        for _tag, attr, value in parsed.attributes:
            if attr in ("href", "action", "hx-post", "hx-get", "hx-put", "hx-delete", "formaction"):
                assert not FORBIDDEN.search(value or ""), (name, attr, value)
        labels = re.findall(r">([^<>{}]*)</(?:button|a)>", source)
        for label in labels:
            assert not FORBIDDEN.search(label), (name, label)
        assert "mailto:" not in source
        assert "navigator.share" not in source
        assert "<form" not in source or all(
            re.search(r'action="/alerts', form) for form in re.findall(r"<form[^>]*>", source)
        )


def test_the_rendered_pages_have_no_link_or_button_that_sends_anything_anywhere(hostile: Web):
    ids = re.findall(r'<tr id="alert-(alt_[0-9a-f]{32})"', hostile.client.get("/alerts").text)
    pages = [hostile.client.get("/alerts").text, hostile.client.get(f"/alerts/{ids[0]}").text]
    for page in pages:
        parsed = parse(page)
        for tag, attr, value in parsed.attributes:
            if attr in ("href", "action", "hx-post", "src"):
                assert (value or "").startswith(("/", "#")), (tag, attr, value)
        for label in re.findall(r">([^<>]+)</(?:button|a)>", page):
            assert not FORBIDDEN.search(label), label
        assert "mailto:" not in page


def test_the_pages_say_in_words_that_nothing_leaves_this_machine(hostile: Web):
    assert "this machine only" in visible_text(hostile.client.get("/alerts").text)


# ---- offline ----


def test_the_alert_pages_load_only_their_own_assets(hostile: Web):
    ids = re.findall(r'<tr id="alert-(alt_[0-9a-f]{32})"', hostile.client.get("/alerts").text)
    for path in ("/alerts", f"/alerts/{ids[0]}"):
        page = hostile.client.get(path).text
        for _tag, attr, value in parse(page).attributes:
            if attr in ("href", "src", "action", "hx-post", "hx-get"):
                assert (value or "").startswith(("/", "#")), (attr, value)
        for src in re.findall(r'(?:src|href)="([^"]+)"', page):
            assert src.startswith(("/", "#")), src
        assert hostile.client.get("/static/htmx.min.js").status_code == 200
        assert hostile.client.get("/static/app.css").status_code == 200
