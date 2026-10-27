"""One user journey through the whole UI, following redirects like a browser would."""

import json
import re
from contextlib import closing

from tests.entrypoints.web.conftest import TOKEN_FIELD, Web, row_of, visible_text


def submit(web: Web, path: str, data: dict[str, str]):
    """POST a form with the token of the page the browser would be showing, then follow."""
    token = TOKEN_FIELD.search(web.client.get("/").text)
    assert token
    return web.client.post(path, data={"csrf_token": token.group(1), **data})


def test_add_list_edit_disable_enable_global_toggle_delete(web: Web):
    # an empty system points to the how-to
    home = web.client.get("/")
    assert "No sources yet" in visible_text(home.text)
    assert "Analysis is off" in visible_text(home.text)

    # add
    form = {
        "id": "city_news",
        "name": "City News",
        "credibility": "4",
        "feed_url": "https://example.org/rss.xml",
        "poll_interval_s": "600",
        "keywords": "flood\nfire",
        "reason": "first feed",
    }
    added = submit(web, "/sources/new/news_rss", form)
    assert added.status_code == 200
    assert "Source added" in visible_text(added.text)
    row = visible_text(row_of(added.text, "city_news"))
    assert "City News" in row
    assert "disabled" in row

    # edit
    page = web.client.get("/sources/city_news/edit").text
    assert 'name="name" value="City News"' in page
    version = re.search(r'name="updated_at" value="([^"]+)"', page)
    assert version  # the page carries the version it was opened at
    changes = {**form, "name": "City News Daily", "updated_at": version.group(1)}
    edited = submit(web, "/sources/city_news/edit", changes)
    assert "Source updated" in visible_text(edited.text)
    assert "City News Daily" in visible_text(row_of(edited.text, "city_news"))

    # enable, then the analysis switches
    enabled = submit(web, "/sources/city_news/enable", {})
    assert "idle" in visible_text(row_of(enabled.text, "city_news"))
    submit(web, "/sources/city_news/analysis", {"on": "1"})
    system = submit(web, "/analysis", {"on": "1"})
    assert "Analysis is on" in visible_text(system.text)
    assert "(running)" in visible_text(row_of(system.text, "city_news"))

    # turning the system off stops the source from running without disabling it
    off = submit(web, "/analysis", {"on": "0"})
    assert "(running)" not in visible_text(row_of(off.text, "city_news"))
    assert "idle" in visible_text(row_of(off.text, "city_news"))

    # disable
    disabled = submit(web, "/sources/city_news/disable", {})
    assert "disabled" in visible_text(row_of(disabled.text, "city_news"))

    # delete, after the confirmation page
    confirm = web.client.get("/sources/city_news/delete")
    assert "alerts and derived data are kept" in visible_text(confirm.text)
    deleted = submit(web, "/sources/city_news/delete", {"reason": "retired"})
    assert "Source deleted" in visible_text(deleted.text)
    assert "No sources yet" in visible_text(deleted.text)

    # the audit trail tells the story, all attributed to the UI
    story = [(e.kind, json.loads(e.detail)["actor"]) for e in web.events()]
    assert [kind for kind, _ in story] == [
        "source_added",
        "source_edited",
        "source_enabled",
        "source_analysis_changed",
        "global_analysis_changed",
        "global_analysis_changed",
        "source_disabled",
        "source_deleted",
    ]
    assert {actor for _, actor in story} == {"web:local"}
    with closing(web.services()) as services:
        assert services.sources.list() == []
