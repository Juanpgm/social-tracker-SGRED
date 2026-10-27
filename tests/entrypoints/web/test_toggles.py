"""Enable/disable, per-source and global analysis switches (UI-6) and their audit trail."""

import dataclasses
import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Any

from fastapi.testclient import TestClient

from social_tracker.bootstrap import Container
from social_tracker.control.domain import CollectionState
from social_tracker.entrypoints.web.app import create_app
from social_tracker.shared.errors import StorageError
from tests.entrypoints.web.conftest import BASE, TOKEN_FIELD, Web, row_of, visible_text

HTMX = {"HX-Request": "true"}


def details(web: Web, kind: str) -> list[dict[str, Any]]:
    return [json.loads(e.detail) for e in web.events(kind)]


def state_of(web: Web, source_id: str = "src_one") -> tuple[bool, CollectionState]:
    with closing(web.services()) as services:
        return services.sources.get(source_id).enabled, services.control.get_state(source_id).state


# ---- global switch ----


def test_the_global_switch_turns_analysis_on_and_off_and_audits_the_actor_and_reason(web: Web):
    on = web.post("/analysis", {"on": "1"})
    assert on.status_code == 303
    assert on.headers["location"] == "/?notice=analysis_on"
    assert "Analysis is on" in visible_text(web.client.get("/").text)
    off = web.post("/analysis", {"on": "0", "reason": "maintenance window"})
    assert off.headers["location"] == "/?notice=analysis_off"
    assert "Analysis is off" in visible_text(web.client.get("/").text)
    first, second = details(web, "global_analysis_changed")
    assert (first["actor"], first["reason"], first["to"]) == ("web:local", "via web UI", "active")
    assert (second["actor"], second["reason"], second["to"]) == (
        "web:local",
        "maintenance window",
        "paused",
    )


def test_a_repeated_global_request_is_an_idempotent_no_op(web: Web):
    web.post("/analysis", {"on": "1"})
    again = web.post("/analysis", {"on": "1"})
    assert again.status_code == 303
    assert len(web.events("global_analysis_changed")) == 1


def test_the_global_switch_rejects_a_missing_or_unknown_value_and_a_blank_reason(web: Web):
    for bad in ({}, {"on": "maybe"}, {"on": ""}, {"on": "1", "reason": "   "}):
        response = web.post("/analysis", bad)
        assert response.status_code == 400, bad
    assert web.events("global_analysis_changed") == []
    assert "Analysis is off" in visible_text(web.client.get("/").text)


def test_global_analysis_with_an_enabled_source_makes_it_effective_only_when_its_flag_is_on(
    web: Web,
):
    web.seed()
    web.post("/sources/src_one/enable")
    web.post("/analysis", {"on": "1"})
    with closing(web.services()) as services:
        assert services.control.overview().sources[0].effective is False
    web.post("/sources/src_one/analysis", {"on": "1"})
    with closing(web.services()) as services:
        assert services.control.overview().sources[0].effective is True
    web.post("/analysis", {"on": "0"})
    with closing(web.services()) as services:
        overview = services.control.overview()
    assert overview.sources[0].analysis is True
    assert overview.sources[0].effective is False


# ---- enable / disable ----


def test_enable_then_disable_moves_the_collection_state_and_audits(web: Web):
    web.seed()
    enabled = web.post("/sources/src_one/enable")
    assert enabled.status_code == 303
    assert enabled.headers["location"] == "/?notice=enabled"
    assert state_of(web) == (True, CollectionState.IDLE)
    disabled = web.post("/sources/src_one/disable", {"reason": "feed is down"})
    assert disabled.headers["location"] == "/?notice=disabled"
    assert state_of(web) == (False, CollectionState.DISABLED)
    (on,) = details(web, "source_enabled")
    (off,) = details(web, "source_disabled")
    assert (on["actor"], on["reason"]) == ("web:local", "via web UI")
    assert (off["actor"], off["reason"]) == ("web:local", "feed is down")


def test_a_double_submit_changes_nothing_the_second_time(web: Web):
    web.seed()
    token = web.csrf()
    for _ in range(3):
        assert web.post("/sources/src_one/enable", token=token).status_code == 303
    assert len(web.events("source_enabled")) == 1
    assert state_of(web) == (True, CollectionState.IDLE)


def test_the_enabled_flag_and_the_collection_state_stay_in_step_after_any_sequence(web: Web):
    web.seed()
    for action in ("enable", "enable", "disable", "disable", "enable", "disable", "enable"):
        web.post(f"/sources/src_one/{action}")
        enabled, state = state_of(web)
        assert enabled == (state is not CollectionState.DISABLED), action


def test_the_row_shows_the_opposite_action(web: Web):
    web.seed()
    assert 'action="/sources/src_one/enable"' in row_of(web.client.get("/").text, "src_one")
    web.post("/sources/src_one/enable")
    assert 'action="/sources/src_one/disable"' in row_of(web.client.get("/").text, "src_one")


def test_an_htmx_toggle_returns_the_real_row_and_a_status_message_without_a_redirect(web: Web):
    web.seed(name="City News")
    response = web.post_htmx("/sources/src_one/enable")
    assert response.status_code == 200
    assert 'action="/sources/src_one/disable"' in row_of(response.text, "src_one")
    assert "idle" in visible_text(row_of(response.text, "src_one"))
    assert 'hx-swap-oob="innerHTML:#live-region"' in response.text
    assert 'id="live-region"' not in response.text  # the region itself stays; only its text moves
    assert "City News is enabled" in visible_text(response.text)
    assert "location" not in response.headers


def test_a_toggle_for_a_source_deleted_elsewhere_is_a_not_found_notice(web: Web):
    plain = web.post("/sources/src_gone/enable")
    assert plain.status_code == 303
    assert plain.headers["location"] == "/?notice=not_found"
    htmx = web.post_htmx("/sources/src_gone/disable")
    assert htmx.status_code == 404
    assert htmx.headers["hx-refresh"] == "true"
    assert "no longer exists" in visible_text(htmx.text)
    assert web.events() == []


def test_an_invalid_id_in_the_url_is_not_found_and_never_echoed(web: Web):
    for bad in ("%3Cscript%3Ealert(1)", "UPPER", "a" * 200, "x"):
        response = web.post(f"/sources/{bad}/enable")
        assert response.status_code == 303, bad
        assert response.headers["location"] == "/?notice=not_found"
    unroutable = web.post("/sources/%3Cb%3Emarker%2Fx/enable")
    assert unroutable.status_code == 404
    assert "marker" not in unroutable.text


# ---- failures revert to the true state ----


def failing_web(web: Web) -> TestClient:
    class Failing:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            return self._inner.__getattribute__(name)

        def enable(self, *args: Any, **kwargs: Any) -> Any:
            raise StorageError("database error: SQLITE_BUSY")

        disable = enable

    def factory() -> Container:
        real = web.services()
        return dataclasses.replace(real, sources=Failing(real.sources))  # type: ignore[arg-type]

    client = TestClient(create_app(factory, host="127.0.0.1"), base_url=BASE)
    return client


def test_a_failed_toggle_shows_an_error_and_the_unchanged_row_for_htmx(web: Web):
    web.seed(name="City News")
    client = failing_web(web)
    token = TOKEN_FIELD.search(client.get("/").text).group(1)  # type: ignore[union-attr]
    response = client.post("/sources/src_one/enable", data={"csrf_token": token}, headers=HTMX)
    assert response.status_code == 503
    row = visible_text(row_of(response.text, "src_one"))
    assert "disabled" in row
    assert 'action="/sources/src_one/enable"' in response.text  # still offers the same action
    assert "database is busy" in visible_text(response.text)
    assert state_of(web) == (False, CollectionState.DISABLED)


def test_a_failed_plain_toggle_redirects_with_an_error_notice_and_changes_nothing(web: Web):
    web.seed()
    client = failing_web(web)
    token = TOKEN_FIELD.search(client.get("/").text).group(1)  # type: ignore[union-attr]
    response = client.post(
        "/sources/src_one/enable", data={"csrf_token": token}, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/?notice=busy"
    assert "database is busy" in visible_text(client.get("/?notice=busy").text)
    assert web.events("source_enabled") == []


# ---- per-source analysis ----


def test_the_per_source_analysis_flag_is_set_idempotently_and_audited(web: Web):
    web.seed()
    first = web.post("/sources/src_one/analysis", {"on": "1"})
    assert first.headers["location"] == "/?notice=analysis_on"
    web.post("/sources/src_one/analysis", {"on": "1"})
    (event,) = details(web, "source_analysis_changed")
    assert (event["actor"], event["to"]) == ("web:local", "on")
    assert "on" in visible_text(row_of(web.client.get("/").text, "src_one"))
    bad = web.post("/sources/src_one/analysis", {"on": "2"})
    assert bad.status_code == 400


# ---- concurrency on one database ----


def test_many_concurrent_enables_of_one_source_write_exactly_one_event(web: Web):
    web.seed()

    def enable(_: int) -> int:
        client = TestClient(web.client.app, base_url=BASE)
        token = TOKEN_FIELD.search(client.get("/").text).group(1)  # type: ignore[union-attr]
        return client.post(
            "/sources/src_one/enable", data={"csrf_token": token}, follow_redirects=False
        ).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(enable, range(8)))
    assert codes == [303] * 8
    assert len(web.events("source_enabled")) == 1
    assert state_of(web) == (True, CollectionState.IDLE)


def test_row_buttons_work_without_script_and_update_in_place_with_it(web: Web):
    web.seed()
    row = row_of(web.client.get("/").text, "src_one")
    forms = re.findall(r'<form method="post" action="([^"]+)" hx-post="([^"]+)"([^>]*)>', row)
    assert len(forms) == 2  # enable/disable and the analysis switch
    for action, hx_post, rest in forms:
        assert action == hx_post
        assert 'hx-target="#row-src_one"' in rest
        assert 'hx-swap="outerHTML"' in rest


def test_the_per_source_analysis_flag_can_be_turned_off_again(web: Web):
    web.seed()
    web.post("/sources/src_one/analysis", {"on": "1"})
    off = web.post("/sources/src_one/analysis", {"on": "0"})
    assert off.headers["location"] == "/?notice=analysis_off"
    with closing(web.services()) as services:
        assert services.control.overview().sources[0].analysis is False
    assert [d["to"] for d in details(web, "source_analysis_changed")] == ["on", "off"]


def test_an_htmx_answer_is_the_row_of_the_source_that_was_toggled(web: Web):
    web.seed("src_a", name="Alpha")
    web.seed("src_b", name="Bravo")
    response = web.post_htmx("/sources/src_b/enable")
    assert 'id="row-src_b"' in response.text
    assert 'id="row-src_a"' not in response.text
    assert "Bravo is enabled" in visible_text(response.text)
