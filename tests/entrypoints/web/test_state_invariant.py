"""Whatever the operator does in the UI, `Source.enabled` agrees with the collection state (R9).

A random sequence of the actions the UI offers (enable and disable over htmx and without, edits,
the per-source and the global analysis switches, deletes, re-adds), mixed with a worker that fails
collections and with writes that fail with a busy database, runs against the real app and SQLite.
After every step each listed source satisfies `enabled <=> state is not disabled`, the list page
tells the same story, and a disabled source is never shown as running analysis.
"""

import re
import shutil
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from social_tracker.bootstrap import BUSY_TIMEOUT_ENV, DB_PATH_ENV, Container, build_container
from social_tracker.control.domain import CollectionState
from social_tracker.entrypoints.web.app import create_app
from social_tracker.entrypoints.web.security import CSRF_COOKIE_NAME, CsrfTokens
from social_tracker.shared.errors import InvalidTransition, NotFoundError, StorageError
from tests.actors import WHO
from tests.entrypoints.web.conftest import BASE

SECRET = b"h" * 32
COOKIE = "H" * 43
TOKEN = CsrfTokens(SECRET).token_for(COOKIE)
FORM = {"Content-Type": "application/x-www-form-urlencoded"}
HTMX = {"HX-Request": "true"}
IDS = ["src_a", "src_b", "src_c"]
CELLS = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)


class Proxy:
    def __init__(self, inner: Any, **overrides: Any) -> None:
        self._inner, self._overrides = inner, overrides

    def __getattr__(self, name: str) -> Any:
        return self._overrides.get(name, getattr(self._inner, name))


def busy(*_: Any, **__: Any) -> Any:
    raise StorageError("database error: SQLITE_BUSY")


class UiMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.root = Path(tempfile.mkdtemp(prefix="st-invariant-"))
        self.env = {DB_PATH_ENV: str(self.root / "ui.db"), BUSY_TIMEOUT_ENV: "2000"}
        self.fail_writes = False
        app = create_app(self._container, host="127.0.0.1", secret=SECRET)
        self.client = TestClient(app, base_url=BASE, raise_server_exceptions=False)
        self.client.cookies.set(CSRF_COOKIE_NAME, COOKIE)

    def _container(self) -> Any:
        real = build_container(self.env)
        if not self.fail_writes:
            return real
        sources = Proxy(real.sources, enable=busy, disable=busy, edit=busy, delete=busy)
        control = Proxy(real.control, set_source_analysis=busy, set_global_analysis=busy)
        return Proxy(real, sources=sources, control=control)

    def teardown(self) -> None:
        self.client.close()
        shutil.rmtree(self.root, ignore_errors=True)

    # ---- helpers ----

    def post(self, path: str, data: dict[str, str], *, htmx: bool = False) -> Any:
        headers = {**FORM, **(HTMX if htmx else {})}
        body = urlencode({"csrf_token": TOKEN, **data})
        return self.client.post(path, content=body, headers=headers, follow_redirects=False)

    def real(self) -> Container:
        return build_container(self.env)

    # ---- the operator's actions ----

    @rule(source_id=st.sampled_from(IDS))
    def add(self, source_id: str) -> None:
        form = {
            "id": source_id,
            "name": f"Name {source_id}",
            "credibility": "3",
            "feed_url": f"https://example.org/{source_id}.xml",
        }
        response = self.post("/sources/new/news_rss", form)
        assert response.status_code in (303, 409)

    @rule(
        source_id=st.sampled_from(IDS),
        action=st.sampled_from(["enable", "disable"]),
        htmx=st.booleans(),
    )
    def toggle(self, source_id: str, action: str, htmx: bool) -> None:
        response = self.post(f"/sources/{source_id}/{action}", {}, htmx=htmx)
        assert response.status_code in (200, 303, 404, 503)

    @rule(source_id=st.sampled_from(IDS), name=st.sampled_from(["One", "Two", "Three"]))
    def edit(self, source_id: str, name: str) -> None:
        page = self.client.get(f"/sources/{source_id}/edit")
        if page.status_code != 200:
            return
        version = re.search(r'name="updated_at" value="([^"]*)"', page.text)
        assert version
        form = {
            "name": f"{name} {source_id}",
            "credibility": "4",
            "feed_url": f"https://example.org/{source_id}.xml",
            "poll_interval_s": "600",
            "updated_at": version.group(1),
        }
        response = self.post(f"/sources/{source_id}/edit", form)
        assert response.status_code in (303, 409, 503)

    @rule(source_id=st.sampled_from(IDS), on=st.booleans(), htmx=st.booleans())
    def source_analysis(self, source_id: str, on: bool, htmx: bool) -> None:
        response = self.post(
            f"/sources/{source_id}/analysis", {"on": "1" if on else "0"}, htmx=htmx
        )
        assert response.status_code in (200, 303, 404, 503)

    @rule(on=st.booleans())
    def global_analysis(self, on: bool) -> None:
        response = self.post("/analysis", {"on": "1" if on else "0"})
        assert response.status_code in (303, 503)

    @rule(source_id=st.sampled_from(IDS))
    def delete(self, source_id: str) -> None:
        response = self.post(f"/sources/{source_id}/delete", {"reason": "cleanup"})
        assert response.status_code in (303, 503)

    @rule(source_id=st.sampled_from(IDS))
    def worker_fails_a_collection(self, source_id: str) -> None:
        with closing(self.real()) as services:
            try:
                services.control.start_collection(source_id, **WHO)
                services.control.fail_collection(source_id, **WHO)
            except (InvalidTransition, NotFoundError):
                pass  # an unknown source, or a state that does not allow it: nothing changes

    @rule(failing=st.booleans())
    def writes_start_or_stop_failing(self, failing: bool) -> None:
        self.fail_writes = failing

    # ---- the invariant ----

    @invariant()
    def enabled_agrees_with_the_collection_state(self) -> None:
        with closing(self.real()) as services:
            overview = services.control.overview()
        for row in overview.sources:
            assert row.source.enabled == (row.state is not CollectionState.DISABLED), row
            if not row.source.enabled:
                assert row.effective is False, row

    @invariant()
    def the_list_page_says_the_same(self) -> None:
        with closing(self.real()) as services:
            overview = services.control.overview()
        page = self.client.get("/").text
        listed = re.findall(r'<tr id="row-([a-z0-9_]+)">(.*?)</tr>', page, re.DOTALL)
        assert [source_id for source_id, _ in listed] == [o.source.id for o in overview.sources]
        for (_, html), row in zip(listed, overview.sources, strict=True):
            cells = [re.sub(r"<[^>]+>", "", c).strip() for c in CELLS.findall(html)]
            assert cells[3] == ("yes" if row.source.enabled else "no"), cells
            assert cells[5] == row.state.value, cells
            if not row.source.enabled:
                assert "(running)" not in cells[4], cells


UiMachine.TestCase.settings = settings(
    max_examples=30,
    stateful_step_count=14,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
TestUiStateInvariant = UiMachine.TestCase


def test_the_rules_reach_the_paths_the_random_runs_rely_on():
    """A guard against a vacuous machine: driven by hand, its rules do take these paths."""
    machine = UiMachine()
    try:
        machine.add("src_a")
        machine.toggle("src_a", "enable", htmx=True)
        assert machine.real().control.get_state("src_a").state is CollectionState.IDLE
        machine.worker_fails_a_collection("src_a")
        assert machine.real().control.get_state("src_a").state is CollectionState.BACKOFF
        machine.writes_start_or_stop_failing(True)
        machine.toggle("src_a", "disable", htmx=True)  # 503: the row keeps its true state
        machine.toggle("src_a", "disable", htmx=False)  # redirect with the busy notice
        machine.writes_start_or_stop_failing(False)
        assert machine.real().sources.get("src_a").enabled is True
        machine.toggle("src_a", "disable", htmx=False)
        machine.edit("src_a", "One")
        machine.delete("src_a")
        machine.add("src_a")
        machine.enabled_agrees_with_the_collection_state()
        machine.the_list_page_says_the_same()
    finally:
        machine.teardown()
