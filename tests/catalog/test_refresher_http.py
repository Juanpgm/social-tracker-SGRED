"""The refresher on top of the REAL fetcher (mock transport, fake DNS): the network failures that
only the fetcher can produce, confined to the two catalog hosts over https."""

import gzip
from datetime import timedelta

import pytest

from social_tracker.catalog.domain import CATALOG_HOSTS, CatalogSource
from social_tracker.catalog.service import CatalogRefresher, Outcome
from social_tracker.collector.http_fetcher import FetchLimits
from social_tracker.persistence.catalog_repo import SqliteCatalogRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.unit_of_work import SqliteUnitOfWork
from social_tracker.shared.clock import FixedClock
from tests.catalog.support import NOW, emsc_payload, usgs_payload
from tests.collector.test_http_fetcher import FakeResolver, build, redirect, reply

USGS, EMSC = CatalogSource.USGS, CatalogSource.EMSC
LIMITS = FetchLimits(max_bytes=64 * 1024, max_redirects=2)
RESOLVER_TABLE = {"earthquake.usgs.gov": ["8.8.4.4"], "www.seismicportal.eu": ["93.184.216.34"],
                  "evil.example.net": ["8.8.8.8"]}  # fmt: skip


class Setup:
    def __init__(self, tmp_path, handler):
        self.db = Database(tmp_path / "h.db")
        self.repo = SqliteCatalogRepository(self.db)
        self.clock = FixedClock(NOW)
        self.fetcher, self.net, self.resolver = build(
            handler,
            FakeResolver(RESOLVER_TABLE),
            limits=LIMITS,
            allowed_hosts=CATALOG_HOSTS,
            https_only=True,
            allowed_ports=frozenset({443}),
        )
        self.refresher = CatalogRefresher(
            self.fetcher, self.repo, SqliteUnitOfWork(self.db), self.clock
        )

    def results(self):
        return {r.source: r for r in self.refresher.refresh(force=True).results}


def answer(request):
    host = request.headers["host"]
    return reply(200, content=usgs_payload({"id": "us1"}) if "usgs" in host else emsc_payload())


@pytest.fixture
def make(tmp_path):
    made = []

    def factory(handler=answer):
        setup = Setup(tmp_path, handler)
        made.append(setup)
        return setup

    yield factory
    for setup in made:
        setup.db.close()


def test_the_good_path_goes_through_the_real_fetcher_to_the_two_hosts_only(make):
    s = make()
    results = s.results()
    assert {r.outcome for r in results.values()} == {Outcome.REFRESHED}
    assert {request.headers["host"] for request in s.net.requests} == CATALOG_HOSTS
    assert all(request.url.scheme == "https" for request in s.net.requests)
    assert len(s.net.requests) == 4


def test_a_redirect_off_the_two_hosts_is_refused_and_nothing_follows_it(make):
    def handler(request):
        if "usgs" in request.headers["host"]:
            return redirect("https://evil.example.net/fdsnws/event/1/query")
        return answer(request)

    s = make(handler)
    results = s.results()
    assert results[USGS].outcome is Outcome.FAILED
    assert results[USGS].error == "host not allowed"
    assert results[EMSC].outcome is Outcome.REFRESHED
    assert all("evil" not in request.headers["host"] for request in s.net.requests)
    assert ("evil.example.net", 443) not in s.resolver.calls


def test_a_downgrade_to_plain_http_is_refused(make):
    s = make(lambda r: redirect("http://earthquake.usgs.gov/fdsnws/event/1/query"))
    results = s.results()
    assert results[USGS].outcome is Outcome.FAILED
    assert results[USGS].error == "redirect to plain http refused"


def test_a_redirect_between_the_two_catalogs_is_the_only_one_followed(make):
    def handler(request):
        if request.headers["host"] == "earthquake.usgs.gov":
            return redirect("https://www.seismicportal.eu/fdsnws/event/1/query")
        return reply(200, content=emsc_payload({"id": "e"}))

    s = make(handler)
    assert s.results()[EMSC].outcome is Outcome.REFRESHED


def test_a_redirect_loop_is_too_many_redirects_not_a_hang(make):
    s = make(lambda r: redirect("https://earthquake.usgs.gov/loop"))
    assert s.results()[USGS].error == "too many redirects"


def test_an_oversized_body_is_refused_by_size(make):
    s = make(lambda r: reply(200, content=b"x" * (LIMITS.max_bytes + 1)))
    results = s.results()
    assert {r.error for r in results.values()} == {"response too large"}
    assert s.repo.count_by_source() == {}


def test_a_gzip_bomb_is_stopped_at_the_cap(make):
    bomb = gzip.compress(b"0" * (10 * 1024 * 1024))
    s = make(lambda r: reply(200, content=bomb, headers={"content-encoding": "gzip"}))
    assert {r.error for r in s.results().values()} == {"response too large"}


def test_a_truncated_gzip_body_is_an_undecodable_answer(make):
    body = gzip.compress(usgs_payload({"id": "us1"}))[:-8]
    s = make(lambda r: reply(200, content=body, headers={"content-encoding": "gzip"}))
    assert {r.error for r in s.results().values()} == {"undecodable response"}


def test_a_json_body_cut_in_the_middle_is_an_unreadable_answer(make):
    s = make(lambda r: reply(200, content=usgs_payload({"id": "us1"})[:-20]))
    assert {r.error for r in s.results().values()} == {"unreadable answer"}
    assert s.repo.count_by_source() == {}


def test_an_address_that_resolves_to_a_private_range_is_refused(make):
    s = make()
    s.resolver.table["earthquake.usgs.gov"] = ["10.1.2.3"]
    results = s.results()
    assert results[USGS].error == "address not allowed"
    assert results[EMSC].outcome is Outcome.REFRESHED


def test_a_dns_failure_is_a_failure_of_that_catalog(make):
    s = make()
    del s.resolver.table["www.seismicportal.eu"]
    results = s.results()
    assert results[EMSC].error == "name lookup failed"
    assert results[USGS].outcome is Outcome.REFRESHED


def test_a_429_with_retry_after_delays_the_next_attempt(make):
    s = make(lambda r: reply(429, headers={"retry-after": "1200"}))
    s.refresher.refresh(force=True)
    for source in CatalogSource:
        state = s.repo.get_state(source)
        assert state.last_error == "http 429 (rate limited)"
        assert state.next_attempt_at == NOW + timedelta(seconds=1200)


def test_a_5xx_then_a_200_recovers(make):
    replies = iter([503, 503])

    def handler(request):
        status = next(replies, 200)
        return answer(request) if status == 200 else reply(status)

    s = make(handler)
    first = s.refresher.refresh(force=True)
    assert {r.outcome for r in first.results} == {Outcome.FAILED}
    s.clock.advance(timedelta(seconds=60))
    second = s.refresher.refresh()
    assert {r.outcome for r in second.results} == {Outcome.REFRESHED}
    assert s.repo.get_state(USGS).consecutive_failures == 0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 500, 502, 503, 504])
def test_every_error_status_is_a_failure_with_a_fixed_message(make, status):
    s = make(lambda r: reply(status, content=b"<html>SECRET-TOKEN</html>"))
    for result in s.results().values():
        assert result.outcome is Outcome.FAILED
        assert result.error.startswith(f"http {status}")
        assert "SECRET" not in result.error


def test_a_conditional_status_without_having_asked_is_not_a_good_answer(make):
    s = make(lambda r: reply(304))
    assert {r.outcome for r in s.results().values()} == {Outcome.FAILED}


def test_a_utf8_bom_in_the_body_is_tolerated(make):
    s = make(lambda r: reply(200, content=b"\xef\xbb\xbf" + usgs_payload({"id": "us1"})))
    assert s.results()[USGS].outcome is Outcome.REFRESHED
