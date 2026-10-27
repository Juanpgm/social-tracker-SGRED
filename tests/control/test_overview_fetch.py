"""The overview carries each source's last successful fetch and last error."""

from datetime import UTC, datetime

from social_tracker.control.domain import FetchStatus
from social_tracker.control.service import ControlService
from social_tracker.shared.clock import FixedClock
from social_tracker.sources.service import SourceRegistry
from tests.actors import WHO
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.fakes.unit_of_work import InMemoryUnitOfWork
from tests.sources.factories import payload

NOW = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def build():
    sources, control = InMemorySourceRepository(), InMemoryControlRepository()
    uow, clock = InMemoryUnitOfWork([sources, control]), FixedClock(NOW)
    return (
        SourceRegistry(sources, control, uow, clock),
        ControlService(control, sources, uow, clock),
        control,
    )


def test_a_source_never_fetched_has_nothing_to_show():
    registry, service, _ = build()
    registry.add(payload(), **WHO)
    (row,) = service.overview().sources
    assert (row.last_fetch_at, row.last_error) == (None, None)


def test_the_overview_shows_the_stored_fetch_status_of_each_source():
    registry, service, control = build()
    registry.add(payload(id="src_one", name="One"), **WHO)
    registry.add(payload(id="src_two", name="Two"), **WHO)
    registry.add(payload(id="src_three", name="Three"), **WHO)
    control.save_fetch_status(FetchStatus("src_one", NOW, None))
    control.save_fetch_status(FetchStatus("src_two", NOW, "http 503"))
    rows = {row.source.id: row for row in service.overview().sources}
    assert (rows["src_one"].last_fetch_at, rows["src_one"].last_error) == (NOW, None)
    assert (rows["src_two"].last_fetch_at, rows["src_two"].last_error) == (NOW, "http 503")
    assert (rows["src_three"].last_fetch_at, rows["src_three"].last_error) == (None, None)


def test_deleting_a_source_forgets_its_fetch_status():
    registry, service, control = build()
    registry.add(payload(), **WHO)
    control.save_fetch_status(FetchStatus("src_one", NOW, "timeout"))
    registry.delete("src_one", **WHO)
    assert control.get_fetch_status("src_one") is None
    assert service.overview().sources == ()
