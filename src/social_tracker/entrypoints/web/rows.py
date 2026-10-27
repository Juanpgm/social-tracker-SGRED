"""How one source is listed: its overview plus a health summary derived from what is stored.

Health is a reading of the collection state, nothing more: `ok` while it is idle or collecting,
`degraded` in backoff (failures so far, the next attempt), `failed` once collection stopped, and
no health at all for a disabled source. The reason is the stored failure count and retry time.
The last fetch (UTC) and the last error are what the collector recorded; a source that never
fetched, or never failed, shows the explicit placeholder `n/a` (done by the template).
"""

from dataclasses import dataclass

from social_tracker.control.domain import CollectionState
from social_tracker.control.service import SourceOverview

OK, DEGRADED, FAILED, OFF = "ok", "degraded", "failed", "n/a"


@dataclass(frozen=True)
class Row:
    """One source as the list shows it."""

    overview: SourceOverview
    health: str
    reason: str

    @property
    def last_fetch(self) -> str | None:
        moment = self.overview.last_fetch_at
        return moment.strftime("%Y-%m-%d %H:%M:%S UTC") if moment else None

    @property
    def last_error(self) -> str | None:
        return self.overview.last_error


def _failures(count: int) -> str:
    return f"{count} failure{'' if count == 1 else 's'}"


def row_for(overview: SourceOverview) -> Row:
    state = overview.state
    if state is CollectionState.BACKOFF:
        retry = overview.next_run_at
        reason = _failures(overview.consecutive_failures)
        if retry:
            reason += f", retry after {retry.isoformat()}"
        return Row(overview, DEGRADED, reason)
    if state is CollectionState.FAILED:
        reason = f"{_failures(overview.consecutive_failures)}, collection stopped"
        return Row(overview, FAILED, reason)
    if state is CollectionState.DISABLED:
        return Row(overview, OFF, "the source is disabled")
    return Row(overview, OK, "")
