"""The pages and actions of the source manager."""

from dataclasses import dataclass

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from social_tracker.control.domain import CollectionState
from social_tracker.control.service import SourceOverview
from social_tracker.entrypoints.web.pages import WebContext, render
from social_tracker.shared.errors import NotFoundError
from social_tracker.sources.domain import SOURCE_TYPES

router = APIRouter()

NOTICES = {
    "not_found": "That source no longer exists. The list below is up to date.",
}


@dataclass(frozen=True)
class Row:
    """One source as the list shows it."""

    overview: SourceOverview
    failures: int
    retry_after: str | None


def _row(ctx: WebContext, overview: SourceOverview) -> Row:
    failures, retry_after = 0, None
    if overview.state in (CollectionState.BACKOFF, CollectionState.FAILED):
        try:
            with ctx.services() as services:
                state = services.control.get_state(overview.source.id)
        except NotFoundError:  # deleted since the overview was read
            return Row(overview, 0, None)
        failures = state.consecutive_failures
        retry_after = state.next_run_at.isoformat() if state.next_run_at else None
    return Row(overview, failures, retry_after)


@router.get("/", response_class=HTMLResponse)
def index(request: Request, notice: str = "") -> HTMLResponse:
    ctx: WebContext = request.app.state.ctx
    with ctx.services() as services:
        overview = services.control.overview()
    rows = [_row(ctx, o) for o in overview.sources]
    return render(
        request,
        "index.html",
        rows=rows,
        global_on=overview.global_analysis,
        notice=NOTICES.get(notice),
        source_types=SOURCE_TYPES,
    )
