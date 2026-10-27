"""The pages and actions of the source manager.

Every route runs behind `read_form` (size, content type and CSRF for anything but a read). Row
buttons post an explicit target state (`on=1`/`on=0`, enable/disable), never "flip", so a repeated
or stale submit is an idempotent no-op and the response always shows the state the server holds.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException

from social_tracker.bootstrap import Container
from social_tracker.control.domain import CollectionState
from social_tracker.control.service import SourceOverview
from social_tracker.entrypoints.web.formdata import read_form
from social_tracker.entrypoints.web.forms import ACTOR, reason_of
from social_tracker.entrypoints.web.pages import BUSY_TEXT, WebContext, render
from social_tracker.shared.errors import NotFoundError, StorageError
from social_tracker.shared.ids import is_valid_slug
from social_tracker.sources.domain import SOURCE_TYPES

router = APIRouter(dependencies=[Depends(read_form)])
Form = Annotated[dict[str, str], Depends(read_form)]

NOTICES = {
    "not_found": "That source no longer exists. The list below is up to date.",
    "busy": BUSY_TEXT,
    "analysis_on": "Analysis is now on.",
    "analysis_off": "Analysis is now off.",
    "enabled": "Source enabled.",
    "disabled": "Source disabled.",
}
GONE_TEXT = NOTICES["not_found"]


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


def _redirect(code: str) -> RedirectResponse:
    return RedirectResponse(f"/?notice={code}", status_code=303)


def _is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def _on(form: dict[str, str]) -> bool:
    value = form.get("on")
    if value not in ("0", "1"):
        raise HTTPException(400)
    return value == "1"


@router.get("/", response_class=HTMLResponse)
def index(request: Request, notice: str = "") -> HTMLResponse:
    ctx: WebContext = request.app.state.ctx
    with ctx.services() as services:
        overview = services.control.overview()
    return render(
        request,
        "index.html",
        rows=[_row(ctx, o) for o in overview.sources],
        global_on=overview.global_analysis,
        notice=NOTICES.get(notice),
        source_types=SOURCE_TYPES,
    )


@router.post("/analysis")
def set_global_analysis(request: Request, form: Form) -> Response:
    on = _on(form)
    ctx: WebContext = request.app.state.ctx
    with ctx.services() as services:
        services.control.set_global_analysis(on, actor=ACTOR, reason=reason_of(form))
    return _redirect("analysis_on" if on else "analysis_off")


# ---- per-source switches ----


def _fragment(
    request: Request, source_id: str, message: str, *, status: int = 200, refresh: bool = False
) -> Response:
    """The htmx answer: the row as the server holds it plus the status line for screen readers."""
    ctx: WebContext = request.app.state.ctx
    headers = {"HX-Refresh": "true"} if refresh else {}
    try:
        with ctx.services() as services:
            found = [o for o in services.control.overview().sources if o.source.id == source_id]
    except StorageError:
        found = []
        headers["HX-Reswap"] = "none"
    if not found:
        headers.setdefault("HX-Reswap", "none")
    row = _row(ctx, found[0]) if found else None
    response = render(request, "_toggle_result.html", status=status, row=row, message=message)
    response.headers.update(headers)
    return response


def _switch(
    request: Request,
    source_id: str,
    form: dict[str, str],
    act: Callable[[Container, str], str],
    code: str,
) -> Response:
    """Apply one switch; `act` returns the confirmation text. The answer shows the true state."""
    ctx: WebContext = request.app.state.ctx
    if not is_valid_slug(source_id):
        return _gone(request, source_id)
    try:
        with ctx.services() as services:
            message = act(services, reason_of(form))
    except NotFoundError:
        return _gone(request, source_id)
    except StorageError:
        if _is_htmx(request):
            return _fragment(request, source_id, BUSY_TEXT, status=503)
        return _redirect("busy")
    return _fragment(request, source_id, message) if _is_htmx(request) else _redirect(code)


def _gone(request: Request, source_id: str) -> Response:
    if _is_htmx(request):
        return _fragment(request, "", GONE_TEXT, status=404, refresh=True)
    return _redirect("not_found")


@router.post("/sources/{source_id}/enable")
def enable(request: Request, source_id: str, form: Form) -> Response:
    def act(services: Container, reason: str) -> str:
        services.sources.enable(source_id, actor=ACTOR, reason=reason)
        return f"{services.sources.get(source_id).name} is enabled."

    return _switch(request, source_id, form, act, "enabled")


@router.post("/sources/{source_id}/disable")
def disable(request: Request, source_id: str, form: Form) -> Response:
    def act(services: Container, reason: str) -> str:
        services.sources.disable(source_id, actor=ACTOR, reason=reason)
        return f"{services.sources.get(source_id).name} is disabled."

    return _switch(request, source_id, form, act, "disabled")


@router.post("/sources/{source_id}/analysis")
def set_source_analysis(request: Request, source_id: str, form: Form) -> Response:
    on = _on(form)

    def act(services: Container, reason: str) -> str:
        services.control.set_source_analysis(source_id, on, actor=ACTOR, reason=reason)
        name = services.sources.get(source_id).name
        return f"Analysis for {name} is {'on' if on else 'off'}."

    return _switch(request, source_id, form, act, "analysis_on" if on else "analysis_off")
