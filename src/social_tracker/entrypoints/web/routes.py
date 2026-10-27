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
from social_tracker.entrypoints.web.forms import (
    ACTOR,
    CHECKBOXES,
    INITIAL_CHECKED,
    INITIAL_VALUES,
    NO_ERRORS,
    TYPE_LABELS,
    FormErrors,
    build_data,
    check_reason,
    fields_for,
    map_conflict,
    map_validation,
    reason_of,
    values_from_source,
    view_fields,
)
from social_tracker.entrypoints.web.pages import BUSY_TEXT, WebContext, render
from social_tracker.shared.errors import (
    ConflictError,
    NotFoundError,
    StorageError,
    ValidationError,
)
from social_tracker.shared.ids import is_valid_slug
from social_tracker.sources.domain import (
    CHUNK_SECONDS_MAX,
    CHUNK_SECONDS_MIN,
    INSTAGRAM_MIN_POLL_S,
    NEWS_RSS_MIN_POLL_S,
    SOURCE_TYPES,
    Source,
)

router = APIRouter(dependencies=[Depends(read_form)])
Form = Annotated[dict[str, str], Depends(read_form)]

GONE_TEXT = "That source no longer exists. The list below is up to date."
NOTICES = {
    "not_found": GONE_TEXT,
    "busy": BUSY_TEXT,
    "analysis_on": "Analysis is now on.",
    "analysis_off": "Analysis is now off.",
    "enabled": "Source enabled.",
    "disabled": "Source disabled.",
    "added": "Source added. It stays disabled until you enable it.",
    "updated": "Source updated.",
    "unchanged": "Nothing to change: the source already has these values.",
    "deleted": "Source deleted. Its alerts and derived data are kept.",
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
    if not found:
        headers["HX-Reswap"] = "none"  # nothing true to draw: leave the page as it is
    row = _row(ctx, found[0]) if found else None
    response = render(request, "_toggle_result.html", status=status, row=row, message=message)
    response.headers.update(headers)
    return response


def _gone(request: Request) -> Response:
    if _is_htmx(request):
        return _fragment(request, "", GONE_TEXT, status=404, refresh=True)
    return _redirect("not_found")


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
        return _gone(request)
    try:
        with ctx.services() as services:
            message = act(services, reason_of(form))
    except NotFoundError:
        return _gone(request)
    except StorageError:
        if _is_htmx(request):
            return _fragment(request, source_id, BUSY_TEXT, status=503)
        return _redirect("busy")
    return _fragment(request, source_id, message) if _is_htmx(request) else _redirect(code)


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


# ---- add and edit ----


def _type_or_404(source_type: str) -> str:
    if source_type not in SOURCE_TYPES:
        raise HTTPException(404)
    return source_type


def _load(ctx: WebContext, source_id: str) -> Source:
    if not is_valid_slug(source_id):
        raise NotFoundError("source not found")
    with ctx.services() as services:
        return services.sources.get(source_id)


def _names(source_type: str, *, editing: bool) -> set[str]:
    return {f.name for f in fields_for(source_type, editing=editing)} | {"reason"}


def _form_page(
    request: Request,
    source_type: str,
    *,
    source_id: str | None,
    values: dict[str, str],
    checked: dict[str, bool],
    errors: FormErrors = NO_ERRORS,
    status: int = 200,
) -> HTMLResponse:
    editing = source_id is not None
    return render(
        request,
        "source_form.html",
        status=status,
        source_type=source_type,
        type_label=TYPE_LABELS[source_type],
        source_id=source_id,
        action=f"/sources/{source_id}/edit" if editing else f"/sources/new/{source_type}",
        fields=view_fields(source_type, values, errors.fields, editing=editing, checked=checked),
        form_errors=errors.form,
        editing=editing,
        has_reference=editing and source_type == "instagram",
    )


def _failure(
    request: Request,
    source_type: str,
    source_id: str | None,
    form: dict[str, str],
    errors: FormErrors,
    *,
    conflict: bool,
) -> HTMLResponse:
    return _form_page(
        request,
        source_type,
        source_id=source_id,
        values=form,
        checked={name: name in form for name in CHECKBOXES},
        errors=errors,
        status=409 if conflict else 422,
    )


@router.get("/sources/new", response_class=HTMLResponse)
def choose_type(request: Request) -> HTMLResponse:
    return render(request, "choose_type.html", types=TYPE_LABELS)


@router.get("/sources/new/{source_type}", response_class=HTMLResponse)
def new_form(request: Request, source_type: str) -> HTMLResponse:
    _type_or_404(source_type)
    return _form_page(
        request, source_type, source_id=None, values=INITIAL_VALUES, checked=INITIAL_CHECKED
    )


@router.post("/sources/new/{source_type}")
def create_source(request: Request, source_type: str, form: Form) -> Response:
    _type_or_404(source_type)
    ctx: WebContext = request.app.state.ctx
    errors, conflict = check_reason(form), False
    if not errors:
        try:
            with ctx.services() as services:
                data = build_data(source_type, form, editing=False)
                services.sources.add(data, actor=ACTOR, reason=reason_of(form))
            return _redirect("added")
        except ValidationError as exc:
            errors = map_validation(exc, _names(source_type, editing=False))
        except ConflictError as exc:
            errors, conflict = map_conflict(exc), True
    return _failure(request, source_type, None, form, errors, conflict=conflict)


@router.get("/sources/{source_id}/edit", response_class=HTMLResponse)
def edit_form(request: Request, source_id: str) -> HTMLResponse:
    source = _load(request.app.state.ctx, source_id)
    values, checked = values_from_source(source)
    return _form_page(request, source.type, source_id=source.id, values=values, checked=checked)


@router.post("/sources/{source_id}/edit")
def update_source(request: Request, source_id: str, form: Form) -> Response:
    ctx: WebContext = request.app.state.ctx
    source = _load(ctx, source_id)
    errors, conflict = check_reason(form), False
    if not errors:
        try:
            with ctx.services() as services:
                changes = build_data(source.type, form, editing=True)
                change = services.sources.edit(
                    source.id, changes, actor=ACTOR, reason=reason_of(form)
                )
            return _redirect("updated" if change.changed else "unchanged")
        except ValidationError as exc:
            errors = map_validation(exc, _names(source.type, editing=True))
        except ConflictError as exc:
            errors, conflict = map_conflict(exc), True
    return _failure(request, source.type, source.id, form, errors, conflict=conflict)


# ---- delete ----


def _confirm_page(request: Request, source: Source, error: str = "", status: int = 200) -> Response:
    return render(request, "confirm_delete.html", status=status, source=source, error=error)


@router.get("/sources/{source_id}/delete", response_class=HTMLResponse)
def confirm_delete(request: Request, source_id: str) -> Response:
    return _confirm_page(request, _load(request.app.state.ctx, source_id))


@router.post("/sources/{source_id}/delete")
def delete_source(request: Request, source_id: str, form: Form) -> Response:
    ctx: WebContext = request.app.state.ctx
    if not is_valid_slug(source_id):
        return _redirect("not_found")
    errors = check_reason(form)
    try:
        if errors:
            return _confirm_page(request, _load(ctx, source_id), errors.fields["reason"], 422)
        with ctx.services() as services:
            services.sources.delete(source_id, actor=ACTOR, reason=reason_of(form))
    except NotFoundError:
        return _redirect("not_found")
    return _redirect("deleted")


@router.get("/howto", response_class=HTMLResponse)
def howto(request: Request) -> HTMLResponse:
    return render(
        request,
        "howto.html",
        types=TYPE_LABELS,
        news_poll_min=NEWS_RSS_MIN_POLL_S,
        instagram_poll_min=INSTAGRAM_MIN_POLL_S,
        chunk_min=CHUNK_SECONDS_MIN,
        chunk_max=CHUNK_SECONDS_MAX,
    )
