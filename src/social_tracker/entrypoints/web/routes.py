"""The pages and actions of the source manager.

Every route runs behind `read_form` (size, content type and CSRF for anything but a read). Row
buttons post an explicit target state (`on=1`/`on=0`, enable/disable), never "flip", so a repeated
or stale submit is an idempotent no-op and the response always shows the state the server holds.
"""

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException

from social_tracker.bootstrap import Container
from social_tracker.entrypoints.web.formdata import read_form
from social_tracker.entrypoints.web.forms import (
    ACTOR,
    CHECKBOXES,
    DEFAULT_REASON,
    INITIAL_CHECKED,
    INITIAL_VALUES,
    NO_ERRORS,
    QUERY_MASK,
    TYPE_LABELS,
    VERSION_FIELD,
    FormErrors,
    build_data,
    check_reason,
    fields_for,
    keep_masked_urls,
    map_conflict,
    map_validation,
    parse_version,
    reason_of,
    values_from_source,
    view_fields,
    without_unvalidated_secrets,
)
from social_tracker.entrypoints.web.pages import (
    BUSY_TEXT,
    WebContext,
    is_htmx,
    render,
    status_fragment,
)
from social_tracker.entrypoints.web.rows import row_for
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
    RECORDED_AUDIO_MIN_POLL_S,
    SOURCE_TYPES,
    InstagramSource,
    LiveStreamSource,
    NewsRssSource,
    RecordedAudioSource,
    Source,
    parse_source,
)
from social_tracker.sources.service import STALE_SOURCE

router = APIRouter(dependencies=[Depends(read_form)])
Form = Annotated[dict[str, str], Depends(read_form)]

GONE_TEXT = "That source no longer exists. The list below is up to date."
STALE_TEXT = (
    "This source was changed by someone else since you opened it. The form now shows the "
    "current values: check them and save again."
)
NOTICES = {
    "not_found": GONE_TEXT,
    "busy": BUSY_TEXT,
    "analysis_on": "Analysis is now on.",
    "analysis_off": "Analysis is now off.",
    "enabled": "Source enabled.",
    "disabled": "Source disabled.",
    "reset": "Source reset: a failed source goes back to work, any other one was left as it is.",
    "added": "Source added. It stays disabled until you enable it.",
    "updated": "Source updated.",
    "unchanged": "Nothing to change: the source already has these values.",
    "deleted": "Source deleted. Its alerts and derived data are kept.",
}


def _redirect(code: str) -> RedirectResponse:
    return RedirectResponse(f"/?notice={code}", status_code=303)


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
        rows=[row_for(o) for o in overview.sources],
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
    request: Request, services: Container, source_id: str, message: str, *, status: int = 200
) -> Response:
    """The htmx answer: the row as the server holds it plus the status line for screen readers.
    One read of the overview (a single snapshot) on the container the request already opened."""
    found = [o for o in services.control.overview().sources if o.source.id == source_id]
    row = row_for(found[0]) if found else None
    return status_fragment(request, message, status=status, row=row)


def _gone(request: Request) -> Response:
    if is_htmx(request):
        return status_fragment(request, GONE_TEXT, status=404, refresh=True)
    return _redirect("not_found")


def _busy(request: Request) -> Response:
    if is_htmx(request):
        return status_fragment(request, BUSY_TEXT, status=503)
    return _redirect("busy")


def _switch(
    request: Request,
    source_id: str,
    form: dict[str, str],
    act: Callable[[Container, str], str],
    code: str,
) -> Response:
    """Apply one switch; `act` returns the confirmation text. The answer shows the true state.

    One container serves the whole request: the write and, for htmx, the read that draws the row
    afterwards (also after a failed write, so the row is never left showing a state it lacks).
    """
    ctx: WebContext = request.app.state.ctx
    if not is_valid_slug(source_id):
        return _gone(request)
    try:
        with ctx.services() as services:
            try:
                message = act(services, reason_of(form))
            except NotFoundError:
                return _gone(request)
            except StorageError:
                if not is_htmx(request):
                    return _redirect("busy")
                return _fragment(request, services, source_id, BUSY_TEXT, status=503)
            if not is_htmx(request):
                return _redirect(code)
            return _fragment(request, services, source_id, message)
    except StorageError:  # the store could not be opened, or the row could not be read back
        return _busy(request)


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


@router.post("/sources/{source_id}/reset")
def reset(request: Request, source_id: str, form: Form) -> Response:
    """Put a FAILED source back to work. Idempotent: a source that is not (or no longer) failed
    is left alone, so a double click or a stale page is harmless."""

    def act(services: Container, reason: str) -> str:
        change = services.control.reset_if_failed(source_id, actor=ACTOR, reason=reason)
        name = services.sources.get(source_id).name
        return f"{name} was reset." if change.changed else f"{name} is not failed: left as it is."

    return _switch(request, source_id, form, act, "reset")


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
    version: str = "",
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
        version=version,
        mask_suffix=f"?{QUERY_MASK}",
    )


def _failure(
    request: Request,
    source_type: str,
    source_id: str | None,
    form: dict[str, str],
    errors: FormErrors,
    *,
    status: int,
    version: str = "",
) -> HTMLResponse:
    return _form_page(
        request,
        source_type,
        source_id=source_id,
        values=form,
        checked={name: name in form for name in CHECKBOXES},
        errors=errors,
        status=status,
        version=version,
    )


def _busy_failure(
    request: Request, source_type: str, source_id: str | None, form: dict[str, str], version: str
) -> HTMLResponse:
    """The database was locked: the form comes back as typed, minus anything never validated
    that could be a credential, with a message that asks to try again."""
    return _failure(
        request,
        source_type,
        source_id,
        without_unvalidated_secrets(source_type, form),
        FormErrors({}, [BUSY_TEXT]),
        status=503,
        version=version,
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


def _is_repeat(services: Container, data: dict[str, Any]) -> bool:
    """True when the source `data` describes already exists exactly as described (enabled flag and
    timestamps aside): the add form was submitted twice, the second submit has nothing to add."""
    try:
        existing = services.sources.get(str(data.get("id")))
        candidate = parse_source(
            {
                **data,
                "enabled": existing.enabled,
                "created_at": existing.created_at,
                "updated_at": existing.updated_at,
            }
        )
    except (NotFoundError, ValidationError):
        return False
    return candidate == existing


@router.post("/sources/new/{source_type}")
def create_source(request: Request, source_type: str, form: Form) -> Response:
    _type_or_404(source_type)
    ctx: WebContext = request.app.state.ctx
    errors, status = check_reason(form), 422
    if not errors:
        try:
            with ctx.services() as services:
                data = build_data(source_type, form, editing=False)
                try:
                    services.sources.add(data, actor=ACTOR, reason=reason_of(form))
                except ConflictError:
                    if not _is_repeat(services, data):
                        raise
            return _redirect("added")
        except ValidationError as exc:
            errors = map_validation(exc, _names(source_type, editing=False))
        except ConflictError as exc:
            errors, status = map_conflict(exc), 409
        except StorageError:
            return _busy_failure(request, source_type, None, form, "")
    return _failure(request, source_type, None, form, errors, status=status)


@router.get("/sources/{source_id}/edit", response_class=HTMLResponse)
def edit_form(request: Request, source_id: str) -> HTMLResponse:
    source = _load(request.app.state.ctx, source_id)
    values, checked = values_from_source(source)
    return _form_page(
        request,
        source.type,
        source_id=source.id,
        values=values,
        checked=checked,
        version=source.updated_at.isoformat(),
    )


def _stale_form(request: Request, ctx: WebContext, source_id: str) -> HTMLResponse:
    """The edit form as the source is stored now, with a message that someone else changed it."""
    source = _load(ctx, source_id)
    values, checked = values_from_source(source)
    return _form_page(
        request,
        source.type,
        source_id=source.id,
        values=values,
        checked=checked,
        errors=FormErrors({}, [STALE_TEXT]),
        status=409,
        version=source.updated_at.isoformat(),
    )


@router.post("/sources/{source_id}/edit")
def update_source(request: Request, source_id: str, form: Form) -> Response:
    ctx: WebContext = request.app.state.ctx
    source = _load(ctx, source_id)
    expected = parse_version(form.get(VERSION_FIELD))
    if expected is None:  # no version, or one that cannot be compared: treat it as stale
        return _stale_form(request, ctx, source_id)
    effective = keep_masked_urls(source, form)  # what is saved; `form` is what is shown back
    version = expected.isoformat()
    errors, status = check_reason(form), 422
    if not errors:
        try:
            with ctx.services() as services:
                changes = build_data(source.type, effective, editing=True)
                change = services.sources.edit(
                    source.id,
                    changes,
                    actor=ACTOR,
                    reason=reason_of(form),
                    expected_updated_at=expected,
                )
            return _redirect("updated" if change.changed else "unchanged")
        except ValidationError as exc:
            errors = map_validation(exc, _names(source.type, editing=True))
        except ConflictError as exc:
            if exc.message == STALE_SOURCE:
                return _stale_form(request, ctx, source_id)
            errors, status = map_conflict(exc), 409
        except StorageError:
            return _busy_failure(request, source.type, source.id, form, version)
    return _failure(request, source.type, source.id, form, errors, status=status, version=version)


# ---- delete ----


def _confirm_page(
    request: Request,
    source: Source,
    *,
    error: str = "",
    busy: str = "",
    reason: str = DEFAULT_REASON,
    status: int = 200,
) -> Response:
    return render(
        request,
        "confirm_delete.html",
        status=status,
        source=source,
        error=error,
        busy=busy,
        reason=reason,
    )


@router.get("/sources/{source_id}/delete", response_class=HTMLResponse)
def confirm_delete(request: Request, source_id: str) -> Response:
    return _confirm_page(request, _load(request.app.state.ctx, source_id))


@router.post("/sources/{source_id}/delete")
def delete_source(request: Request, source_id: str, form: Form) -> Response:
    ctx: WebContext = request.app.state.ctx
    if not is_valid_slug(source_id):
        return _redirect("not_found")
    reason = reason_of(form)
    errors = check_reason(form)
    try:
        if errors:
            source = _load(ctx, source_id)
            return _confirm_page(
                request, source, error=errors.fields["reason"], reason=reason, status=422
            )
        with ctx.services() as services:
            services.sources.delete(source_id, actor=ACTOR, reason=reason)
    except NotFoundError:
        return _redirect("not_found")
    except StorageError:
        try:
            source = _load(ctx, source_id)
        except NotFoundError:
            return _redirect("not_found")
        return _confirm_page(request, source, busy=BUSY_TEXT, reason=reason, status=503)
    return _redirect("deleted")


def _default(model: type[Source], field: str) -> Any:
    """A default as the domain declares it, so the how-to cannot drift from the code."""
    return model.model_fields[field].default


@router.get("/howto", response_class=HTMLResponse)
def howto(request: Request) -> HTMLResponse:
    return render(
        request,
        "howto.html",
        types=TYPE_LABELS,
        news_poll_min=NEWS_RSS_MIN_POLL_S,
        news_poll_default=_default(NewsRssSource, "poll_interval_s"),
        instagram_poll_min=INSTAGRAM_MIN_POLL_S,
        instagram_poll_default=_default(InstagramSource, "poll_interval_s"),
        recorded_poll_min=RECORDED_AUDIO_MIN_POLL_S,
        recorded_poll_default=_default(RecordedAudioSource, "poll_interval_s"),
        chunk_min=CHUNK_SECONDS_MIN,
        chunk_max=CHUNK_SECONDS_MAX,
        chunk_default=_default(LiveStreamSource, "chunk_seconds"),
        language_default=_default(LiveStreamSource, "language"),
    )
