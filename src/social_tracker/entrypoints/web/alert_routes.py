"""The alerts page and the marks on it (UI-7).

Alerts are read here and marked seen, dismissed or reopened; nothing else can be done with one:
no route publishes, shares, mails or approves anything (G-1, UI-11), and every string of an alert
is escaped by the template that draws it. The list is a plain GET form (filters in the query), so
a filtered page can be bookmarked and works without script; the marks are POST forms that htmx
upgrades to fragments. After any mark, also a refused one, the answer shows the alert as the
server holds it (UI-6): a stale page is corrected, never believed.
"""

import re
from math import ceil
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.exceptions import HTTPException

from social_tracker.alerts.domain import (
    KNOWN_TAGS,
    WITHDRAWN,
    AlertFilter,
    AlertStatus,
    Severity,
    parse_moment,
)
from social_tracker.alerts.service import AlertDetail
from social_tracker.bootstrap import Container
from social_tracker.classification.domain import RISK_LABELS
from social_tracker.entrypoints.web.formdata import read_form
from social_tracker.entrypoints.web.forms import ACTOR, check_reason, reason_of
from social_tracker.entrypoints.web.pages import (
    BUSY_TEXT,
    WebContext,
    error_page,
    is_htmx,
    render,
    status_fragment,
)
from social_tracker.shared.errors import (
    InvalidTransition,
    NotFoundError,
    StorageError,
    ValidationError,
)
from social_tracker.shared.ids import is_valid_slug

router = APIRouter(dependencies=[Depends(read_form)])
Form = Annotated[dict[str, str], Depends(read_form)]

PAGE_SIZE = 25
FILTER_FIELDS = ("severity", "label", "status", "source", "tag", "since", "until")
MAX_PARAM_CHARS = 100
VIEWS = ("list", "detail")
GONE_TEXT = "That alert no longer exists. The list below is up to date."
INVALID_REASON_TEXT = "The reason is not valid: use plain text of at most 500 characters."
NOTICES = {
    "marked_seen": "Alert marked as seen.",
    "marked_dismissed": "Alert dismissed.",
    "marked_reopened": "Alert reopened.",
    "unchanged": "Nothing to change: the alert already had that status.",
    "gone": GONE_TEXT,
    "busy": BUSY_TEXT,
}
_PAGE_RE = re.compile(r"[0-9]{1,7}", re.ASCII)
_BACK_RE = re.compile(r"/alerts(?:\?[A-Za-z0-9_=&%.:+\-]{0,500})?", re.ASCII)
_MARKS = {
    "seen": ("marked_seen", "Alert marked as seen."),
    "dismiss": ("marked_dismissed", "Alert dismissed."),
    "reopen": ("marked_reopened", "Alert reopened."),
}


# ---- reading the query ----


def _param(request: Request, name: str) -> str:
    value = request.query_params.get(name, "")
    if len(value) > MAX_PARAM_CHARS:
        raise ValidationError("a filter value is too long")
    return value.strip()


def _filter(request: Request) -> tuple[AlertFilter, dict[str, str]]:
    values = {name: _param(request, name) for name in FILTER_FIELDS}
    chosen = AlertFilter(
        severity=values["severity"] or None,  # type: ignore[arg-type]
        label=values["label"] or None,  # type: ignore[arg-type]
        source_id=values["source"] or None,
        status=values["status"] or None,
        tag=values["tag"] or None,
        since=parse_moment(values["since"], end=False) if values["since"] else None,
        until=parse_moment(values["until"], end=True) if values["until"] else None,
    )
    return chosen, values


def _page(request: Request) -> int:
    text = request.query_params.get("page", "").strip()
    if not text:
        return 1
    if not _PAGE_RE.fullmatch(text) or int(text) < 1:
        raise ValidationError("the page must be a whole number from 1")
    return int(text)


def _url(values: dict[str, str], page: int) -> str:
    params = {name: value for name, value in values.items() if value}
    if page > 1:
        params["page"] = str(page)
    return "/alerts" + (f"?{urlencode(params)}" if params else "")


def _safe_back(raw: str, alert_id: str, view: str) -> str:
    """Where a plain form returns to: the page of alerts it came from, else the list. Only an
    address of this page (`/alerts` and a short plain query) is believed; nothing else is."""
    if view == "detail":
        return f"/alerts/{alert_id}" if is_valid_slug(alert_id) else "/alerts"
    return raw if _BACK_RE.fullmatch(raw) else "/alerts"


def _with_notice(path: str, code: str) -> str:
    separator = "&" if "?" in path else "?"
    return f"{path}{separator}{urlencode({'notice': code})}"


# ---- the pages ----


@router.get("/alerts", response_class=HTMLResponse)
def alerts_page(request: Request, notice: str = "") -> HTMLResponse:
    ctx: WebContext = request.app.state.ctx
    chosen, values = _filter(request)
    page = _page(request)
    with ctx.services() as services:
        listing = services.alerts.list(chosen, limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
        removed = services.alerts.removed_sources({a.source_id for a in listing.alerts})
        new = services.alerts.counts().new
        source_ids = [o.source.id for o in services.control.overview().sources]
    pages = max(1, ceil(listing.total / PAGE_SIZE))
    return render(
        request,
        "alerts.html",
        notice=NOTICES.get(notice),
        alerts_new=new,
        rows=listing.alerts,
        removed=removed,
        total=listing.total,
        page=page,
        pages=pages,
        f=values,
        filtered=any(values.values()),
        back=_url(values, page),
        prev_url=_url(values, page - 1) if listing.has_previous and page > 1 else "",
        next_url=_url(values, page + 1) if listing.has_next else "",
        first_page_url=_url(values, 1),
        severities=[s.value for s in Severity],
        labels=[label.value for label in RISK_LABELS],
        statuses=[*(s.value for s in AlertStatus), WITHDRAWN],
        tags=KNOWN_TAGS,
        source_ids=source_ids,
    )


@router.get("/alerts/{alert_id}", response_class=HTMLResponse)
def alert_page(request: Request, alert_id: str, notice: str = "") -> HTMLResponse:
    ctx: WebContext = request.app.state.ctx
    try:
        with ctx.services() as services:
            found = services.alerts.detail(alert_id)
            new = services.alerts.counts().new
    except NotFoundError:
        return error_page(request, 404, "That alert does not exist.")
    return render(
        request,
        "alert_detail.html",
        notice=NOTICES.get(notice),
        alerts_new=new,
        alert=found.alert,
        members=found.members,
        events=found.events,
        removed=found.removed_sources,
    )


# ---- the marks ----


def _answer(
    request: Request,
    services: Container,
    alert_id: str,
    view: str,
    back: str,
    message: str,
    *,
    status: int,
) -> Response:
    """The htmx answer: the alert as it is now (a row, or the status panel of the detail page),
    the message for the live region and the counter of the badge."""
    found: AlertDetail = services.alerts.detail(alert_id)
    return render(
        request,
        "_alert_result.html",
        status=status,
        alert=found.alert,
        removed=found.removed_sources,
        view=view,
        back=back,
        message=message,
        failed=status >= 400,
        alerts_new=services.alerts.counts().new,
    )


def _gone(request: Request) -> Response:
    if is_htmx(request):
        return status_fragment(request, GONE_TEXT, status=404, refresh=True)
    return RedirectResponse(_with_notice("/alerts", "gone"), status_code=303)


def _busy(request: Request, back: str) -> Response:
    if is_htmx(request):
        return status_fragment(request, BUSY_TEXT, status=503)
    return RedirectResponse(_with_notice(back, "busy"), status_code=303)


def _mark(request: Request, alert_id: str, form: dict[str, str], verb: str) -> Response:
    view = form.get("view", "list")
    if view not in VIEWS:
        raise HTTPException(400)
    back = _safe_back(form.get("back", ""), alert_id, view)
    if not is_valid_slug(alert_id):
        return _gone(request)
    code, done = _MARKS[verb]
    ctx: WebContext = request.app.state.ctx
    try:
        with ctx.services() as services:
            message, status = done, 200
            if check_reason(form):
                message, status = INVALID_REASON_TEXT, 422
            else:
                try:
                    message, status = _apply(services, alert_id, verb, form), 200
                except NotFoundError:
                    return _gone(request)
                except InvalidTransition as exc:
                    message, status = exc.message, 409
                except StorageError:
                    message, status = BUSY_TEXT, 503
            if status == 200 and not is_htmx(request):
                changed = message == done
                return RedirectResponse(
                    _with_notice(back, code if changed else "unchanged"), status_code=303
                )
            if not is_htmx(request):
                if status == 503:
                    return RedirectResponse(_with_notice(back, "busy"), status_code=303)
                return error_page(request, status, message)
            return _answer(request, services, alert_id, view, back, message, status=status)
    except NotFoundError:
        return _gone(request)
    except StorageError:  # the store could not be opened, or the alert could not be read back
        return _busy(request, back)


def _apply(services: Container, alert_id: str, verb: str, form: dict[str, str]) -> str:
    """Do the mark and return the message to show: the confirmation, or that nothing changed."""
    reason = reason_of(form)
    if verb == "reopen":
        result = services.alerts.reopen(alert_id, actor=ACTOR, reason=reason)
    else:
        target = "seen" if verb == "seen" else "dismissed"
        result = services.alerts.set_status(alert_id, target, actor=ACTOR, reason=reason)
    if result.changed:
        return _MARKS[verb][1]
    return f"That alert is already {result.alert.status.value}. Nothing changed."


@router.post("/alerts/{alert_id}/seen")
def mark_seen(request: Request, alert_id: str, form: Form):
    return _mark(request, alert_id, form, "seen")


@router.post("/alerts/{alert_id}/dismiss")
def mark_dismissed(request: Request, alert_id: str, form: Form):
    return _mark(request, alert_id, form, "dismiss")


@router.post("/alerts/{alert_id}/reopen")
def mark_reopened(request: Request, alert_id: str, form: Form):
    return _mark(request, alert_id, form, "reopen")
