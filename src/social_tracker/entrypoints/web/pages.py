"""Shared page plumbing: the per-app context, the template environment and rendering."""

from collections.abc import Callable
from contextlib import AbstractContextManager, closing
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, PackageLoader, StrictUndefined

from social_tracker.bootstrap import Container
from social_tracker.entrypoints.web.security import CsrfTokens

ContainerFactory = Callable[[], Container]
BUSY_TEXT = "The database is busy or unavailable. Wait a moment and try again."


@dataclass
class WebContext:
    container_factory: ContainerFactory
    tokens: CsrfTokens
    hosts: frozenset[str]
    env: Environment

    def services(self) -> AbstractContextManager[Container]:
        return closing(self.container_factory())


def build_environment() -> Environment:
    return Environment(
        loader=PackageLoader("social_tracker.entrypoints.web", "templates"),
        autoescape=True,
        undefined=StrictUndefined,
    )


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


_REFUSED_TEXT = "The request was refused. Reload the page and try again."
_FALLBACK_TEXT = {
    403: _REFUSED_TEXT,
    409: "That conflicts with the data already stored. Reload the page and try again.",
    422: "The values were not valid. Open the form and correct them.",
}
_FALLBACK_DEFAULT = "The request could not be completed."


def render(request: Request, template: str, *, status: int = 200, **context: Any) -> HTMLResponse:
    """Render a template. An htmx caller never receives a document for a failure: the page would
    be pasted inside the row that was clicked, so it gets the status fragment instead."""
    if status >= 400 and is_htmx(request) and not template.startswith("_"):
        text = context.get("message") or _FALLBACK_TEXT.get(status, _FALLBACK_DEFAULT)
        return status_fragment(request, text, status=status, refresh=status == 403)
    ctx: WebContext = request.app.state.ctx
    page = ctx.env.get_template(template).render(
        {"notice": None, "csrf_token": request.state.csrf_token, **context}
    )
    return HTMLResponse(page, status_code=status)


def error_page(request: Request, status: int, message: str) -> HTMLResponse:
    return render(request, "error.html", status=status, message=message)


def status_fragment(
    request: Request,
    message: str,
    *,
    status: int = 200,
    row: Any = None,
    refresh: bool = False,
) -> HTMLResponse:
    """The htmx answer: the row as the server holds it (if any) and the message for the live
    region. Without a row there is nothing true to draw, so htmx is told to leave the page alone
    (`HX-Reswap: none`; the out-of-band message is still applied). `refresh` asks for a reload."""
    failed = status >= 400
    response = render(
        request, "_toggle_result.html", status=status, row=row, message=message, failed=failed
    )
    if row is None:
        response.headers["HX-Reswap"] = "none"
    if refresh:
        response.headers["HX-Refresh"] = "true"
    return response
