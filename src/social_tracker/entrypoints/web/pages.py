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


def render(request: Request, template: str, *, status: int = 200, **context: Any) -> HTMLResponse:
    ctx: WebContext = request.app.state.ctx
    page = ctx.env.get_template(template).render(
        {"notice": None, "csrf_token": request.state.csrf_token, **context}
    )
    return HTMLResponse(page, status_code=status)


def error_page(request: Request, status: int, message: str) -> HTMLResponse:
    return render(request, "error.html", status=status, message=message)
