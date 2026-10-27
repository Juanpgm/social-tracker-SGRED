"""The ASGI application: middleware, error pages and the template environment.

Every response, errors included, goes through `guard`, which checks the Host header, issues the
CSRF cookie and adds the security headers. Failures never reach the client as framework JSON or a
traceback: a refused operation is a short page, an unexpected one is logged by exception type only.
"""

import logging
from collections.abc import Awaitable, Callable, Iterable

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from social_tracker.entrypoints.web import alert_routes, routes
from social_tracker.entrypoints.web.pages import (
    BUSY_TEXT,
    ContainerFactory,
    WebContext,
    build_environment,
    error_page,
    is_htmx,
)
from social_tracker.entrypoints.web.security import (
    CSRF_COOKIE_NAME,
    SECURITY_HEADERS,
    CsrfTokens,
    allowed_hosts,
    host_allowed,
)
from social_tracker.shared.errors import ConflictError, DomainError, NotFoundError, StorageError

logger = logging.getLogger("social_tracker.web")

_STATUS_TEXT = {
    400: "The request was not valid.",
    403: "The request was refused. Reload the page and try again.",
    404: "That page does not exist.",
    405: "That action is not available here.",
    408: "The request took too long to arrive.",
    413: "The request was too large.",
    415: "The request used an unsupported content type.",
}


def _domain_error(request: Request, exc: Exception) -> Response:
    if isinstance(exc, NotFoundError):
        return error_page(request, 404, "That source does not exist.")
    if isinstance(exc, StorageError):
        return error_page(request, 503, BUSY_TEXT)
    status = 409 if isinstance(exc, ConflictError) else 400
    assert isinstance(exc, DomainError)
    return error_page(request, status, exc.message or _STATUS_TEXT[400])


def _http_error(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, HTTPException)
    text = _STATUS_TEXT.get(exc.status_code, "The request could not be completed.")
    return error_page(request, exc.status_code, text)


def create_app(
    container_factory: ContainerFactory,
    *,
    host: str,
    secret: bytes | None = None,
    extra_hosts: Iterable[str] = (),
) -> FastAPI:
    """Build the UI. `host` is the address the server binds to; it is served besides loopback,
    and so is every name in `extra_hosts` (the names clients use to reach a wildcard bind)."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ctx = WebContext(
        container_factory, CsrfTokens(secret), allowed_hosts(host, extra_hosts), build_environment()
    )
    app.mount("/static", StaticFiles(packages=[("social_tracker.entrypoints.web", "static")]))
    app.include_router(routes.router)
    app.include_router(alert_routes.router)
    app.add_exception_handler(DomainError, _domain_error)
    app.add_exception_handler(HTTPException, _http_error)

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        ctx: WebContext = request.app.state.ctx
        cookie = request.cookies.get(CSRF_COOKIE_NAME)
        issue = not ctx.tokens.well_formed(cookie)
        cookie = ctx.tokens.new_cookie() if issue or cookie is None else cookie
        request.state.csrf_token = ctx.tokens.token_for(cookie)
        if not host_allowed(request.headers.get("host"), ctx.hosts):
            response: Response = PlainTextResponse("Invalid host header", status_code=400)
            if is_htmx(request):
                response.headers["HX-Reswap"] = "none"  # plain text must not replace a row
        else:
            try:
                response = await call_next(request)
            except Exception as exc:
                # The type only: the message of a failure can carry user input or a path.
                logger.error("unexpected failure (%s)", type(exc).__name__)
                response = error_page(request, 500, "Something went wrong.")
            if issue:
                response.set_cookie(CSRF_COOKIE_NAME, cookie, httponly=True, samesite="strict")
        response.headers.update(SECURITY_HEADERS)
        return response

    return app
