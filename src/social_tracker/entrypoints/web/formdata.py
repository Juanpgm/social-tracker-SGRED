"""Reading a state-changing request: bounded, strictly decoded and CSRF-checked.

`read_form` is a router-level dependency, so no POST route can skip it: the body is only read
after its declared size and content type pass, at most `MAX_BODY_BYTES` of it are ever buffered,
and the CSRF token is verified before any route code sees a field.
"""

import asyncio
from urllib.parse import parse_qsl

from fastapi import Request
from starlette.exceptions import HTTPException

from social_tracker.entrypoints.web.pages import WebContext
from social_tracker.entrypoints.web.security import CSRF_COOKIE_NAME, CSRF_FIELD

MAX_BODY_BYTES = 16 * 1024
MAX_FIELDS = 100
BODY_TIMEOUT_S = 10  # the whole body, not each chunk: a trickle cannot keep a request open
FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


async def _body(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        if not declared.isdecimal():
            raise HTTPException(400)
        if int(declared) > MAX_BODY_BYTES:
            raise HTTPException(413)
    body = bytearray()
    try:
        async with asyncio.timeout(BODY_TIMEOUT_S):
            async for chunk in request.stream():
                body += chunk
                if len(body) > MAX_BODY_BYTES:
                    raise HTTPException(413)
    except TimeoutError:
        raise HTTPException(408) from None
    return bytes(body)


async def read_form(request: Request) -> dict[str, str]:
    """The decoded form of a POST (first value of each field); `{}` for a read."""
    if request.method in SAFE_METHODS:
        return {}
    content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if content_type != FORM_CONTENT_TYPE:
        raise HTTPException(415)
    raw = await _body(request)
    try:
        pairs = parse_qsl(
            raw.decode("utf-8"),
            keep_blank_values=True,
            strict_parsing=True,
            errors="strict",
            max_num_fields=MAX_FIELDS,
        )
    except ValueError:  # UnicodeDecodeError is one: bad bytes, a bare field, too many fields
        raise HTTPException(400) from None
    form: dict[str, str] = {}
    for key, value in pairs:
        form.setdefault(key, value)
    ctx: WebContext = request.app.state.ctx
    if not ctx.tokens.valid(request.cookies.get(CSRF_COOKIE_NAME), form.get(CSRF_FIELD)):
        raise HTTPException(403)
    return form
