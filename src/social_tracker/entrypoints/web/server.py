"""Start the web UI. The only place that binds a socket."""

from contextlib import closing

import uvicorn

from social_tracker.bootstrap import build_container
from social_tracker.entrypoints.web.app import create_app
from social_tracker.entrypoints.web.security import check_bind_host


def run_server(host: str, port: int, *, allow_non_loopback: bool) -> None:
    """Serve until interrupted. A host beyond this machine needs `allow_non_loopback`.

    The database is opened once first, so a corrupt or unreachable store stops the start with the
    usual refusal instead of showing an error page on every request.
    """
    check_bind_host(host, allow_non_loopback=allow_non_loopback)
    with closing(build_container()):
        pass
    uvicorn.run(create_app(build_container, host=host), host=host, port=port, server_header=False)
