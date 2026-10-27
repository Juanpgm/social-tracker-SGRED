"""Ports of the collector module."""

from typing import Protocol, runtime_checkable

from social_tracker.collector.domain import FetchRequest, FetchResponse


@runtime_checkable
class Fetcher(Protocol):
    """Fetches one document over HTTP(S) on behalf of the collector.

    The only outbound path of the application (G-1): nothing is ever sent but a GET to a
    configured source. An implementation must raise `FetchError` (never another exception) for
    every failure of the request itself, with a fixed message that carries no URL or response
    text, and must refuse addresses `collector.netpolicy` blocks, redirects included.
    A `not_modified` response has an empty body.
    """

    def fetch(self, request: FetchRequest) -> FetchResponse: ...
