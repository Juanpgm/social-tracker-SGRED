"""Boundaries of the fetcher's size, status and decompression guards."""

import gzip
import zlib
from collections.abc import Iterator

import pytest

from social_tracker.collector import http_fetcher
from social_tracker.collector.domain import FetchErrorKind
from social_tracker.collector.http_fetcher import FetchLimits
from tests.collector.test_http_fetcher import build, failure, get, reply

LIMITS = FetchLimits(max_bytes=1000)


def test_a_declared_length_equal_to_the_limit_is_accepted_and_one_more_is_not():
    fetcher, _, _ = build(
        lambda r: reply(content=b"x" * 1000, headers={"content-length": "1000"}), limits=LIMITS
    )
    assert len(get(fetcher).body) == 1000
    fetcher, _, _ = build(
        lambda r: reply(content=b"x", headers={"content-length": "1001"}), limits=LIMITS
    )
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_LARGE


@pytest.mark.parametrize("status", [200, 203, 204, 299])
def test_every_2xx_status_is_a_success(status):
    fetcher, _, _ = build(lambda r: reply(status, content=b"<rss/>"))
    assert get(fetcher).body == b"<rss/>"


@pytest.mark.parametrize("status", [199, 300, 305, 400])
def test_statuses_outside_2xx_that_are_not_redirects_are_http_errors(status):
    fetcher, _, _ = build(lambda r: reply(status))
    error = failure(lambda: get(fetcher))
    assert (error.kind, error.status) == (FetchErrorKind.HTTP, status)


class DecompressSpy:
    """Wraps a real decompressor and records every call's output cap and output size."""

    def __init__(self, real) -> None:
        self._real = real
        self.calls: list[tuple[int, int]] = []

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        produced = self._real.decompress(data, max_length)
        self.calls.append((max_length, len(produced)))
        return produced

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def test_decompression_is_capped_on_every_call_so_a_bomb_never_inflates_far_past_the_limit(
    monkeypatch,
):
    spies: list[DecompressSpy] = []
    real_factory = zlib.decompressobj

    def spying(wbits: int) -> DecompressSpy:
        spy = DecompressSpy(real_factory(wbits))
        spies.append(spy)
        return spy

    monkeypatch.setattr(http_fetcher.zlib, "decompressobj", spying)
    limit = 20_000
    packed = gzip.compress(b"lorem ipsum dolor " * 20_000)

    def trickle() -> Iterator[bytes]:
        for start in range(0, len(packed), 40):
            yield packed[start : start + 40]

    fetcher, _, _ = build(
        lambda r: reply(content=trickle(), headers={"content-encoding": "gzip"}),
        limits=FetchLimits(max_bytes=limit),
    )
    assert failure(lambda: get(fetcher)).kind is FetchErrorKind.TOO_LARGE
    (spy,) = spies
    assert len(spy.calls) >= 2  # the bomb needed several calls, so the running cap matters
    assert all(cap > 0 for cap, _ in spy.calls)  # zero would mean "unlimited" to zlib
    assert sum(size for _, size in spy.calls) <= limit + 1
