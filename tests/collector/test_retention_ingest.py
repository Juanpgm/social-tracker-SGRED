"""A purged item must not come back, and one source cannot grow the database without bound.

Retention deletes old items, but a feed that still lists them would bring them back as "new" at
the next fetch (and the purge would delete them again, forever). Dated entries older than the
retention horizon are skipped at ingest; undated ones (first-seen semantics) and everything
removed under retention 0 are remembered by a tombstone. Quotas cap what a source may keep.
"""

from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.collector.domain import IngestLimits
from social_tracker.collector.feed import FeedEntry, parse_feed
from social_tracker.collector.normalize import normalize_entries
from social_tracker.collector.service import CollectorService, CollectOutcome
from social_tracker.items.domain import ItemStatus, item_size_bytes
from social_tracker.items.service import ItemService
from tests.collector.harness import START, rss

FETCHED = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def undated_feed(*guids: str) -> bytes:
    items = "".join(
        f"<item><guid>{g}</guid><title>Title {g}</title><description>Body of {g}</description>"
        "</item>"
        for g in guids
    )
    return (
        f'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>{items}</channel></rss>'
    ).encode()


def dated_feed(*entries: tuple[str, str]) -> bytes:
    """`(guid, pubDate)` entries."""
    items = "".join(
        f"<item><guid>{g}</guid><title>Title {g}</title><description>Body of {g}</description>"
        f"<pubDate>{date}</pubDate></item>"
        for g, date in entries
    )
    return (
        f'<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>{items}</channel></rss>'
    ).encode()


OLD, RECENT = "Sun, 01 Mar 2026 07:00:00 GMT", "Wed, 01 Apr 2026 07:00:00 GMT"


def configure(world, **options) -> None:
    world.collector = CollectorService(
        world.sources_repo,
        world.control,
        world.control_repo,
        world.items,
        world.fetcher,
        world.uow,
        world.clock,
        **options,
    )


def purge(world, retention_days: int):
    return ItemService(world.items, world.uow, world.clock, retention_days).purge_expired()


# ---- ingest -> purge -> re-ingest of a feed that keeps listing its entries ----


def test_a_dated_entry_that_was_purged_is_not_ingested_again(world):
    configure(world, retention_days=30)
    world.ready()
    assert world.collector.collect_once("src_news").new == 2
    world.clock.advance(timedelta(days=31))
    assert purge(world, 30).deleted == 2
    result = world.collector.collect_once("src_news")  # the feed still lists both entries
    assert (result.outcome, result.new, result.duplicate) == (CollectOutcome.COLLECTED, 0, 0)
    assert dict(result.discarded) == {"older_than_retention": 2}
    assert world.items.count() == 0


def test_the_cycle_stays_closed_over_many_fetches_and_purges(world):
    configure(world, retention_days=30)
    world.ready(poll_interval_s=120)
    world.collector.collect_once("src_news")
    for _ in range(3):
        world.clock.advance(timedelta(days=31))
        purge(world, 30)
        assert world.collector.collect_once("src_news").new == 0
        assert purge(world, 30).deleted == 0
    assert world.items.count() == 0


def test_an_entry_that_is_still_inside_the_retention_is_kept(world):
    configure(world, retention_days=30)
    world.ready()
    world.collector.collect_once("src_news")
    world.clock.advance(timedelta(days=29))
    assert purge(world, 30).deleted == 0
    result = world.collector.collect_once("src_news")
    assert (result.new, result.duplicate) == (0, 2)
    assert dict(result.discarded) == {}
    assert world.items.count() == 2


def test_a_new_entry_next_to_purged_ones_is_still_collected(world):
    configure(world, retention_days=30)
    world.ready()
    world.collector.collect_once("src_news")
    world.clock.advance(timedelta(days=31))
    purge(world, 30)
    # "a" is still listed with its old date (skipped); "c" is new and recent
    world.fetcher.then(dated_feed(("a", OLD), ("c", RECENT)))
    result = world.collector.collect_once("src_news")
    assert (result.new, dict(result.discarded)) == (1, {"older_than_retention": 1})
    assert [i.external_id for i in world.items.list()] == ["c"]


def test_an_undated_entry_that_was_purged_is_remembered_and_not_ingested_again(world):
    configure(world, retention_days=30)
    world.fetcher.then(undated_feed("u1", "u2"))
    world.ready()
    first = world.collector.collect_once("src_news")
    assert first.new == 2
    assert all("published_at_missing" in i.flags for i in world.items.list())
    world.clock.advance(timedelta(days=31))
    assert purge(world, 30).deleted == 2
    again = world.collector.collect_once("src_news")
    assert (again.new, again.duplicate) == (0, 2)
    assert world.items.count() == 0
    world.clock.advance(timedelta(days=31))
    assert purge(world, 30).deleted == 0
    assert world.collector.collect_once("src_news").new == 0


def test_a_new_undated_entry_is_ingested_beside_remembered_ones(world):
    configure(world, retention_days=30)
    world.fetcher.then(undated_feed("u1"))
    world.ready()
    world.collector.collect_once("src_news")
    world.clock.advance(timedelta(days=31))
    purge(world, 30)
    world.fetcher.then(undated_feed("u1", "u2"))
    result = world.collector.collect_once("src_news")
    assert (result.new, result.duplicate) == (1, 1)


def test_retention_zero_removes_classified_items_once_and_they_stay_gone(world):
    configure(world, retention_days=0)
    world.ready()
    world.collector.collect_once("src_news")
    for item in world.items.list():
        world.items.set_status(item.id, ItemStatus.CLASSIFIED)
    assert purge(world, 0).deleted == 2
    world.clock.advance(timedelta(days=400))
    result = world.collector.collect_once("src_news")
    assert (result.new, result.duplicate) == (0, 2)
    assert dict(result.discarded) == {}  # retention 0 has no horizon: nothing is "too old"
    assert world.items.count() == 0


def test_retention_zero_keeps_old_dated_entries_that_were_never_classified(world):
    configure(world, retention_days=0)
    world.ready()
    world.clock.advance(timedelta(days=4000))  # the feed's entries are ancient
    result = world.collector.collect_once("src_news")
    assert (result.new, dict(result.discarded)) == (2, {})


def test_without_a_configured_retention_no_horizon_applies(world):
    world.ready()
    world.clock.advance(timedelta(days=4000))
    assert world.collector.collect_once("src_news").new == 2


def test_the_horizon_does_not_hide_the_discard_of_empty_entries(world):
    configure(world, retention_days=30)
    world.fetcher.then(rss(("a", "", ""), ("b", "Real", "Text")))
    world.ready()
    world.clock.advance(timedelta(days=400))
    result = world.collector.collect_once("src_news")
    assert (result.new, dict(result.discarded)) == (0, {"empty": 1, "older_than_retention": 1})


# ---- the horizon itself: boundaries ----


def entry(published: str | None, guid: str = "g") -> FeedEntry:
    return FeedEntry(
        title=f"T {guid}",
        link=None,
        guid=guid,
        published=published,
        body_html=f"body {guid}",
        author=None,
        raw={"guid": guid},
    )


@pytest.mark.parametrize(
    ("published", "kept"),
    [
        ("Sun, 01 Mar 2026 06:59:59 GMT", False),  # a second before the horizon
        ("Sun, 01 Mar 2026 07:00:00 GMT", True),  # exactly on it
        ("Sun, 01 Mar 2026 07:00:01 GMT", True),
    ],
)
def test_an_entry_published_exactly_at_the_horizon_is_kept(published, kept):
    horizon = datetime(2026, 3, 1, 7, 0, tzinfo=UTC)
    report = normalize_entries(
        [entry(published)],
        source_id="src_news",
        source_name="N",
        fetched_at=FETCHED,
        horizon=horizon,
    )
    assert len(report.items) == (1 if kept else 0)
    assert report.discarded == ({} if kept else {"older_than_retention": 1})


def test_an_undated_entry_is_never_older_than_the_horizon():
    horizon = FETCHED + timedelta(days=1)  # even a horizon in the future
    report = normalize_entries(
        [entry(None), entry("not a date", "g2")],
        source_id="src_news",
        source_name="N",
        fetched_at=FETCHED,
        horizon=horizon,
    )
    assert len(report.items) == 2
    assert report.discarded == {}


def test_a_future_dated_entry_is_clamped_and_kept():
    report = normalize_entries(
        [entry("Sun, 01 Mar 2037 07:00:00 GMT")],
        source_id="src_news",
        source_name="N",
        fetched_at=FETCHED,
        horizon=FETCHED - timedelta(days=30),
    )
    assert len(report.items) == 1


# ---- per-fetch byte cap ----


def sized(n: int) -> list[FeedEntry]:
    return [entry("Sun, 01 Mar 2026 08:00:00 GMT", f"g{i}") for i in range(n)]


def test_the_per_fetch_cap_boundaries():
    entries = sized(3)
    weights = [
        item_size_bytes(i)
        for i in normalize_entries(
            entries, source_id="src_news", source_name="N", fetched_at=FETCHED
        ).items
    ]
    total = sum(weights)
    for cap, stored in ((total + 1, 3), (total, 3), (total - 1, 2)):
        report = normalize_entries(
            entries,
            source_id="src_news",
            source_name="N",
            fetched_at=FETCHED,
            max_total_bytes=cap,
        )
        assert len(report.items) == stored
        assert report.discarded == ({} if stored == 3 else {"fetch_quota": 3 - stored})


def test_an_item_bigger_than_the_whole_cap_is_discarded_not_fatal():
    report = normalize_entries(
        sized(2), source_id="src_news", source_name="N", fetched_at=FETCHED, max_total_bytes=1
    )
    assert report.items == ()
    assert report.discarded == {"fetch_quota": 2}


def test_the_per_fetch_cap_is_optional():
    report = normalize_entries(
        sized(3), source_id="src_news", source_name="N", fetched_at=FETCHED, max_total_bytes=None
    )
    assert len(report.items) == 3


# ---- per-source quota through the service ----


def test_the_source_quota_stops_a_flood_and_says_so(world, caplog):
    configure(world, limits=IngestLimits(max_items_per_source=3))
    entries = [(f"g{n}", f"Title {n}", f"Body {n}") for n in range(5)]
    world.fetcher.then(rss(*entries))
    world.ready()
    with caplog.at_level("WARNING", logger="social_tracker.collector"):
        result = world.collector.collect_once("src_news")
    assert (result.outcome, result.new) == (CollectOutcome.COLLECTED, 3)
    assert dict(result.discarded) == {"source_quota": 2}
    assert world.items.count() == 3
    assert "source_quota" in caplog.text
    assert "src_news" in caplog.text
    assert world.state().state.value == "idle"  # a full source is not a failing source


@pytest.mark.parametrize(("cap", "stored"), [(4, 3), (3, 3), (2, 2)], ids=["cap+1", "cap", "cap-1"])
def test_the_source_item_quota_boundaries(world, cap, stored):
    configure(world, limits=IngestLimits(max_items_per_source=cap))
    world.fetcher.then(rss(*[(f"g{n}", f"T{n}", f"B{n}") for n in range(3)]))
    world.ready()
    result = world.collector.collect_once("src_news")
    assert result.new == stored
    assert dict(result.discarded) == ({} if stored == 3 else {"source_quota": 3 - stored})


@pytest.mark.parametrize(
    ("slack", "stored"), [(1, 2), (0, 2), (-1, 1)], ids=["cap+1", "cap", "cap-1"]
)
def test_the_source_byte_quota_boundaries(world, slack, stored):
    document = rss(("g0", "T0", "B0"), ("g1", "T1", "B1"))
    weight = sum(
        item_size_bytes(i)
        for i in normalize_entries(
            parse_feed(document).entries, source_id="src_news", source_name="N", fetched_at=START
        ).items
    )
    configure(world, limits=IngestLimits(max_bytes_per_source=weight + slack))
    world.fetcher.then(document)
    world.ready()
    result = world.collector.collect_once("src_news")
    assert result.new == stored
    assert dict(result.discarded) == ({} if stored == 2 else {"source_quota": 1})


def test_the_per_fetch_byte_cap_discards_the_rest_and_says_so(world):
    document = rss(("g0", "T0", "B0"), ("g1", "T1", "B1"), ("g2", "T2", "B2"))
    one = normalize_entries(
        parse_feed(document).entries[:1], source_id="src_news", source_name="N", fetched_at=START
    ).items[0]
    configure(world, limits=IngestLimits(max_bytes_per_fetch=item_size_bytes(one) * 2))
    world.fetcher.then(document)
    world.ready()
    result = world.collector.collect_once("src_news")
    assert (result.new, dict(result.discarded)) == (2, {"fetch_quota": 1})


def test_the_quota_counts_what_earlier_fetches_stored(world):
    configure(world, limits=IngestLimits(max_items_per_source=3))
    world.fetcher.then(rss(("a", "A", "a"), ("b", "B", "b")))
    world.ready()
    world.collector.collect_once("src_news")
    world.fetcher.then(rss(("a", "A", "a"), ("b", "B", "b"), ("c", "C", "c"), ("d", "D", "d")))
    world.clock.advance(timedelta(seconds=601))
    result = world.collector.collect_once("src_news")
    assert (result.new, result.duplicate) == (1, 2)
    assert dict(result.discarded) == {"source_quota": 1}


def test_a_purge_makes_room_under_the_quota_again(world):
    configure(world, limits=IngestLimits(max_items_per_source=2), retention_days=30)
    world.fetcher.then(rss(("a", "A", "a"), ("b", "B", "b")))
    world.ready()
    world.collector.collect_once("src_news")
    world.clock.advance(timedelta(days=31))
    purge(world, 30)
    world.fetcher.then(dated_feed(("c", RECENT), ("d", RECENT)))
    assert world.collector.collect_once("src_news").new == 2


def test_the_default_limits_are_generous_but_finite():
    limits = IngestLimits()
    assert limits.max_items_per_source >= 50_000
    assert limits.max_bytes_per_source >= 256 * 1024 * 1024
    assert limits.max_bytes_per_fetch >= 8 * 1024 * 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_items_per_source": 0},
        {"max_bytes_per_source": -1},
        {"max_bytes_per_fetch": True},
        {"max_items_per_source": 1.5},
        {"max_bytes_per_fetch": "5"},
    ],
)
def test_invalid_limits_are_refused(kwargs):
    with pytest.raises(ValueError, match="must be a positive whole number"):
        IngestLimits(**kwargs)
