"""Reusable contract for any `ItemRepository` implementation (in-memory fake and SQLite).

Rules every implementation must honour (and that this suite pins):
- Items are stored once per `(source_id, external_id)` AND once per `(source_id, content_hash)`;
  the report counts new and duplicate items, and a batch is all or nothing.
- Listing is newest first (`published_at`, then id, descending); search matches whole words,
  ignoring case and accents, and treats user text as plain words, never as operators.
- Retention purges strictly older than the cutoff, by `fetched_at`.
"""

from datetime import timedelta

import pytest

from social_tracker.items.domain import (
    AddReport,
    ItemKind,
    ItemStatus,
    SourceUsage,
    item_size_bytes,
)
from social_tracker.items.ports import ItemRepository
from social_tracker.shared.errors import ValidationError
from tests.items.factories import NOW, make_item

PORT_SURFACE = frozenset(
    name
    for name, member in vars(ItemRepository).items()
    if not name.startswith("_") and callable(member)
)

HOSTILE_QUERIES = [
    '"',
    '""',
    '"unbalanced',
    'unbalanced"',
    "*",
    "***",
    "term*",
    "-",
    "-minus",
    "a -b",
    "AND",
    "OR",
    "NOT",
    "a AND",
    "OR b",
    "NOT NOT",
    "NEAR",
    "NEAR(a b)",
    "NEAR/2",
    "a NEAR/2 b",
    "col:term",
    "title:quake",
    "{title}: quake",
    "^quake",
    "(",
    ")",
    "((quake)",
    "quake))",
    "quake OR (",
    "'; DROP TABLE items; --",
    "\x00",
    "quake\x00",
    "\U0001f30b",
    "%",
    "_",
    "\\",
    "x" * 10_000,
]


class ItemRepositoryContract:
    ALLOWED_EXTRA_PUBLIC: frozenset[str] = frozenset()

    def make_repo(self, **options: int) -> ItemRepository:
        """A fresh repository; `options` are constructor settings such as
        `max_tombstones_per_source`."""
        raise NotImplementedError

    @pytest.fixture
    def repo(self) -> ItemRepository:
        return self.make_repo()

    def test_the_public_surface_is_exactly_the_port(self, repo):
        public = {n for n in dir(repo) if not n.startswith("_") and callable(getattr(repo, n))}
        assert public - PORT_SURFACE == self.ALLOWED_EXTRA_PUBLIC
        assert public >= PORT_SURFACE

    # ---- adding ----

    def test_new_items_are_stored_and_counted(self, repo):
        report = repo.add_new([make_item("a"), make_item("b", title="Other", text="Other text")])
        assert report == AddReport(new=2, duplicate=0)
        assert repo.count() == 2

    def test_every_field_survives_a_round_trip(self, repo):
        item = make_item(
            "guid-ñ",
            title="Sismo \U0001f30b en Cali áéí",
            text="Texto con acentos y emoji \U0001f468‍\U0001f469‍\U0001f467\nSegunda línea",
            author="Ana Pérez",
            language="es-CO",
            url="https://example.org/a?x=1&y=2",
            kind=ItemKind.COMMENT,
            parent_id="itm_parent",
            flags=("truncated", "published_at_missing"),
            status=ItemStatus.CLASSIFIED,
            raw_payload={"description": "<p>hola</p>", "nested": {"n": [1, 2, None]}},
            published_at=NOW + timedelta(microseconds=123456),
        )
        repo.add_new([item])
        assert repo.get(item.id) == item

    def test_an_unknown_item_is_none(self, repo):
        repo.add_new([make_item("a")])
        assert repo.get("itm_unknown") is None

    def test_re_ingesting_the_same_items_reports_zero_new(self, repo):
        batch = [make_item("a"), make_item("b", title="Other", text="Other text")]
        repo.add_new(batch)
        assert repo.add_new(batch) == AddReport(new=0, duplicate=2)
        assert repo.count() == 2

    def test_the_same_content_under_another_id_and_url_is_stored_once(self, repo):
        repo.add_new([make_item("first")])
        again = make_item("second", url="https://other.example/second")
        assert again.content_hash == make_item("first").content_hash
        assert repo.add_new([again]) == AddReport(new=0, duplicate=1)
        assert repo.get(again.id) is None
        assert repo.count() == 1

    def test_the_same_external_id_with_other_content_is_a_duplicate(self, repo):
        repo.add_new([make_item("a")])
        edited = make_item("a", text="The feed edited this item afterwards")
        assert repo.add_new([edited]) == AddReport(new=0, duplicate=1)
        assert "edited" not in repo.get(make_item("a").id).text

    def test_the_same_content_in_two_sources_is_stored_in_both(self, repo):
        report = repo.add_new(
            [make_item("a", source_id="src_one"), make_item("a", source_id="src_two")]
        )
        assert report == AddReport(new=2, duplicate=0)

    def test_duplicates_inside_one_batch_are_stored_once(self, repo):
        batch = [
            make_item("a"),
            make_item("a", text="same id, other text"),  # same external id
            make_item("b"),  # same content as "a"
            make_item("c", title="Different", text="Different text"),
        ]
        assert repo.add_new(batch) == AddReport(new=2, duplicate=2)
        assert repo.count() == 2

    def test_an_empty_batch_is_fine(self, repo):
        assert repo.add_new([]) == AddReport(new=0, duplicate=0)
        assert repo.count() == 0

    def test_a_batch_with_a_non_item_stores_nothing(self, repo):
        with pytest.raises(ValidationError):
            repo.add_new([make_item("a"), {"id": "x"}])  # type: ignore[list-item]
        assert repo.count() == 0

    # ---- listing ----

    def test_listing_is_newest_first_and_ties_break_by_id(self, repo):
        old = make_item("old", title="Old", text="old text", published_at=NOW - timedelta(days=2))
        new = make_item("new", title="New", text="new text", published_at=NOW)
        mid_a = make_item(
            "mid_a", title="Mid a", text="mid a", published_at=NOW - timedelta(days=1)
        )
        mid_b = make_item(
            "mid_b", title="Mid b", text="mid b", published_at=NOW - timedelta(days=1)
        )
        repo.add_new([old, mid_a, new, mid_b])
        tied = sorted([mid_a, mid_b], key=lambda i: i.id, reverse=True)
        assert repo.list() == [new, *tied, old]

    def test_listing_filters_by_source_and_applies_the_limit(self, repo):
        repo.add_new(
            [
                make_item("a", source_id="src_one", text="one a", title="one a"),
                make_item(
                    "b",
                    source_id="src_one",
                    text="one b",
                    title="one b",
                    published_at=NOW + timedelta(hours=1),
                ),
                make_item("c", source_id="src_two", text="two c", title="two c"),
            ]
        )
        assert [i.external_id for i in repo.list(source_id="src_one")] == ["b", "a"]
        assert [i.external_id for i in repo.list(source_id="src_one", limit=1)] == ["b"]
        assert repo.list(source_id="src_none") == []
        assert len(repo.list()) == 3

    @pytest.mark.parametrize("limit", [0, -1, 1001, True, "5", 2.5, None])
    def test_an_invalid_limit_is_rejected(self, repo, limit):
        repo.add_new([make_item("a")])
        with pytest.raises(ValidationError):
            repo.list(limit=limit)
        with pytest.raises(ValidationError):
            repo.search("quake", limit=limit)

    def test_the_limit_boundaries_are_accepted(self, repo):
        repo.add_new([make_item("a")])
        assert len(repo.list(limit=1)) == 1
        assert len(repo.list(limit=1000)) == 1

    def test_count_by_source(self, repo):
        repo.add_new(
            [
                make_item("a", source_id="src_one"),
                make_item("b", source_id="src_two", title="Other", text="other"),
            ]
        )
        assert (repo.count(), repo.count("src_one"), repo.count("src_two")) == (2, 1, 1)
        assert repo.count("src_none") == 0

    # ---- search ----

    def test_search_finds_whole_words_ignoring_case_and_accents(self, repo):
        quake = make_item("q", title="Sismo fuerte", text="Un temblor sacudió Cali esta mañana")
        other = make_item("o", title="Economía", text="El dólar sube")
        repo.add_new([quake, other])
        for query in ("sismo", "SISMO", "Sísmo", "temblor", "manana", "mañana", "cali sismo"):
            assert repo.search(query) == [quake], query
        assert repo.search("economia") == [other]
        assert repo.search("sism") == []  # whole words: no prefix matching
        assert repo.search("volcan") == []

    def test_every_word_must_match(self, repo):
        both = make_item("both", title="Quake Cali", text="shaking")
        one = make_item("one", title="Quake Bogota", text="shaking")
        repo.add_new([both, one])
        assert repo.search("quake cali") == [both]
        assert {i.id for i in repo.search("quake")} == {both.id, one.id}

    def test_operator_words_are_plain_words(self, repo):
        dogs = make_item("dogs", title="dog", text="walks")
        cats = make_item("cats", title="cat", text="sleeps")
        mixed = make_item("mixed", title="dog and cat", text="together or not near")
        repo.add_new([dogs, cats, mixed])
        # Each operator word is an ordinary word that must be present, so only the item that
        # really contains "or", "and", "not" and "near" is found; operators would match others.
        assert repo.search("dog OR cat") == [mixed]
        assert repo.search("dog AND cat") == [mixed]
        assert repo.search("dog NOT cat") == [mixed]
        assert repo.search("NEAR") == [mixed]
        assert {i.id for i in repo.search("cat*")} == {cats.id, mixed.id}

    def test_search_covers_the_title_and_the_text(self, repo):
        in_title = make_item("t", title="volcano", text="nothing")
        in_text = make_item("x", title="nothing", text="volcano erupts")
        repo.add_new([in_title, in_text])
        assert {i.id for i in repo.search("volcano")} == {in_title.id, in_text.id}

    @pytest.mark.parametrize("query", ["", "   ", "\n", '""', "*", "\U0001f30b"])
    def test_a_search_without_words_finds_nothing(self, repo, query):
        repo.add_new([make_item("a")])
        assert repo.search(query) == []

    @pytest.mark.parametrize("query", HOSTILE_QUERIES, ids=lambda q: repr(q)[:20])
    def test_hostile_search_text_never_raises(self, repo, query):
        item = make_item("a", title="AND OR NOT NEAR quake", text="plain text")
        repo.add_new([item])
        result = repo.search(query)
        assert isinstance(result, list)
        assert all(found == item for found in result)

    def test_search_applies_the_limit_newest_first(self, repo):
        items = [
            make_item(f"g{i}", title=f"quake {i}", text="x", published_at=NOW + timedelta(hours=i))
            for i in range(5)
        ]
        repo.add_new(items)
        assert [i.external_id for i in repo.search("quake", limit=2)] == ["g4", "g3"]

    def test_search_results_with_the_same_date_come_in_descending_id_order(self, repo):
        twins = [make_item(f"twin{i}", title="quake", text=f"same {i}") for i in range(4)]
        repo.add_new(twins)
        found = repo.search("quake")
        assert [i.id for i in found] == sorted((i.id for i in twins), reverse=True)

    def test_search_rejects_non_text(self, repo):
        with pytest.raises(ValidationError):
            repo.search(None)  # type: ignore[arg-type]

    def test_a_purged_item_is_no_longer_found(self, repo):
        repo.add_new([make_item("a", fetched_at=NOW)])
        assert len(repo.search("quake")) == 1
        repo.purge_fetched_before(NOW + timedelta(days=1))
        assert repo.search("quake") == []

    # ---- status ----

    def test_set_status_changes_the_stage_and_reports_unknown_ids(self, repo):
        item = make_item("a")
        repo.add_new([item])
        assert repo.set_status(item.id, ItemStatus.CLASSIFIED) is True
        assert repo.get(item.id).status is ItemStatus.CLASSIFIED
        assert repo.set_status("itm_unknown", ItemStatus.CLASSIFIED) is False

    def test_set_status_rejects_an_unknown_stage(self, repo):
        item = make_item("a")
        repo.add_new([item])
        with pytest.raises(ValidationError):
            repo.set_status(item.id, "bogus")  # type: ignore[arg-type]
        assert repo.get(item.id).status is ItemStatus.NEW

    def test_a_status_change_keeps_the_item_searchable(self, repo):
        item = make_item("a")
        repo.add_new([item])
        repo.set_status(item.id, ItemStatus.CLASSIFIED)
        assert [i.id for i in repo.search("quake")] == [item.id]

    # ---- retention ----

    def test_purge_fetched_before_removes_only_strictly_older_items(self, repo):
        cutoff = NOW
        kept_equal = make_item("equal", title="equal", text="e", fetched_at=cutoff)
        kept_newer = make_item(
            "newer", title="newer", text="n", fetched_at=cutoff + timedelta(seconds=1)
        )
        gone_older = make_item(
            "older", title="older", text="o", fetched_at=cutoff - timedelta(microseconds=1)
        )
        gone_much = make_item(
            "much", title="much", text="m", fetched_at=cutoff - timedelta(days=90)
        )
        repo.add_new([kept_equal, kept_newer, gone_older, gone_much])
        assert repo.purge_fetched_before(cutoff) == 2
        assert {i.external_id for i in repo.list()} == {"equal", "newer"}

    def test_purge_on_an_empty_store_deletes_nothing(self, repo):
        assert repo.purge_fetched_before(NOW) == 0
        assert repo.purge_classified() == 0

    def test_purge_ignores_the_publication_date(self, repo):
        ancient = make_item("a", published_at=NOW - timedelta(days=900), fetched_at=NOW)
        repo.add_new([ancient])
        assert repo.purge_fetched_before(NOW - timedelta(days=30)) == 0
        assert repo.count() == 1

    def test_purge_classified_removes_only_classified_items(self, repo):
        done = make_item("done", title="done", text="d")
        pending = make_item("pending", title="pending", text="p")
        repo.add_new([done, pending])
        repo.set_status(done.id, ItemStatus.CLASSIFIED)
        assert repo.purge_classified() == 1
        assert [i.external_id for i in repo.list()] == ["pending"]

    def test_a_purged_dated_item_is_not_remembered_by_the_repository(self, repo):
        # A dated entry is kept out by the collector's retention horizon instead (it is older
        # than the horizon by construction), so the store keeps no tombstone for it.
        item = make_item("a", fetched_at=NOW - timedelta(days=60))
        repo.add_new([item])
        repo.purge_fetched_before(NOW)
        assert repo.add_new([item]) == AddReport(new=1, duplicate=0)

    # ---- tombstones: what a purge must not let come back ----

    def test_an_undated_item_purged_by_age_is_not_ingested_again(self, repo):
        item = make_item("a", flags=("published_at_missing",), fetched_at=NOW - timedelta(days=60))
        repo.add_new([item])
        assert repo.purge_fetched_before(NOW) == 1
        assert repo.count() == 0
        assert repo.add_new([item]) == AddReport(new=0, duplicate=1)
        assert repo.count() == 0

    def test_a_classified_item_purged_for_retention_zero_is_not_ingested_again(self, repo):
        dated, undated = (
            make_item("a", title="a", text="a"),
            make_item("b", title="b", text="b", flags=("published_at_missing",)),
        )
        repo.add_new([dated, undated])
        repo.set_status(dated.id, ItemStatus.CLASSIFIED)
        repo.set_status(undated.id, ItemStatus.CLASSIFIED)
        assert repo.purge_classified() == 2
        assert repo.add_new([dated, undated]) == AddReport(new=0, duplicate=2)
        assert repo.count() == 0

    def test_a_tombstone_blocks_the_same_external_id_even_with_other_content(self, repo):
        repo.add_new([make_item("a", flags=("published_at_missing",), fetched_at=NOW)])
        repo.purge_fetched_before(NOW + timedelta(days=1))
        edited = make_item("a", text="the publisher edited it", flags=("published_at_missing",))
        assert repo.add_new([edited]) == AddReport(new=0, duplicate=1)

    def test_a_tombstone_only_covers_its_own_source(self, repo):
        repo.add_new([make_item("a", source_id="src_one", flags=("published_at_missing",))])
        repo.purge_fetched_before(NOW + timedelta(days=1))
        other = make_item("a", source_id="src_two", flags=("published_at_missing",))
        assert repo.add_new([other]) == AddReport(new=1, duplicate=0)

    def test_new_items_next_to_a_tombstoned_one_are_still_stored(self, repo):
        old = make_item("old", title="old", text="old", flags=("published_at_missing",))
        repo.add_new([old])
        repo.purge_fetched_before(NOW + timedelta(days=1))
        fresh = make_item("fresh", title="fresh", text="fresh")
        assert repo.add_new([old, fresh]) == AddReport(new=1, duplicate=1)
        assert [i.external_id for i in repo.list()] == ["fresh"]

    def test_purging_twice_keeps_one_tombstone_and_nothing_breaks(self, repo):
        item = make_item("a", flags=("published_at_missing",))
        repo.add_new([item])
        repo.purge_fetched_before(NOW + timedelta(days=1))
        repo.purge_fetched_before(NOW + timedelta(days=1))
        repo.purge_classified()
        assert repo.add_new([item]) == AddReport(new=0, duplicate=1)

    def test_the_oldest_tombstones_of_a_source_are_forgotten_beyond_the_cap(self):
        repo = self.make_repo(max_tombstones_per_source=3)
        items = [
            make_item(f"g{n}", title=f"t{n}", text=f"x{n}", flags=("published_at_missing",))
            for n in range(5)
        ]
        for item in items:  # one purge each, so their tombstones are written in this order
            repo.add_new([item])
            repo.purge_fetched_before(NOW + timedelta(days=1))
        report = repo.add_new(items)
        assert report == AddReport(new=2, duplicate=3)  # g0 and g1 were forgotten
        assert {i.external_id for i in repo.list()} == {"g0", "g1"}

    def test_the_tombstone_cap_is_per_source(self):
        repo = self.make_repo(max_tombstones_per_source=1)
        one = make_item("a", source_id="src_one", flags=("published_at_missing",))
        two = make_item("a", source_id="src_two", flags=("published_at_missing",))
        repo.add_new([one, two])
        repo.purge_fetched_before(NOW + timedelta(days=1))
        assert repo.add_new([one, two]) == AddReport(new=0, duplicate=2)

    # ---- per-source quota ----

    def test_usage_counts_the_items_and_bytes_of_one_source(self, repo):
        a = make_item("a", title="Title", text="Body")
        b = make_item("b", source_id="src_two", title="Other", text="Other body")
        repo.add_new([a, b])
        usage = repo.source_usage("src_news")
        assert usage.items == 1
        assert usage.bytes == item_size_bytes(a)
        assert repo.source_usage("src_two").items == 1
        assert repo.source_usage("src_none") == SourceUsage(0, 0)

    def test_usage_counts_bytes_not_characters(self, repo):
        item = make_item("a", title="ñandú", text="日本語のテキスト")
        repo.add_new([item])
        assert repo.source_usage("src_news").bytes == item_size_bytes(item)
        assert item_size_bytes(item) > len(item.title) + len(item.text)

    @pytest.mark.parametrize(
        ("room", "stored", "refused"), [(4, 3, 0), (3, 3, 0), (2, 2, 1), (1, 1, 2)]
    )
    def test_the_item_quota_stops_new_items_at_the_cap(self, repo, room, stored, refused):
        batch = [make_item(f"g{n}", title=f"t{n}", text=f"x{n}") for n in range(3)]
        report = repo.add_new(batch, max_items=room)
        assert report == AddReport(new=stored, duplicate=0, over_quota=refused)
        assert [i.external_id for i in sorted(repo.list(), key=lambda i: i.external_id)] == [
            f"g{n}" for n in range(stored)
        ]

    def test_existing_items_of_the_source_use_up_the_item_quota(self, repo):
        repo.add_new([make_item("old", title="old", text="old")])
        batch = [make_item(f"g{n}", title=f"t{n}", text=f"x{n}") for n in range(3)]
        assert repo.add_new(batch, max_items=3) == AddReport(new=2, duplicate=0, over_quota=1)
        assert repo.source_usage("src_news").items == 3

    def test_other_sources_do_not_use_up_a_quota(self, repo):
        repo.add_new([make_item("x", source_id="src_other", title="x", text="x")])
        assert repo.add_new([make_item("a")], max_items=1) == AddReport(new=1, duplicate=0)

    def test_duplicates_are_not_charged_to_the_quota_even_when_it_is_full(self, repo):
        a, b = make_item("a", title="a", text="a"), make_item("b", title="b", text="b")
        repo.add_new([a])
        report = repo.add_new([a, b], max_items=1)  # a is a duplicate, b does not fit
        assert report == AddReport(new=0, duplicate=1, over_quota=1)

    def test_a_tombstoned_item_is_a_duplicate_not_an_over_quota_one(self, repo):
        old = make_item("old", title="old", text="old", flags=("published_at_missing",))
        repo.add_new([old])
        repo.purge_fetched_before(NOW + timedelta(days=1))
        assert repo.add_new([old], max_items=1) == AddReport(new=0, duplicate=1, over_quota=0)

    @pytest.mark.parametrize(
        ("slack", "stored"), [(1, 2), (0, 2), (-1, 1)], ids=["cap+1", "cap", "cap-1"]
    )
    def test_the_byte_quota_boundaries(self, repo, slack, stored):
        batch = [make_item(f"g{n}", title=f"t{n}", text=f"x{n}") for n in range(2)]
        cap = sum(item_size_bytes(i) for i in batch) + slack
        report = repo.add_new(batch, max_bytes=cap)
        assert (report.new, report.over_quota) == (stored, 2 - stored)

    def test_the_byte_quota_counts_what_is_already_stored(self, repo):
        first = make_item("a", title="a", text="a")
        second = make_item("b", title="b", text="b")
        repo.add_new([first])
        cap = item_size_bytes(first) + item_size_bytes(second)
        assert repo.add_new([second], max_bytes=cap - 1) == AddReport(0, 0, over_quota=1)
        assert repo.add_new([second], max_bytes=cap) == AddReport(1, 0)

    def test_a_smaller_later_item_may_still_fit_when_a_bigger_one_did_not(self, repo):
        big = make_item("big", title="big", text="x" * 500)
        small = make_item("small", title="small", text="s")
        report = repo.add_new([big, small], max_bytes=item_size_bytes(small) + 10)
        assert (report.new, report.over_quota) == (1, 1)
        assert [i.external_id for i in repo.list()] == ["small"]

    def test_both_quotas_apply_together(self, repo):
        batch = [make_item(f"g{n}", title=f"t{n}", text=f"x{n}") for n in range(4)]
        report = repo.add_new(batch, max_items=3, max_bytes=item_size_bytes(batch[0]) * 2 + 1)
        assert report.new == 2
        assert report.over_quota == 2

    def test_without_quotas_nothing_is_refused(self, repo):
        batch = [make_item(f"g{n}", title=f"t{n}", text=f"x{n}") for n in range(30)]
        assert repo.add_new(batch, max_items=None, max_bytes=None).new == 30

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
    def test_an_invalid_quota_is_rejected_and_nothing_is_stored(self, repo, bad):
        with pytest.raises(ValidationError):
            repo.add_new([make_item("a")], max_items=bad)
        with pytest.raises(ValidationError):
            repo.add_new([make_item("a")], max_bytes=bad)
        assert repo.count() == 0

    def test_purge_rejects_a_naive_cutoff(self, repo):
        with pytest.raises(ValidationError):
            repo.purge_fetched_before(NOW.replace(tzinfo=None))
