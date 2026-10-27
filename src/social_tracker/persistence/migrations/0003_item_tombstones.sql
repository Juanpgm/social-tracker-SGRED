-- Migration 0003: tombstones of purged items.
--
-- Retention deletes items, but a feed that keeps listing an entry would bring it back as "new"
-- at the next fetch, only to be purged again later (an endless cycle that also re-triggers every
-- downstream stage). An entry WITH a publication date is kept out by the collector's retention
-- horizon (anything published before it is skipped at ingest). A tombstone covers the rest: the
-- entries whose date is unknown (first-seen semantics: the fetch time stood in for it) and every
-- item removed once classified (retention of 0 days, where no time horizon exists).
--
-- A tombstone is just the dedup key `(source_id, external_id)`: no text, no payload. It is
-- written by the purge in the same transaction that deletes the item. `seq` orders them so the
-- repository can forget the OLDEST ones beyond its per-source cap and the table cannot grow
-- without bound under a feed that invents new ids forever.
--
-- SR-9: no foreign key to `sources`, like the other tables of collected data.

CREATE TABLE item_tombstones (
    seq         INTEGER PRIMARY KEY,
    source_id   TEXT NOT NULL
        CHECK (length(source_id) BETWEEN 3 AND 64 AND source_id NOT GLOB '*[^a-z0-9_]*'),
    external_id TEXT NOT NULL CHECK (length(external_id) BETWEEN 1 AND 512)
);

CREATE UNIQUE INDEX ux_item_tombstones_source_external ON item_tombstones (source_id, external_id);
