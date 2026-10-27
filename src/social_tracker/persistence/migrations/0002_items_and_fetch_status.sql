-- Migration 0002: collected items, their full-text index and per-source fetch status.
--
-- Same rules as 0001: Python validates first, these CHECKs are the last line of defence, and
-- timestamps are ISO-8601 text in UTC ("+00:00").
--
-- SR-9: `items` and `source_fetch_status` deliberately have NO foreign key to `sources`. An item
-- carries `source_id` plus a `source_name` snapshot, so deleting a source keeps what was
-- collected from it until retention removes it.
--
-- `seq` is the table's rowid and the key of the full-text index: an explicit INTEGER PRIMARY KEY
-- keeps rowids stable under VACUUM, which a table keyed only by a text `id` would not.
--
-- Text limits: `text` is at most 100 KiB of UTF-8 (CI-8), `raw_payload` (the feed entry as received,
-- JSON) at most 256 KiB. Both are measured in bytes, not characters.

CREATE TABLE items (
    seq          INTEGER PRIMARY KEY,
    id           TEXT NOT NULL UNIQUE
        CHECK (length(id) BETWEEN 3 AND 64 AND id NOT GLOB '*[^a-z0-9_]*'),
    source_id    TEXT NOT NULL
        CHECK (length(source_id) BETWEEN 3 AND 64 AND source_id NOT GLOB '*[^a-z0-9_]*'),
    source_name  TEXT NOT NULL CHECK (length(source_name) BETWEEN 1 AND 100),
    kind         TEXT NOT NULL CHECK (kind IN ('post', 'comment', 'article', 'segment')),
    external_id  TEXT NOT NULL CHECK (length(external_id) BETWEEN 1 AND 512),
    parent_id    TEXT
        CHECK (parent_id IS NULL
               OR (length(parent_id) BETWEEN 3 AND 64 AND parent_id NOT GLOB '*[^a-z0-9_]*')),
    url          TEXT
        CHECK (url IS NULL
               OR (length(url) <= 2048 AND (url GLOB 'http://*' OR url GLOB 'https://*'))),
    title        TEXT NOT NULL CHECK (length(title) <= 1000),
    author       TEXT CHECK (author IS NULL OR length(author) BETWEEN 1 AND 200),
    text         TEXT NOT NULL CHECK (length(CAST(text AS BLOB)) <= 102400),
    language     TEXT CHECK (language IS NULL OR length(language) BETWEEN 2 AND 35),
    published_at TEXT NOT NULL CHECK (published_at GLOB '????-??-??T??:??:??*+00:00'),
    fetched_at   TEXT NOT NULL CHECK (fetched_at GLOB '????-??-??T??:??:??*+00:00'),
    content_hash TEXT NOT NULL
        CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    -- JSON array of short lowercase names: truncated, published_at_missing, ...
    flags        TEXT NOT NULL
        CHECK (CASE WHEN json_valid(flags) THEN json_type(flags) = 'array' ELSE 0 END),
    -- Pipeline stage. The domain owns the list of stages; the CHECK only keeps the shape, so a
    -- later stage does not need a table rebuild.
    status       TEXT NOT NULL
        CHECK (length(status) BETWEEN 1 AND 32 AND status NOT GLOB '*[^a-z_]*'),
    raw_payload  TEXT NOT NULL
        CHECK ((CASE WHEN json_valid(raw_payload) THEN json_type(raw_payload) = 'object' ELSE 0 END)
               AND length(CAST(raw_payload AS BLOB)) <= 262144)
);

-- Dedup by external id (CI-3). The content-hash dedup is a lookup, not a constraint, because
-- the same text may legitimately be stored again after a purge; the index serves it, and
-- the cross-source "seen before" lookups of the later classification stage.
CREATE UNIQUE INDEX ux_items_source_external ON items (source_id, external_id);
CREATE INDEX ix_items_content_hash ON items (content_hash, source_id);
CREATE INDEX ix_items_published_at ON items (published_at);
CREATE INDEX ix_items_fetched_at ON items (fetched_at);
CREATE INDEX ix_items_parent_id ON items (parent_id) WHERE parent_id IS NOT NULL;

-- Full-text index over title and text. It is an external-content table (the text is stored
-- once, in `items`) kept in step by triggers rather than by application code: every insert,
-- delete and text edit reaches the index in the same statement and transaction, so no code
-- path (the retention purge, a manual repair, a future writer) can leave it stale.
-- `unicode61 remove_diacritics 2` makes "sismo" and "sísmo" the same word.
CREATE VIRTUAL TABLE items_fts USING fts5(
    title, text, content = 'items', content_rowid = 'seq',
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER items_fts_after_insert AFTER INSERT ON items
BEGIN
    INSERT INTO items_fts (rowid, title, text) VALUES (new.seq, new.title, new.text);
END;

CREATE TRIGGER items_fts_after_delete AFTER DELETE ON items
BEGIN
    INSERT INTO items_fts (items_fts, rowid, title, text)
    VALUES ('delete', old.seq, old.title, old.text);
END;

CREATE TRIGGER items_fts_after_update AFTER UPDATE OF title, text ON items
BEGIN
    INSERT INTO items_fts (items_fts, rowid, title, text)
    VALUES ('delete', old.seq, old.title, old.text);
    INSERT INTO items_fts (rowid, title, text) VALUES (new.seq, new.title, new.text);
END;

-- What the collector last learned about a source: the time of the last successful fetch, the
-- last error (a short fixed-vocabulary phrase, never a URL or a secret) and the validators
-- for conditional requests. No foreign key: see the SR-9 note above.
CREATE TABLE source_fetch_status (
    source_id     TEXT NOT NULL PRIMARY KEY
        CHECK (length(source_id) BETWEEN 3 AND 64 AND source_id NOT GLOB '*[^a-z0-9_]*'),
    last_fetch_at TEXT CHECK (last_fetch_at GLOB '????-??-??T??:??:??*+00:00'),
    last_error    TEXT CHECK (last_error IS NULL OR length(last_error) BETWEEN 1 AND 300),
    etag          TEXT CHECK (etag IS NULL OR length(etag) BETWEEN 1 AND 512),
    last_modified TEXT CHECK (last_modified IS NULL OR length(last_modified) BETWEEN 1 AND 100)
);
