-- Migration 0004: the classification cascade (RC-1 .. RC-5, RC-7, RC-11).
--
-- Same rules as before: Python validates first, these CHECKs are the last line of defence, and
-- timestamps are ISO-8601 text in UTC ("+00:00").
--
-- `classifications` holds ONE verdict per item (item_id is UNIQUE, so reprocessing or two racing
-- classifiers can never store two). Like `items`, it has no foreign key to anything: an item may
-- be purged by retention while its verdict, which later alert records refer to, stays.
-- Booleans are INTEGER 0/1. `signals` is the evidence found by the rules (names and weights,
-- never item text), capped at 4 KiB; `topics` the topics the filter found.
--
-- `item_signatures` keeps, for every classified item that was in scope, what the near-duplicate
-- check compares against: the content hash, the instant the content counts from (first seen or
-- published, whichever is earlier) and the 80-value MinHash signature (320 bytes, big-endian
-- unsigned 32-bit). No text, so the row outlives the item. `item_signature_bands` is the LSH
-- index over it: 16 bands per signature, looked up by (band, key), so finding the candidates of
-- an item never scans the table. Deleting a signature deletes its bands.
--
-- `topic_config` is the history of the editable topic vocabulary: every change appends a version
-- (the latest one is in force; none stored means the defaults shipped with the application).

CREATE TABLE classifications (
    seq                     INTEGER PRIMARY KEY,
    item_id                 TEXT NOT NULL UNIQUE
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    label                   TEXT NOT NULL
        CHECK (label IN ('unintentional_rumor', 'deliberate_disinformation', 'recycled_content',
                         'no_risk', 'unclassified', 'out_of_scope')),
    confidence              REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    stage                   TEXT NOT NULL CHECK (stage IN ('topic', 'rules', 'dedup', 'catalog', 'llm')),
    rationale               TEXT NOT NULL CHECK (length(rationale) BETWEEN 1 AND 300),
    signals                 TEXT NOT NULL
        CHECK ((CASE WHEN json_valid(signals) THEN json_type(signals) = 'array' ELSE 0 END)
               AND length(CAST(signals AS BLOB)) <= 4096),
    topics                  TEXT NOT NULL
        CHECK (CASE WHEN json_valid(topics) THEN json_type(topics) = 'array' ELSE 0 END),
    low_confidence          INTEGER NOT NULL CHECK (low_confidence IN (0, 1)),
    needs_llm               INTEGER NOT NULL CHECK (needs_llm IN (0, 1)),
    duplicate_of            TEXT
        CHECK (duplicate_of IS NULL
               OR (length(duplicate_of) BETWEEN 3 AND 64 AND duplicate_of NOT GLOB '*[^a-z0-9_]*'
                   AND duplicate_of <> item_id)),
    recycled_of             TEXT
        CHECK (recycled_of IS NULL
               OR (length(recycled_of) BETWEEN 3 AND 64 AND recycled_of NOT GLOB '*[^a-z0-9_]*'
                   AND recycled_of <> item_id)),
    truncated               INTEGER NOT NULL CHECK (truncated IN (0, 1)),
    catalog_status          TEXT NOT NULL
        CHECK (length(catalog_status) BETWEEN 1 AND 32 AND catalog_status NOT GLOB '*[^a-z_]*'),
    priority                REAL NOT NULL CHECK (priority >= 0 AND priority <= 1),
    scope_reason            TEXT
        CHECK (scope_reason IS NULL
               OR scope_reason IN ('no_topic_match', 'crime_public_order', 'empty_text')),
    source_credibility_used INTEGER NOT NULL CHECK (source_credibility_used BETWEEN 1 AND 5),
    classified_at           TEXT NOT NULL CHECK (classified_at GLOB '????-??-??T??:??:??*+00:00'),
    classifier_version      TEXT NOT NULL CHECK (length(classifier_version) BETWEEN 1 AND 64),
    CHECK (duplicate_of IS NULL OR recycled_of IS NULL),
    CHECK (duplicate_of IS NULL OR needs_llm = 0),
    CHECK ((label = 'out_of_scope') = (scope_reason IS NOT NULL)),
    CHECK (label <> 'out_of_scope' OR needs_llm = 0)
);

CREATE INDEX ix_classifications_label ON classifications (label);
-- The LLM queue (stage 3c): the highest priority first, the oldest first among equals.
CREATE INDEX ix_classifications_queue ON classifications (priority DESC, classified_at, item_id)
    WHERE needs_llm = 1;

CREATE TABLE item_signatures (
    item_id      TEXT NOT NULL PRIMARY KEY
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    content_hash TEXT NOT NULL
        CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    ref_at       TEXT NOT NULL CHECK (ref_at GLOB '????-??-??T??:??:??*+00:00'),
    minhash      BLOB NOT NULL CHECK (typeof(minhash) = 'blob' AND length(minhash) = 320)
);

CREATE INDEX ix_item_signatures_hash ON item_signatures (content_hash);
CREATE INDEX ix_item_signatures_ref ON item_signatures (ref_at);

CREATE TABLE item_signature_bands (
    item_id TEXT NOT NULL REFERENCES item_signatures (item_id) ON DELETE CASCADE,
    band    INTEGER NOT NULL CHECK (band BETWEEN 0 AND 15),
    key     INTEGER NOT NULL,
    PRIMARY KEY (item_id, band)
) WITHOUT ROWID;

CREATE INDEX ix_item_signature_bands_lookup ON item_signature_bands (band, key, item_id);

CREATE TABLE topic_config (
    version    INTEGER PRIMARY KEY CHECK (version >= 1),
    config     TEXT NOT NULL
        CHECK ((CASE WHEN json_valid(config) THEN json_type(config) = 'object' ELSE 0 END)
               AND length(CAST(config AS BLOB)) <= 262144),
    created_at TEXT NOT NULL CHECK (created_at GLOB '????-??-??T??:??:??*+00:00'),
    actor      TEXT NOT NULL CHECK (length(actor) BETWEEN 1 AND 100),
    reason     TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 500)
);

-- The classification queue: pending items of the enabled sources, oldest published first.
CREATE INDEX ix_items_status_published ON items (status, published_at, id);
