-- Migration 0005: hardening of the classification cascade (unit 3b1 review).
--
-- Same rules as before: Python validates first, these CHECKs are the last line of defence, and
-- timestamps are ISO-8601 text in UTC ("+00:00"). Nothing is lost: every stored verdict and every
-- stored signature survives this migration unchanged.
--
-- 1. `classifications` is REBUILT (create, copy, drop, rename; no other table refers to it and it
--    has no trigger), because a CHECK cannot be altered in SQLite and two things change in them:
--    `scope_reason` may now also be 'non_spanish', and the new links are constrained. New columns:
--      near_of            a near copy within the recycle window; its verdict is its own, the link
--                         is only a prior for later stages. At most one of duplicate_of,
--                         recycled_of and near_of is set.
--      awaiting_original  1 on a duplicate whose original is still waiting for the LLM: the verdict
--                         is provisional and the later stage replaces it when the original is
--                         judged. Only a duplicate can wait.
--    The queue index keeps its shape; `ix_classifications_awaiting` finds the waiting duplicates
--    of an original.
--
-- 2. `item_signatures` gets three columns by ALTER (all existing rows stay valid): the digests of
--    the folded text and of the numbers in it, and the word count. A signature stored before this
--    migration has no digests and 0 words: it can only match by identical content hash, because a
--    near copy is required to have the same numbers and a minimum of words. `text_hash` is
--    indexed to find an identical text whose raw content hash differs.
--
-- 3. `classification_failures` counts the consecutive per-item errors of an item (a "poison"
--    item); after the configured number the service parks the item with the status
--    'classification_failed'. The status needs no schema change (`items.status` only checks
--    its shape). No foreign key, like the other tables of collected data.

PRAGMA defer_foreign_keys = ON;

CREATE TABLE classifications_rebuilt (
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
    near_of                 TEXT
        CHECK (near_of IS NULL
               OR (length(near_of) BETWEEN 3 AND 64 AND near_of NOT GLOB '*[^a-z0-9_]*'
                   AND near_of <> item_id)),
    awaiting_original       INTEGER NOT NULL DEFAULT 0 CHECK (awaiting_original IN (0, 1)),
    truncated               INTEGER NOT NULL CHECK (truncated IN (0, 1)),
    catalog_status          TEXT NOT NULL
        CHECK (length(catalog_status) BETWEEN 1 AND 32 AND catalog_status NOT GLOB '*[^a-z_]*'),
    priority                REAL NOT NULL CHECK (priority >= 0 AND priority <= 1),
    scope_reason            TEXT
        CHECK (scope_reason IS NULL
               OR scope_reason IN ('no_topic_match', 'crime_public_order', 'empty_text',
                                   'non_spanish')),
    source_credibility_used INTEGER NOT NULL CHECK (source_credibility_used BETWEEN 1 AND 5),
    classified_at           TEXT NOT NULL CHECK (classified_at GLOB '????-??-??T??:??:??*+00:00'),
    classifier_version      TEXT NOT NULL CHECK (length(classifier_version) BETWEEN 1 AND 64),
    CHECK ((duplicate_of IS NOT NULL) + (recycled_of IS NOT NULL) + (near_of IS NOT NULL) <= 1),
    CHECK (duplicate_of IS NULL OR needs_llm = 0),
    CHECK (awaiting_original = 0 OR duplicate_of IS NOT NULL),
    CHECK ((label = 'out_of_scope') = (scope_reason IS NOT NULL)),
    CHECK (label <> 'out_of_scope' OR needs_llm = 0)
);

INSERT INTO classifications_rebuilt (
    seq, item_id, label, confidence, stage, rationale, signals, topics, low_confidence, needs_llm,
    duplicate_of, recycled_of, truncated, catalog_status, priority, scope_reason,
    source_credibility_used, classified_at, classifier_version
)
SELECT
    seq, item_id, label, confidence, stage, rationale, signals, topics, low_confidence, needs_llm,
    duplicate_of, recycled_of, truncated, catalog_status, priority, scope_reason,
    source_credibility_used, classified_at, classifier_version
FROM classifications;

DROP TABLE classifications;
ALTER TABLE classifications_rebuilt RENAME TO classifications;

CREATE INDEX ix_classifications_label ON classifications (label);
-- The LLM queue (stage 3c): the highest priority first, the oldest first among equals.
CREATE INDEX ix_classifications_queue ON classifications (priority DESC, classified_at, item_id)
    WHERE needs_llm = 1;
-- The duplicates that wait for their original to be judged (stage 3c propagates its verdict).
CREATE INDEX ix_classifications_awaiting ON classifications (duplicate_of)
    WHERE awaiting_original = 1;
-- The history pruning walks the verdicts oldest first.
CREATE INDEX ix_classifications_classified_at ON classifications (classified_at, item_id);

ALTER TABLE item_signatures ADD COLUMN text_hash TEXT
    CHECK (text_hash IS NULL OR (length(text_hash) = 64 AND text_hash NOT GLOB '*[^0-9a-f]*'));
ALTER TABLE item_signatures ADD COLUMN digits_hash TEXT
    CHECK (digits_hash IS NULL OR (length(digits_hash) = 64 AND digits_hash NOT GLOB '*[^0-9a-f]*'));
ALTER TABLE item_signatures ADD COLUMN n_tokens INTEGER NOT NULL DEFAULT 0
    CHECK (n_tokens >= 0 AND n_tokens <= 1000000);

CREATE INDEX ix_item_signatures_text_hash ON item_signatures (text_hash)
    WHERE text_hash IS NOT NULL;

CREATE TABLE classification_failures (
    item_id  TEXT NOT NULL PRIMARY KEY
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    failures INTEGER NOT NULL CHECK (failures BETWEEN 1 AND 1000),
    last_at  TEXT NOT NULL CHECK (last_at GLOB '????-??-??T??:??:??*+00:00')
);
