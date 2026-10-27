-- Migration 0011: the local gold set (unit 3d).
--
-- Same rules as before: Python validates first, these CHECKs are the last line of defence, and
-- timestamps are ISO-8601 text in UTC ("+00:00"). One new table and one index: nothing that exists
-- is touched.
--
-- gold_examples: the texts a person labeled, to measure the cascade (`classify calibrate`) and to
--   tune the alert threshold. The id is made from the text (`gld_` and 16 hex digits), so the same
--   text is the same row whoever adds it and however many times. The text may be REAL content
--   collected from real sources: it lives in the local database like the collected items do and
--   never in the repository; `gold export` writes it to a file only when asked to.
--   `seq` keeps the order the examples were added in.

CREATE TABLE gold_examples (
    seq         INTEGER PRIMARY KEY,
    id          TEXT NOT NULL UNIQUE
        CHECK (length(id) = 20 AND id GLOB 'gld_[0-9a-f]*' AND id NOT GLOB '*[^a-z0-9_]*'),
    text        TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND 20000),
    label       TEXT NOT NULL
        CHECK (label IN ('unintentional_rumor', 'deliberate_disinformation', 'recycled_content',
                         'no_risk', 'unclassified', 'out_of_scope')),
    note        TEXT NOT NULL CHECK (length(note) <= 500),
    credibility INTEGER NOT NULL CHECK (credibility BETWEEN 1 AND 5),
    url         TEXT
        CHECK (url IS NULL
               OR (length(url) <= 2048 AND (url GLOB 'http://*' OR url GLOB 'https://*'))),
    added_at    TEXT NOT NULL CHECK (added_at GLOB '????-??-??T??:??:??*+00:00'),
    actor       TEXT NOT NULL CHECK (length(actor) BETWEEN 1 AND 100)
);

CREATE INDEX ix_gold_examples_label ON gold_examples (label);
