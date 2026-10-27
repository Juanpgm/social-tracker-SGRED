-- Migration 0010: the alert store (unit 3d, AL-1 .. AL-11).
--
-- Same rules as before: Python validates first, these CHECKs are the last line of defence, and
-- timestamps are ISO-8601 text in UTC ("+00:00"). Nothing existing is touched: three new tables
-- and their indexes, so every stored item, verdict and judgment survives unchanged.
--
-- alerts: ONE row per alert, a projection of its members (see `alerts.domain.project`): the label,
--   severity, confidence, rationale, source, item reference, excerpt and evidence are those of the
--   leading member (the most severe active one), `occurrences` counts the active members and
--   `tags` merge theirs. `status` is the operator's (new, seen, dismissed); `withdrawn_at` and
--   `withdrawn_reason` are set together when no member qualifies any more (the verdict fell under
--   the threshold or became no_risk): the alert is kept, hidden from the default list, and comes
--   back by itself if a verdict qualifies again. There is no foreign key to `items` or `sources`:
--   an alert outlives both, keeping the excerpt, the evidence and the snapshot of the source name,
--   until the retention purge forgets it (by the time of its latest occurrence).
--
-- alert_items: the items behind an alert. `item_id` is the primary key, so an item can raise at
--   most one alert however many times it is reprocessed, reclassified or settled by the model, and
--   two workers racing on the same item store one row between them. It also keeps the verdict the
--   member was last evaluated on (`label`, `confidence`, `stage`, `settled_at`), which is what the
--   sync compares with the stored classification to notice an upgrade or a downgrade.
--
-- alert_events: the append-only trail of an alert (created, merged, updated, withdrawn,
--   reactivated, status_changed) with actor and reason. It goes with its alert.

CREATE TABLE alerts (
    id                TEXT NOT NULL PRIMARY KEY
        CHECK (length(id) BETWEEN 3 AND 64 AND id NOT GLOB '*[^a-z0-9_]*'),
    label             TEXT NOT NULL
        CHECK (label IN ('unintentional_rumor', 'deliberate_disinformation', 'recycled_content')),
    severity          TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    confidence        REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    rationale         TEXT NOT NULL CHECK (length(rationale) BETWEEN 1 AND 300),
    source_id         TEXT NOT NULL
        CHECK (length(source_id) BETWEEN 3 AND 64 AND source_id NOT GLOB '*[^a-z0-9_]*'),
    source_name       TEXT NOT NULL CHECK (length(source_name) BETWEEN 1 AND 100),
    item_id           TEXT NOT NULL
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    excerpt           TEXT NOT NULL CHECK (length(excerpt) <= 500),
    tags              TEXT NOT NULL
        CHECK (CASE WHEN json_valid(tags) THEN json_type(tags) = 'array' ELSE 0 END),
    evidence          TEXT NOT NULL
        CHECK ((CASE WHEN json_valid(evidence) THEN json_type(evidence) = 'object' ELSE 0 END)
               AND length(CAST(evidence AS BLOB)) <= 4096),
    linked_sources    TEXT NOT NULL
        CHECK (CASE WHEN json_valid(linked_sources) THEN json_type(linked_sources) = 'array'
                    ELSE 0 END),
    status            TEXT NOT NULL CHECK (status IN ('new', 'seen', 'dismissed')),
    occurrences       INTEGER NOT NULL CHECK (occurrences >= 0),
    first_seen_at     TEXT NOT NULL CHECK (first_seen_at GLOB '????-??-??T??:??:??*+00:00'),
    last_seen_at      TEXT NOT NULL CHECK (last_seen_at GLOB '????-??-??T??:??:??*+00:00'),
    created_at        TEXT NOT NULL CHECK (created_at GLOB '????-??-??T??:??:??*+00:00'),
    updated_at        TEXT NOT NULL CHECK (updated_at GLOB '????-??-??T??:??:??*+00:00'),
    status_changed_at TEXT
        CHECK (status_changed_at IS NULL
               OR status_changed_at GLOB '????-??-??T??:??:??*+00:00'),
    withdrawn_at      TEXT
        CHECK (withdrawn_at IS NULL OR withdrawn_at GLOB '????-??-??T??:??:??*+00:00'),
    withdrawn_reason  TEXT
        CHECK (withdrawn_reason IS NULL OR length(withdrawn_reason) BETWEEN 1 AND 200),
    CHECK ((withdrawn_at IS NULL) = (withdrawn_reason IS NULL))
);

-- The list: newest occurrence first, the id making the order total.
CREATE INDEX ix_alerts_last_seen ON alerts (last_seen_at DESC, id DESC);
CREATE INDEX ix_alerts_status ON alerts (status, last_seen_at DESC);
CREATE INDEX ix_alerts_severity ON alerts (severity, last_seen_at DESC);

CREATE TABLE alert_items (
    item_id      TEXT NOT NULL PRIMARY KEY
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    alert_id     TEXT NOT NULL REFERENCES alerts (id) ON DELETE CASCADE,
    source_id    TEXT NOT NULL
        CHECK (length(source_id) BETWEEN 3 AND 64 AND source_id NOT GLOB '*[^a-z0-9_]*'),
    source_name  TEXT NOT NULL CHECK (length(source_name) BETWEEN 1 AND 100),
    content_hash TEXT NOT NULL
        CHECK (length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'),
    excerpt      TEXT NOT NULL CHECK (length(excerpt) <= 500),
    label        TEXT NOT NULL
        CHECK (label IN ('unintentional_rumor', 'deliberate_disinformation', 'recycled_content',
                         'no_risk', 'unclassified', 'out_of_scope')),
    confidence   REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    stage        TEXT NOT NULL CHECK (length(stage) BETWEEN 1 AND 64),
    rationale    TEXT NOT NULL CHECK (length(rationale) BETWEEN 1 AND 300),
    severity     TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    tags         TEXT NOT NULL
        CHECK (CASE WHEN json_valid(tags) THEN json_type(tags) = 'array' ELSE 0 END),
    evidence     TEXT NOT NULL
        CHECK ((CASE WHEN json_valid(evidence) THEN json_type(evidence) = 'object' ELSE 0 END)
               AND length(CAST(evidence AS BLOB)) <= 4096),
    active       INTEGER NOT NULL CHECK (active IN (0, 1)),
    occurred_at  TEXT NOT NULL CHECK (occurred_at GLOB '????-??-??T??:??:??*+00:00'),
    linked_at    TEXT NOT NULL CHECK (linked_at GLOB '????-??-??T??:??:??*+00:00'),
    event_age_s  INTEGER NOT NULL CHECK (event_age_s >= 0),
    settled_at   TEXT CHECK (settled_at IS NULL OR settled_at GLOB '????-??-??T??:??:??*+00:00')
);

CREATE INDEX ix_alert_items_alert ON alert_items (alert_id, occurred_at, item_id);
-- Finding the alert a copy belongs to: same content, close in time.
CREATE INDEX ix_alert_items_hash ON alert_items (content_hash, occurred_at);
CREATE INDEX ix_alert_items_source ON alert_items (source_id);

CREATE TABLE alert_events (
    seq        INTEGER PRIMARY KEY,
    alert_id   TEXT NOT NULL REFERENCES alerts (id) ON DELETE CASCADE,
    at         TEXT NOT NULL CHECK (at GLOB '????-??-??T??:??:??*+00:00'),
    kind       TEXT NOT NULL
        CHECK (kind IN ('created', 'merged', 'updated', 'withdrawn', 'reactivated',
                        'status_changed')),
    actor      TEXT NOT NULL CHECK (length(actor) BETWEEN 1 AND 100),
    reason     TEXT NOT NULL CHECK (length(reason) BETWEEN 1 AND 500),
    from_value TEXT NOT NULL CHECK (length(from_value) <= 100),
    to_value   TEXT NOT NULL CHECK (length(to_value) <= 100)
);

CREATE INDEX ix_alert_events_alert ON alert_events (alert_id, seq);

-- alert_forgotten: the items whose alert the retention purge removed. Without it the next sync
-- would find the same risk verdict of an item that still exists and raise the alert all over again
-- (with a retention of 0 days, at once). It holds ids only, never text, and a row goes when the
-- verdict of its item does (the purge cleans up after itself).
CREATE TABLE alert_forgotten (
    item_id      TEXT NOT NULL PRIMARY KEY
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    forgotten_at TEXT NOT NULL CHECK (forgotten_at GLOB '????-??-??T??:??:??*+00:00')
);
