-- Migration 0001: initial schema.
--
-- Column rules mirror the domain models (sources.domain, control.domain). Values are
-- validated in Python first; these CHECKs are the last line of defence, so a bug or a manual
-- edit cannot persist an incoherent row. Timestamps are ISO-8601 text in UTC ("+00:00").
-- `schema_migrations` is created by the migration runner itself, before any migration runs.

CREATE TABLE sources (
    id              TEXT    NOT NULL PRIMARY KEY
        CHECK (length(id) BETWEEN 3 AND 64 AND id NOT GLOB '*[^a-z0-9_]*'),
    name            TEXT    NOT NULL CHECK (length(name) BETWEEN 1 AND 100),
    normalized_name TEXT    NOT NULL CHECK (length(normalized_name) >= 1),
    type            TEXT    NOT NULL
        CHECK (type IN ('news_rss', 'instagram', 'live_stream', 'recorded_audio')),
    mode            TEXT    NOT NULL CHECK (mode IN ('recorded', 'streaming')),
    credibility     INTEGER NOT NULL
        CHECK (typeof(credibility) = 'integer' AND credibility BETWEEN 1 AND 5),
    enabled         INTEGER NOT NULL CHECK (typeof(enabled) = 'integer' AND enabled IN (0, 1)),
    secret_ref      TEXT,
    -- Type-specific fields (feed_url, keywords, handle, ...) live here as a JSON object.
    config_json     TEXT    NOT NULL
        CHECK (CASE WHEN json_valid(config_json) THEN json_type(config_json) = 'object' ELSE 0 END),
    created_at      TEXT    NOT NULL CHECK (created_at GLOB '????-??-??T??:??:??*+00:00'),
    updated_at      TEXT    NOT NULL CHECK (updated_at GLOB '????-??-??T??:??:??*+00:00'),
    CHECK ((type = 'live_stream') = (mode = 'streaming')),
    CHECK (type <> 'instagram' OR secret_ref IS NOT NULL)
);

CREATE UNIQUE INDEX ux_sources_normalized_name ON sources (normalized_name);

-- No foreign key to sources on purpose: sources and control state are separate stores.
CREATE TABLE source_state (
    source_id            TEXT    NOT NULL PRIMARY KEY
        CHECK (length(source_id) BETWEEN 3 AND 64 AND source_id NOT GLOB '*[^a-z0-9_]*'),
    state                TEXT    NOT NULL
        CHECK (state IN ('disabled', 'idle', 'collecting', 'backoff', 'failed')),
    consecutive_failures INTEGER NOT NULL
        CHECK (typeof(consecutive_failures) = 'integer' AND consecutive_failures BETWEEN 0 AND 5),
    next_run_at          TEXT CHECK (next_run_at GLOB '????-??-??T??:??:??*+00:00'),
    -- FAILED <=> exactly the maximum failures; DISABLED has none; BACKOFF <=> next_run_at set.
    CHECK ((consecutive_failures = 5) = (state = 'failed')),
    CHECK (state <> 'disabled' OR consecutive_failures = 0),
    CHECK ((state = 'backoff') = (next_run_at IS NOT NULL))
);

CREATE TABLE settings (
    key   TEXT NOT NULL PRIMARY KEY CHECK (length(key) BETWEEN 1 AND 100),
    value TEXT NOT NULL CHECK (length(value) <= 1000),
    CHECK (key <> 'global_analysis_enabled' OR value IN ('true', 'false'))
);

CREATE TABLE control_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    at        TEXT NOT NULL CHECK (at GLOB '????-??-??T??:??:??*+00:00'),
    kind      TEXT NOT NULL CHECK (length(kind) BETWEEN 1 AND 64),
    source_id TEXT
        CHECK (source_id IS NULL
               OR (length(source_id) BETWEEN 3 AND 64 AND source_id NOT GLOB '*[^a-z0-9_]*')),
    detail    TEXT NOT NULL CHECK (length(detail) <= 2000)
);

-- The event log is an audit trail: rows can be inserted, never changed or removed.
CREATE TRIGGER control_events_no_update BEFORE UPDATE ON control_events
BEGIN
    SELECT RAISE(ABORT, 'control_events is append-only');
END;

CREATE TRIGGER control_events_no_delete BEFORE DELETE ON control_events
BEGIN
    SELECT RAISE(ABORT, 'control_events is append-only');
END;

-- INSERT OR REPLACE deletes the old row without firing the DELETE trigger, and an explicit
-- low id would reorder the log (it is read back by id). Reject both: BEFORE INSERT cannot see
-- the id of an auto-assigned row, so the replacement check runs before and the ordering check
-- after the insert (the failed statement is rolled back as a whole).
CREATE TRIGGER control_events_no_replace BEFORE INSERT ON control_events
WHEN EXISTS (SELECT 1 FROM control_events WHERE id = NEW.id)
BEGIN
    SELECT RAISE(ABORT, 'control_events is append-only');
END;

CREATE TRIGGER control_events_ids_only_grow AFTER INSERT ON control_events
WHEN EXISTS (SELECT 1 FROM control_events WHERE id > NEW.id)
BEGIN
    SELECT RAISE(ABORT, 'control_events is append-only');
END;
