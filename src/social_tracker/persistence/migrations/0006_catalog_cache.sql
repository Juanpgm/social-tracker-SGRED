-- Migration 0006: the official-catalog cache (unit 3b2, RC-6).
--
-- Events of the two official catalogs (USGS and EMSC) that the worker keeps for the cross-check of
-- the classifier. The classifier reads ONLY these rows; it never reaches the network. Same rules as
-- before: Python validates first, these CHECKs are the last line of defence, and timestamps are
-- ISO-8601 text in UTC ("+00:00"). No foreign keys: nothing here refers to collected content.
--
-- catalog_events: one row per (source, external id). A later fetch of the same id replaces the row
--   in place (catalogs revise magnitudes and places). Rows older than the retention are purged.
-- catalog_refresh_state: one row per catalog with the last good refresh (what tells a stale cache
--   from a quiet day), the failures since, when to try again, and a claim (claimed_until) that
--   keeps two refreshers from fetching the same catalog at once.

CREATE TABLE catalog_events (
    seq          INTEGER PRIMARY KEY,
    source       TEXT NOT NULL CHECK (source IN ('usgs', 'emsc')),
    external_id  TEXT NOT NULL
        CHECK (length(external_id) BETWEEN 1 AND 64
               AND external_id NOT GLOB '*[^A-Za-z0-9_.:-]*'),
    origin_time  TEXT NOT NULL CHECK (origin_time GLOB '????-??-??T??:??:??*+00:00'),
    latitude     REAL NOT NULL CHECK (latitude >= -90 AND latitude <= 90),
    longitude    REAL NOT NULL CHECK (longitude >= -180 AND longitude <= 180),
    magnitude    REAL NOT NULL CHECK (magnitude >= -2 AND magnitude <= 10),
    place        TEXT NOT NULL CHECK (length(place) <= 200),
    fetched_at   TEXT NOT NULL CHECK (fetched_at GLOB '????-??-??T??:??:??*+00:00'),
    UNIQUE (source, external_id)
);

CREATE INDEX ix_catalog_events_origin ON catalog_events (origin_time, source, external_id);

CREATE TABLE catalog_refresh_state (
    source               TEXT NOT NULL PRIMARY KEY CHECK (source IN ('usgs', 'emsc')),
    last_success_at      TEXT CHECK (last_success_at IS NULL
                                     OR last_success_at GLOB '????-??-??T??:??:??*+00:00'),
    last_attempt_at      TEXT CHECK (last_attempt_at IS NULL
                                     OR last_attempt_at GLOB '????-??-??T??:??:??*+00:00'),
    consecutive_failures INTEGER NOT NULL DEFAULT 0
        CHECK (consecutive_failures >= 0 AND consecutive_failures <= 1000000),
    next_attempt_at      TEXT CHECK (next_attempt_at IS NULL
                                     OR next_attempt_at GLOB '????-??-??T??:??:??*+00:00'),
    claimed_until        TEXT CHECK (claimed_until IS NULL
                                     OR claimed_until GLOB '????-??-??T??:??:??*+00:00'),
    last_error           TEXT CHECK (last_error IS NULL OR length(last_error) <= 200)
);
