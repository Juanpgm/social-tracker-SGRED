-- Migration 0007: a refresh whose answer may have been cut short (review of unit 3b2).
--
-- The services return at most `limit` events, newest first. An answer that reaches the limit may
-- be missing older events inside the window it covers, so the cache cannot call the absence of an
-- event a contradiction. `last_truncated` is 1 when the last good refresh of the catalog hit the
-- limit; the check then reads the cache as not complete (no "no event" mismatch).

ALTER TABLE catalog_refresh_state
    ADD COLUMN last_truncated INTEGER NOT NULL DEFAULT 0 CHECK (last_truncated IN (0, 1));
