-- Migration 0012: the alert remediation (unit 3d, second round).
--
-- Same rules as before: Python validates first, these CHECKs are the last line of defence, and
-- timestamps are ISO-8601 text in UTC ("+00:00"). Nothing that exists is rewritten: columns with a
-- safe default are added, five small tables are created, and every stored alert, member, verdict
-- and item survives unchanged.
--
-- alerts.downgraded_by_model: the language model settled a verdict LOWER than the provisional one
--   this alert was raised on (or made every member stop qualifying). Such an alert is NOT hidden:
--   it stays in the default list with status `new` and a visible flag, until a person looks at it.
-- alerts.overflow: copies that were counted but are not stored as members, because the alert is
--   at its member cap (a flood of near copies must not grow an alert without bound).
-- alerts.events_dropped: routine trail entries (merged, updated) not stored because the alert is
--   at its trail cap; the detail page says how many.
-- alert_events.from_absorbed: the entry was copied from an alert that was absorbed into this one,
--   with its original time, so merging never erases an audit trail.
--
-- alert_forgotten.label, confidence, settled_at: the signature of the verdict that was forgotten.
--   The item is not raised again while its verdict is the same; a changed or upgraded verdict
--   clears the entry (NULL = a row from before this migration: it keeps suppressing).
--
-- alert_overflow: the items counted in `alerts.overflow` (item, alert, verdict signature), so that
--   a repeated look at the same item does not count it twice.
--
-- alert_failures: items the alert step could not evaluate, with a backoff and a park flag, so that
--   one poison item cannot be retried at every tick of the worker.
--
-- alert_dirty: the items whose verdict was written since the alert step last looked at them. The
--   step reads this small set instead of comparing every member with its verdict at every tick.
--   Two triggers fill it (a risk verdict, or the verdict of a member, that is inserted or
--   updated); the first pass after the upgrade finds the verdicts that already existed.

ALTER TABLE alerts ADD COLUMN downgraded_by_model INTEGER NOT NULL DEFAULT 0
    CHECK (downgraded_by_model IN (0, 1));
ALTER TABLE alerts ADD COLUMN overflow INTEGER NOT NULL DEFAULT 0 CHECK (overflow >= 0);
ALTER TABLE alerts ADD COLUMN events_dropped INTEGER NOT NULL DEFAULT 0
    CHECK (events_dropped >= 0);
ALTER TABLE alert_events ADD COLUMN from_absorbed INTEGER NOT NULL DEFAULT 0
    CHECK (from_absorbed IN (0, 1));

ALTER TABLE alert_forgotten ADD COLUMN label TEXT;
ALTER TABLE alert_forgotten ADD COLUMN confidence REAL
    CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));
ALTER TABLE alert_forgotten ADD COLUMN settled_at TEXT
    CHECK (settled_at IS NULL OR settled_at GLOB '????-??-??T??:??:??*+00:00');

-- Finding the alert of an ORIGINAL whose copies were seen first: the copies point at it.
CREATE INDEX ix_alert_items_near ON alert_items (json_extract(evidence, '$.near_of'), occurred_at);
CREATE INDEX ix_alert_items_duplicate
    ON alert_items (json_extract(evidence, '$.duplicate_of'), occurred_at);

CREATE TABLE alert_overflow (
    item_id    TEXT NOT NULL PRIMARY KEY
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    alert_id   TEXT NOT NULL REFERENCES alerts (id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    settled_at TEXT CHECK (settled_at IS NULL OR settled_at GLOB '????-??-??T??:??:??*+00:00')
);

CREATE INDEX ix_alert_overflow_alert ON alert_overflow (alert_id);

CREATE TABLE alert_failures (
    item_id        TEXT NOT NULL PRIMARY KEY
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    failures       INTEGER NOT NULL CHECK (failures >= 1),
    last_failed_at TEXT NOT NULL CHECK (last_failed_at GLOB '????-??-??T??:??:??*+00:00'),
    next_try_at    TEXT NOT NULL CHECK (next_try_at GLOB '????-??-??T??:??:??*+00:00'),
    parked         INTEGER NOT NULL CHECK (parked IN (0, 1))
);

CREATE TABLE alert_dirty (
    item_id TEXT NOT NULL PRIMARY KEY
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*')
);

INSERT OR IGNORE INTO alert_dirty (item_id)
SELECT item_id FROM classifications
WHERE label IN ('unintentional_rumor', 'deliberate_disinformation', 'recycled_content')
   OR item_id IN (SELECT item_id FROM alert_items);

CREATE TRIGGER classifications_alert_dirty_insert AFTER INSERT ON classifications
WHEN NEW.label IN ('unintentional_rumor', 'deliberate_disinformation', 'recycled_content')
  OR EXISTS (SELECT 1 FROM alert_items m WHERE m.item_id = NEW.item_id)
BEGIN
    INSERT OR IGNORE INTO alert_dirty (item_id) VALUES (NEW.item_id);
END;

CREATE TRIGGER classifications_alert_dirty_update AFTER UPDATE ON classifications
WHEN NEW.label IN ('unintentional_rumor', 'deliberate_disinformation', 'recycled_content')
  OR EXISTS (SELECT 1 FROM alert_items m WHERE m.item_id = NEW.item_id)
BEGIN
    INSERT OR IGNORE INTO alert_dirty (item_id) VALUES (NEW.item_id);
END;
