-- Migration 0008: the LLM escalation (unit 3c, RC-7, RC-9, RC-12).
--
-- Same rules as before: Python validates first, these CHECKs are the last line of defence, and
-- timestamps are ISO-8601 text in UTC ("+00:00"). Nothing is rebuilt and every stored verdict
-- survives unchanged: the escalation settles a queued verdict IN PLACE (`classifications` keeps
-- one row per item) and keeps the rest here. No foreign keys, like the other tables of collected
-- data: a verdict may outlive its item, and so may its judgment, until the history pruning
-- forgets both together.
--
-- llm_judgments: ONE record per item that went through the judge, for the audit and for
--   reproducibility. `outcome` is 'judged' (a valid answer: label, confidence, rationale, evidence,
--   the model that gave it and the prompt version), 'invalid' (two invalid answers in a row,
--   RC-9: nothing from the model is applied) or 'reused' (an exact copy that took the answer of
--   its original; `reused_from` names it, and the label is absent when the original was
--   'invalid'). The prior_* columns keep the verdict the earlier stage had given before. The raw
--   model output is kept truncated to 2048 characters, for the audit only: it is never logged.
--
-- llm_escalations: bookkeeping of an item that has NO judgment (yet): 'escalation_failed' (the
--   call failed; retry after next_attempt_at, doubling per failure), 'parked' (item-specific
--   failures reached the limit: it stays as the CPU stages left it) or 'dropped' (pushed out of
--   a full queue, the oldest first, and counted). last_error is a fixed code, never text.
--
-- llm_budget: judgments started per UTC hour (the hourly cap).
-- llm_lease: the single in-flight slot (one judgment at a time on the GPU), shared by every
--   process that uses this database; it expires, so a crashed holder frees it.

CREATE TABLE llm_judgments (
    item_id         TEXT NOT NULL PRIMARY KEY
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    outcome         TEXT NOT NULL CHECK (outcome IN ('judged', 'invalid', 'reused')),
    label           TEXT
        CHECK (label IS NULL
               OR label IN ('unintentional_rumor', 'deliberate_disinformation', 'recycled_content',
                            'no_risk')),
    confidence      REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    rationale       TEXT CHECK (rationale IS NULL OR length(rationale) BETWEEN 1 AND 300),
    evidence        TEXT NOT NULL
        CHECK ((CASE WHEN json_valid(evidence) THEN json_type(evidence) = 'array' ELSE 0 END)
               AND length(CAST(evidence AS BLOB)) <= 4096),
    llm_model       TEXT CHECK (llm_model IS NULL OR length(llm_model) BETWEEN 1 AND 100),
    prompt_version  TEXT NOT NULL CHECK (length(prompt_version) BETWEEN 1 AND 64),
    raw_output      TEXT CHECK (raw_output IS NULL OR length(raw_output) <= 2048),
    prior_label     TEXT NOT NULL
        CHECK (prior_label IN ('unintentional_rumor', 'deliberate_disinformation',
                               'recycled_content', 'no_risk', 'unclassified', 'out_of_scope')),
    prior_confidence REAL NOT NULL CHECK (prior_confidence >= 0 AND prior_confidence <= 1),
    prior_stage     TEXT NOT NULL CHECK (prior_stage IN ('topic', 'rules', 'dedup', 'catalog', 'llm')),
    reused_from     TEXT
        CHECK (reused_from IS NULL
               OR (length(reused_from) BETWEEN 3 AND 64 AND reused_from NOT GLOB '*[^a-z0-9_]*'
                   AND reused_from <> item_id)),
    judged_at       TEXT NOT NULL CHECK (judged_at GLOB '????-??-??T??:??:??*+00:00'),
    CHECK (outcome <> 'judged'
           OR (label IS NOT NULL AND confidence IS NOT NULL AND rationale IS NOT NULL
               AND llm_model IS NOT NULL AND reused_from IS NULL)),
    CHECK (outcome <> 'invalid' OR (label IS NULL AND reused_from IS NULL)),
    CHECK ((outcome = 'reused') = (reused_from IS NOT NULL))
);

CREATE INDEX ix_llm_judgments_outcome ON llm_judgments (outcome);

CREATE TABLE llm_escalations (
    item_id         TEXT NOT NULL PRIMARY KEY
        CHECK (length(item_id) BETWEEN 3 AND 64 AND item_id NOT GLOB '*[^a-z0-9_]*'),
    state           TEXT NOT NULL CHECK (state IN ('escalation_failed', 'parked', 'dropped')),
    attempts        INTEGER NOT NULL CHECK (attempts BETWEEN 0 AND 1000000),
    next_attempt_at TEXT
        CHECK (next_attempt_at IS NULL OR next_attempt_at GLOB '????-??-??T??:??:??*+00:00'),
    last_error      TEXT CHECK (last_error IS NULL OR length(last_error) <= 64),
    updated_at      TEXT NOT NULL CHECK (updated_at GLOB '????-??-??T??:??:??*+00:00')
);

CREATE INDEX ix_llm_escalations_state ON llm_escalations (state);

CREATE TABLE llm_budget (
    hour_start TEXT NOT NULL PRIMARY KEY CHECK (hour_start GLOB '????-??-??T??:00:00+00:00'),
    calls      INTEGER NOT NULL CHECK (calls BETWEEN 0 AND 1000000)
);

CREATE TABLE llm_lease (
    id     INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
    holder TEXT NOT NULL CHECK (length(holder) BETWEEN 1 AND 100),
    until  TEXT NOT NULL CHECK (until GLOB '????-??-??T??:??:??*+00:00')
);
