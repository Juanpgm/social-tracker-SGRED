-- Migration 0009: hardening of the LLM escalation (unit 3c, remediation round).
--
-- Same rules as before: Python validates first, these CHECKs are the last line of defence, and
-- timestamps are ISO-8601 text in UTC ("+00:00"). Nothing is lost: every stored verdict survives
-- unchanged and no table is rebuilt (two columns are added by ALTER).
--
-- classifications.settled_at / settle_reason: WHEN and WHY a verdict left the LLM queue, for the
--   alert layer, which must tell a settled verdict from one that still waits.
--     'llm'        the model judged it
--     'duplicate'  an exact copy that took the verdict of its original (or was released from it)
--     'invalid'    the model answered twice with something that is not a valid verdict
--     'parked'     given up on without a judgment (repeated item failures, or dropped from a
--                  full queue); only these can be reopened (`llm requeue-parked`)
--   Both are NULL for a verdict the CPU stages settled themselves, and for a verdict that still
--   waits. They are set together: a time without a reason (or the reverse) is refused.
--   Verdicts settled before this migration are given what the judgments and the bookkeeping say.
--
-- Two partial indexes serve the new reads: the parked items by age (the time-to-live that sends
-- them back to the queue) and the valid judgments by time (the judge has been answering).

ALTER TABLE classifications ADD COLUMN settled_at TEXT
    CHECK (settled_at IS NULL OR settled_at GLOB '????-??-??T??:??:??*+00:00');
ALTER TABLE classifications ADD COLUMN settle_reason TEXT
    CHECK (settle_reason IS NULL OR settle_reason IN ('llm', 'duplicate', 'invalid', 'parked'))
    CHECK ((settle_reason IS NULL) = (settled_at IS NULL));

UPDATE classifications
SET settled_at = (SELECT j.judged_at FROM llm_judgments j WHERE j.item_id = classifications.item_id),
    settle_reason = (
        SELECT CASE j.outcome WHEN 'judged' THEN 'llm' WHEN 'invalid' THEN 'invalid'
                              ELSE 'duplicate' END
        FROM llm_judgments j WHERE j.item_id = classifications.item_id)
WHERE needs_llm = 0
  AND EXISTS (SELECT 1 FROM llm_judgments j WHERE j.item_id = classifications.item_id);

UPDATE classifications
SET settled_at = (SELECT e.updated_at FROM llm_escalations e WHERE e.item_id = classifications.item_id),
    settle_reason = 'parked'
WHERE needs_llm = 0
  AND settled_at IS NULL
  AND EXISTS (SELECT 1 FROM llm_escalations e
              WHERE e.item_id = classifications.item_id AND e.state IN ('parked', 'dropped'));

CREATE INDEX ix_llm_escalations_parked ON llm_escalations (updated_at, item_id)
    WHERE state = 'parked';
CREATE INDEX ix_llm_judgments_judged ON llm_judgments (judged_at)
    WHERE outcome = 'judged';
