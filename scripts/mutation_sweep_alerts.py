"""A mutation sweep of the boundaries of the alerts and the calibration (unit 3d).

Each mutant changes ONE operator or constant of the production code (a `>=` into a `>`, a window
by one, a dropped guard); the targeted tests must fail. A mutant that leaves them green is a
survivor: a boundary no test pins. The sweep works on COPIES of the source in temporary folders
(see `scripts/mutation_copy.py`): the real tree is never written to, so an interrupt cannot leave
it mutated.

    uv run python scripts/mutation_sweep_alerts.py            # every mutant
    uv run python scripts/mutation_sweep_alerts.py retention  # mutants named "...retention..."
    uv run python scripts/mutation_sweep_alerts.py --jobs 6   # more copies at once (default 4)
    uv run python scripts/mutation_sweep_alerts.py --check    # only look for stale patterns

The second round of the alerts has its own sweep, `scripts/mutation_sweep_alerts_remediation.py`.
The browser and web tests are left out of the sweep (they cost minutes and repeat what the service
tests pin through the same code). Exit status: 0 when every mutant was killed, 1 when some
survived or their pattern is stale, 2 when the targeted tests do not pass on the unmutated copy.
"""

# ruff: noqa: E501 - a pattern is a piece of the source and keeps the line it has there
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mutation_copy import Mutant, m, run_sweep  # noqa: E402,F401  (the script folder is not a package)

TESTS = [
    "tests/alerts",
    "tests/gold",
    "tests/classification/test_calibration.py",
    "tests/entrypoints/test_cli_alerts.py",
    "tests/entrypoints/test_cli_gold.py",
]


DOMAIN = "alerts/domain.py"
EXCERPT = "alerts/excerpt.py"
SERVICE = "alerts/service.py"
REPO = "persistence/alerts_repo.py"
CALIBRATION = "classification/calibration.py"
GOLD = "gold/service.py"
GOLD_REPO = "persistence/gold_repo.py"

# Known equivalent mutants, left out on purpose (no input can tell them apart):
# - `_SCAN_CHARS` and `_WORK_CHARS` of the excerpt: they only bound the work on a hostile text
#   (tests/alerts/test_excerpt.py pins the time and the result, not the exact window);
# - the order of the clauses of a WHERE and of the indexes of the migration.
MUTANTS = [
    # ---- AL-1: which verdicts alert ----
    m(
        "alert threshold inclusive",
        DOMAIN,
        "chosen in RISK_LABELS and score >= threshold",
        "chosen in RISK_LABELS and score > threshold",
    ),  # fmt: skip
    m(
        "alert labels",
        DOMAIN,
        "chosen in RISK_LABELS and score >= threshold",
        "chosen is not Label.NO_RISK and score >= threshold",
    ),  # fmt: skip
    # ---- AL-2: severity ----
    m(
        "critical deliberate inclusive",
        DOMAIN,
        "score >= settings.critical_disinformation",
        "score > settings.critical_disinformation",
    ),  # fmt: skip
    m(
        "critical window inclusive",
        DOMAIN,
        "event_age <= settings.critical_window",
        "event_age < settings.critical_window",
    ),  # fmt: skip
    m(
        "critical catalog inclusive",
        DOMAIN,
        "score >= settings.critical_catalog",
        "score > settings.critical_catalog",
    ),  # fmt: skip
    m(
        "critical needs a mismatch",
        DOMAIN,
        "catalog_status == CATALOG_MISMATCH",
        "catalog_status != CATALOG_MISMATCH",
    ),  # fmt: skip
    m("high inclusive", DOMAIN, "if score >= settings.high:", "if score > settings.high:"),
    m("medium inclusive", DOMAIN, "if score >= settings.medium:", "if score > settings.medium:"),
    m(
        "event age never negative",
        DOMAIN,
        "max(timedelta(0), seen - published)",
        "seen - published",
    ),
    m(
        "default critical deliberate",
        DOMAIN,
        "critical_disinformation: float = 0.85",
        "critical_disinformation: float = 0.86",
    ),  # fmt: skip
    m(
        "default critical catalog",
        DOMAIN,
        "critical_catalog: float = 0.8\n",
        "critical_catalog: float = 0.81\n",
    ),  # fmt: skip
    m("default high", DOMAIN, "    high: float = 0.7\n", "    high: float = 0.71\n"),
    m("default medium", DOMAIN, "    medium: float = 0.5\n", "    medium: float = 0.51\n"),
    m(
        "default window",
        DOMAIN,
        "dedup_window: timedelta = timedelta(hours=24)",
        "dedup_window: timedelta = timedelta(hours=23)",
    ),  # fmt: skip
    m("default retention", DOMAIN, "DEFAULT_RETENTION_DAYS = 180", "DEFAULT_RETENTION_DAYS = 181"),
    m("excerpt length", DOMAIN, "MAX_EXCERPT_CHARS = 500", "MAX_EXCERPT_CHARS = 501"),
    # ---- AL-5: the status machine ----
    m(
        "reopen is legal",
        DOMAIN,
        "(AlertStatus.DISMISSED, AlertStatus.SEEN),",
        "(AlertStatus.SEEN, AlertStatus.NEW),",
    ),  # fmt: skip
    m(
        "dismiss from seen is legal",
        DOMAIN,
        "(AlertStatus.SEEN, AlertStatus.DISMISSED),",
        "(AlertStatus.SEEN, AlertStatus.NEW),",
    ),  # fmt: skip
    # ---- AL-6: merging ----
    m(
        "the most severe leads",
        DOMAIN,
        "-severity_rank(member.severity), -member.confidence",
        "severity_rank(member.severity), -member.confidence",
    ),  # fmt: skip
    m(
        "the most confident leads",
        DOMAIN,
        "-severity_rank(member.severity), -member.confidence",
        "-severity_rank(member.severity), member.confidence",
    ),  # fmt: skip
    m(
        "merge window",
        SERVICE,
        "window=self._settings.dedup_window,",
        "window=self._settings.dedup_window / 2,",
    ),  # fmt: skip
    m(
        "bridging alerts",
        SERVICE,
        "if len(targets) > 1:\n            return self._bridge(",
        "if False:\n            return self._bridge(",
    ),  # fmt: skip
    m(
        "bridging status",
        SERVICE,
        "for status in (AlertStatus.NEW, AlertStatus.SEEN, AlertStatus.DISMISSED):",
        "for status in (AlertStatus.DISMISSED, AlertStatus.SEEN, AlertStatus.NEW):",
    ),  # fmt: skip
    m("merge window lower bound", REPO, "m.occurred_at >= ?", "m.occurred_at > ?"),
    m("merge window upper bound", REPO, "m.occurred_at <= ?", "m.occurred_at < ?"),
    # ---- AL-10: idempotence, withdrawal ----
    m(
        "a withdrawn alert never comes back",
        SERVICE,
        "if keep_visible or any(m.active for m in members):",
        "if keep_visible or not any(m.active for m in members):",
    ),  # fmt: skip
    m(
        "the age of an event",
        SERVICE,
        "age = event_age(\n            item.published_at, item.fetched_at, date_known=not _UNDATED.intersection(item.flags)\n        )",
        "age = timedelta(0)",
    ),  # fmt: skip
    m(
        "forgotten items",
        SERVICE,
        "        if sig.matches(verdict.label, verdict.confidence, verdict.settled_at):\n            return True\n",
        "        if False:\n            return True\n",
    ),  # fmt: skip
    m(
        "pending ignores the forgotten",
        REPO,
        '    "AND NOT EXISTS (SELECT 1 FROM alert_forgotten f WHERE f.item_id = c.item_id "\n    f"AND {_SAME_VERDICT})"',
        '    ""',
    ),  # fmt: skip
    m("pending threshold inclusive", REPO, "c.confidence >= ?", "c.confidence > ?"),
    m("changed verdicts", REPO, "OR m.stage <> c.stage ", ""),
    # ---- AL-8: retention ----
    m("retention zero", SERVICE, "if days == 0:", "if days == 1:"),
    m(
        "retention cutoff",
        SERVICE,
        "before=now - timedelta(days=days)",
        "before=now - timedelta(days=days + 1)",
    ),  # fmt: skip
    m(
        "purge is strict",
        REPO,
        'conditions.append("alerts.last_seen_at < ?")',
        'conditions.append("alerts.last_seen_at <= ?")',
    ),  # fmt: skip
    m(
        "purge handled",
        REPO,
        "((alerts.status <> 'new' OR alerts.withdrawn_at IS NOT NULL) AND NOT EXISTS",
        "((alerts.status <> 'new') AND NOT EXISTS",
    ),
    # ---- AL-4: the list ----
    m(
        "since inclusive",
        REPO,
        'clauses.append(f"{column} >= ?")',
        'clauses.append(f"{column} > ?")',
    ),  # fmt: skip
    m(
        "until inclusive",
        REPO,
        'clauses.append(f"{column} <= ?")',
        'clauses.append(f"{column} < ?")',
    ),  # fmt: skip
    m("withdrawn hidden", REPO, 'clauses.append("a.withdrawn_at IS NULL")', 'clauses.append("1")'),
    m(
        "newest first",
        REPO,
        "ORDER BY a.last_seen_at DESC, a.id DESC LIMIT ? OFFSET ?",
        "ORDER BY a.last_seen_at ASC, a.id DESC LIMIT ? OFFSET ?",
    ),  # fmt: skip
    m("source filter", REPO, "(a.source_id = ? OR EXISTS", "(a.source_id <> ? OR EXISTS"),
    # ---- AL-5: the service ----
    m(
        "a redundant request is a no-op",
        SERVICE,
        "if not change.changed:\n                return StatusResult",
        "if False:\n                return StatusResult",
    ),  # fmt: skip
    m(
        "reopen needs a looked-at alert",
        SERVICE,
        "if alert.status is AlertStatus.NEW:\n                raise InvalidTransition",
        "if False:\n                raise InvalidTransition",
    ),  # fmt: skip
    # ---- AL-11: the excerpt ----
    m(
        "excerpt bound inclusive",
        EXCERPT,
        "if len(flat) <= MAX_EXCERPT_CHARS:",
        "if len(flat) < MAX_EXCERPT_CHARS:",
    ),  # fmt: skip
    m(
        "excerpt cut",
        EXCERPT,
        "flat[: MAX_EXCERPT_CHARS - 1] + ELLIPSIS",
        "flat[:MAX_EXCERPT_CHARS] + ELLIPSIS",
    ),  # fmt: skip
    m(
        "excerpt links",
        EXCERPT,
        "text = _URL_RE.sub(_link, text)",
        "text = text",
    ),  # fmt: skip
    m(
        "excerpt cookies",
        EXCERPT,
        'text = _LINE_SECRET_RE.sub(lambda m: f"{m.group(1)}: {REDACTED}", text)',
        "text = text",
    ),  # fmt: skip
    m(
        "excerpt pairs",
        EXCERPT,
        'text = _PAIR_SECRET_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)',
        "text = text",
    ),  # fmt: skip
    m("excerpt hidden characters", EXCERPT, 'return HIDDEN_RE.sub("", window)', "return window"),
    m(
        "excerpt spaces",
        EXCERPT,
        '_SPACES_RE.sub(" ", redact(joined)).strip()',
        "redact(joined).strip()",
    ),
    # ---- calibration ----
    m("sweep inclusive", CALIBRATION, "p.confidence >= threshold", "p.confidence > threshold"),
    m("suggestion prefers the lower", CALIBRATION, "-r.threshold))", "r.threshold))"),
    m("sweep grid", CALIBRATION, "for step in range(14))", "for step in range(13))"),
    m(
        "acceptable labels",
        CALIBRATION,
        "return prediction.label if prediction.label in sample.expected else sample.expected[0]",
        "return sample.expected[0]",
    ),  # fmt: skip
    m(
        "escalation rate",
        CALIBRATION,
        "escalated = sum(1 for _, p in scored if p.escalated)",
        "escalated = 0",
    ),  # fmt: skip
    m(
        "off-topic is out of scope",
        CALIBRATION,
        'return Prediction(Label.OUT_OF_SCOPE, 1.0, False, Stage.TOPIC, "out of scope")',
        'return Prediction(Label.UNCLASSIFIED, 1.0, False, Stage.TOPIC, "out of scope")',
    ),  # fmt: skip
    # ---- the gold set ----
    m(
        "give up after three failures",
        GOLD,
        "streak < MAX_CONSECUTIVE_FAILURES",
        "streak <= MAX_CONSECUTIVE_FAILURES",
    ),  # fmt: skip
    m(
        "only escalated texts are asked", GOLD, "prediction.escalated and streak", "True and streak"
    ),  # fmt: skip
    m(
        "a success resets the failures",
        GOLD,
        "                    streak = 0\n",
        "                    pass\n",
    ),
    m(
        "same text another label", GOLD, "if existing.label is not example.label:", "if False:"
    ),  # fmt: skip
    m("the set is bounded", GOLD, "if self._repo.count() >= self._max:", "if False:"),
    m(
        "gold keeps the order added",
        GOLD_REPO,
        "ORDER BY seq LIMIT ? OFFSET ?",
        "ORDER BY seq DESC LIMIT ? OFFSET ?",
    ),  # fmt: skip
]


if __name__ == "__main__":
    sys.exit(run_sweep(MUTANTS, TESTS, sys.argv[1:]))
