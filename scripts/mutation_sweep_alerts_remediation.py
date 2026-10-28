# ruff: noqa: E501 - a pattern is a piece of the source and keeps the line it has there
"""A mutation sweep of the boundaries added by the second round of the alerts (remediation).

Same idea as `mutation_sweep_alerts.py`: each mutant changes ONE operator, constant or guard of the
production code and the targeted tests must fail; a survivor is a boundary no test pins. It works
on COPIES of the source in temporary folders (see `scripts/mutation_copy.py`): the real tree is
never written to, so an interrupt cannot leave it mutated, and the sweep checks that the real tree
is unchanged when it ends.

    uv run python scripts/mutation_sweep_alerts_remediation.py             # every mutant
    uv run python scripts/mutation_sweep_alerts_remediation.py redact      # names with "redact"
    uv run python scripts/mutation_sweep_alerts_remediation.py --jobs 6    # more copies at once
    uv run python scripts/mutation_sweep_alerts_remediation.py --check     # stale patterns only

Exit status: 0 when every mutant was killed, 1 when some survived or their pattern is stale, 2 when
the targeted tests do not pass on the unmutated copy, 3 when the real tree changed (a bug here).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mutation_copy import m, run_sweep  # noqa: E402  (the script folder is not a package)

TESTS = [
    "tests/alerts",
    "tests/control/test_liveness.py",
    "tests/collector/test_worker_heartbeat.py",
    "tests/shared/test_timezone.py",
    "tests/persistence/test_migration_0012.py",
    "tests/entrypoints/test_cli_alerts.py",
    "tests/entrypoints/test_cli_alerts_remediation.py",
    "tests/entrypoints/test_cli_gold_export_path.py",
    "tests/entrypoints/web/test_alerts_operator.py",
]

DOMAIN = "alerts/domain.py"
EXCERPT = "alerts/excerpt.py"
SERVICE = "alerts/service.py"
REPO = "persistence/alerts_repo.py"
INPUTS = "entrypoints/inputs.py"
TIMEZONE = "shared/timezone.py"
WORKER = "collector/worker.py"

# Known equivalent mutants, left out on purpose (no input can tell them apart): the largest sizes
# (`MAX_FORGOTTEN`, the bounds of the settings that only a hostile environment reaches) and the
# order of the clauses of a WHERE.
MUTANTS = [
    # ---- A1: a rise puts a handled alert back to new ----
    m(
        "rise needs a higher severity",
        SERVICE,
        "if severity_rank(alert.severity) <= severity_rank(basis):",
        "if severity_rank(alert.severity) < severity_rank(basis):",
    ),
    m(
        "rise only for a handled alert",
        SERVICE,
        "if basis is None or alert.withdrawn or alert.status is AlertStatus.NEW:",
        "if basis is None or alert.withdrawn:",
    ),
    m(
        "rise resets the status",
        SERVICE,
        "        rise = self._rise(alert, basis)\n        if rise is not None:",
        "        rise = self._rise(alert, basis)\n        if rise is None:",
    ),
    m(
        "rise is audited",
        SERVICE,
        "        if reason is not None and was is not None:",
        "        if False:",
    ),
    m(
        "merged status follows the highest basis",
        SERVICE,
        "basis = max((a.severity for a in held if a.status is status), key=severity_rank)",
        "basis = min((a.severity for a in held if a.status is status), key=severity_rank)",
    ),
    m(
        "a model upgrade resets too",
        SERVICE,
        '            reason = self._rise(alert, previous.severity, "verdict upgraded" if model_made else "")',
        "            reason = None",
    ),
    # ---- A11: what the model says never hides an alert ----
    m(
        "the model made the verdict",
        SERVICE,
        "old.stage != Stage.LLM.value or old.settled_at != verdict.settled_at",
        "old.stage != Stage.LLM.value and old.settled_at != verdict.settled_at",
    ),
    m(
        "a downgrade is dark or lowered",
        SERVICE,
        "downgrade = model_made and (dark or lowered)",
        "downgrade = model_made and dark",
    ),
    m(
        "lowered is strict",
        SERVICE,
        "severity_rank(projected.severity) < severity_rank(previous.severity)",
        "severity_rank(projected.severity) <= severity_rank(previous.severity)",
    ),
    m(
        "the model confirming clears the flag",
        SERVICE,
        "flag = True if downgrade else (False if model_made else previous.downgraded_by_model)",
        "flag = True if downgrade else previous.downgraded_by_model",
    ),
    m(
        "a dark flagged alert stays visible",
        SERVICE,
        "keep_visible=dark and flag",
        "keep_visible=False",
    ),
    m(
        "a person looking clears the flag",
        SERVICE,
        "alert, status=wanted, status_changed_at=now, downgraded_by_model=False",
        "alert, status=wanted, status_changed_at=now, downgraded_by_model=alert.downgraded_by_model",
    ),
    m(
        "the model's words are shown",
        SERVICE,
        "            alert = replace(alert, rationale=fresh.rationale)  # what the model said",
        "            alert = alert",
    ),
    m(
        "a live copy clears the flag",
        SERVICE,
        "            if lead is None or not lead.active:\n                alert = replace(alert, downgraded_by_model=False)",
        "            if False:\n                alert = replace(alert, downgraded_by_model=False)",
    ),
    # ---- A10: what a model writes is cleaned ----
    m(
        "rationale of a new member",
        SERVICE,
        "            rationale=make_rationale(verdict.rationale),\n            severity=severity,\n            tags=tags_for(",
        "            rationale=verdict.rationale,\n            severity=severity,\n            tags=tags_for(",
    ),
    m(
        "rationale of an updated member",
        SERVICE,
        "            stage=verdict.stage.value,\n            rationale=make_rationale(verdict.rationale),\n            severity=severity if severity is not None else old.severity,",
        "            stage=verdict.stage.value,\n            rationale=verdict.rationale,\n            severity=severity if severity is not None else old.severity,",
    ),
    m(
        "evidence strings",
        SERVICE,
        "            settle_reason=_tidy(verdict.settle_reason),",
        "            settle_reason=verdict.settle_reason,",
    ),
    m(
        "model name",
        SERVICE,
        "            llm_model=_tidy(judgment.llm_model) if judged and judgment else None,",
        "            llm_model=judgment.llm_model if judged and judgment else None,",
    ),
    m("redacted text is cut after it grows", EXCERPT, "    if len(flat) > limit:", "    if False:"),
    # ---- A8: caps ----
    m(
        "member cap",
        SERVICE,
        "if self._alerts.count_members(previous.id) >= self._settings.max_members:",
        "if self._alerts.count_members(previous.id) > self._settings.max_members:",
    ),
    m(
        "the lead is kept",
        SERVICE,
        "victim = self._alerts.weakest_member(previous.id, protect=previous.item_id)",
        "victim = self._alerts.weakest_member(previous.id, protect=None)",
    ),
    m(
        "a full alert counts what it gave up",
        SERVICE,
        "alert = replace(alert, overflow=previous.overflow + gave_up)",
        "alert = replace(alert, overflow=previous.overflow)",
    ),
    m(
        "a copy only counted is counted",
        SERVICE,
        "occurrences=previous.occurrences + 1,\n                overflow=previous.overflow + 1,",
        "occurrences=previous.occurrences + 1,\n                overflow=previous.overflow,",
    ),
    m(
        "an overflow item is remembered",
        SERVICE,
        "if self._alerts.overflow_of(item_id) is not None:",
        "if False:",
    ),
    m(
        "the routine trail cap",
        SERVICE,
        "(kind in _ROUTINE and stored >= cap)",
        "(kind in _ROUTINE and stored > cap)",
    ),
    m("the hard trail cap", SERVICE, "if stored >= 2 * cap or", "if stored >= 4 * cap or"),
    m(
        "trail entries left out are counted",
        SERVICE,
        "            self._alerts.drop_events(alert_id, 1)\n            return",
        "            return",
    ),
    m(
        "the trail shown",
        SERVICE,
        "hidden = stored - len(events) + alert.events_dropped",
        "hidden = 0",
    ),
    m(
        "an incremental merge keeps the lead",
        DOMAIN,
        "leads = restart or lead_key(member) < lead_key(lead)",
        "leads = restart or lead_key(member) > lead_key(lead)",
    ),
    m(
        "an incremental merge counts",
        DOMAIN,
        "occurrences=(1 + previous.overflow) if restart else previous.occurrences + 1,",
        "occurrences=previous.occurrences,",
    ),
    m(
        "an incremental merge keeps the times",
        DOMAIN,
        "first_seen_at=min(previous.first_seen_at, member.occurred_at),",
        "first_seen_at=previous.first_seen_at,",
    ),
    m("a copy joins the tags", DOMAIN, "    tags |= set(member.tags)\n", "    tags |= set()\n"),
    m(
        "member cap lower bound",
        DOMAIN,
        "_whole(self.max_members, 1, MAX_MEMBERS_LIMIT)",
        "_whole(self.max_members, 0, MAX_MEMBERS_LIMIT)",
    ),
    m(
        "event cap lower bound",
        DOMAIN,
        "_whole(self.max_events, 10, MAX_EVENTS_LIMIT)",
        "_whole(self.max_events, 9, MAX_EVENTS_LIMIT)",
    ),
    m(
        "weakest is the inactive first",
        REPO,
        "ORDER BY active, {_SEVERITY_RANK}, confidence, occurred_at DESC, item_id LIMIT 1",
        "ORDER BY active DESC, {_SEVERITY_RANK}, confidence, occurred_at DESC, item_id LIMIT 1",
    ),
    m(
        "weakest is the least severe",
        REPO,
        "ORDER BY active, {_SEVERITY_RANK}, confidence, occurred_at DESC, item_id LIMIT 1",
        "ORDER BY active, {_SEVERITY_RANK} DESC, confidence, occurred_at DESC, item_id LIMIT 1",
    ),
    m(
        "the first member stays",
        REPO,
        'AND item_id <> (SELECT item_id FROM alert_items WHERE alert_id = ? "\n            "ORDER BY occurred_at, item_id LIMIT 1) "',
        'AND ? IS NOT NULL "',
    ),
    # ---- A12: links, absorbed trails, order ----
    m(
        "a copy is found from its original",
        SERVICE,
        "            window=self._settings.dedup_window,\n            item_id=item.id,\n",
        "            window=self._settings.dedup_window,\n",
    ),
    m(
        "the trail of an absorbed alert",
        SERVICE,
        "            left_out = self._alerts.copy_events(other_id, keep, limit=room)",
        "            left_out = 0",
    ),
    m(
        "reverse links by near copy",
        REPO,
        'for key in ("near_of", "duplicate_of"):',
        'for key in ("duplicate_of",):',
    ),
    m(
        "reverse links by duplicate",
        REPO,
        'for key in ("near_of", "duplicate_of"):',
        'for key in ("near_of",):',
    ),
    m(
        "a given up member is found by content",
        REPO,
        '"SELECT o.alert_id FROM alert_overflow o WHERE o.content_hash = ? "',
        '"SELECT o.alert_id FROM alert_overflow o WHERE o.content_hash <> ? "',
    ),
    m(
        "the sync is chronological",
        REPO,
        "WHERE (u.ts, u.item_id) > (?, ?) {waiting} ORDER BY u.ts, u.item_id LIMIT ?",
        "WHERE (u.ts, u.item_id) > (?, ?) {waiting} ORDER BY u.item_id LIMIT ?",
    ),
    m(
        "the dirty list is chronological",
        REPO,
        '"ORDER BY ts, d.item_id LIMIT ?"',
        '"ORDER BY d.item_id LIMIT ?"',
    ),
    m(
        "absorbed trail keeps what matters",
        REPO,
        "ORDER BY (kind IN ('merged', 'updated')), seq LIMIT ?",
        "ORDER BY seq LIMIT ?",
    ),
    m(
        "absorbed trail is marked",
        REPO,
        "SELECT ?, at, kind, actor, reason, from_value, to_value, 1 FROM alert_events",
        "SELECT ?, at, kind, actor, reason, from_value, to_value, 0 FROM alert_events",
    ),
    # ---- A2: forgetting a verdict ----
    m(
        "a forgotten verdict stays forgotten",
        SERVICE,
        "        if sig.matches(verdict.label, verdict.confidence, verdict.settled_at):\n            return True\n",
        "        if False:\n            return True\n",
    ),
    m(
        "a changed verdict clears the memory",
        SERVICE,
        "        self._alerts.clear_forgotten(item_id)\n        return False",
        "        return False",
    ),
    m(
        "the signature compares the confidence",
        DOMAIN,
        "            and confidence == self.confidence\n",
        "            and True\n",
    ),
    m(
        "the signature compares the settlement",
        DOMAIN,
        "            and settled_at == self.settled_at\n",
        "            and True\n",
    ),
    m(
        "an old entry suppresses",
        DOMAIN,
        "        if self.label is None:\n            return True",
        "        if self.label is None:\n            return False",
    ),
    m(
        "the memory lasts a year",
        REPO,
        "FORGOTTEN_KEEP = timedelta(days=365)",
        "FORGOTTEN_KEEP = timedelta(days=364)",
    ),
    m(
        "the memory is dropped by age",
        REPO,
        '"DELETE FROM alert_forgotten WHERE forgotten_at < ?"',
        '"DELETE FROM alert_forgotten WHERE forgotten_at > ?"',
    ),
    m(
        "the purge remembers the signature",
        REPO,
        'SELECT item_id, label, confidence, settled_at FROM alert_items "',
        'SELECT item_id, NULL, NULL, NULL FROM alert_items "',
    ),
    m(
        "a waiting alert is not purged",
        REPO,
        "AND json_extract(w.evidence, '$.provisional') = 1))",
        "AND json_extract(w.evidence, '$.provisional') = 0))",
    ),
    m(
        "only an active waiting member spares",
        REPO,
        'WHERE w.alert_id = alerts.id AND w.active = 1 "',
        'WHERE w.alert_id = alerts.id AND w.active = 0 "',
    ),
    m(
        "the spare is for the handled purge only",
        REPO,
        "        if handled:\n            conditions.append(\n                \"((alerts.status <> 'new' OR alerts.withdrawn_at IS NOT NULL) AND NOT EXISTS \"",
        "        if handled:\n            conditions.append(\n                \"((alerts.status <> 'new' OR alerts.withdrawn_at IS NOT NULL) OR NOT EXISTS \"",
    ),
    # ---- A14: a poison item ----
    m(
        "a failure is recorded",
        SERVICE,
        "            failures = self._alerts.record_failure(",
        "            failures = PARK_AFTER - 1 if True else self._alerts.record_failure(",
    ),
    m(
        "an item that works is forgiven",
        SERVICE,
        "            self._alerts.clear_failure(item_id)\n",
        "            pass\n",
    ),
    m(
        "a looked-at item is not dirty",
        SERVICE,
        "            self._alerts.clear_dirty(item_id)\n",
        "            pass\n",
    ),
    m("parked at the limit", REPO, "int(failures >= park_after),", "int(failures > park_after),"),
    m(
        "backoff doubles",
        REPO,
        "delay = min(backoff_base * (2 ** min(failures - 1, 30)), backoff_cap)",
        "delay = min(backoff_base * failures, backoff_cap)",
    ),
    m(
        "backoff is capped",
        REPO,
        "delay = min(backoff_base * (2 ** min(failures - 1, 30)), backoff_cap)",
        "delay = backoff_base * (2 ** min(failures - 1, 30))",
    ),
    m(
        "backoff ends exactly",
        REPO,
        "AND (x.parked = 1 OR x.next_try_at > ?))",
        "AND (x.parked = 1 OR x.next_try_at >= ?))",
    ),
    m(
        "parked items wait for a rebuild",
        REPO,
        "AND (x.parked = 1 OR x.next_try_at > ?))",
        "AND (x.next_try_at > ?))",
    ),
    m(
        "a rebuild retries at once",
        REPO,
        "SET parked = 0, next_try_at = last_failed_at",
        "SET parked = 0",
    ),
    m(
        "a rebuild releases the parked",
        SERVICE,
        "            self._alerts.release_parked()",
        "            pass",
    ),
    m(
        "the failing items are counted",
        SERVICE,
        "return replace(counts, retrying=retrying, parked=parked)",
        "return counts",
    ),
    # ---- A6: the threshold ----
    m(
        "a moved threshold sweeps",
        SERVICE,
        "sweep = full or self._control.alerts_synced_threshold() != threshold",
        "sweep = full",
    ),
    m(
        "a clean sweep marks the threshold",
        SERVICE,
        'self._remember(threshold if sweep and not tally["errors"] else None)',
        "self._remember(threshold if sweep else None)",
    ),
    m(
        "a refresh leaves its time",
        SERVICE,
        "            self._control.record_alert_refresh()\n",
        "            pass\n",
    ),
    m(
        "a normal pass reads the dirty marks",
        SERVICE,
        "                refs = self._alerts.dirty_ids(after=after, limit=batch, now=now)",
        "                refs = self._alerts.sync_ids(\n                    min_confidence=threshold, full=False, after=after, limit=batch, now=now\n                )",
    ),
    m(
        "threshold set applies at once",
        "entrypoints/cli.py",
        "report = services.alerts.rebuild() if change.changed else None",
        "report = None",
    ),
    # ---- A3: the age of an event ----
    m(
        "an unknown age is not fresh",
        DOMAIN,
        "    if not date_known:\n        return UNKNOWN_AGE",
        "    if not date_known:\n        return timedelta(0)",
    ),
    m(
        "a missing date is unknown",
        SERVICE,
        '_UNDATED = frozenset({"published_at_missing", "published_at_future"})',
        '_UNDATED = frozenset({"published_at_future"})',
    ),
    m(
        "a future date is unknown",
        SERVICE,
        '_UNDATED = frozenset({"published_at_missing", "published_at_future"})',
        '_UNDATED = frozenset({"published_at_missing"})',
    ),
    m(
        "an unknown age has no publication time",
        DOMAIN,
        "        if self.event_age_s >= UNKNOWN_AGE.total_seconds():",
        "        if self.event_age_s > UNKNOWN_AGE.total_seconds():",
    ),
    # ---- A5, A7: the operator pages and dates ----
    m(
        "the worker leaves its time",
        WORKER,
        "            self._control.record_worker_pass()",
        "            pass",
    ),
    m(
        "a day ends at its last instant",
        DOMAIN,
        "day = day.replace(hour=23, minute=59, second=59, microsecond=999999)",
        "day = day.replace(hour=23, minute=59, second=59, microsecond=0)",
    ),
    m(
        "a date is read in the zone",
        DOMAIN,
        "ensure_utc(moment.replace(tzinfo=zone) if moment.tzinfo is None else moment)",
        "ensure_utc(moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment)",
    ),
    m(
        "an offset beats the zone",
        DOMAIN,
        "ensure_utc(moment.replace(tzinfo=zone) if moment.tzinfo is None else moment)",
        "ensure_utc(moment.replace(tzinfo=zone))",
    ),
    m(
        "the first sighting can be chosen",
        REPO,
        '_WHEN_COLUMN = {"first_seen": "a.first_seen_at", "last_seen": "a.last_seen_at"}',
        '_WHEN_COLUMN = {"first_seen": "a.last_seen_at", "last_seen": "a.last_seen_at"}',
    ),
    m(
        "provisional can be hidden",
        REPO,
        "COALESCE(json_extract(a.evidence, '$.provisional'), 0) = 0",
        "COALESCE(json_extract(a.evidence, '$.provisional'), 0) = 1",
    ),
    m("withdrawn can be shown", REPO, "        if not flt.include_withdrawn:", "        if True:"),
    m(
        "the zone has a default",
        "shared/timezone.py",
        'DEFAULT_TIMEZONE = "America/Bogota"',
        'DEFAULT_TIMEZONE = "UTC"',
    ),
    m(
        "a naive time is not guessed",
        TIMEZONE,
        "    local = ensure_utc(moment).astimezone(zone)\n    offset = local.utcoffset()",
        "    local = moment.astimezone(zone)\n    offset = local.utcoffset()",
    ),
    m(
        "the compact form knows UTC",
        TIMEZONE,
        '    if minutes == 0:\n        suffix = "Z"',
        '    if minutes == 1:\n        suffix = "Z"',
    ),
    m(
        "a page past the end is empty",
        "entrypoints/web/alert_routes.py",
        "        if offset > MAX_OFFSET:",
        "        if offset > MAX_OFFSET * 1000:",
    ),
    # ---- A9: redaction ----
    m(
        "a random token needs 24 characters",
        EXCERPT,
        "len(token) >= _MIN_RANDOM_TOKEN",
        "len(token) >= _MIN_RANDOM_TOKEN + 1",
    ),
    m(
        "a random token has three classes",
        EXCERPT,
        "and _classes(token) == 3\n        and _CAMEL_WORDS_RE",
        "and _classes(token) >= 2\n        and _CAMEL_WORDS_RE",
    ),
    m(
        "camel case words are prose",
        EXCERPT,
        "and _CAMEL_WORDS_RE.fullmatch(token) is None",
        "and True",
    ),
    m("a random token has entropy", EXCERPT, "and _entropy(token) >= _MIN_ENTROPY", "and True"),
    m("a path token needs 16 characters", EXCERPT, "_MIN_PATH_TOKEN = 16", "_MIN_PATH_TOKEN = 17"),
    m(
        "a hex path token",
        EXCERPT,
        "        return any(c.isdigit() for c in segment) and any(c.isalpha() for c in segment)",
        "        return False",
    ),
    m(
        "link credentials of any scheme",
        EXCERPT,
        '    text = _USERINFO_RE.sub(r"\\1", text)\n',
        "    text = text\n",
    ),
    m(
        "a link path loses tokens",
        EXCERPT,
        "    text = _URL_RE.sub(_link, text)",
        "    text = _URL_RE.sub(lambda m: m.group(1) + m.group(2) + m.group(3), text)",
    ),
    m(
        "a spoken secret",
        EXCERPT,
        "    text = _SPOKEN_SECRET_RE.sub(_spoken, text)\n",
        "    text = text\n",
    ),
    m("a spoken secret needs 6 characters", EXCERPT, "[^\\s;,&]{{6,}})", "[^\\s;,&]{{7,}})"),
    m(
        "a spoken word is not a secret",
        EXCERPT,
        "    secret = quoted or (",
        "    secret = True or (",
    ),
    m(
        "bearer needs 6 characters",
        EXCERPT,
        "bearer\\s+([A-Za-z0-9._~+/=-]{6,})",
        "bearer\\s+([A-Za-z0-9._~+/=-]{7,})",
    ),
    m(
        "a long bearer value",
        EXCERPT,
        "    if len(value) >= 10 or any(",
        "    if len(value) >= 11 or any(",
    ),
    m("vendor keys", EXCERPT, "    text = _VENDOR_RE.sub(REDACTED, text)\n", "    text = text\n"),
    m(
        "github keys",
        EXCERPT,
        'r"\\bgh[pousr]_[A-Za-z0-9]{20,}"',
        'r"\\bgh[pousr]_[A-Za-z0-9]{40,}"',
    ),
    m("aws keys", EXCERPT, 'r"\\b(?:AKIA|ASIA)[0-9A-Z]{16}\\b"', 'r"\\b(?:AKIA)[0-9A-Z]{16}\\b"'),
    m("google keys", EXCERPT, 'r"\\bAIza[0-9A-Za-z_-]{30,}"', 'r"\\bAIza[0-9A-Za-z_-]{40,}"'),
    m(
        "slack tokens",
        EXCERPT,
        'r"\\bxox[abprs]-[A-Za-z0-9-]{10,}"',
        'r"\\bxox[abpr]-[A-Za-z0-9-]{10,}"',
    ),
    m("openai keys", EXCERPT, 'r"\\bsk-[A-Za-z0-9_-]{16,}"', 'r"\\bsk-[A-Za-z0-9]{16,}"'),
    # ---- A17: the export path ----
    m(
        "a git file marks a work tree",
        INPUTS,
        '        if (candidate / ".git").exists():',
        '        if (candidate / ".git").is_dir():',
    ),
    m(
        "dot dot cannot hide a work tree",
        INPUTS,
        "    folder = path.resolve().parent",
        "    folder = path.absolute().parent",
    ),
    m(
        "an export in a work tree is refused",
        "entrypoints/cli.py",
        "            if in_tree and not allow_repo_path:",
        "            if in_tree and allow_repo_path:",
    ),
    m(
        "the calibration says what it measured",
        "entrypoints/cli.py",
        '    lines.append(f"measured with {CALIBRATION_CAVEAT}"',
        '    lines.append(f"measured {CALIBRATION_CAVEAT}"',
    ),
]

if __name__ == "__main__":
    sys.exit(run_sweep(MUTANTS, TESTS, sys.argv[1:]))
