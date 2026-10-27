"""A mutation sweep of the boundaries added by the remediation round of the LLM escalation.

Each mutant changes ONE operator or constant of the production code; the tests that target it must
fail. A mutant that leaves them green is a survivor: a boundary no test pins. The source is always
restored, even on an interrupt.

    uv run python scripts/mutation_sweep_llm_remediation.py            # every mutant
    uv run python scripts/mutation_sweep_llm_remediation.py backoff    # names containing "backoff"

Each mutant runs only the test files of its area (`AREAS`), the real-socket deadline tests
included, so a full sweep takes minutes, not hours. Exit status: 0 when every mutant was killed, 1
when some survived, 2 when the targeted tests do not pass on the unmutated code.
"""

# ruff: noqa: E501 (the patterns of the mutants are source lines and cannot be wrapped)

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "social_tracker"

REPOS = [
    "tests/llm/test_fake_repositories.py",
    "tests/persistence/test_sqlite_escalation_repository.py",
    "tests/persistence/test_sqlite_classification_repositories.py",
    "tests/classification/test_fake_repositories.py",
    "tests/persistence/test_purge_llm_records.py",
]
AREAS = {
    "domain": [
        "tests/llm/test_remediation_domain.py",
        "tests/llm/test_remediation_verdicts.py",
        "tests/llm/test_domain.py",
        "tests/llm/test_remediation_service.py",
    ],
    "prompt": ["tests/llm/test_remediation_prompt.py", "tests/llm/test_prompt.py"],
    "schema": ["tests/llm/test_remediation_schema.py", "tests/llm/test_schema.py"],
    "ollama": [
        "tests/llm/test_remediation_adapter.py",
        "tests/llm/test_ollama.py",
        "tests/llm/test_ollama_proxy.py",
    ],
    "service": [
        "tests/llm/test_remediation_service.py",
        "tests/llm/test_service_failures.py",
        "tests/llm/test_service_flow.py",
        "tests/llm/test_service_links.py",
        "tests/llm/test_service_edges.py",
    ],
    "repos": REPOS,
    "classifier": ["tests/classification/test_service_fold_duplicates.py"],
}


@dataclass(frozen=True)
class Mutant:
    name: str
    area: str
    path: str
    old: str
    new: str


def m(name: str, area: str, path: str, old: str, new: str) -> Mutant:
    return Mutant(name, area, path, old, new)


DOMAIN = "llm/domain.py"
PROMPT = "llm/prompt.py"
SCHEMA = "llm/schema.py"
SERVICE = "llm/service.py"
OLLAMA = "llm/ollama.py"
LLM_REPO = "persistence/llm_repo.py"
CLASS_REPO = "persistence/classification_repo.py"
ITEMS_REPO = "persistence/items_repo.py"
CLASSIFIER = "classification/service.py"

# Known equivalent mutants, left out on purpose (no input can tell them apart):
# - `keep = min(keep - 1, keep - over)` of the prompt cut: `keep - 1` alone reaches the same cut,
#   only more slowly;
# - the exact number of seconds of the margin of a lease beyond the longest judgment.
MUTANTS = [
    # ---- settings ----
    m(
        "settings room for the prompt",
        "domain",
        DOMAIN,
        "and self.num_ctx - self.max_output_tokens >= MIN_PROMPT_TOKENS",
        "and self.num_ctx - self.max_output_tokens > MIN_PROMPT_TOKENS",
    ),
    m(
        "settings num_ctx floor",
        "domain",
        DOMAIN,
        "_whole(self.num_ctx, 2048, 262_144)",
        "_whole(self.num_ctx, 2049, 262_144)",
    ),
    m(
        "settings num_ctx ceiling",
        "domain",
        DOMAIN,
        "_whole(self.num_ctx, 2048, 262_144)",
        "_whole(self.num_ctx, 2048, 262_145)",
    ),
    m(
        "settings load deadline ceiling",
        "domain",
        DOMAIN,
        "_number(self.load_deadline_s, 0.0, 1800.0)",
        "_number(self.load_deadline_s, 0.0, 1801.0)",
    ),
    m(
        "settings load deadline positive",
        "domain",
        DOMAIN,
        "and self.load_deadline_s > 0",
        "and self.load_deadline_s >= 0",
    ),
    m(
        "settings pass budget positive",
        "domain",
        DOMAIN,
        "and self.pass_budget_s > 0",
        "and self.pass_budget_s >= 0",
    ),
    m(
        "settings ttl floor",
        "domain",
        DOMAIN,
        "_whole(self.parked_ttl_s, 60, 2_592_000)",
        "_whole(self.parked_ttl_s, 59, 2_592_000)",
    ),
    m(
        "settings share positive",
        "domain",
        DOMAIN,
        "and self.source_share > 0",
        "and self.source_share >= 0",
    ),
    m(
        "settings share ceiling",
        "domain",
        DOMAIN,
        "_number(self.source_share, 0.0, 1.0)",
        "_number(self.source_share, 0.0, 1.5)",
    ),
    m(
        "cold deadline is the longer one",
        "domain",
        DOMAIN,
        "return max(self.load_deadline_s, self.call_deadline_s)",
        "return min(self.load_deadline_s, self.call_deadline_s)",
    ),
    m(
        "longest judgment adds the retry",
        "domain",
        DOMAIN,
        "return self.cold_deadline_s + self.call_deadline_s",
        "return self.cold_deadline_s",
    ),
    m(
        "source cap at least one",
        "domain",
        DOMAIN,
        "return max(1, int(self.queue_max * self.source_share))",
        "return int(self.queue_max * self.source_share)",
    ),
    # ---- who is to blame ----
    m(
        "a status makes a bad response the judge's",
        "domain",
        DOMAIN,
        "self.kind is JudgeErrorKind.BAD_RESPONSE and self.status is not None\n        )\n\n    @property\n    def item_specific",
        "self.kind is JudgeErrorKind.BAD_RESPONSE and self.status is None\n        )\n\n    @property\n    def item_specific",
    ),
    m(
        "a load timeout is the judge's",
        "domain",
        DOMAIN,
        "        JudgeErrorKind.LOAD_TIMEOUT,\n        JudgeErrorKind.SERVER_ERROR,",
        "        JudgeErrorKind.SERVER_ERROR,",
    ),
    m(
        "a 5xx is the judge's",
        "domain",
        DOMAIN,
        "        JudgeErrorKind.SERVER_ERROR,\n        JudgeErrorKind.MODEL_NOT_FOUND,",
        "        JudgeErrorKind.MODEL_NOT_FOUND,",
    ),
    m(
        "an invalid output is not an item failure",
        "domain",
        DOMAIN,
        "return not self.judge_level and self.kind is not JudgeErrorKind.INVALID_OUTPUT",
        "return not self.judge_level",
    ),
    m(
        "retry hint needs an attempt",
        "domain",
        DOMAIN,
        'and (self.attempt > 0 or self.previous_problem == "")',
        "and True",
    ),
    # ---- how a judgment lands ----
    m(
        "confidence cap",
        "domain",
        DOMAIN,
        "MAX_LLM_CONFIDENCE: Final = 0.95",
        "MAX_LLM_CONFIDENCE: Final = 0.96",
    ),
    m(
        "suspicious cap",
        "domain",
        DOMAIN,
        "SUSPICIOUS_NO_RISK_CAP: Final = 0.7",
        "SUSPICIOUS_NO_RISK_CAP: Final = 0.71",
    ),
    m(
        "strong evidence threshold",
        "domain",
        DOMAIN,
        "STRONG_EVIDENCE: Final = 0.6",
        "STRONG_EVIDENCE: Final = 0.61",
    ),
    m(
        "prior verdict strength inclusive",
        "domain",
        DOMAIN,
        "if prior.label in _RISK and prior.confidence >= STRONG_EVIDENCE:",
        "if prior.label in _RISK and prior.confidence > STRONG_EVIDENCE:",
    ),
    m(
        "signal strength inclusive",
        "domain",
        DOMAIN,
        "if signal.label in _RISK and signal.weight >= STRONG_EVIDENCE:",
        "if signal.label in _RISK and signal.weight > STRONG_EVIDENCE:",
    ),
    m(
        "obfuscation is suspicious",
        "domain",
        DOMAIN,
        '_SUSPICIOUS_SIGNALS: Final = frozenset({"obfuscated_text"})',
        "_SUSPICIOUS_SIGNALS: Final = frozenset()",
    ),
    m(
        "catalog mismatch is suspicious",
        "domain",
        DOMAIN,
        '    if prior.catalog_status == "mismatch":\n        return True',
        "    if False:\n        return True",
    ),
    m(
        "catalog mismatch signal is suspicious",
        "domain",
        DOMAIN,
        '        if signal.name == "catalog_mismatch":\n            return True\n',
        "",
    ),
    m(
        "injection signal is suspicious",
        "domain",
        DOMAIN,
        'or signal.name.startswith("injection")',
        "or False",
    ),
    m(
        "only a no_risk contradicts",
        "domain",
        DOMAIN,
        "if answer.label is not Label.NO_RISK:\n        return False",
        "if False:\n        return False",
    ),
    m(
        "a contradiction is flagged",
        "domain",
        DOMAIN,
        "low_confidence=flagged or llm_low_confidence(confidence, alert_min),",
        "low_confidence=llm_low_confidence(confidence, alert_min),",
    ),
    m(
        "a requeued copy never has priority zero",
        "domain",
        DOMAIN,
        "priority=max(priority, REQUEUED_PRIORITY_FLOOR),",
        "priority=priority,",
    ),
    m(
        "a released copy is low confidence",
        "domain",
        DOMAIN,
        "duplicate, awaiting_original=False, low_confidence=True, **_settlement(reason, now)",
        "duplicate, awaiting_original=False, **_settlement(reason, now)",
    ),
    m(
        "a reused copy inherits the low confidence",
        "domain",
        DOMAIN,
        "low_confidence=original.low_confidence\n        or llm_low_confidence(original.confidence, alert_min),",
        "low_confidence=llm_low_confidence(original.confidence, alert_min),",
    ),
    # ---- the prompt ----
    m(
        "token estimate rounds up",
        "prompt",
        PROMPT,
        "return -(-chars * 2 // 5) if CHARS_PER_TOKEN == 2.5 else int(chars / CHARS_PER_TOKEN) + 1",
        "return (chars * 2 // 5) if CHARS_PER_TOKEN == 2.5 else int(chars / CHARS_PER_TOKEN) + 1",
    ),
    m(
        "prompt budget in characters",
        "prompt",
        PROMPT,
        "budget = max(0, num_ctx - output_tokens) * 5 // 2",
        "budget = max(0, num_ctx - output_tokens) * 2",
    ),
    m(
        "a prompt that fits is untouched",
        "prompt",
        PROMPT,
        "if _prompt_chars(probe) <= budget and _prompt_chars(request) <= budget:",
        "if _prompt_chars(probe) < budget and _prompt_chars(request) < budget:",
    ),
    m(
        "room for the retry is kept",
        "prompt",
        PROMPT,
        'probe = replace(request, text_truncated=True, attempt=1, previous_problem="not_object")',
        "probe = replace(request, text_truncated=True)",
    ),
    m(
        "a retry appends a message",
        "prompt",
        PROMPT,
        "if request.attempt > 0:\n        messages.append",
        "if request.attempt > 5:\n        messages.append",
    ),
    m(
        "the correction names the problem",
        "prompt",
        PROMPT,
        "(problem: {problem or 'invalid'})",
        "(problem: invalid)",
    ),
    # ---- the schema ----
    m(
        "empty answers are their own class",
        "schema",
        SCHEMA,
        'raise _invalid("empty", "empty or non-text output", raw)',
        'raise _invalid("not_json", "empty or non-text output", raw)',
    ),
    m(
        "schema errors are their own class",
        "schema",
        SCHEMA,
        'raise _invalid("schema", "does not match the schema", raw) from None',
        'raise _invalid("not_json", "does not match the schema", raw) from None',
    ),
    # ---- the adapter ----
    m(
        "output budget cap",
        "ollama",
        OLLAMA,
        "_MAX_PREDICT: Final = 8192",
        "_MAX_PREDICT: Final = 8193",
    ),
    m(
        "a cut answer gets double the room",
        "ollama",
        OLLAMA,
        "predict = min(2 * predict, _MAX_PREDICT, max(predict, room))",
        "predict = min(3 * predict, _MAX_PREDICT, max(predict, room))",
    ),
    m(
        "only a cut answer gets more room",
        "ollama",
        OLLAMA,
        'if request.previous_problem == "cut":',
        'if request.previous_problem != "cut":',
    ),
    m(
        "extra room stays inside the window",
        "ollama",
        OLLAMA,
        "predict = min(2 * predict, _MAX_PREDICT, max(predict, room))",
        "predict = min(2 * predict, _MAX_PREDICT)",
    ),
    m(
        "a retry has another seed",
        "ollama",
        OLLAMA,
        '"seed": JUDGE_SEED + request.attempt,',
        '"seed": JUDGE_SEED,',
    ),
    m("the first seed", "domain", DOMAIN, "JUDGE_SEED: Final = 0", "JUDGE_SEED: Final = 1"),
    m(
        "think is always off",
        "ollama",
        OLLAMA,
        '        if model not in self._no_think:\n            payload["think"] = False',
        "        pass",
    ),
    m(
        "think is remembered",
        "ollama",
        OLLAMA,
        "            self._no_think.add(model)\n",
        "            pass\n",
    ),
    m(
        "a 422 also rejects think",
        "ollama",
        OLLAMA,
        "_THINK_STATUSES: Final = (400, 422)",
        "_THINK_STATUSES: Final = (400,)",
    ),
    m(
        "only a body that names think",
        "ollama",
        OLLAMA,
        'and status in _THINK_STATUSES and "think" in lowered:',
        "and status in _THINK_STATUSES:",
    ),
    m(
        "think is a chat matter",
        "ollama",
        OLLAMA,
        "if path == _CHAT and status in _THINK_STATUSES",
        "if status in _THINK_STATUSES",
    ),
    m(
        "the model is cold at the instant it expires",
        "ollama",
        OLLAMA,
        "return now >= self._warm_until.get(model, 0.0)",
        "return now > self._warm_until.get(model, 0.0)",
    ),
    m(
        "a call keeps the model loaded",
        "ollama",
        OLLAMA,
        "self._warm_until[model] = self._monotonic() + self._settings.keep_alive_s",
        "self._warm_until[model] = self._monotonic()",
    ),
    m(
        "a cold call gets the load deadline",
        "ollama",
        OLLAMA,
        "budget = _Budget(start + (settings.cold_deadline_s if cold else settings.call_deadline_s))",
        "budget = _Budget(start + settings.call_deadline_s)",
    ),
    m(
        "the fallback shares the hard cap",
        "ollama",
        OLLAMA,
        "budget = _Budget(start + settings.cold_deadline_s, cold=True)",
        "budget = _Budget(start + settings.call_deadline_s, cold=True)",
    ),
    m(
        "a timeout while loading is its own kind",
        "ollama",
        OLLAMA,
        "kind = JudgeErrorKind.LOAD_TIMEOUT if budget.cold else JudgeErrorKind.TIMEOUT",
        "kind = JudgeErrorKind.TIMEOUT",
    ),
    m(
        "only a 404 or an unloadable model is a missing model",
        "ollama",
        OLLAMA,
        "if path == _CHAT and (status == 404 or (status >= 500 and names_model)):",
        "if path == _CHAT and (status == 404 or names_model):",
    ),
    m(
        "a 404 is a missing model",
        "ollama",
        OLLAMA,
        "if path == _CHAT and (status == 404 or (status >= 500 and names_model)):",
        "if path == _CHAT and (status >= 500 and names_model):",
    ),
    m(
        "a redirect carries its status",
        "ollama",
        OLLAMA,
        '"a redirect was refused", status=status)',
        '"a redirect was refused")',
    ),
    m(
        "an error carries its status",
        "ollama",
        OLLAMA,
        'raise _error(JudgeErrorKind.BAD_RESPONSE, f"the server answered {status}", status=status)',
        'raise _error(JudgeErrorKind.BAD_RESPONSE, f"the server answered {status}")',
    ),
    m(
        "a cut answer is a class of its own",
        "ollama",
        OLLAMA,
        '                problem="cut",',
        '                problem="schema",',
    ),
    m(
        "the banner must be exact",
        "ollama",
        OLLAMA,
        'if self._exchange("GET", _ROOT, None, budget).strip() == _ROOT_BANNER:',
        'if self._exchange("GET", _ROOT, None, budget).strip().startswith(_ROOT_BANNER):',
    ),
    m(
        "a version must be text",
        "ollama",
        OLLAMA,
        "if isinstance(version, str) and version:",
        "if version:",
    ),
    m(
        "identify passes real failures on",
        "ollama",
        OLLAMA,
        "        except JudgeError as exc:\n            if exc.kind is not JudgeErrorKind.BAD_RESPONSE:\n                raise\n        try:\n            data = json.loads",
        "        except JudgeError:\n            pass\n        try:\n            data = json.loads",
    ),
    # ---- the service ----
    m(
        "the success window",
        "service",
        SERVICE,
        "SUCCESS_WINDOW = timedelta(hours=1)",
        "SUCCESS_WINDOW = timedelta(hours=2)",
    ),
    m(
        "a success in this pass vouches",
        "service",
        SERVICE,
        "return tally.judged > 0 or self._escalations.judged_since(now - SUCCESS_WINDOW) > 0",
        "return self._escalations.judged_since(now - SUCCESS_WINDOW) > 0",
    ),
    m(
        "a recent success vouches",
        "service",
        SERVICE,
        "return tally.judged > 0 or self._escalations.judged_since(now - SUCCESS_WINDOW) > 0",
        "return tally.judged > 0",
    ),
    m("the breaker", "service", SERVICE, "BREAKER = 2", "BREAKER = 3"),
    m(
        "a failed item waits behind the judge",
        "service",
        SERVICE,
        "step = attempts + 1 if behind else attempts",
        "step = attempts",
    ),
    m(
        "nothing there is not the item's",
        "service",
        SERVICE,
        "{JudgeErrorKind.UNREACHABLE, JudgeErrorKind.MODEL_NOT_FOUND, JudgeErrorKind.NOT_OLLAMA}",
        "{JudgeErrorKind.MODEL_NOT_FOUND, JudgeErrorKind.NOT_OLLAMA}",
    ),
    m(
        "a failed call is refunded",
        "service",
        SERVICE,
        "            self._escalations.release_call(refund)",
        "            pass",
    ),
    m(
        "a judge-level failure refunds",
        "service",
        SERVICE,
        "return self._judge_failed(current, exc.code, tally, refund=hour, behind=not nowhere)",
        "return self._judge_failed(current, exc.code, tally, refund=None, behind=not nowhere)",
    ),
    m(
        "the pass budget is inclusive",
        "service",
        SERVICE,
        "if tally.deadline is not None and self._clock.now() >= tally.deadline:\n                tally.out_of_time",
        "if tally.deadline is not None and self._clock.now() > tally.deadline:\n                tally.out_of_time",
    ),
    m(
        "the budget is checked between the two calls",
        "service",
        SERVICE,
        "        if tally.deadline is not None and self._clock.now() >= tally.deadline:\n            raise _Interrupted(stopped=False)\n",
        "",
    ),
    m(
        "the stop flag is checked between the two calls",
        "service",
        SERVICE,
        "        if self._should_stop():\n            raise _Interrupted(stopped=True)\n",
        "",
    ),
    m(
        "a retry says what was wrong",
        "service",
        SERVICE,
        "replace(request, attempt=1, previous_problem=first.problem)",
        "replace(request, attempt=1)",
    ),
    m(
        "a retry is another attempt",
        "service",
        SERVICE,
        "replace(request, attempt=1, previous_problem=first.problem)",
        "replace(request, attempt=0, previous_problem='')",
    ),
    m(
        "the identity is trusted until its moment",
        "service",
        SERVICE,
        "if self._identified_until is not None and now < self._identified_until:",
        "if self._identified_until is not None and now <= self._identified_until:",
    ),
    m("the identity lasts", "service", SERVICE, "IDENTITY_TTL_S = 300", "IDENTITY_TTL_S = 0"),
    m(
        "a failure forgets the identity",
        "service",
        SERVICE,
        "        self._identified_until = None\n        tally.unreachable = True",
        "        tally.unreachable = True",
    ),
    m(
        "a success ends the rest",
        "service",
        SERVICE,
        "        self._judge_failures = 0\n        self._rests_until = None",
        "        pass",
    ),
    m(
        "the rest ends at its moment",
        "service",
        SERVICE,
        "if self._rests_until is not None and now < self._rests_until:",
        "if self._rests_until is not None and now <= self._rests_until:",
    ),
    m(
        "the rest doubles",
        "service",
        SERVICE,
        "        self._judge_failures += 1\n",
        "        self._judge_failures = 1\n",
    ),
    m(
        "the lease covers the longest judgment",
        "service",
        SERVICE,
        "span = self._settings.longest_judgment_s + LEASE_MARGIN_S",
        "span = 2 * self._settings.call_deadline_s + LEASE_MARGIN_S",
    ),
    m(
        "a requeued item queues last",
        "service",
        SERVICE,
        "REQUEUE_PRIORITY = 0.1",
        "REQUEUE_PRIORITY = 0.9",
    ),
    m(
        "the ttl is counted from when it was parked",
        "service",
        SERVICE,
        "cutoff = now - timedelta(seconds=self._settings.parked_ttl_s)",
        "cutoff = now + timedelta(seconds=self._settings.parked_ttl_s)",
    ),
    m(
        "an orphan is checked for eligibility",
        "service",
        SERVICE,
        "                if not self._eligible(self._items.get(copy_id)):\n                    continue\n",
        "",
    ),
    m(
        "a copy is checked for eligibility",
        "service",
        SERVICE,
        "            if not self._eligible(copy_item):\n                continue\n",
        "",
    ),
    m(
        "only identical text is shared",
        "service",
        SERVICE,
        "shares = answer is not None and _same_text(original_item, copy_item)",
        "shares = answer is not None",
    ),
    m(
        "the queue bound is exclusive",
        "service",
        SERVICE,
        "        if excess <= 0:\n            return\n        cap =",
        "        if excess < 0:\n            return\n        cap =",
    ),
    m(
        "the fair share only binds above it",
        "service",
        SERVICE,
        "if over > 0 and excess > 0:",
        "if over >= 0 and excess > 0:",
    ),
    m(
        "the fair share never drops more than the excess",
        "service",
        SERVICE,
        "min(over, excess), source_id)",
        "over, source_id)",
    ),
    m("the page of drops", "service", SERVICE, "DROP_PAGE = 1000", "DROP_PAGE = 1001"),
    m(
        "a page never asks for more than the excess",
        "service",
        SERVICE,
        "limit=min(wanted - dropped, DROP_PAGE),",
        "limit=DROP_PAGE,",
    ),
    m(
        "an invalid answer settles as invalid",
        "service",
        SERVICE,
        'left_unjudged(fresh, now, "invalid")\n            ):',
        'left_unjudged(fresh, now, "parked")\n            ):',
    ),
    m(
        "copies of an invalid original are recorded",
        "service",
        SERVICE,
        '(answer is None and reason != "invalid")',
        "(answer is None)",
    ),
    m(
        "an item parks at its limit",
        "service",
        SERVICE,
        "if failures >= settings.max_failures:",
        "if failures > settings.max_failures:",
    ),
    m(
        "a request that cannot be built is no judge failure",
        "service",
        SERVICE,
        "        if not called:\n            return _CONTINUE\n",
        "",
    ),
    m(
        "the prompt is fitted",
        "service",
        SERVICE,
        "        return fit_request(\n            request, num_ctx=settings.num_ctx, output_tokens=settings.max_output_tokens\n        )",
        "        return request",
    ),
    m(
        "an invalid verdict resets the streak",
        "service",
        SERVICE,
        "        tally.invalid += 1\n        tally.reused += copies\n        tally.streak = 0\n",
        "        tally.invalid += 1\n        tally.reused += copies\n",
    ),
    m(
        "an invalid verdict clears the failure record",
        "service",
        SERVICE,
        '                    judged_at=now,\n                )\n            )\n            self._escalations.clear_state(current.item_id)\n            copies = self._settle_copies(\n                left_unjudged(fresh, now, "invalid"),',
        '                    judged_at=now,\n                )\n            )\n            copies = self._settle_copies(\n                left_unjudged(fresh, now, "invalid"),',
    ),
    m(
        "a vanished source is an ineligible one",
        "service",
        SERVICE,
        "        except NotFoundError:\n            return False\n\n    def _request",
        "        except KeyError:\n            return False\n\n    def _request",
    ),
    m(
        "a judged answer counts a contradiction",
        "service",
        SERVICE,
        "        if contradicts_prior(fresh, answer):\n            tally.contradicted += 1\n",
        "",
    ),
    # ---- the repositories ----
    m(
        "a backoff ends at its moment",
        "repos",
        LLM_REPO,
        'AND (e.next_attempt_at IS NULL OR e.next_attempt_at <= ?)))"\n)\n_ORPHANS',
        'AND (e.next_attempt_at IS NULL OR e.next_attempt_at < ?)))"\n)\n_ORPHANS',
    ),
    m(
        "the lowest priority is dropped first",
        "repos",
        LLM_REPO,
        "ORDER BY c.priority, c.classified_at, c.item_id LIMIT ?",
        "ORDER BY c.priority DESC, c.classified_at, c.item_id LIMIT ?",
    ),
    m(
        "the oldest is dropped first among equals",
        "repos",
        LLM_REPO,
        "ORDER BY c.priority, c.classified_at, c.item_id LIMIT ?",
        "ORDER BY c.priority, c.classified_at DESC, c.item_id LIMIT ?",
    ),
    m(
        "a refund never goes below zero",
        "repos",
        LLM_REPO,
        "SET calls = calls - 1 WHERE hour_start = ? AND calls > 0",
        "SET calls = calls - 1 WHERE hour_start = ? AND calls >= 0",
    ),
    m("judged since is inclusive", "repos", LLM_REPO, "AND judged_at >= ?", "AND judged_at > ?"),
    m(
        "parked up to is inclusive",
        "repos",
        LLM_REPO,
        'sql += " AND updated_at <= ?"',
        'sql += " AND updated_at < ?"',
    ),
    m(
        "parked go oldest first",
        "repos",
        LLM_REPO,
        "ORDER BY updated_at, item_id LIMIT ?",
        "ORDER BY updated_at DESC, item_id LIMIT ?",
    ),
    m(
        "an original that no longer waits is gone",
        "repos",
        LLM_REPO,
        "OR oi.status <> 'classified' OR o.needs_llm = 0 \"",
        "OR oi.status <> 'classified' \"",
    ),
    m(
        "an original that is not classified is gone",
        "repos",
        LLM_REPO,
        "OR oi.status <> 'classified' OR o.needs_llm = 0 \"",
        'OR o.needs_llm = 0 "',
    ),
    m(
        "an original in an ineligible source is gone",
        "repos",
        LLM_REPO,
        '"OR oi.source_id NOT IN ({marks}) ORDER BY c.item_id LIMIT ?"',
        '"ORDER BY c.item_id LIMIT ?"',
    ),
    m(
        "only a parked verdict reopens",
        "repos",
        CLASS_REPO,
        "needs_llm = 0 AND settle_reason = 'parked'\"",
        'needs_llm = 0"',
    ),
    m(
        "released copies queue at a floor",
        "repos",
        CLASS_REPO,
        "priority = MAX(0.1, COALESCE(",
        "priority = MAX(0.0, COALESCE(",
    ),
    m(
        "released copies lose the link",
        "repos",
        CLASS_REPO,
        'awaiting_original = 0, duplicate_of = NULL, needs_llm = 1, "',
        'awaiting_original = 0, needs_llm = 1, "',
    ),
    m(
        "orphans are counted apart",
        "repos",
        CLASS_REPO,
        'f"COALESCE(SUM(awaiting_original AND NOT {_LIVE_ORIGINAL}), 0) FROM classifications"',
        'f"COALESCE(SUM(0), 0) FROM classifications"',
    ),
    m(
        "a settlement writes its reason",
        "repos",
        CLASS_REPO,
        'settled_at = ?, settle_reason = ? "\n    "WHERE item_id = ? AND (needs_llm = 1',
        'settled_at = ?, settle_reason = settle_reason "\n    "WHERE item_id = ? AND (needs_llm = 1',
    ),
    m(
        "a waiting item is spared until the horizon",
        "repos",
        ITEMS_REPO,
        "(NOT {_WAITING_FOR_LLM} OR fetched_at < ?)",
        "(NOT {_WAITING_FOR_LLM} OR fetched_at <= ?)",
    ),
    m(
        "a waiting item is spared by default",
        "repos",
        ITEMS_REPO,
        'return f" AND NOT {_WAITING_FOR_LLM}", ()',
        'return "", ()',
    ),
    m(
        "the audit record goes with the item",
        "repos",
        ITEMS_REPO,
        '        conn.execute(f"DELETE FROM llm_judgments WHERE item_id IN ({doomed})", params)\n',
        "",
    ),
    m(
        "the bookkeeping goes with the item",
        "repos",
        ITEMS_REPO,
        '        conn.execute(f"DELETE FROM llm_escalations WHERE item_id IN ({doomed})", params)\n',
        "",
    ),
    # ---- the classifier ----
    m(
        "a model verdict is shared only on identical text",
        "classifier",
        CLASSIFIER,
        "        if match.verdict.stage is not Stage.LLM:\n            return True\n",
        "        return True\n",
    ),
    m(
        "the original must be there to vouch",
        "classifier",
        CLASSIFIER,
        "return earlier is not None and text_identity(",
        "return earlier is None or text_identity(",
    ),
]


def run_tests(area: str) -> bool:
    """True when the tests of `area` pass."""
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *AREAS[area]],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},  # no stale .pyc after a restore
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode == 0


def sweep(selected: list[Mutant]) -> list[Mutant]:
    survivors = []
    for mutant in selected:
        path = SRC / mutant.path
        original = path.read_bytes()
        text = original.decode("utf-8")
        newline = "\r\n" if "\r\n" in text else "\n"
        normal = text.replace("\r\n", "\n")
        if mutant.old not in normal:
            print(f"STALE     {mutant.name}: pattern not found in {mutant.path}")
            survivors.append(mutant)
            continue
        mutated = normal.replace(mutant.old, mutant.new, 1).replace("\n", newline)
        try:
            path.write_bytes(mutated.encode("utf-8"))
            killed = not run_tests(mutant.area)
        finally:
            path.write_bytes(original)
        print(f"{'killed  ' if killed else 'SURVIVED'}  {mutant.name}", flush=True)
        if not killed:
            survivors.append(mutant)
    return survivors


def main(argv: list[str]) -> int:
    selected = [x for x in MUTANTS if not argv or any(a in x.name for a in argv)]
    for area in sorted({x.area for x in selected}):
        if not run_tests(area):
            print(f"the targeted tests of '{area}' do not pass on the unmutated code")
            return 2
    survivors = sweep(selected)
    print(f"\n{len(selected) - len(survivors)} of {len(selected)} mutants killed")
    for mutant in survivors:
        print(f"  survivor: {mutant.name}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
