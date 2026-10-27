"""A mutation sweep of the boundaries of the LLM escalation (unit 3c).

Each mutant changes ONE operator or constant of the production code (a `<=` into a `<`, a limit by
one, a dropped guard); the targeted tests must fail. A mutant that leaves them green is a survivor:
a boundary no test pins. The source is always restored, even on an interrupt.

    uv run python scripts/mutation_sweep_llm.py            # every mutant
    uv run python scripts/mutation_sweep_llm.py backoff    # mutants whose name contains "backoff"

The real-socket tests are left out of the sweep (they only repeat what the scripted transports
already pin and cost seconds each), except the one that pins the ignored proxy variables
(`test_ollama_proxy.py`). Exit status: 0 when every mutant was killed, 1 when some
survived, 2 when the targeted tests do not pass on the unmutated code.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "social_tracker"
TESTS = [
    "tests/llm",
    "--ignore=tests/llm/test_ollama_http.py",
    "tests/llm/test_ollama_proxy.py",
    "tests/persistence/test_sqlite_escalation_repository.py",
    "tests/classification/test_fake_repositories.py",
    "tests/persistence/test_sqlite_classification_repositories.py",
]


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str


def m(name: str, path: str, old: str, new: str) -> Mutant:
    return Mutant(name, path, old, new)


DOMAIN = "llm/domain.py"
SCHEMA = "llm/schema.py"
SERVICE = "llm/service.py"
OLLAMA = "llm/ollama.py"
LLM_REPO = "persistence/llm_repo.py"
CLASS_REPO = "persistence/classification_repo.py"

# Known equivalent mutants, left out on purpose (no input can tell them apart):
# - the number of seconds a lease lasts beyond one call (the tests run on a fixed clock and a
#   lease only has to outlive a call);
# - `remaining <= 0` vs `< 0` in the call deadline (a monotonic clock never stands still between
#   two reads in practice);
# - `allow_inf_nan=False` of the confidence and the `parse_constant` hook of the JSON reader: each
#   is defence in depth for the other (NaN fails `ge/le` anyway, and an infinity is over `le`);
# - `not fresh.needs_llm` in the re-check before a write: `ClassificationRepository.settle`
#   refuses a verdict that no longer waits, and every write sits behind it.
MUTANTS = [
    # ---- the vocabulary ----
    m(
        "low confidence floor inclusive",
        DOMAIN,
        "return confidence < LOW_CONFIDENCE_FLOOR or confidence < alert_min",
        "return confidence <= LOW_CONFIDENCE_FLOOR or confidence < alert_min",
    ),
    m(
        "low confidence alert inclusive",
        DOMAIN,
        "return confidence < LOW_CONFIDENCE_FLOOR or confidence < alert_min",
        "return confidence < LOW_CONFIDENCE_FLOOR or confidence <= alert_min",
    ),
    m(
        "low confidence ignores the floor",
        DOMAIN,
        "return confidence < LOW_CONFIDENCE_FLOOR or confidence < alert_min",
        "return confidence < alert_min",
    ),
    m(
        "backoff base doubles",
        DOMAIN,
        "base_s * 2**exponent",
        "base_s * 3**exponent",
    ),
    m(
        "backoff cap",
        DOMAIN,
        "min(cap_s, base_s * 2**exponent)",
        "max(cap_s, base_s * 2**exponent)",
    ),
    m(
        "backoff first failure",
        DOMAIN,
        "exponent = min(failures - 1, 40)",
        "exponent = min(failures, 40)",
    ),
    m(
        "budget hour microseconds",
        DOMAIN,
        ".replace(minute=0, second=0, microsecond=0)",
        ".replace(minute=0, second=0)",
    ),
    m(
        "budget hour minutes",
        DOMAIN,
        ".replace(minute=0, second=0, microsecond=0)",
        ".replace(second=0, microsecond=0)",
    ),
    m(
        "prompt text boundary",
        DOMAIN,
        "if len(joined) <= max_chars:",
        "if len(joined) < max_chars:",
    ),
    m("prompt text head", DOMAIN, "head = max_chars // 2", "head = max_chars // 2 + 1"),
    m("sanitize cap", DOMAIN, "[:limit].rstrip()", "[: limit + 1].rstrip()"),
    m(  # the first of two identical lines in the file: the one of JudgeResponse
        "rationale length",
        DOMAIN,
        '_text(self.rationale, 1, MAX_RATIONALE_CHARS, "the rationale")',
        '_text(self.rationale, 1, MAX_RATIONALE_CHARS + 1, "the rationale")',
    ),
    m(  # same: the first of two identical lines
        "evidence count",
        DOMAIN,
        "len(self.evidence) > MAX_EVIDENCE_ITEMS",
        "len(self.evidence) >= MAX_EVIDENCE_ITEMS",
    ),
    m(
        "hourly cap range",
        DOMAIN,
        "_whole(self.hourly_cap, 1, 100_000)",
        "_whole(self.hourly_cap, 0, 100_000)",
    ),
    m("port range", DOMAIN, "if not 1 <= port <= 65535:", "if not 0 <= port <= 65535:"),
    m("port default", DOMAIN, "if match.group(2) else 80", "if match.group(2) else 11434"),
    m("connect host", DOMAIN, "        return LOOPBACK_ADDRESS", "        return self.host"),
    m(
        "fallback equals model",
        DOMAIN,
        "if self.fallback_model and self.fallback_model != self.model:",
        "if self.fallback_model:",
    ),
    m(
        "timeout is item specific",
        DOMAIN,
        "{JudgeErrorKind.TIMEOUT, JudgeErrorKind.SERVER_ERROR, JudgeErrorKind.BAD_RESPONSE}",
        "{JudgeErrorKind.SERVER_ERROR, JudgeErrorKind.BAD_RESPONSE}",
    ),
    m(
        "copy stops waiting",
        DOMAIN,
        "        awaiting_original=False,\n    )",
        "        awaiting_original=True,\n    )",
    ),
    m(
        "released copy stops waiting",
        DOMAIN,
        "return replace(duplicate, awaiting_original=False)",
        "return replace(duplicate, awaiting_original=True)",
    ),
    m(
        "judgment leaves the queue",
        DOMAIN,
        "        needs_llm=False,\n        priority=0.0,\n    )\n\n\ndef left_unjudged",
        "        needs_llm=True,\n        priority=0.0,\n    )\n\n\ndef left_unjudged",
    ),
    m(
        "unjudged is low confidence",
        DOMAIN,
        "return replace(prior, needs_llm=False, low_confidence=True, priority=0.0)",
        "return replace(prior, needs_llm=False, low_confidence=False, priority=0.0)",
    ),
    # ---- the strict schema ----
    m("output cap", SCHEMA, "if len(raw) > MAX_OUTPUT_CHARS:", "if len(raw) >= MAX_OUTPUT_CHARS:"),
    m(
        "confidence upper bound",
        SCHEMA,
        "Field(ge=0.0, le=1.0, allow_inf_nan=False)",
        "Field(ge=0.0, le=1.1, allow_inf_nan=False)",
    ),
    m(
        "confidence lower bound",
        SCHEMA,
        "Field(ge=0.0, le=1.0, allow_inf_nan=False)",
        "Field(ge=-0.1, le=1.0, allow_inf_nan=False)",
    ),
    m("confidence nan", SCHEMA, "allow_inf_nan=False", "allow_inf_nan=True"),
    m("extra fields", SCHEMA, 'extra="forbid"', 'extra="ignore"'),
    m("strict types", SCHEMA, "strict=True", "strict=False"),
    m(
        "rationale input bound",
        SCHEMA,
        "_MAX_RATIONALE_INPUT: Final = 2000",
        "_MAX_RATIONALE_INPUT: Final = 2001",
    ),
    m("rationale cap", SCHEMA, "RATIONALE_CHARS: Final = 300", "RATIONALE_CHARS: Final = 301"),
    m(
        "evidence items",
        SCHEMA,
        "Field(max_length=MAX_EVIDENCE_ITEMS)",
        "Field(max_length=MAX_EVIDENCE_ITEMS + 1)",
    ),
    m("duplicate keys", SCHEMA, "if len(set(keys)) != len(keys):", "if False:"),
    m("non finite constants", SCHEMA, ", parse_constant=_no_constant", ""),
    # ---- the service ----
    m(
        "parking threshold",
        SERVICE,
        "if failures >= settings.max_failures:",
        "if failures > settings.max_failures:",
    ),
    m("breaker", SERVICE, "BREAKER = 2", "BREAKER = 3"),
    m("breaker boundary", SERVICE, "tally.streak >= BREAKER", "tally.streak > BREAKER"),
    m("every failure is the items", SERVICE, "+ int(item_specific)", "+ 1"),
    m("no failure is the items", SERVICE, "+ int(item_specific)", "+ 0"),
    m(
        "judge down ends the batch",
        SERVICE,
        "        if not item_specific:\n            tally.unreachable = True",
        "        if False:\n            tally.unreachable = True",
    ),
    m(
        "attempts count",
        SERVICE,
        "attempts = (before.attempts if before else 0) + 1",
        "attempts = (before.attempts if before else 0) + 2",
    ),
    m("drop boundary", SERVICE, "if excess <= 0:", "if excess < 0:"),
    m(
        "dry run cap",
        SERVICE,
        "cap_reached=len(entries) > room,",
        "cap_reached=len(entries) >= room,",
    ),
    m("batch bound", SERVICE, "or not 1 <= limit <= MAX_BATCH:", "or not 1 <= limit < MAX_BATCH:"),
    m(
        "recycled copy of a judged item",
        SERVICE,
        "if current.recycled_of is not None and self._already_judged(current.recycled_of):",
        "if False:",
    ),
    m(
        "invalid output kind",
        SERVICE,
        "if exc.kind is JudgeErrorKind.INVALID_OUTPUT:",
        "if exc.kind is not JudgeErrorKind.INVALID_OUTPUT:",
    ),
    m(
        "copies take the verdict",
        SERVICE,
        "            if answer is not None:",
        "            if answer is None:",
    ),
    m(
        "still waiting check",
        SERVICE,
        "if fresh is None or not fresh.needs_llm or not self._eligible(self._items.get(item_id)):",
        "if fresh is None or not self._eligible(self._items.get(item_id)):",
    ),
    m(
        "classified only",
        SERVICE,
        "item.status is not ItemStatus.CLASSIFIED:",
        "item.status is ItemStatus.OUT_OF_SCOPE:",
    ),
    m("latest tag", SERVICE, 'f"{name}:latest"', 'f"{name}:newest"'),
    # ---- the adapter ----
    m("redirect refused", OLLAMA, "if 300 <= status < 400:", "if 300 <= status < 300:"),
    m("http errors", OLLAMA, "        if status >= 400:", "        if status > 400:"),
    m("model not found status", OLLAMA, "(status == 404 or", "(status == 403 or"),
    m(
        "load failure falls back",
        OLLAMA,
        "(status >= 500 and names_model)",
        "(status >= 600 and names_model)",
    ),
    m("declared size", OLLAMA, "int(declared) > limit", "int(declared) >= limit"),
    m("streamed size", OLLAMA, "            if size > limit:", "            if size >= limit:"),
    m("qwen thinking switch", OLLAMA, '"qwen" in model.lower()', '"gemma" in model.lower()'),
    m(
        "keep alive unit",
        OLLAMA,
        '{self._settings.keep_alive_s}s"',
        '{self._settings.keep_alive_s}m"',
    ),
    m("environment proxies", OLLAMA, "            trust_env=False,", "            trust_env=True,"),
    m("redirects followed", OLLAMA, "follow_redirects=False", "follow_redirects=True"),
    m("temperature", OLLAMA, '"temperature": 0,', '"temperature": 1,'),
    m("last model", OLLAMA, "last = index == len(models) - 1", "last = index == 0"),
    m("finished answers only", OLLAMA, "done is not True", "done is None"),
    # ---- the repositories ----
    m("budget cap", LLM_REPO, "WHERE calls < ?", "WHERE calls <= ?"),
    m("backoff moment", LLM_REPO, "e.next_attempt_at <= ?", "e.next_attempt_at < ?"),
    m("lease expiry", LLM_REPO, "OR llm_lease.until <= ?", "OR llm_lease.until < ?"),
    m(
        "queue order",
        LLM_REPO,
        "ORDER BY c.priority DESC, c.classified_at, c.item_id LIMIT ?",
        "ORDER BY c.priority ASC, c.classified_at, c.item_id LIMIT ?",
    ),
    m(
        "settle only what waits",
        CLASS_REPO,
        "AND (needs_llm = 1 OR awaiting_original = 1)",
        "AND needs_llm = 1",
    ),
]


def run_tests() -> bool:
    """True when the targeted tests pass."""
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS],
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
            killed = not run_tests()
        finally:
            path.write_bytes(original)
        print(f"{'killed  ' if killed else 'SURVIVED'}  {mutant.name}", flush=True)
        if not killed:
            survivors.append(mutant)
    return survivors


def main(argv: list[str]) -> int:
    selected = [x for x in MUTANTS if not argv or any(a in x.name for a in argv)]
    if not run_tests():
        print("the targeted tests do not pass on the unmutated code")
        return 2
    survivors = sweep(selected)
    print(f"\n{len(selected) - len(survivors)} of {len(selected)} mutants killed")
    for mutant in survivors:
        print(f"  survivor: {mutant.name}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
