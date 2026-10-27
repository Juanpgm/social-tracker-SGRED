"""A mutation sweep of the boundaries added by the review of the official-catalog cross-check.

Companion of `mutation_sweep_catalog.py` (the boundaries of unit 3b2). Each mutant changes ONE
operator or constant of the production code; the targeted tests must fail. A mutant that leaves
them green is a survivor: a boundary no test pins. The source is always restored, even on an
interrupt.

    uv run python scripts/mutation_sweep_catalog_review.py            # every mutant
    uv run python scripts/mutation_sweep_catalog_review.py grace      # ids containing "grace"
    uv run python scripts/mutation_sweep_catalog_review.py --check    # only verify the patterns

Exit status: 0 when every mutant was killed, 1 when some survived or a pattern is stale.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "social_tracker"
TESTS = [
    "tests/catalog",
    "tests/classification/test_claims.py",
    "tests/classification/test_claims_boundaries.py",
    "tests/classification/test_claims_remediation.py",
    "tests/classification/test_catalog_match.py",
    "tests/classification/test_catalog_match_boundaries.py",
    "tests/classification/test_catalog_match_remediation.py",
    "tests/classification/test_rules_catalog.py",
    "tests/entrypoints/test_cli_catalog.py",
]


# The tests that exercise each file first: a mutant they kill is not run against the rest. A
# mutant that survives them is run against all of TESTS before it is called a survivor.
NARROW = {
    "classification/claims.py": [
        "tests/classification/test_claims.py",
        "tests/classification/test_claims_boundaries.py",
        "tests/classification/test_claims_remediation.py",
    ],
    "classification/catalog_match.py": [
        "tests/classification/test_catalog_match.py",
        "tests/classification/test_catalog_match_boundaries.py",
        "tests/classification/test_catalog_match_remediation.py",
        "tests/catalog/test_check_remediation.py",
    ],
    "catalog/service.py": [
        "tests/catalog/test_refresher_remediation.py",
        "tests/catalog/test_check_remediation.py",
        "tests/catalog/test_check.py",
        "tests/catalog/test_refresher.py",
    ],
    "catalog/parsers.py": [
        "tests/catalog/test_parsers_remediation.py",
        "tests/catalog/test_parsers.py",
    ],
    "persistence/catalog_repo.py": [
        "tests/catalog/test_repository_remediation.py",
        "tests/catalog/test_repository.py",
    ],
    "catalog/domain.py": [
        "tests/catalog/test_check_remediation.py",
        "tests/catalog/test_domain.py",
        "tests/catalog/test_bootstrap.py",
    ],
    "bootstrap.py": ["tests/catalog/test_bootstrap.py"],
    "collector/worker.py": ["tests/catalog/test_worker_catalog.py"],
}


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str


def m(name: str, path: str, old: str, new: str) -> Mutant:
    return Mutant(name, path, old, new)


CLAIMS = "classification/claims.py"
MATCH = "classification/catalog_match.py"
SERVICE = "catalog/service.py"
PARSERS = "catalog/parsers.py"
REPO = "persistence/catalog_repo.py"
DOMAIN = "catalog/domain.py"
BOOT = "bootstrap.py"
WORKER = "collector/worker.py"

# Known equivalent mutants, left out on purpose:
# - `settled = claimed` after a recorded failure: the failure transaction already cleared the
#   claim, so releasing it again changes nothing;
# - the `[a-z]` boundary after the digit of the `M` magnitude pattern: every text that reaches it
#   is also stopped by the preceding `(?![.,]?[0-9])`.
MUTANTS = [
    # ---- claims: ranges, lists, units ----
    m(
        "claims range span inclusive",
        CLAIMS,
        "if round(upper - low, 2) > MAX_RANGE_SPAN:",
        "if round(upper - low, 2) >= MAX_RANGE_SPAN:",
    ),
    m(
        "claims range span rounding",
        CLAIMS,
        "if round(upper - low, 2) > MAX_RANGE_SPAN:",
        "if upper - low > MAX_RANGE_SPAN:",
    ),
    m(
        "claims range span value",
        CLAIMS,
        "MAX_RANGE_SPAN: Final = 1.0",
        "MAX_RANGE_SPAN: Final = 1.5",
    ),
    m(
        "claims list needs a decimal",
        CLAIMS,
        'elif "." in tail.group(3) or "," in tail.group(3):',
        "elif True:",
    ),
    m(
        "claims list second dropped",
        CLAIMS,
        "listed, consumed = upper, tail.end(3)",
        "consumed = tail.end(3)",
    ),
    m("claims square metres unit", CLAIMS, "|m\\b|m[23]\\b|", "|m\\b|"),
    m("claims square metres lookbehind", CLAIMS, "(?<![0-9]\\s)|", ""),
    m("claims blanks collapsed", CLAIMS, '_BLANKS.sub(" ", bare)', "bare"),
    m("claims days ago is historic", CLAIMS, "semanas?|dias?)\\b", "semanas?)\\b"),
    m("claims speculative text", CLAIMS, "if _SPECULATIVE.search(plain):", "if False:"),
    # ---- claims: time of day ----
    m(
        "claims leading zero is 24 hours",
        CLAIMS,
        'or hour_text.startswith("0")',
        "",
    ),
    m("claims thirteen is 24 hours", CLAIMS, "twenty_four = hour >= 13", "twenty_four = hour > 13"),
    m("claims utc is 24 hours", CLAIMS, 'or groups.get("utc")', ""),
    m(
        "claims bare number is no time",
        CLAIMS,
        'if groups.get("m") is None and groups.get("hs") is None:',
        "if False:",
    ),
    m(
        "claims part: twelve of the morning",
        CLAIMS,
        "return None if hour == 12 else hour",
        "return hour",
    ),
    m(
        "claims part: twelve of the afternoon",
        CLAIMS,
        "return 12 if hour == 12 else hour + 12",
        "return hour + 12",
    ),
    m(
        "claims part: twelve of the night",
        CLAIMS,
        "    if hour == 12:  # noqa",
        "    if False:  # noqa",
    ),
    m(
        "claims part: night starts at six",
        CLAIMS,
        "return hour + 12 if hour >= 6",
        "return hour + 12 if hour >= 5",
    ),
    m(
        "claims part: night six included",
        CLAIMS,
        "return hour + 12 if hour >= 6",
        "return hour + 12 if hour > 6",
    ),
    m("claims part: dawn twelve", CLAIMS, "        return hour % 12\n", "        return hour\n"),
    m(
        "claims part: hour low",
        CLAIMS,
        '    """`hour` of "de la mañana/tarde/noche/madrugada" on a 24-hour clock."""\n'
        "    if not 1 <= hour <= 12:",
        '    """`hour` of "de la mañana/tarde/noche/madrugada" on a 24-hour clock."""\n'
        "    if not 1 < hour <= 12:",
    ),
    m(
        "claims part: hour high",
        CLAIMS,
        '    """`hour` of "de la mañana/tarde/noche/madrugada" on a 24-hour clock."""\n'
        "    if not 1 <= hour <= 12:",
        '    """`hour` of "de la mañana/tarde/noche/madrugada" on a 24-hour clock."""\n'
        "    if not 1 <= hour < 12:",
    ),
    m(
        "claims meridiem twelve",
        CLAIMS,
        "return (hour % 12) + (12 if meridiem",
        "return (hour % 13) + (12 if meridiem",
    ),
    m(
        "claims clock needs the earthquake sentence",
        CLAIMS,
        "if not any(anchored):\n        return lambda _position: True",
        "if not any(anchored):\n        return lambda _position: False",
    ),
    m(
        "claims clock sentence lookup",
        CLAIMS,
        "anchored[bisect_right(cuts, position) - 1]",
        "anchored[0]",
    ),
    m(
        "claims clock filter applied",
        CLAIMS,
        "if clock is None or not in_quake_sentence(position):",
        "if clock is None:",
    ),
    m(
        "claims extreme publication time",
        CLAIMS,
        "    except OverflowError:  # a publication time at the edge",
        "    except ZeroDivisionError:  # a publication time at the edge",
    ),
    # ---- the matcher ----
    m(
        "match grace boundary",
        MATCH,
        "reference > ensure_utc(covered_until) - settings.ingest_grace",
        "reference >= ensure_utc(covered_until) - settings.ingest_grace",
    ),
    m(
        "match lagging softens a mismatch",
        MATCH,
        "if finding.status == MISMATCH and lagging:",
        "if False:",
    ),
    m(
        "match lagging softens no event",
        MATCH,
        "MISMATCH if complete and local and not lagging else UNAVAILABLE",
        "MISMATCH if complete and local else UNAVAILABLE",
    ),
    m(
        "match foreign claims never say no event",
        MATCH,
        "MISMATCH if complete and local and not lagging else UNAVAILABLE",
        "MISMATCH if complete and not lagging else UNAVAILABLE",
    ),
    m(
        "match incomplete cache says unavailable",
        MATCH,
        "MISMATCH if complete and local and not lagging else UNAVAILABLE",
        "MISMATCH if local and not lagging else UNAVAILABLE",
    ),
    m(
        "match floor inclusive",
        MATCH,
        "if max(m.high for m in claimed) < floor:",
        "if max(m.high for m in claimed) <= floor:",
    ),
    m(
        "match floor by place",
        MATCH,
        "floor = settings.min_magnitude if local else settings.global_min_magnitude",
        "floor = settings.global_min_magnitude",
    ),
    m(
        "match floor regional",
        MATCH,
        "floor = settings.min_magnitude if local else settings.global_min_magnitude",
        "floor = settings.min_magnitude",
    ),
    m(
        "match tier inclusive",
        MATCH,
        "row[0] <= candidates[0][0] + pad",
        "row[0] < candidates[0][0] + pad",
    ),
    m("match tier only with a named instant", MATCH, "    if explicit:  #", "    if True:  #"),
    m("match tier needs a named instant", MATCH, "    if explicit:  #", "    if False:  #"),
    m(
        "match matched inclusive",
        MATCH,
        "row[0] < settings.max_magnitude_delta for row",
        "row[0] <= settings.max_magnitude_delta for row",
    ),
    m("match partial match", MATCH, "if matched < len(claimed):", "if False:"),
    m("match nothing matched", MATCH, "if matched == 0:", "if matched < 0:"),
    m(
        "match foreign relevance",
        MATCH,
        "    if not local:\n        return False",
        "    if False:\n        return False",
    ),
    m(
        "match no place is local",
        MATCH,
        "    if place is None:\n        return True",
        "    if place is None:\n        return False",
    ),
    m("match whole words", MATCH, 'f" {name} " in padded', "name in place"),
    m(
        "match wide range inclusive",
        MATCH,
        "round(m.high - m.low, 2) <= MAX_RANGE_SPAN",
        "round(m.high - m.low, 2) < MAX_RANGE_SPAN",
    ),
    m(
        "match grace upper limit",
        MATCH,
        "timedelta(0) <= self.ingest_grace <= _MAX_GRACE",
        "timedelta(0) <= self.ingest_grace < _MAX_GRACE",
    ),
    m(
        "match grace lower limit",
        MATCH,
        "timedelta(0) <= self.ingest_grace <= _MAX_GRACE",
        "timedelta(0) < self.ingest_grace <= _MAX_GRACE",
    ),
    # ---- the cached check ----
    m(
        "check truncated is not complete",
        SERVICE,
        " and not any(s.truncated for s in states),",
        ",",
    ),
    m(
        "check covered until is the oldest refresh",
        SERVICE,
        "covered_until=min(fresh)",
        "covered_until=max(fresh)",
    ),
    m(
        "status truncated is not complete",
        SERVICE,
        "complete=fresh_count == len(rows) and not any(r.truncated for r in rows),",
        "complete=fresh_count == len(rows),",
    ),
    # ---- parsers ----
    m("parser empty body", PARSERS, "if not text.strip():", "if False:"),
    m(
        "parser huge integer",
        PARSERS,
        "    except OverflowError:  # an integer of hundreds of digits",
        "    except ZeroDivisionError:  # an integer of hundreds of digits",
    ),
    m(
        "parser edge of the calendar",
        PARSERS,
        'except (ValueError, OverflowError):  # "0001',
        'except ValueError:  # "0001',
    ),
    m("parser limit minimum", PARSERS, "or limit < 1", "or limit < 0"),
    m("parser truncated inclusive", PARSERS, "len(features) >= limit", "len(features) > limit"),
    m(
        "parser only non earthquakes",
        PARSERS,
        "if features and not kept and not not_earthquake:",
        "if features and not kept:",
    ),
    m(
        "parser usgs type",
        PARSERS,
        'return kind == "earthquake" if source is CatalogSource.USGS else kind in',
        "return True if source is CatalogSource.USGS else kind in",
    ),
    m(
        "parser no type is kept",
        PARSERS,
        "    if kind is None:\n        return True",
        "    if kind is None:\n        return False",
    ),
    m(
        "parser type of the wrong shape",
        PARSERS,
        "    if earthquake is None:\n        return None",
        "    if earthquake is None:\n        pass",
    ),
    # ---- the refresher ----
    m(
        "refresher truncated answers do not reconcile",
        SERVICE,
        "if fetched.truncated or not fetched.events:",
        "if not fetched.events:",
    ),
    m(
        "refresher empty answers do not reconcile",
        SERVICE,
        "if fetched.truncated or not fetched.events:",
        "if fetched.truncated:",
    ),
    m(
        "refresher reconcile window",
        SERVICE,
        "start = end - timedelta(hours=self._settings.lookback_hours)",
        "start = end",
    ),
    m(
        "refresher reconcile own source",
        SERVICE,
        "if e.source is source and e.external_id not in listed",
        "if e.external_id not in listed",
    ),
    m(
        "refresher reconcile in the transaction",
        SERVICE,
        "removed = self._repo.delete_events(source, gone) if gone else 0",
        "removed = 0",
    ),
    m(
        "refresher truncated mark",
        SERVICE,
        "truncated = truncated or parsed.truncated",
        "truncated = parsed.truncated",
    ),
    m("refresher passes the limit", SERVICE, "limit=self._settings.max_events", "limit=None"),
    m(
        "refresher rereads the state after the claim",
        SERVICE,
        "skipped = self._skip(source, state, self._clock.now()) if state else None",
        "skipped = None",
    ),
    m("refresher releases the claim", SERVICE, "if claimed and not settled:", "if False:"),
    m(
        "refresher stops between requests",
        SERVICE,
        "            if self._should_stop():\n                raise _StopRequested",
        "            if False:\n                raise _StopRequested",
    ),
    m(
        "refresher stops between catalogs",
        SERVICE,
        "            if self._should_stop():\n                results.append",
        "            if False:\n                results.append",
    ),
    m(
        "worker classifies before refreshing",
        WORKER,
        "        classification, failed = self._classify_pending()\n"
        "        refreshed = self._refresh_catalog()\n",
        "        refreshed = self._refresh_catalog()\n"
        "        classification, failed = self._classify_pending()\n",
    ),
    # ---- storage ----
    m("repo truncated mark", REPO, "1 if truncated else 0", "0"),
    m(
        "repo delete batch",
        REPO,
        "batch = unique[start : start + _DELETE_BATCH]",
        "batch = unique[start : start + _DELETE_BATCH - 1]",
    ),
    m(
        "repo delete by source",
        REPO,
        '"DELETE FROM catalog_events WHERE source = ? AND external_id IN',
        '"DELETE FROM catalog_events WHERE 1 = ? AND external_id IN',
    ),
    # ---- settings ----
    m(
        "domain grace lower limit",
        DOMAIN,
        "_whole(self.ingest_grace_s, 0, 86_400)",
        "_whole(self.ingest_grace_s, 1, 86_400)",
    ),
    m(
        "domain grace upper limit",
        DOMAIN,
        "_whole(self.ingest_grace_s, 0, 86_400)",
        "_whole(self.ingest_grace_s, 0, 86_399)",
    ),
    m(
        "domain grace reaches the matcher",
        DOMAIN,
        "ingest_grace=timedelta(seconds=self.ingest_grace_s),",
        "",
    ),
    m("domain floor reaches the matcher", DOMAIN, "min_magnitude=self.min_magnitude,", ""),
    m(
        "boot grace lower limit",
        BOOT,
        "defaults.ingest_grace_s, 0, 86_400",
        "defaults.ingest_grace_s, 1, 86_400",
    ),
    m(
        "boot grace upper limit",
        BOOT,
        "defaults.ingest_grace_s, 0, 86_400",
        "defaults.ingest_grace_s, 0, 86_399",
    ),
    m("boot stop reaches the refresher", BOOT, "stop_event.is_set).refresh", "None).refresh"),
]


def run_tests(tests: list[str] | None = None) -> bool:
    """True when the targeted tests pass."""
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *(tests or TESTS)],
        cwd=ROOT,
        # no bytecode: a mutant restored within the same second would otherwise leave a stale .pyc
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    return done.returncode == 0


def stale(mutant: Mutant) -> bool:
    text = (SRC / mutant.path).read_bytes().decode("utf-8").replace("\r\n", "\n")
    return mutant.old not in text


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
            killed = not run_tests(NARROW.get(mutant.path)) or not run_tests()
        finally:
            path.write_bytes(original)
        print(f"{'killed  ' if killed else 'SURVIVED'}  {mutant.name}", flush=True)
        if not killed:
            survivors.append(mutant)
    return survivors


def main(argv: list[str]) -> int:
    if "--check" in argv:
        missing = [x for x in MUTANTS if stale(x)]
        for mutant in missing:
            print(f"STALE     {mutant.name}: pattern not found in {mutant.path}")
        print(f"{len(MUTANTS) - len(missing)} of {len(MUTANTS)} patterns found")
        return 1 if missing else 0
    names = [a for a in argv if not a.startswith("-")]
    selected = [x for x in MUTANTS if not names or any(a in x.name for a in names)]
    if not run_tests():
        print("the targeted tests do not pass on the unmutated code")
        return 2
    survivors = sweep(selected)
    print(f"\n{len(selected) - len(survivors)} of {len(selected)} mutants killed")
    for mutant in survivors:
        print(f"  survivor: {mutant.name} ({mutant.path})")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
