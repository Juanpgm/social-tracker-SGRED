"""A mutation sweep of the boundaries of the official-catalog cross-check (unit 3b2).

Each mutant changes ONE operator or constant of the production code (a `<=` into a `<`, a limit by
one, a dropped guard); the targeted tests must fail. A mutant that leaves them green is a survivor:
a boundary no test pins. The source is always restored, even on an interrupt.

    uv run python scripts/mutation_sweep_catalog.py            # every mutant
    uv run python scripts/mutation_sweep_catalog.py claims     # mutants whose id contains "claims"

Exit status: 0 when every mutant was killed, 1 when some survived.
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
    "tests/collector/test_http_fetcher_allowlist.py",
    "tests/entrypoints/test_cli_catalog.py",
]


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
RULES = "classification/rules.py"
SERVICE = "catalog/service.py"
PARSERS = "catalog/parsers.py"
REPO = "persistence/catalog_repo.py"
FETCHER = "collector/http_fetcher.py"
DOMAIN = "catalog/domain.py"

# Known equivalent mutants, left out on purpose (no input can tell them apart):
# - `start < consumed` vs `<=` in the magnitude scan: two numbers never touch, the regex needs a
#   non-digit between them;
# - `not days_back and at > ...` vs `at > ...`: a time of "yesterday" is never later than now;
# - the early `no_claim` return of the cached check: `cross_check` answers `no_claim` itself.
MUTANTS = [
    m(
        "claims range span inclusive",
        CLAIMS,
        "if round(upper - low, 2) > MAX_RANGE_SPAN:",
        "if round(upper - low, 2) >= MAX_RANGE_SPAN:",
    ),
    m(
        "claims range may be backwards",
        CLAIMS,
        "                if upper >= low:",
        "                if True:",
    ),
    m(
        "claims magnitude max inclusive",
        CLAIMS,
        "0.0 <= number <= MAX_MAGNITUDE",
        "0.0 <= number < MAX_MAGNITUDE",
    ),
    m(
        "claims magnitude min inclusive",
        CLAIMS,
        "0.0 <= number <= MAX_MAGNITUDE",
        "0.0 < number <= MAX_MAGNITUDE",
    ),
    m(
        "claims future tolerance",
        CLAIMS,
        "at > published + FUTURE_TOLERANCE",
        "at >= published + FUTURE_TOLERANCE",
    ),
    m("claims twelve hour clock low", CLAIMS, "if not 1 <= hour <= 12:", "if not 1 < hour <= 12:"),
    m("claims twelve hour clock high", CLAIMS, "if not 1 <= hour <= 12:", "if not 1 <= hour < 12:"),
    m("claims minute limit", CLAIMS, "if minute > 59:", "if minute >= 59:"),
    m("claims hour limit", CLAIMS, "if hour > 23:", "if hour >= 23:"),
    m("claims text cap", CLAIMS, "text[:MAX_TEXT_CHARS]", "text[: MAX_TEXT_CHARS + 1]"),
    m("claims offset low", CLAIMS, "not -12 <= hours <= 14", "not -12 < hours <= 14"),
    m("claims offset high", CLAIMS, "not -12 <= hours <= 14", "not -12 <= hours < 14"),
    m("claims place cap", CLAIMS, "[:MAX_PLACE_CHARS]", "[: MAX_PLACE_CHARS + 1]"),
    m(
        "claims magnitude cap",
        CLAIMS,
        "return tuple(claims[:MAX_MAGNITUDES])",
        "return tuple(claims[: MAX_MAGNITUDES + 1])",
    ),
    m(
        "claims utc marker",
        CLAIMS,
        'zone = timedelta(0) if match.group("utc") else offset',
        "zone = offset",
    ),
    m(
        "claims historic year",
        CLAIMS,
        "int(match.group(1)) != local_day.year",
        "int(match.group(1)) < local_day.year",
    ),
    m(
        "match delta threshold",
        MATCH,
        "row[0] < settings.max_magnitude_delta for row",
        "row[0] <= settings.max_magnitude_delta for row",
    ),
    m(
        "match delta rounding",
        MATCH,
        "round(claim.distance(magnitude), _DELTA_DIGITS)",
        "claim.distance(magnitude)",
    ),
    m("match radius inclusive", MATCH, "<= settings.radius_km", "< settings.radius_km"),
    m(
        "match global floor inclusive",
        MATCH,
        "max(event.magnitudes) >= settings.global_min_magnitude",
        "max(event.magnitudes) > settings.global_min_magnitude",
    ),
    m(
        "match window inclusive",
        MATCH,
        "            if abs(event.origin_time - reference) <= window\n",
        "            if abs(event.origin_time - reference) < window\n",
    ),
    m(
        "match future inclusive",
        MATCH,
        "reference > current + FUTURE_TOLERANCE",
        "reference >= current + FUTURE_TOLERANCE",
    ),
    m(
        "match coverage inclusive",
        MATCH,
        "reference - window < ensure_utc(covered_from)",
        "reference - window <= ensure_utc(covered_from)",
    ),
    m(
        "match duplicate seconds",
        MATCH,
        "<= settings.duplicate_seconds",
        "< settings.duplicate_seconds",
    ),
    m(
        "match duplicate magnitude",
        MATCH,
        "<= settings.duplicate_magnitude",
        "< settings.duplicate_magnitude",
    ),
    m("match duplicate km", MATCH, "<= settings.duplicate_km", "< settings.duplicate_km"),
    m("match same source merge", MATCH, "point.source not in {p.source for p in group}", "True"),
    m(
        "match complete flag",
        MATCH,
        "MISMATCH if complete and local and not lagging else UNAVAILABLE",
        "MISMATCH",
    ),
    m("match historic", MATCH, "or claims.when.kind is TimeKind.HISTORIC", ""),
    m("match tie by delta", MATCH, "key=lambda row: (row[0], row[1],", "key=lambda row: (row[0],"),
    m(
        "match nearest in time",
        MATCH,
        "key=lambda row: (row[0], row[1],",
        "key=lambda row: (row[1], row[0],",
    ),
    m(
        "rules consistent factor applied",
        RULES,
        "if catalog_status == CONSISTENT:",
        "if catalog_status != MISMATCH:",
    ),
    m(
        "rules mismatch signal",
        RULES,
        '"catalog_mismatch": catalog_status == MISMATCH',
        '"catalog_mismatch": catalog_status != CONSISTENT',
    ),
    m(
        "check stale inclusive",
        SERVICE,
        "now - s.last_success_at <= max_age",
        "now - s.last_success_at < max_age",
    ),
    m(
        "check complete needs all",
        SERVICE,
        "complete=len(fresh) == len(CatalogSource)",
        "complete=len(fresh) >= 1",
    ),
    m("check coverage uses newest", SERVICE, "covered_from=max(fresh)", "covered_from=min(fresh)"),
    m("check empty cache", SERVICE, "or not self._repo.count_by_source()", ""),
    m(
        "refresher backoff inclusive",
        SERVICE,
        "now < state.next_attempt_at",
        "now <= state.next_attempt_at",
    ),
    m(
        "refresher due inclusive",
        SERVICE,
        "last <= now < last + interval",
        "last <= now <= last + interval",
    ),
    m("refresher retry-after wins", SERVICE, "delay = max(delay, min(", "delay = min(delay, min("),
    m(
        "refresher retry-after cap",
        SERVICE,
        "MAX_RETRY_AFTER_S = 6 * 3600.0",
        "MAX_RETRY_AFTER_S = 7 * 3600.0",
    ),
    m(
        "refresher claims before fetch",
        SERVICE,
        "if claimed and not self._repo.try_claim",
        "if False and not self._repo.try_claim",
    ),
    m(
        "refresher jitter base",
        SERVICE,
        "backoff_delay_s(failures,",
        "backoff_delay_s(failures + 1,",
    ),
    m("parser future inclusive", PARSERS, "when > now + MAX_FUTURE", "when >= now + MAX_FUTURE"),
    m("parser old inclusive", PARSERS, "when < now - MAX_AGE", "when <= now - MAX_AGE"),
    m("parser feature cap", PARSERS, "> MAX_FEATURES", ">= MAX_FEATURES"),
    m(
        "parser nothing readable",
        PARSERS,
        "if features and not kept and not not_earthquake:",
        "if False:",
    ),
    m("parser json constants", PARSERS, "parse_constant=_reject_constant", "parse_constant=None"),
    m("parser place length", PARSERS, "[:MAX_PLACE_LENGTH]", "[: MAX_PLACE_LENGTH + 1]"),
    m(
        "repo range lower inclusive",
        REPO,
        "origin_time >= ? AND origin_time <= ?",
        "origin_time > ? AND origin_time <= ?",
    ),
    m(
        "repo range upper inclusive",
        REPO,
        "origin_time >= ? AND origin_time <= ?",
        "origin_time >= ? AND origin_time < ?",
    ),
    m(
        "repo purge strict",
        REPO,
        "DELETE FROM catalog_events WHERE origin_time < ?",
        "DELETE FROM catalog_events WHERE origin_time <= ?",
    ),
    m(
        "repo claim boundary",
        REPO,
        "OR catalog_refresh_state.claimed_until <= ?",
        "OR catalog_refresh_state.claimed_until < ?",
    ),
    m(
        "repo failure counter",
        REPO,
        "catalog_refresh_state.consecutive_failures + 1",
        "catalog_refresh_state.consecutive_failures + 2",
    ),
    m("fetcher host allowlist", FETCHER, "target.host not in self._allowed_hosts", "False"),
    m("fetcher https only", FETCHER, 'self._https_only and target.scheme != "https"', "False"),
    m("domain max age vs interval", DOMAIN, "self.interval_s, 7 * 86_400", "0, 7 * 86_400"),
    m("domain lookback vs window", DOMAIN, "math.ceil(2 * window_hours)", "1"),
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
        print(f"{'killed  ' if killed else 'SURVIVED'}  {mutant.name}")
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
        print(f"  survivor: {mutant.name} ({mutant.path})")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
