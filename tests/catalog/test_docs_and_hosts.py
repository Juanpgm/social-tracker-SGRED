"""The documented settings are the real ones, and the code reaches exactly two hosts."""

import re
from pathlib import Path

import social_tracker.bootstrap as bootstrap
from social_tracker.catalog.domain import CATALOG_HOSTS, CatalogSettings

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
SRC = ROOT / "src" / "social_tracker"


def row(variable: str) -> str:
    found = [line for line in README.splitlines() if line.startswith(f"| `{variable}`")]
    assert len(found) == 1, variable
    return found[0]


def test_every_catalog_variable_is_documented_with_its_real_default():
    defaults = CatalogSettings()
    expected = {
        bootstrap.CATALOG_CENTER_ENV: "`3.4516,-76.5320`",
        bootstrap.CATALOG_RADIUS_ENV: f"`{defaults.radius_km:g}`",
        bootstrap.CATALOG_MIN_MAGNITUDE_ENV: f"`{defaults.min_magnitude:g}`",
        bootstrap.CATALOG_GLOBAL_MIN_MAGNITUDE_ENV: f"`{defaults.global_min_magnitude:g}`",
        bootstrap.CATALOG_INTERVAL_ENV: f"`{defaults.interval_s}`",
        bootstrap.CATALOG_MAX_AGE_ENV: f"`{defaults.max_age_s}`",
    }
    for variable, default in expected.items():
        assert default in row(variable).split("|")[-2], variable


def test_the_readme_names_the_two_hosts_and_the_exclusion():
    for host in CATALOG_HOSTS:
        assert host in README
    assert "Servicio Geológico Colombiano is excluded" in " ".join(README.split())


def test_the_source_code_contains_no_url_but_the_two_catalog_hosts_and_known_examples():
    allowed = CATALOG_HOSTS | {"example.org", "www.w3.org", "purl.org"}
    seen: set[str] = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        seen |= {m.lower() for m in re.findall(r"https?://([A-Za-z0-9.-]+)", text)}
    unexpected = {h for h in seen if h not in allowed and "example" not in h}
    assert unexpected <= {"localhost", "127.0.0.1"}, unexpected


def test_no_colombian_survey_endpoint_or_setting_exists_in_the_code():
    for path in SRC.rglob("*"):
        if path.suffix not in {".py", ".sql", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8").lower()
        assert "sgc.gov.co" not in text, path
        assert "catalog_sgc" not in text, path
