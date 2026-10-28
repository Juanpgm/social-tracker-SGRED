"""The calibration report says what it measured: the CPU cascade without history or catalogs, so a
reader of the printed report alone is not misled about what the numbers mean."""

from pathlib import Path

import pytest

from tests.entrypoints.conftest import Cli
from tests.entrypoints.test_cli_gold import OFF_TOPIC, PREDICTION, REPORT, add

CAVEAT = "no history, no catalog; CPU stages only"


@pytest.fixture
def labeled(cli: Cli, tmp_path: Path) -> Cli:
    texts = [(PREDICTION, "unintentional_rumor"), (REPORT, "no_risk"), (OFF_TOPIC, "out_of_scope")]
    for n, (text, label) in enumerate(texts):
        assert add(cli, tmp_path, text, label, name=f"t{n}.txt").exit_code == 0
    return cli


def test_the_report_prints_the_caveat(labeled: Cli):
    out = labeled.ok("classify", "calibrate").stdout
    assert f"measured with {CAVEAT}" in out


def test_the_caveat_is_in_a_report_from_a_file_too(labeled: Cli, tmp_path: Path):
    target = tmp_path / "gold.yaml"
    labeled.ok("gold", "export", "-o", str(target))
    out = labeled.ok("classify", "calibrate", "--file", str(target)).stdout
    assert f"measured with {CAVEAT}" in out


def test_the_caveat_comes_before_the_numbers_it_qualifies(labeled: Cli):
    lines = labeled.ok("classify", "calibrate").stdout.splitlines()
    caveat = next(i for i, line in enumerate(lines) if CAVEAT in line)
    accuracy = next(i for i, line in enumerate(lines) if line.startswith("accuracy:"))
    assert caveat < accuracy


def test_an_empty_set_prints_no_report_and_so_no_caveat(cli: Cli):
    result = cli("classify", "calibrate")
    assert CAVEAT not in result.stdout
