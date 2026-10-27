"""Architecture contract of the catalog module: ports and pure logic, no I/O of its own.

Same technique as `test_import_contracts`: the real `.importlinter`, retargeted at a throwaway
package with one deliberate violation (or one legal import), must break (or hold).
"""

from pathlib import Path

import pytest

from tests.architecture.test_import_contracts import _make_package, _retarget_config, _run

CONTRACT = "The catalog module must do no I/O of its own"
BASE = {"domain/model.py": "X = 1\n"}
OUTSIDE = {
    "persistence/thing.py": "Y = 2\n",
    "entrypoints/thing.py": "Y = 2\n",
    "collector/http_fetcher.py": "Y = 2\n",
    "collector/ports.py": "Y = 2\n",
    "bootstrap.py": "Y = 2\n",
}


def lint(repo_root: Path, tmp_path: Path, files: dict[str, str]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _make_package(tmp_path, "fixpkg", {**BASE, **OUTSIDE, **files})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    return _run(["--config", str(config)], cwd=tmp_path)


@pytest.mark.parametrize("module", ["service", "parsers", "queries", "domain", "ports"])
@pytest.mark.parametrize(
    "statement",
    [
        "import sqlite3",
        "import socket",
        "import httpx2",
        "import os",
        "from fixpkg.persistence import thing",
        "from fixpkg.collector import http_fetcher",
    ],
)
def test_the_catalog_may_not_do_io_or_reach_an_adapter(
    repo_root: Path, tmp_path: Path, module: str, statement: str
):
    result = lint(repo_root, tmp_path, {f"catalog/{module}.py": statement + "\n"})
    assert result.returncode != 0, result.stdout
    assert CONTRACT in result.stdout


def test_the_catalog_may_use_the_fetcher_port_json_and_the_classification_core(
    repo_root: Path, tmp_path: Path
):
    files = {
        "classification/catalog_match.py": "Z = 3\n",
        "catalog/service.py": (
            "import json\nimport logging\nfrom urllib.parse import urlencode\n"
            "from fixpkg.collector import ports\nfrom fixpkg.classification import catalog_match\n"
        ),
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_wiring_may_build_the_catalog_with_the_adapters(repo_root: Path, tmp_path: Path):
    files = {
        "catalog/service.py": "S = 1\n",
        "bootstrap.py": "from fixpkg.catalog import service\nfrom fixpkg.persistence import thing\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr
