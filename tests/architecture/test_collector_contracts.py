"""Architecture contracts of the collector: one outbound path, a core without I/O.

Same technique as `test_import_contracts`: the real `.importlinter`, retargeted at a throwaway
package with one deliberate violation (or one legal import), must break (or hold).
"""

from pathlib import Path

import pytest

from tests.architecture.test_import_contracts import (
    COLLECTOR_CORE_MODULES as CORE_MODULE_NAMES,
)
from tests.architecture.test_import_contracts import (
    WEB_OUTBOUND_CONTRACT,
    _make_package,
    _retarget_config,
    _run,
)

OUTBOUND_CONTRACT = "Only the fetcher adapter may open network connections"
OUTBOUND_IMPORTS = [
    "import httpx2",
    "import httpx",
    "import socket",
    "import requests",
    "import aiohttp",
]
FETCHER = "collector/http_fetcher.py"
BASE = {"domain/model.py": "X = 1\n"}


def lint(repo_root: Path, tmp_path: Path, files: dict[str, str]):
    _make_package(tmp_path, "fixpkg", {**BASE, **files})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    return _run(["--config", str(config)], cwd=tmp_path)


@pytest.mark.parametrize("statement", OUTBOUND_IMPORTS)
@pytest.mark.parametrize(
    "module_path",
    [
        "bootstrap.py",
        "sources/service.py",
        "items/ports.py",
        "persistence/db.py",
        "entrypoints/cli.py",
        "collector/netpolicy.py",
        "collector/other_adapter.py",
    ],
)
def test_network_libraries_are_confined_to_the_fetcher_adapter(
    repo_root: Path, tmp_path: Path, module_path: str, statement: str
):
    result = lint(repo_root, tmp_path, {module_path: statement + "\n"})
    assert result.returncode != 0, result.stdout
    assert OUTBOUND_CONTRACT in result.stdout
    assert "BROKEN" in result.stdout


@pytest.mark.parametrize("statement", ["import httpx2", "import socket"])
def test_the_fetcher_adapter_may_use_the_http_client_and_sockets(
    repo_root: Path, tmp_path: Path, statement: str
):
    files = {
        FETCHER: statement + "\n",
        "bootstrap.py": "from fixpkg.collector import http_fetcher\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("statement", ["import requests", "import aiohttp", "import httpx"])
def test_the_fetcher_adapter_may_not_use_other_network_clients(
    repo_root: Path, tmp_path: Path, statement: str
):
    result = lint(repo_root, tmp_path, {FETCHER: statement + "\n"})
    assert result.returncode != 0, result.stdout
    assert OUTBOUND_CONTRACT in result.stdout


@pytest.mark.parametrize("module_path", ["entrypoints/web/app.py", "entrypoints/web/sub/views.py"])
@pytest.mark.parametrize(
    "statement",
    ["from fixpkg.collector import http_fetcher", "import fixpkg.collector.service"],
)
def test_the_web_ui_may_not_import_the_collector(
    repo_root: Path, tmp_path: Path, module_path: str, statement: str
):
    files = {FETCHER: "Y = 2\n", "collector/service.py": "Z = 3\n", module_path: statement + "\n"}
    result = lint(repo_root, tmp_path, files)
    assert result.returncode != 0, result.stdout
    assert WEB_OUTBOUND_CONTRACT in result.stdout


def test_the_web_ui_may_reach_the_worker_side_only_through_the_composition_root(
    repo_root: Path, tmp_path: Path
):
    files = {
        FETCHER: "import httpx2\n",
        "bootstrap.py": "from fixpkg.collector import http_fetcher\n",
        "entrypoints/web/app.py": "from fixpkg import bootstrap\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


COLLECTOR_CORE_MODULES = [f"collector/{name}.py" for name in CORE_MODULE_NAMES]


@pytest.mark.parametrize("module_path", COLLECTOR_CORE_MODULES)
@pytest.mark.parametrize(
    "statement",
    ["import sqlite3", "from fixpkg.persistence import thing", "import ssl", "import subprocess"],
)
def test_the_collector_core_must_not_do_io(
    repo_root: Path, tmp_path: Path, module_path: str, statement: str
):
    files = {"persistence/thing.py": "Y = 2\n", module_path: statement + "\n"}
    result = lint(repo_root, tmp_path, files)
    assert result.returncode != 0, result.stdout
    assert "BROKEN" in result.stdout
