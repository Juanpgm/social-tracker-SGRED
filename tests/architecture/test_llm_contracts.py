"""Architecture contracts of the LLM escalation: a core without I/O and a loopback-only adapter.

Same technique as `test_import_contracts`: the real `.importlinter`, retargeted at a throwaway
package with deliberate violations (or legal imports), must break (or hold). Every forbidden
import of a module goes in ONE file and `lint-imports` reports each of them, so the suite spawns
few linters and no rule is a rule that only looks like one.
"""

import re
from pathlib import Path

import pytest

from tests.architecture.test_import_contracts import _make_package, _retarget_config, _run

CORE_CONTRACT = "The LLM escalation core must do no I/O of its own"
ADAPTER_CONTRACT = "The Ollama adapter must stay a loopback-only HTTP client"
OUTBOUND_CONTRACT = "Only the fetcher adapter may open network connections"
WEB_CONTRACT = "The web UI must not make outbound network calls"
BASE = {"domain/model.py": "X = 1\n"}
OUTSIDE = {
    "persistence/thing.py": "Y = 2\n",
    "entrypoints/thing.py": "Y = 2\n",
    "collector/thing.py": "Y = 2\n",
    "control/thing.py": "Y = 2\n",
    "items/thing.py": "Y = 2\n",
    "sources/thing.py": "Y = 2\n",
    "catalog/thing.py": "Y = 2\n",
    "bootstrap.py": "Y = 2\n",
}
CORE = ["domain", "schema", "prompt", "ports", "service"]
CORE_FORBIDDEN = {
    "import sqlite3": "sqlite3",
    "import socket": "socket",
    "import ssl": "ssl",
    "import subprocess": "subprocess",
    "import httpx2": "httpx2",
    "import requests": "requests",
    "import urllib": "urllib",
    "import os": "os",
    "import pathlib": "pathlib",
    "import time": "time",
    "import random": "random",
    "import yaml": "yaml",
    "from fixpkg.persistence import thing": "fixpkg.persistence.thing",
    "from fixpkg.collector import thing": "fixpkg.collector.thing",
    "from fixpkg.bootstrap import Y": "fixpkg.bootstrap",
    "from fixpkg.llm import ollama": "fixpkg.llm.ollama",
}
ADAPTER_FORBIDDEN = {
    "import socket": "socket",
    "import ssl": "ssl",
    "import subprocess": "subprocess",
    "import requests": "requests",
    "import urllib": "urllib",
    "import http": "http",
    "import os": "os",
    "import pathlib": "pathlib",
    "import sqlite3": "sqlite3",
    "from fixpkg.persistence import thing": "fixpkg.persistence.thing",
    "from fixpkg.collector import thing": "fixpkg.collector.thing",
    "from fixpkg.control import thing": "fixpkg.control.thing",
    "from fixpkg.bootstrap import Y": "fixpkg.bootstrap",
}


def lint(repo_root: Path, tmp_path: Path, files: dict[str, str]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _make_package(tmp_path, "fixpkg", {**BASE, **OUTSIDE, **files})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    return _run(["--config", str(config)], cwd=tmp_path)


def section(output: str, contract: str) -> str:
    """The part of the report that lists what broke `contract` (up to the next title)."""
    chunks = re.split(r"\n(?=[^\n]+\n-{10,}\n)", output)
    return next(chunk for chunk in chunks if chunk.startswith(contract))


def source_lines(statements: dict[str, str]) -> str:
    return "\n".join(statements) + "\n"


@pytest.mark.parametrize("module", CORE)
def test_the_core_may_not_do_io_reach_storage_or_the_adapter(
    repo_root: Path, tmp_path: Path, module: str
):
    result = lint(repo_root, tmp_path, {f"llm/{module}.py": source_lines(CORE_FORBIDDEN)})
    assert result.returncode != 0, result.stdout
    report = section(result.stdout, CORE_CONTRACT)
    for target in CORE_FORBIDDEN.values():
        assert f"fixpkg.llm.{module} -> {target}" in report, target


def test_the_core_may_use_the_standard_library_pydantic_and_the_rest_of_the_application(
    repo_root: Path, tmp_path: Path
):
    files = {
        "llm/service.py": (
            "import json\nimport logging\nimport threading\nimport hashlib\nimport unicodedata\n"
            "from fixpkg.control import thing\nfrom fixpkg.items import thing as t2\n"
            "from fixpkg.classification import domain\nfrom fixpkg.llm import domain as d\n"
        ),
        "llm/schema.py": "import pydantic\nfrom fixpkg.llm import domain\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_adapter_reaches_nothing_but_the_http_client(repo_root: Path, tmp_path: Path):
    result = lint(repo_root, tmp_path, {"llm/ollama.py": source_lines(ADAPTER_FORBIDDEN)})
    assert result.returncode != 0, result.stdout
    report = section(result.stdout, ADAPTER_CONTRACT)
    for target in ADAPTER_FORBIDDEN.values():
        assert f"fixpkg.llm.ollama -> {target}" in report, target


def test_the_adapter_may_import_the_http_client_json_time_and_the_core(
    repo_root: Path, tmp_path: Path
):
    files = {
        "llm/ollama.py": (
            "import json\nimport time\nimport httpx2\n"
            "from fixpkg.llm import domain, prompt, schema\n"
        )
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


def test_only_the_adapter_may_import_the_http_client_in_the_llm_package(
    repo_root: Path, tmp_path: Path
):
    files = {f"llm/{module}.py": "import httpx2\n" for module in CORE}
    result = lint(repo_root, tmp_path, files)
    assert result.returncode != 0, result.stdout
    report = section(result.stdout, OUTBOUND_CONTRACT)
    for module in CORE:
        assert f"fixpkg.llm.{module} -> httpx2" in report, module
    assert "fixpkg.llm.ollama" not in report


def test_the_web_ui_may_not_import_the_adapter(repo_root: Path, tmp_path: Path):
    result = lint(
        repo_root, tmp_path, {"entrypoints/web/page.py": "from fixpkg.llm import ollama\n"}
    )
    assert result.returncode != 0, result.stdout
    assert "fixpkg.entrypoints.web.page -> fixpkg.llm.ollama" in section(
        result.stdout, WEB_CONTRACT
    )


def test_the_wiring_may_build_the_adapter(repo_root: Path, tmp_path: Path):
    files = {
        "llm/ollama.py": "import httpx2\n",
        "bootstrap.py": "from fixpkg.llm import ollama\nfrom fixpkg.persistence import thing\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_adapter_names_no_host_but_the_loopback_one(repo_root: Path):
    """The import rules cannot see addresses: the adapter's source must not spell any."""
    path = repo_root / "src" / "social_tracker" / "llm" / "ollama.py"
    source = path.read_text(encoding="utf-8")
    for needle in ("http://", "https://", "www.", ".com", ".org", ".net"):
        assert needle not in source, needle
