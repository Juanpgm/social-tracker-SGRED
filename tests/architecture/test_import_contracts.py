"""Architecture contract: the domain layer stays pure.

The real contract lives in `.importlinter`. These tests run `lint-imports` against it, and
prove it is not vacuous by running the same contract (retargeted at a throwaway package)
against deliberate violations.

Limitation: import-linter analyses static import statements only. Dynamic imports
(`__import__`, `importlib.import_module`, `builtins.open`, `eval`) cannot be caught by
this contract; importing `importlib` itself is forbidden in the domain to narrow that gap.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CONFIG_NAME = ".importlinter"
FORBIDDEN_EXTERNALS = [
    "sqlite3",
    "yaml",
    "typer",
    "httpx",
    "httpx2",
    "requests",
    "aiohttp",
    "os",
    "pathlib",
    "io",
    "shutil",
    "socket",
    "ssl",
    "subprocess",
    "urllib",
    "http",
    "asyncio",
    "importlib",
    "tempfile",
    "glob",
]
COLLECTOR_CORE_MODULES = ("feed", "normalize", "netpolicy", "service", "worker")
CLASSIFICATION_CORE_MODULES = ("domain", "text", "links", "topics", "rules", "similarity")
LLM_MODULES = ("domain", "schema", "prompt", "ports", "service", "ollama")
ALERTS_MODULES = ("domain", "excerpt", "ports", "service")
GOLD_MODULES = ("domain", "ports", "service", "yaml_io")
FORBIDDEN_LAYERS = ["adapters", "persistence", "entrypoints", "infrastructure", "storage"]


def _lint_imports_cmd() -> list[str]:
    scripts = Path(sys.executable).parent
    for name in ("lint-imports.exe", "lint-imports"):
        candidate = scripts / name
        if candidate.exists():
            return [str(candidate)]
    return [sys.executable, "-m", "importlinter.cli"]


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_lint_imports_cmd(), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def _make_package(root: Path, name: str, files: dict[str, str]) -> None:
    pkg = root / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for rel, body in files.items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        init = target.parent / "__init__.py"
        if not init.exists():
            init.write_text("")
        target.write_text(textwrap.dedent(body))
    # The entrypoints and web contracts name their source packages literally (so a rename cannot
    # silently disable them), hence every fixture package has both, empty unless a test fills them.
    (pkg / "entrypoints" / "web").mkdir(parents=True, exist_ok=True)
    (pkg / "entrypoints" / "__init__.py").touch()
    (pkg / "entrypoints" / "web" / "__init__.py").touch()
    # The same goes for the collector-core contract and the modules it lists.
    (pkg / "collector").mkdir(exist_ok=True)
    (pkg / "collector" / "__init__.py").touch()
    for module in COLLECTOR_CORE_MODULES:
        (pkg / "collector" / f"{module}.py").touch()
    # And the classification-core contract.
    (pkg / "classification").mkdir(exist_ok=True)
    (pkg / "classification" / "__init__.py").touch()
    for module in CLASSIFICATION_CORE_MODULES:
        (pkg / "classification" / f"{module}.py").touch()
    # And the LLM escalation contracts (its core and its loopback-only adapter).
    (pkg / "llm").mkdir(exist_ok=True)
    (pkg / "llm" / "__init__.py").touch()
    for module in LLM_MODULES:
        (pkg / "llm" / f"{module}.py").touch()
    # And the alerts contracts (their module and its pure part).
    (pkg / "alerts").mkdir(exist_ok=True)
    (pkg / "alerts" / "__init__.py").touch()
    for module in ALERTS_MODULES:
        (pkg / "alerts" / f"{module}.py").touch()
    # And the gold-set contracts.
    (pkg / "gold").mkdir(exist_ok=True)
    (pkg / "gold" / "__init__.py").touch()
    for module in GOLD_MODULES:
        (pkg / "gold" / f"{module}.py").touch()


def _retarget_config(repo_root: Path, dest: Path, root_package: str) -> Path:
    text = (repo_root / CONFIG_NAME).read_text(encoding="utf-8")
    config = dest / CONFIG_NAME
    config.write_text(text.replace("social_tracker", root_package), encoding="utf-8")
    return config


def test_real_contract_passes_on_the_repository(repo_root: Path):
    result = _run(["--config", str(repo_root / CONFIG_NAME)], cwd=repo_root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_missing_config_fails_with_a_clear_message(tmp_path: Path):
    result = _run(["--config", str(tmp_path / "does-not-exist.ini")], cwd=tmp_path)
    assert result.returncode != 0
    assert "does-not-exist.ini" in (result.stdout + result.stderr)


def test_contract_tolerates_absent_layers(repo_root: Path, tmp_path: Path):
    """A package with only a clean domain (no adapters/persistence/entrypoints) passes."""
    _make_package(tmp_path, "fixpkg", {"domain/model.py": "X = 1\n"})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    result = _run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", FORBIDDEN_EXTERNALS)
def test_domain_importing_forbidden_external_fails(repo_root: Path, tmp_path: Path, module: str):
    _make_package(tmp_path, "fixpkg", {"domain/model.py": f"import {module}\n"})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    result = _run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode != 0, result.stdout
    assert "BROKEN" in result.stdout


@pytest.mark.parametrize("layer", FORBIDDEN_LAYERS)
def test_domain_importing_internal_layer_fails(repo_root: Path, tmp_path: Path, layer: str):
    _make_package(
        tmp_path,
        "fixpkg",
        {
            f"{layer}/thing.py": "Y = 2\n",
            "domain/model.py": f"from fixpkg.{layer} import thing\n",
        },
    )
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    result = _run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode != 0, result.stdout
    assert "BROKEN" in result.stdout


def test_nested_domain_package_is_also_covered(repo_root: Path, tmp_path: Path):
    """`<anything>.domain` (e.g. sources.domain) is covered, not just the top-level one."""
    _make_package(tmp_path, "fixpkg", {"sources/domain/rules.py": "import sqlite3\n"})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    result = _run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode != 0, result.stdout


SERVICE_MODULES = [
    "sources/service.py",
    "control/service.py",
    "sources/yaml_io.py",
    "items/service.py",
    "collector/service.py",
]


@pytest.mark.parametrize("module_path", SERVICE_MODULES)
@pytest.mark.parametrize(
    "forbidden_import",
    [
        "import sqlite3",
        "from fixpkg.persistence import thing",
        "from fixpkg.entrypoints import cli",
    ],
)
def test_services_may_not_reach_storage_or_entrypoints(
    repo_root: Path, tmp_path: Path, module_path: str, forbidden_import: str
):
    _make_package(
        tmp_path,
        "fixpkg",
        {
            "domain/model.py": "X = 1\n",
            "persistence/thing.py": "Y = 2\n",
            "entrypoints/cli.py": "Z = 3\n",
            module_path: forbidden_import + "\n",
        },
    )
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    result = _run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode != 0, result.stdout
    assert "BROKEN" in result.stdout


def test_the_yaml_codec_may_use_yaml_but_domain_modules_may_not(repo_root: Path, tmp_path: Path):
    _make_package(
        tmp_path, "fixpkg", {"domain/model.py": "X = 1\n", "sources/yaml_io.py": "import yaml\n"}
    )
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    ok = _run(["--config", str(config)], cwd=tmp_path)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    _make_package(
        tmp_path, "fixpkg2", {"domain/model.py": "X = 1\n", "sources/domain.py": "import yaml\n"}
    )
    config2 = _retarget_config(repo_root, tmp_path, "fixpkg2")
    bad = _run(["--config", str(config2)], cwd=tmp_path)
    assert bad.returncode != 0, bad.stdout


ENTRYPOINT_MODULES = ["entrypoints/cli.py", "entrypoints/api.py", "entrypoints/sub/handlers.py"]


@pytest.mark.parametrize("module_path", ENTRYPOINT_MODULES)
@pytest.mark.parametrize(
    "forbidden_import",
    [
        "import sqlite3",
        "from sqlite3 import connect",
        "from fixpkg.persistence import thing",
        "import fixpkg.persistence.thing",
    ],
)
def test_entrypoints_may_not_import_storage_directly(
    repo_root: Path, tmp_path: Path, module_path: str, forbidden_import: str
):
    _make_package(
        tmp_path,
        "fixpkg",
        {
            "domain/model.py": "X = 1\n",
            "persistence/thing.py": "Y = 2\n",
            module_path: forbidden_import + "\n",
        },
    )
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    result = _run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode != 0, result.stdout
    assert "Entrypoints must reach storage only through the composition root" in result.stdout
    assert "BROKEN" in result.stdout


def test_entrypoints_may_use_the_composition_root_that_imports_storage(
    repo_root: Path, tmp_path: Path
):
    _make_package(
        tmp_path,
        "fixpkg",
        {
            "domain/model.py": "X = 1\n",
            "persistence/thing.py": "Y = 2\n",
            "bootstrap.py": "from fixpkg.persistence import thing\nimport sqlite3\n",
            "entrypoints/cli.py": "from fixpkg import bootstrap\n",
        },
    )
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    result = _run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


NETWORK_CLIENTS = [
    "import httpx",
    "import httpx2",
    "import requests",
    "import aiohttp",
    "import websockets",
    "import http.client",
    "import socket",
    "import smtplib",
    "import ftplib",
    "import email",
    "import urllib3",
    "import webbrowser",
    "import xmlrpc",
    "import imaplib",
    "import poplib",
    "import telnetlib",
    "import socketserver",
    "import selectors",
]
WEB_OUTBOUND_CONTRACT = "The web UI must not make outbound network calls"


@pytest.mark.parametrize("statement", NETWORK_CLIENTS)
@pytest.mark.parametrize("module_path", ["entrypoints/web/app.py", "entrypoints/web/sub/views.py"])
def test_the_web_ui_may_not_import_a_network_client(
    repo_root: Path, tmp_path: Path, module_path: str, statement: str
):
    _make_package(tmp_path, "fixpkg", {"domain/model.py": "X = 1\n", module_path: statement + "\n"})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    result = _run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode != 0, result.stdout
    assert WEB_OUTBOUND_CONTRACT in result.stdout
    assert "BROKEN" in result.stdout


def test_the_web_ui_may_parse_urls(repo_root: Path, tmp_path: Path):
    _make_package(
        tmp_path,
        "fixpkg",
        {
            "domain/model.py": "X = 1\n",
            "entrypoints/web/app.py": "from urllib.parse import parse_qsl\n",
        },
    )
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    result = _run(["--config", str(config)], cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
