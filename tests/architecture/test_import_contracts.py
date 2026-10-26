"""Architecture contract: the domain layer stays pure.

The real contract lives in `.importlinter`. These tests run `lint-imports` against it, and
prove it is not vacuous by running the same contract (retargeted at a throwaway package)
against deliberate violations.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CONFIG_NAME = ".importlinter"
FORBIDDEN_EXTERNALS = ["sqlite3", "yaml", "typer", "httpx", "requests"]


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


def _retarget_config(repo_root: Path, dest: Path, root_package: str) -> Path:
    text = (repo_root / CONFIG_NAME).read_text(encoding="utf-8")
    config = dest / CONFIG_NAME
    config.write_text(text.replace("social_tracker", root_package), encoding="utf-8")
    return config


def test_real_contract_passes_on_the_repository(repo_root: Path):
    result = _run(["--config", str(repo_root / CONFIG_NAME)], cwd=repo_root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_config_declares_every_forbidden_external(repo_root: Path):
    text = (repo_root / CONFIG_NAME).read_text(encoding="utf-8")
    for module in FORBIDDEN_EXTERNALS:
        assert module in text, f"{module} missing from the domain purity contract"
    for layer in ("adapters", "persistence", "entrypoints"):
        assert layer in text, f"{layer} missing from the domain purity contract"


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


@pytest.mark.parametrize("layer", ["adapters", "persistence", "entrypoints"])
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
