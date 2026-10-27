"""Architecture contract of the classification core: pure, deterministic, self-contained.

Same technique as `test_import_contracts`: the real `.importlinter`, retargeted at a throwaway
package with one deliberate violation (or one legal import), must break (or hold).
"""

from pathlib import Path

import pytest

from tests.architecture.test_import_contracts import (
    CLASSIFICATION_CORE_MODULES,
    _make_package,
    _retarget_config,
    _run,
)

CONTRACT = "The classification core must be pure and deterministic"
CORE = list(CLASSIFICATION_CORE_MODULES)
BASE = {"domain/model.py": "X = 1\n"}
FORBIDDEN = [
    "import sqlite3",
    "import yaml",
    "import random",
    "import time",
    "import secrets",
    "import uuid",
    "import os",
    "import socket",
    "import subprocess",
    "from fixpkg.persistence import thing",
    "from fixpkg.control import thing",
    "from fixpkg.items import thing",
    "from fixpkg.classification import service",
    "from fixpkg.classification import ports",
    "from fixpkg.classification import yaml_io",
]
OUTSIDE = {
    "persistence/thing.py": "Y = 2\n",
    "control/thing.py": "Y = 2\n",
    "items/thing.py": "Y = 2\n",
    "classification/service.py": "Y = 2\n",
    "classification/ports.py": "Y = 2\n",
    "classification/yaml_io.py": "Y = 2\n",
}


def lint(repo_root: Path, tmp_path: Path, files: dict[str, str]):
    _make_package(tmp_path, "fixpkg", {**BASE, **files})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    return _run(["--config", str(config)], cwd=tmp_path)


@pytest.mark.parametrize("module", CORE)
@pytest.mark.parametrize("statement", FORBIDDEN)
def test_the_core_modules_may_not_import_io_nondeterminism_or_the_outer_layers(
    repo_root: Path, tmp_path: Path, module: str, statement: str
):
    files = {**OUTSIDE, f"classification/{module}.py": statement + "\n"}
    result = lint(repo_root, tmp_path, files)
    assert result.returncode != 0, result.stdout
    assert CONTRACT in result.stdout
    assert "BROKEN" in result.stdout


def test_the_core_modules_may_import_each_other_and_the_shared_kernel(
    repo_root: Path, tmp_path: Path
):
    files = {
        "shared/errors.py": "E = 1\n",
        "classification/text.py": "import re\nimport unicodedata\nimport hashlib\n",
        "classification/similarity.py": "from fixpkg.shared import errors\nimport struct\n",
        "classification/domain.py": "from fixpkg.classification import similarity\nimport json\n",
        "classification/topics.py": "from fixpkg.classification import domain, text\n",
        "classification/rules.py": "from fixpkg.classification import domain, text, topics\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_service_may_use_the_core_and_the_ports_but_not_persistence(
    repo_root: Path, tmp_path: Path
):
    files = {
        "classification/rules.py": "R = 1\n",
        "classification/ports.py": "P = 1\n",
        "classification/service.py": "from fixpkg.classification import rules, ports\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr
