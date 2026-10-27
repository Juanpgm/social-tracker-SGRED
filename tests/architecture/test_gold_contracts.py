"""Architecture contracts of the gold set (G-1): labeled texts stay on this machine. Same technique
as `test_alerts_contracts`: the real `.importlinter`, retargeted at a throwaway package with
deliberate violations (or legal imports), must break (or hold)."""

from pathlib import Path

import pytest

from tests.architecture.test_alerts_contracts import lines
from tests.architecture.test_import_contracts import _make_package, _retarget_config, _run
from tests.architecture.test_llm_contracts import section

IO_CONTRACT = "The gold set must do no I/O of its own and send nothing out"
PURE_CONTRACT = "The gold vocabulary and ports must be pure and deterministic"
BASE = {"domain/model.py": "X = 1\n"}
OUTSIDE = {
    "persistence/thing.py": "Y = 2\n",
    "entrypoints/thing.py": "Y = 2\n",
    "collector/thing.py": "Y = 2\n",
    "control/thing.py": "Y = 2\n",
    "items/thing.py": "Y = 2\n",
    "llm/ollama.py": "Y = 2\n",
    "bootstrap.py": "Y = 2\n",
}
MODULES = ["domain", "ports", "service", "yaml_io"]
EVERYWHERE = {
    "import sqlite3": "sqlite3",
    "import socket": "socket",
    "import ssl": "ssl",
    "import subprocess": "subprocess",
    "import smtplib": "smtplib",
    "import httpx2": "httpx2",
    "import requests": "requests",
    "import urllib": "urllib",
    "import http": "http",
    "import os": "os",
    "import pathlib": "pathlib",
    "import typer": "typer",
    "from fixpkg.persistence import thing": "fixpkg.persistence.thing",
    "from fixpkg.entrypoints import thing": "fixpkg.entrypoints.thing",
    "from fixpkg.collector import thing": "fixpkg.collector.thing",
    "from fixpkg.llm import ollama": "fixpkg.llm.ollama",
    "from fixpkg.bootstrap import Y": "fixpkg.bootstrap",
}
PURE_ONLY = {
    "import yaml": "yaml",
    "import random": "random",
    "import time": "time",
    "import logging": "logging",
    "import threading": "threading",
    "from fixpkg.control import thing": "fixpkg.control.thing",
    "from fixpkg.llm import service": "fixpkg.llm.service",
}


def lint(repo_root: Path, tmp_path: Path, files: dict[str, str]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _make_package(tmp_path, "fixpkg", {**BASE, **OUTSIDE, **files})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    return _run(["--config", str(config)], cwd=tmp_path)


@pytest.mark.parametrize("module", MODULES)
def test_the_gold_set_does_no_io_and_reaches_neither_storage_nor_the_model_adapter(
    repo_root: Path, tmp_path: Path, module: str
):
    result = lint(repo_root, tmp_path, {f"gold/{module}.py": lines(EVERYWHERE)})
    assert result.returncode != 0, result.stdout
    report = section(result.stdout, IO_CONTRACT)
    for target in EVERYWHERE.values():
        assert f"fixpkg.gold.{module} -> {target}" in report, target


@pytest.mark.parametrize("module", ["domain", "ports"])
def test_the_vocabulary_and_the_ports_are_pure(repo_root: Path, tmp_path: Path, module: str):
    result = lint(repo_root, tmp_path, {f"gold/{module}.py": lines(PURE_ONLY)})
    assert result.returncode != 0, result.stdout
    report = section(result.stdout, PURE_CONTRACT)
    for target in PURE_ONLY.values():
        assert f"fixpkg.gold.{module} -> {target}" in report, target


def test_the_codec_may_use_yaml_and_the_service_the_rest_of_the_application(
    repo_root: Path, tmp_path: Path
):
    files = {
        "gold/yaml_io.py": "import yaml\nfrom fixpkg.gold import domain\n",
        "gold/service.py": (
            "import logging\n"
            "from fixpkg.control import thing\n"
            "from fixpkg.items import thing as t2\n"
            "from fixpkg.llm import domain as d\n"
            "from fixpkg.gold import domain, ports, yaml_io\n"
        ),
        "gold/domain.py": "import hashlib\nimport unicodedata\nfrom fixpkg.items import thing\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_wiring_may_build_the_gold_store(repo_root: Path, tmp_path: Path):
    files = {
        "gold/service.py": "X = 1\n",
        "bootstrap.py": "from fixpkg.gold import service\nfrom fixpkg.persistence import thing\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr
