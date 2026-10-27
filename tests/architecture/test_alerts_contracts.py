"""Architecture contracts of the alerts (AL-7, G-1): stored and shown on this machine, nothing sent
out. Same technique as `test_llm_contracts`: the real `.importlinter`, retargeted at a throwaway
package with deliberate violations (or legal imports), must break (or hold)."""

import ast
import re
from pathlib import Path

import pytest

from tests.architecture.test_import_contracts import _make_package, _retarget_config, _run
from tests.architecture.test_llm_contracts import section

IO_CONTRACT = "The alerts module must do no I/O and send nothing out"
PURE_CONTRACT = "The alert vocabulary and excerpt must be pure and deterministic"
OUTBOUND_CONTRACT = "Only the fetcher adapter may open network connections"
BASE = {"domain/model.py": "X = 1\n"}
OUTSIDE = {
    "persistence/thing.py": "Y = 2\n",
    "entrypoints/thing.py": "Y = 2\n",
    "collector/thing.py": "Y = 2\n",
    "control/thing.py": "Y = 2\n",
    "items/thing.py": "Y = 2\n",
    "sources/thing.py": "Y = 2\n",
    "llm/ollama.py": "Y = 2\n",
    "bootstrap.py": "Y = 2\n",
}
MODULES = ["domain", "excerpt", "ports", "service"]
FORBIDDEN_EVERYWHERE = {
    "import sqlite3": "sqlite3",
    "import socket": "socket",
    "import ssl": "ssl",
    "import subprocess": "subprocess",
    "import smtplib": "smtplib",
    "import ftplib": "ftplib",
    "import httpx2": "httpx2",
    "import requests": "requests",
    "import aiohttp": "aiohttp",
    "import urllib": "urllib",
    "import http": "http",
    "import os": "os",
    "import pathlib": "pathlib",
    "import asyncio": "asyncio",
    "import yaml": "yaml",
    "from fixpkg.persistence import thing": "fixpkg.persistence.thing",
    "from fixpkg.entrypoints import thing": "fixpkg.entrypoints.thing",
    "from fixpkg.collector import thing": "fixpkg.collector.thing",
    "from fixpkg.llm import ollama": "fixpkg.llm.ollama",
    "from fixpkg.bootstrap import Y": "fixpkg.bootstrap",
}
FORBIDDEN_IN_THE_PURE_PART = {
    "import random": "random",
    "import time": "time",
    "import secrets": "secrets",
    "import uuid": "uuid",
    "import logging": "logging",
    "import threading": "threading",
    "from fixpkg.control import thing": "fixpkg.control.thing",
    "from fixpkg.items import thing": "fixpkg.items.thing",
    "from fixpkg.sources import thing": "fixpkg.sources.thing",
}


def lint(repo_root: Path, tmp_path: Path, files: dict[str, str]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    _make_package(tmp_path, "fixpkg", {**BASE, **OUTSIDE, **files})
    config = _retarget_config(repo_root, tmp_path, "fixpkg")
    return _run(["--config", str(config)], cwd=tmp_path)


def lines(statements: dict[str, str]) -> str:
    return "\n".join(statements) + "\n"


@pytest.mark.parametrize("module", MODULES)
def test_the_alerts_do_no_io_and_reach_neither_storage_nor_the_model_adapter(
    repo_root: Path, tmp_path: Path, module: str
):
    result = lint(repo_root, tmp_path, {f"alerts/{module}.py": lines(FORBIDDEN_EVERYWHERE)})
    assert result.returncode != 0, result.stdout
    report = section(result.stdout, IO_CONTRACT)
    for target in FORBIDDEN_EVERYWHERE.values():
        assert f"fixpkg.alerts.{module} -> {target}" in report, target


@pytest.mark.parametrize("module", ["domain", "excerpt", "ports"])
def test_the_vocabulary_the_excerpt_and_the_ports_are_pure(
    repo_root: Path, tmp_path: Path, module: str
):
    result = lint(repo_root, tmp_path, {f"alerts/{module}.py": lines(FORBIDDEN_IN_THE_PURE_PART)})
    assert result.returncode != 0, result.stdout
    report = section(result.stdout, PURE_CONTRACT)
    for target in FORBIDDEN_IN_THE_PURE_PART.values():
        assert f"fixpkg.alerts.{module} -> {target}" in report, target


def test_the_service_may_use_the_other_modules_through_their_ports_and_log(
    repo_root: Path, tmp_path: Path
):
    files = {
        "alerts/service.py": (
            "import logging\nimport json\nimport hashlib\nfrom collections import Counter\n"
            "from fixpkg.control import thing\nfrom fixpkg.items import thing as t2\n"
            "from fixpkg.sources import thing as t3\nfrom fixpkg.alerts import domain, ports\n"
            "from fixpkg.domain import model\n"
        ),
        "alerts/domain.py": "import re\nimport json\nimport hashlib\nimport unicodedata\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_wiring_may_build_the_alert_store(repo_root: Path, tmp_path: Path):
    files = {
        "alerts/service.py": "X = 1\n",
        "bootstrap.py": "from fixpkg.alerts import service\nfrom fixpkg.persistence import thing\n",
    }
    result = lint(repo_root, tmp_path, files)
    assert result.returncode == 0, result.stdout + result.stderr


def test_only_the_fetcher_may_open_connections_so_the_alerts_may_not(
    repo_root: Path, tmp_path: Path
):
    files = {f"alerts/{module}.py": "import httpx2\n" for module in MODULES}
    result = lint(repo_root, tmp_path, files)
    assert result.returncode != 0
    report = section(result.stdout, OUTBOUND_CONTRACT)
    for module in MODULES:
        assert f"fixpkg.alerts.{module} -> httpx2" in report


# ---- what the import rules cannot see: no way out is even spelled ----

ALERTS = Path(__file__).parents[2] / "src" / "social_tracker" / "alerts"
# Whole words, taken from snake_case and dotted names: `published_at` (a time of the item) is fine,
# `publish` is not.
NOTIFY_WORDS = frozenset(
    {
        "publish", "publication", "publisher", "webhook", "smtp", "email", "mail", "notify",
        "notification", "notifications", "sendmail", "tweet", "share", "sharing", "debunk",
        "draft", "approve", "approval", "broadcast", "slack", "telegram", "whatsapp", "push",
    }
)  # fmt: skip


def words(name: str) -> set[str]:
    return {w for w in re.split(r"[^a-z]+", name.lower()) if w}


def notifying_names(source: str) -> list[str]:
    """Every name (function, class, variable, attribute, import) in `source` that spells a way to
    tell somebody something."""
    found = []
    for node in ast.walk(ast.parse(source)):
        names: list[str] = []
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.alias):
            names.append(node.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        found += [n for n in names if words(n) & NOTIFY_WORDS]
    return found


def test_the_scanner_finds_what_it_is_for_and_leaves_alone_what_it_is_not():
    nl = chr(10)
    assert notifying_names(nl.join(["def publish_alert(a):", "    send_email(a)"])) == [
        "publish_alert",
        "send_email",
    ]
    assert notifying_names(
        nl.join(["import smtplib", "from x import webhook", "x.draft_reply()"])
    ) == [
        "webhook",
        "draft_reply",
    ]
    assert notifying_names(nl.join(["published_at = shared_clock", "x.notifier_off()"])) == []
    assert notifying_names("class Approval: pass") == ["Approval"]


def test_the_alerts_package_spells_no_way_to_notify_anybody():
    """G-1: no publication, no webhook, no e-mail, no draft of a debunk, no approval workflow."""
    offenders = []
    for path in ALERTS.glob("*.py"):
        offenders += [
            f"{path.name}: {n}" for n in notifying_names(path.read_text(encoding="utf-8"))
        ]
    assert offenders == []


def test_the_alerts_package_has_no_clock_of_its_own():
    """The service takes the clock by injection and the pure part knows no time at all."""
    for path in ALERTS.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for call in ("datetime.now(", "utcnow(", "time.time(", "time.monotonic(", "date.today("):
            assert call not in source, f"{path.name}: {call}"
