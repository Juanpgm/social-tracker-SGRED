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
    "import email": "email",
    "import urllib3": "urllib3",
    "import webbrowser": "webbrowser",
    "import xmlrpc": "xmlrpc",
    "import imaplib": "imaplib",
    "import poplib": "poplib",
    "import telnetlib": "telnetlib",
    "import socketserver": "socketserver",
    "import selectors": "selectors",
    "from fixpkg.persistence import thing": "fixpkg.persistence.thing",
    "from fixpkg.entrypoints import thing": "fixpkg.entrypoints.thing",
    "from fixpkg.collector import thing": "fixpkg.collector.thing",
    "from fixpkg.llm import ollama": "fixpkg.llm.ollama",
    "from fixpkg.bootstrap import Y": "fixpkg.bootstrap",
}
FORBIDDEN_IN_THE_PURE_PART = dict(
    [
        ("import random", "random"),
        ("import time", "time"),
        ("import secrets", "secrets"),
        ("import uuid", "uuid"),
        ("import logging", "logging"),
        ("import threading", "threading"),
        ("from fixpkg.control import thing", "fixpkg.control.thing"),
        ("from fixpkg.items import thing", "fixpkg.items.thing"),
        ("from fixpkg.sources import thing", "fixpkg.sources.thing"),
    ]
)  # pairs, not `key: value`, so that the secret scanner does not mistake `secrets` for a secret


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


# ---- G-1 beyond the alerts package: the gold set, the alert pages and the alert/gold commands ----

SRC = Path(__file__).parents[2] / "src" / "social_tracker"
GOLD = SRC / "gold"
WEB = SRC / "entrypoints" / "web"
CLI = SRC / "entrypoints" / "cli.py"
ALERT_TEMPLATES = sorted(WEB.glob("templates/*alert*.html"))
_JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.DOTALL)
_ATTRIBUTE = re.compile(r'(?<=[\s<])([a-z][a-z0-9-]*)="([^"]*)"')
_PROSE_ATTRIBUTES = frozenset({"aria-label", "title", "alt", "placeholder", "content"})


def template_names(source: str) -> list[str]:
    """Every name a template spells that is not prose: the identifiers inside its `{{ }}` and
    `{% %}` blocks and the values of its attributes (ids, names, links, targets, classes...)."""
    found: list[str] = []
    for block in _JINJA.findall(source):
        found += [
            w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", block) if words(w) & NOTIFY_WORDS
        ]
    markup = _JINJA.sub(" ", source)
    for attribute, value in _ATTRIBUTE.findall(markup):
        if attribute in _PROSE_ATTRIBUTES:
            continue
        found += [w for w in re.split(r"[^A-Za-z0-9_]+", value) if w and words(w) & NOTIFY_WORDS]
    return found


def command_names(source: str) -> list[str]:
    """The notifying names inside the alert and gold commands of the CLI (and their helpers)."""
    tree = ast.parse(source)
    found: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorated = any(
            "alerts_app" in ast.unparse(d)
            or "gold_app" in ast.unparse(d)
            or "threshold_app" in ast.unparse(d)
            for d in node.decorator_list
        )
        if decorated or re.search(r"alert|gold", node.name):
            found += notifying_names(ast.unparse(node))
    return found


def test_the_template_scanner_reads_names_not_prose():
    page = (
        '<form action="/alerts/{{ a.id }}/share" hx-post="/alerts/x/seen" class="push">'
        '<input name="notify_me"><button aria-label="Send an email">Notifications are off'
        "</button></form>{% set webhook = 1 %}{{ publish(a) }}"
    )
    assert template_names(page) == ["webhook", "publish", "share", "push", "notify_me"]
    assert template_names('<p class="hint">There are no external notifications.</p>') == []
    assert template_names('<a href="/alerts?tag=earthquake">earthquake</a>') == []


def test_the_command_scanner_finds_a_command_that_would_notify():
    source = "\n".join(
        [
            "@alerts_app.command('x')",
            "def alerts_x():",
            "    send_email(1)",
            "def unrelated():",
            "    send_email(2)",
            "def _gold_text():",
            "    webhook = 1",
        ]
    )
    assert command_names(source) == ["send_email", "webhook"]


def test_the_gold_package_spells_no_way_to_notify_anybody():
    offenders = []
    for path in GOLD.glob("*.py"):
        offenders += [
            f"{path.name}: {n}" for n in notifying_names(path.read_text(encoding="utf-8"))
        ]
    assert offenders == []


def test_the_alert_pages_route_spells_no_way_to_notify_anybody():
    source = (WEB / "alert_routes.py").read_text(encoding="utf-8")
    assert notifying_names(source) == []


def test_the_alert_templates_spell_no_way_to_notify_anybody():
    assert ALERT_TEMPLATES, "the scan would be vacuous"
    offenders = []
    for path in ALERT_TEMPLATES:
        offenders += [f"{path.name}: {n}" for n in template_names(path.read_text("utf-8"))]
    assert offenders == []


def test_the_alert_and_gold_commands_spell_no_way_to_notify_anybody():
    names = command_names(CLI.read_text(encoding="utf-8"))
    assert names == []


def test_the_scans_really_cover_the_files_they_are_for():
    assert len(list(GOLD.glob("*.py"))) >= 4
    assert any(p.name == "alerts.html" for p in ALERT_TEMPLATES)
    assert any(p.name == "alert_detail.html" for p in ALERT_TEMPLATES)
    cli_source = CLI.read_text(encoding="utf-8")
    assert "gold_app.command" in cli_source
    assert "alerts_app.command" in cli_source
