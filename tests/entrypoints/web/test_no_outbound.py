"""The web UI never opens a connection of its own (G-1).

import-linter cannot name `urllib.request` on its own (and the UI needs `urllib.parse`), so the
one module it misses is checked here from the syntax tree.
"""

import ast
from pathlib import Path

import pytest

import social_tracker.entrypoints.web as web_package

BANNED = ("urllib.request", "http.client", "socket", "ssl", "smtplib", "ftplib")


def imported_modules(source: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def is_banned(module: str) -> bool:
    return any(module == name or module.startswith(name + ".") for name in BANNED)


@pytest.mark.parametrize(
    "snippet",
    [
        "import urllib.request",
        "from urllib.request import urlopen",
        "from urllib import request\nrequest.urlopen",
        "import socket",
    ],
)
def test_the_checker_recognises_outbound_imports(snippet: str):
    assert any(is_banned(m) for m in imported_modules(snippet)), snippet


def test_the_checker_lets_form_parsing_through():
    assert not any(is_banned(m) for m in imported_modules("from urllib.parse import parse_qsl"))


def test_no_module_of_the_web_ui_imports_a_network_module():
    root = Path(web_package.__file__).parent
    files = sorted(root.rglob("*.py"))
    assert len(files) >= 8
    for path in files:
        offending = [m for m in imported_modules(path.read_text("utf-8")) if is_banned(m)]
        assert offending == [], (path.name, offending)
