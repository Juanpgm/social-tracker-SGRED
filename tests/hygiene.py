"""Repository hygiene checker: finds secrets, cookies and runtime data in a working tree.

Pure standard library. Works with or without git: inside a git repository it honors
`.gitignore` (tracked + untracked-but-not-ignored files); otherwise it walks the tree and
skips well-known tooling directories.

Violation messages never include the offending value.
"""

import os
import re
import subprocess
from pathlib import Path

MAX_SCAN_BYTES = 1_000_000
_BINARY_SNIFF_BYTES = 8192

PRUNED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".import_linter_cache",
        ".hypothesis",
        "node_modules",
    }
)

_KEYWORDS = r"(?:password|passwd|secret|token|api[_-]?key)"

# key = "literal" / key: 'literal' / "key": "literal" (quoted literal of 3+ chars)
_QUOTED_CREDENTIAL = re.compile(
    rf"""(?ix) {_KEYWORDS} [\w.-]* ["']? \s* [:=] \s* ["'] [^"'\s]{{3,}} ["']"""
)
# KEY=value in env/ini style files (uppercase keys, unquoted literal)
_UNQUOTED_CREDENTIAL = re.compile(
    rf"""(?x) ^\s* [A-Z0-9_]* {_KEYWORDS.upper()} [A-Z0-9_]* \s* = \s* [^\s#'"]{{3,}} \s*$""",
)
_CODE_SUFFIXES = frozenset({".py", ".pyi", ".md", ".rst", ".txt"})


def _is_pruned(parts: tuple[str, ...]) -> bool:
    return any(part in PRUNED_DIRS for part in parts)


def _git_listed_files(root: Path) -> list[Path] | None:
    """Files git considers part of the tree (honors .gitignore), or None if not usable."""
    if not (root / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    names = [n for n in proc.stdout.decode("utf-8", errors="replace").split("\0") if n]
    return [Path(n) for n in names]


def _walked_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in PRUNED_DIRS]
        base = Path(dirpath).relative_to(root)
        found.extend(base / name for name in filenames)
    return found


def _candidate_files(root: Path) -> list[Path]:
    listed = _git_listed_files(root)
    if listed is None:
        files = _walked_files(root)
    else:
        files = listed
        if not any(not _is_pruned(p.parts) for p in files) and any(
            not _is_pruned(p.parts) for p in _walked_files(root)
        ):
            raise RuntimeError(
                "git listed no files for a non-empty working tree; refusing to pass vacuously"
            )
    return sorted(p for p in files if not _is_pruned(p.parts))


def scanned_files(root: Path) -> list[Path]:
    """Relative paths the checker considers for scanning (lets tests prove it is not vacuous)."""
    return _candidate_files(root)


def _forbidden_path_reason(rel: Path) -> str | None:
    parts = rel.parts
    name = parts[-1]
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "environment file"
    if name.endswith(".cookies"):
        return "cookies file"
    if "secrets" in parts[:-1]:
        return "file under a secrets directory"
    if parts[0] == "data" and len(parts) > 1:
        return "file under the data directory"
    return None


def _read_text(path: Path) -> str | None:
    """Return file text, or None when the file should not be scanned."""
    try:
        if path.is_symlink() or path.stat().st_size > MAX_SCAN_BYTES:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:_BINARY_SNIFF_BYTES]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _env_example_reason(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        value = stripped.split("=", 1)[1].strip().strip("'\"")
        if value:
            return "template env file defines a non-empty value"
    return None


def _credential_reason(rel: Path, text: str) -> str | None:
    scan_unquoted = rel.suffix not in _CODE_SUFFIXES
    for number, line in enumerate(text.splitlines(), start=1):
        if _QUOTED_CREDENTIAL.search(line):
            return f"line {number}: credential-like assignment with a literal value"
        if scan_unquoted and _UNQUOTED_CREDENTIAL.match(line):
            return f"line {number}: credential-like assignment with a literal value"
    return None


def find_violations(root: Path) -> list[str]:
    """Return one human-readable violation per offending file (empty list = clean)."""
    violations: list[str] = []
    for rel in _candidate_files(root):
        label = rel.as_posix()
        reason = _forbidden_path_reason(rel)
        if reason is None:
            text = _read_text(root / rel)
            if text is not None:
                if rel.name == ".env.example":
                    reason = _env_example_reason(text)
                if reason is None:
                    reason = _credential_reason(rel, text)
        if reason is not None:
            violations.append(f"{label}: {reason}")
    return violations
