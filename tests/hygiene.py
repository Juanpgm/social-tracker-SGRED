"""Repository hygiene checker: finds secrets, cookies and runtime data in a working tree.

Pure standard library. Works with or without git: inside a git repository it honors
`.gitignore` (tracked + untracked-but-not-ignored files); otherwise it walks the tree and
skips well-known tooling directories.

Scope and limits:
- The git file list uses `--exclude-standard`, so it includes tracked files AND untracked files
  that are not ignored. Results therefore depend on the local `.gitignore`, `.git/info/exclude`
  and global excludes: an ignored secret is intentionally not reported, but a locally excluded
  pattern also hides files from the scan. If git fails or lists nothing for a non-empty tree,
  the checker falls back to walking the tree or raises instead of passing vacuously.
- Detection is heuristic (path rules plus regexes); it cannot find encoded or split secrets.
- Symlinks are never followed; those resolving outside the repository are reported.

Violation messages never include the offending value.
"""

import fnmatch
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

_SENSITIVE_KEY = re.compile(
    r"(?i)password|passwd|secret|token|api[_-]?key|authorization"
    r"|(?:^|[_.-])auth(?:$|[_.-])|bearer|cookie|session[_-]?id"
)
# Keys that name a reference to a secret (env var, vault path, URL...) rather than the secret.
_BENIGN_KEY_SUFFIX = re.compile(
    r"(?i)[_.-](?:ref|env|var|env_var|url|uri|name|file|path|header|endpoint|count|ttl|type|prefix)$"
)
# key = "literal" / key: 'literal' / "key": "literal" / key = f"..." / key = ("...") / triple quotes
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<key>[\w.-]+) ["']? \s* [:=] \s* [frbu]{0,2} \(? \s*
    (?:\"\"\"|'''|"|') (?P<val>[^"'\s]{3,}) ["']
    """
)
# key = value / key: value on its own line (env, ini, yaml, toml style files)
_UNQUOTED = re.compile(
    r"""(?ix) ^\s* (?P<key>[\w.-]+) \s* (?P<sep>[:=]) \s* (?P<val>[^\s#'"]{3,}) \s* (?:\#.*)? $"""
)
_ENV_VAR_NAME = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_URL_WITHOUT_CREDENTIALS = re.compile(r"(?i)^[a-z][a-z0-9+.-]*://[^@\s]*$")
_NON_SECRET_WORDS = frozenset({"null", "none", "true", "false", "yes", "no"})

_CONTENT_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key material"),
    (re.compile(r"eyJ[\w-]+\.eyJ[\w-]+\."), "JWT-like token"),
    (
        re.compile(r"(?i)[a-z][a-z0-9+.-]*://[^\s/:@\"']+:(?![${<])[^\s/@\"']+@"),
        "URL with inline credentials",
    ),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"), "bearer token"),
)

_CODE_SUFFIXES = frozenset({".py", ".pyi", ".md", ".rst", ".txt"})
_COLON_SUFFIXES = frozenset({".yaml", ".yml", ".ini", ".cfg", ".toml", ".json"})
_KEY_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx"})
_PRIVATE_KEY_PREFIXES = ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")


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
    parts = tuple(part.casefold() for part in rel.parts)
    name = parts[-1]
    suffix = Path(name).suffix
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "environment file"
    if name.endswith(".cookies"):
        return "cookies file"
    if (
        fnmatch.fnmatchcase(name, "cookies*.txt")
        or fnmatch.fnmatchcase(name, "cookies*.json")
        or fnmatch.fnmatchcase(name, "*storage_state*.json")
    ):
        return "browser session file"
    if suffix in _KEY_SUFFIXES or name.startswith(_PRIVATE_KEY_PREFIXES):
        return "private key or certificate file"
    if name == ".netrc":
        return "netrc credentials file"
    if fnmatch.fnmatchcase(name, "*.db") or fnmatch.fnmatchcase(name, "*.sqlite*"):
        return "local database file"
    if "secrets" in parts[:-1]:
        return "file under a secrets directory"
    if parts[0] == "data" and len(parts) > 1:
        return "file under the data directory"
    return None


def _symlink_reason(root: Path, path: Path) -> str | None:
    """Flag symlinks that point outside the repository; links inside are never followed."""
    try:
        target = path.resolve(strict=True)
        inside = target.is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return None  # dangling or looping link: nothing to read, nothing to leak
    return None if inside else "symlink resolving outside the repository"


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


def _is_benign_value(value: str) -> bool:
    return (
        value.casefold() in _NON_SECRET_WORDS
        or value.startswith(("$", "{", "<", "%("))
        or _ENV_VAR_NAME.match(value) is not None
        or _URL_WITHOUT_CREDENTIALS.match(value) is not None
    )


def _is_credential_assignment(key: str, value: str) -> bool:
    return (
        _SENSITIVE_KEY.search(key) is not None
        and _BENIGN_KEY_SUFFIX.search(key) is None
        and not _is_benign_value(value)
    )


def _credential_reason(rel: Path, text: str) -> str | None:
    suffix = rel.suffix.casefold()
    scan_unquoted = suffix not in _CODE_SUFFIXES
    allow_colon = suffix in _COLON_SUFFIXES
    for number, line in enumerate(text.splitlines(), start=1):
        for pattern, reason in _CONTENT_RULES:
            if pattern.search(line):
                return f"line {number}: {reason}"
        for match in _ASSIGNMENT.finditer(line):
            if _is_credential_assignment(match["key"], match["val"]):
                return f"line {number}: credential-like assignment with a literal value"
        if scan_unquoted:
            match = _UNQUOTED.match(line)
            if (
                match
                and (match["sep"] == "=" or allow_colon)
                and _is_credential_assignment(match["key"], match["val"])
            ):
                return f"line {number}: credential-like assignment with a literal value"
    return None


def find_violations(root: Path) -> list[str]:
    """Return one human-readable violation per offending file (empty list = clean)."""
    violations: list[str] = []
    for rel in _candidate_files(root):
        label = rel.as_posix()
        reason = _forbidden_path_reason(rel)
        if reason is None and (root / rel).is_symlink():
            reason = _symlink_reason(root, root / rel)
            if reason is not None:
                violations.append(f"{label}: {reason}")
            continue
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
