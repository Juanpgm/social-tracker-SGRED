"""Repository hygiene: no secrets, cookies or runtime data in the working tree.

The checker lives in `tests/hygiene.py`. It is exercised against fixtures in `tmp_path`
(including its failure modes) and then run against the real repository.

Fixture strings that look like credentials are assembled from fragments so this file does
not trip the very scanner it tests.
"""

import os
import subprocess
from pathlib import Path

import pytest

from tests import hygiene
from tests.hygiene import MAX_SCAN_BYTES, find_violations, scanned_files

PW = "pass" + "word"
TOK = "to" + "ken"
KEY = "api" + "_key"


def _write(root: Path, rel: str, content: str | bytes = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


# --- forbidden paths -------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        ".env",
        ".env.local",
        "session.cookies",
        "nested/dir/browser.cookies",
        "secrets/key.pem",
        "pkg/secrets/inner.txt",
        "data/risk.db",
        "data/sub/dump.json",
    ],
)
def test_forbidden_paths_are_flagged(tmp_path: Path, rel: str):
    _write(tmp_path, rel, "x")
    violations = find_violations(tmp_path)
    assert any(rel.replace("\\", "/") in v for v in violations), violations


@pytest.mark.parametrize(
    "rel",
    [
        ".ENV",
        ".Env.Local",
        "Session.COOKIES",
        "Data/risk.json",
        "DATA/sub/dump.json",
        "Secrets/a.txt",
        "pkg/SECRETS/inner.txt",
        "ID_RSA",
        "Server.PEM",
        "Deploy/Private.KEY",
        "RISK.DB",
        "Risk.SQLite3",
        "Cookies.TXT",
        "Auth/Storage_State.JSON",
        ".NETRC",
    ],
)
def test_forbidden_paths_are_matched_case_insensitively(tmp_path: Path, rel: str):
    _write(tmp_path, rel, "x")
    assert find_violations(tmp_path), rel


@pytest.mark.parametrize(
    "rel",
    ["src/app.py", "docs/data.md", "src/pkg/data/__init__.py", "notes.cookies.md", "env.txt"],
)
def test_ordinary_paths_are_not_flagged(tmp_path: Path, rel: str):
    _write(tmp_path, rel, "x = 1\n")
    assert find_violations(tmp_path) == []


def test_empty_tree_has_no_violations(tmp_path: Path):
    assert find_violations(tmp_path) == []


# --- .env.example ----------------------------------------------------------------------


def test_env_example_with_empty_values_is_allowed(tmp_path: Path):
    _write(tmp_path, ".env.example", f"# comment\n\nDB_PATH=\n{KEY.upper()}=\nOTHER=''\n")
    assert find_violations(tmp_path) == []


def test_env_example_with_a_value_fails(tmp_path: Path):
    _write(tmp_path, ".env.example", "DB_PATH=/var/lib/tracker.db\n")
    violations = find_violations(tmp_path)
    assert len(violations) == 1
    assert ".env.example" in violations[0]


def test_env_example_with_a_secret_looking_value_fails(tmp_path: Path):
    _write(tmp_path, ".env.example", f"{KEY.upper()}=abc123realvalue\n")
    assert find_violations(tmp_path)


# --- credential content ----------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        f'{PW} = "hunter2hunter2"',
        f"{TOK}: 'abcdef123456'",
        f'{KEY}="sk-live-1234567890"',
        f'API_{TOK.upper()} = "abcdef123456"',
        f"{PW.upper()}=hunter2hunter2",
        f'  "{TOK}": "abcdef123456",',
    ],
)
def test_credential_assignments_with_literals_are_flagged(tmp_path: Path, line: str):
    _write(tmp_path, "src/config.env", f"{line}\n")
    violations = find_violations(tmp_path)
    assert violations, line
    assert "src/config.env" in violations[0]
    assert "hunter2" not in violations[0], "must not leak value"
    assert "abcdef" not in violations[0], "must not leak value"


@pytest.mark.parametrize(
    "line",
    [
        f'{PW} = ""',
        f"{PW} = os.environ['X']",
        f"{TOK} = get_{TOK}()",
        f"{KEY} = None",
        f"def check({PW}: str) -> bool:",
        f"# {PW}: describe the field here",
        f"{TOK}_count = 3",
        "x = 1",
    ],
)
def test_benign_code_is_not_flagged(tmp_path: Path, line: str):
    _write(tmp_path, "src/ok.py", f"{line}\n")
    assert find_violations(tmp_path) == []


def test_violations_are_reported_for_every_offending_file(tmp_path: Path):
    _write(tmp_path, "a.py", f'{PW} = "hunter2hunter2"\n')
    _write(tmp_path, "b.py", f'{TOK} = "abcdef123456"\n')
    _write(tmp_path, "c.py", "ok = 1\n")
    violations = find_violations(tmp_path)
    assert len(violations) == 2


# --- realistic secret artifacts ---------------------------------------------------------

_HUNTER = "hunter2" + "hunter2"
_JWT = "ey" + "Jhbgc" + "iOiJIUzI1" + "." + "ey" + "Jzdwii" + "OiIxMjM0" + "." + "c2lnbmF0dXJl"
_PEM_HEADER = "-----" + "BEGIN RSA PRIV" + "ATE KEY-----"
_DB_URL = "postgres://" + "svc_user:" + "s3cr3tpw" + "@db.internal:5432/risk"

SECRET_ARTIFACTS = [
    pytest.param("config.yaml", f"{PW}: {_HUNTER}\n", id="yaml-lowercase-colon"),
    pytest.param("settings.ini", f"{PW} = {_HUNTER}\n", id="ini-lowercase-equals"),
    pytest.param("app.py", f'{PW} = """{_HUNTER}"""\n', id="py-triple-quoted"),
    pytest.param("app.py", f'{PW} = f"{_HUNTER}{{suffix}}"\n', id="py-fstring"),
    pytest.param("app.py", f'{PW} = ("{_HUNTER}")\n', id="py-parenthesised"),
    pytest.param("app.py", f'H = {{"Authorization": "Bearer {_JWT}"}}\n', id="py-bearer-jwt"),
    pytest.param("notes.txt", f"Authorization: Bearer {_JWT}\n", id="txt-bearer-header"),
    pytest.param("app.py", f'SESSIONID = "{_HUNTER}"\n', id="py-sessionid"),
    pytest.param("id_rsa", "x\n", id="path-id_rsa"),
    pytest.param("server.pem", "x\n", id="path-pem"),
    pytest.param(".netrc", "x\n", id="path-netrc"),
    pytest.param("deploy/key.txt", f"{_PEM_HEADER}\nabc\n", id="content-private-key"),
    pytest.param("app.py", f'URL = "{_DB_URL}"\n', id="py-db-url-inline-password"),
    pytest.param("cookies.txt", "x\n", id="path-cookies-txt"),
    pytest.param("auth/storage_state.json", "{}\n", id="path-storage-state"),
    pytest.param("risk.sqlite", "x\n", id="path-sqlite-outside-data"),
    pytest.param("local.db", "x\n", id="path-db-outside-data"),
]


@pytest.mark.parametrize(("rel", "content"), SECRET_ARTIFACTS)
def test_realistic_secret_artifacts_are_flagged(tmp_path: Path, rel: str, content: str):
    _write(tmp_path, rel, content)
    violations = find_violations(tmp_path)
    assert violations, rel
    joined = " ".join(violations)
    for leaked in (_HUNTER, _JWT, "s3cr3tpw"):
        assert leaked not in joined, "must not leak value"


@pytest.mark.parametrize(
    "line",
    [
        'TOKEN_URL = "https://x.example/oauth"',
        'api_key_env = "SOCIAL_API_KEY"',
        'secret_ref = "vault/social/api"',
        'PASSWORD_ENV_VAR = "DB_PASSWORD"',
        'COOKIE_NAME = "sid"',
        'AUTHOR = "Jane Doe"',
        'URL = "http://localhost:8080/path"',
        'URL = "https://user@example.org/path"',
        'URL = "https://user:${DB_PASS}@host/db"',
        "Authorization = 'Bearer ' + value",
        'session = "main"',
    ],
)
def test_config_constants_and_env_var_names_are_not_flagged(tmp_path: Path, line: str):
    _write(tmp_path, "src/settings.py", f"{line}\n")
    assert find_violations(tmp_path) == []


@pytest.mark.parametrize(
    "content",
    [
        f"{PW}: null\n",
        f"{PW}: ${{DB_PASSWORD}}\n",
        f"{TOK}_url: https://x.example/oauth\n",
        f"{KEY}_env: SOCIAL_API_KEY\n",
        f"{PW}:\n",
    ],
)
def test_structured_config_without_literal_secrets_is_not_flagged(tmp_path: Path, content: str):
    _write(tmp_path, "config.yaml", content)
    assert find_violations(tmp_path) == []


# --- robustness ------------------------------------------------------------------------


def test_binary_files_are_skipped_without_crashing(tmp_path: Path):
    _write(tmp_path, "blob.bin", b"\x00\x01\x02" + f'{PW} = "hunter2hunter2"'.encode())
    assert find_violations(tmp_path) == []


def test_non_utf8_files_are_skipped_without_crashing(tmp_path: Path):
    _write(tmp_path, "latin.txt", b"caf\xe9 \xff\xfe broken bytes\n")
    assert find_violations(tmp_path) == []


def test_files_over_the_size_guard_are_not_scanned(tmp_path: Path):
    payload = f'{PW} = "hunter2hunter2"\n' + "a" * (MAX_SCAN_BYTES + 1)
    _write(tmp_path, "huge.txt", payload)
    assert find_violations(tmp_path) == []


def test_file_just_under_the_size_guard_is_scanned(tmp_path: Path):
    line = f'{PW} = "hunter2hunter2"\n'
    _write(tmp_path, "big.txt", line + "a" * (MAX_SCAN_BYTES - len(line) - 1))
    assert find_violations(tmp_path)


def test_tooling_directories_are_pruned(tmp_path: Path):
    _write(tmp_path, ".venv/lib/x.py", f'{PW} = "hunter2hunter2"\n')
    _write(tmp_path, ".git/config", f'{PW} = "hunter2hunter2"\n')
    _write(tmp_path, "pkg/__pycache__/x.py", f'{PW} = "hunter2hunter2"\n')
    _write(tmp_path, ".ruff_cache/x", f'{PW} = "hunter2hunter2"\n')
    assert find_violations(tmp_path) == []


def test_unreadable_entries_do_not_crash(tmp_path: Path):
    """A dangling symlink (or any file that vanishes mid-walk) must not abort the scan."""
    target = tmp_path / "missing-target"
    link = tmp_path / "dangling"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this platform")
    assert find_violations(tmp_path) == []


def test_read_errors_do_not_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Permission errors / files vanishing mid-scan are skipped, not raised."""
    _write(tmp_path, "locked.py", "x = 1\n")

    def deny(self: Path) -> bytes:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", deny)
    assert find_violations(tmp_path) == []


# --- git awareness ---------------------------------------------------------------------


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_gitignored_local_files_are_honored_inside_a_git_repo(tmp_path: Path):
    _git(tmp_path, "init", "-q")
    _write(tmp_path, ".gitignore", ".env\ndata/\n")
    _write(tmp_path, ".env", f"{PW.upper()}=hunter2hunter2\n")
    _write(tmp_path, "data/risk.db", "x")
    assert find_violations(tmp_path) == []


def test_force_added_ignored_file_is_flagged(tmp_path: Path):
    _git(tmp_path, "init", "-q")
    _write(tmp_path, ".gitignore", ".env\n")
    _write(tmp_path, ".env", "A=1\n")
    _git(tmp_path, "add", "-f", ".env")
    assert any(".env" in v for v in find_violations(tmp_path))


def test_works_without_git_history_or_repository(tmp_path: Path):
    _write(tmp_path, "secrets/k.txt", "x")
    assert any("secrets/k.txt" in v for v in find_violations(tmp_path))


def _fake_git(monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: bytes = b"") -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout)

    monkeypatch.setattr(hygiene.subprocess, "run", fake_run)


def test_git_failure_falls_back_to_walking_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".git").mkdir()
    _write(tmp_path, "data/risk.db", "x")
    _fake_git(monkeypatch, returncode=128)
    assert any("data/risk.db" in v for v in find_violations(tmp_path))


def test_git_executable_missing_falls_back_to_walking_the_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".git").mkdir()
    _write(tmp_path, "data/risk.db", "x")

    def boom(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(hygiene.subprocess, "run", boom)
    assert any("data/risk.db" in v for v in find_violations(tmp_path))


def test_git_listing_nothing_for_a_non_empty_tree_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A silent empty listing would make the scan vacuous, so it must fail loudly."""
    (tmp_path / ".git").mkdir()
    _write(tmp_path, "data/risk.db", "x")
    _fake_git(monkeypatch, returncode=0, stdout=b"")
    with pytest.raises(RuntimeError, match="git listed no files"):
        find_violations(tmp_path)


def test_scanned_files_reports_what_was_actually_scanned(tmp_path: Path):
    _write(tmp_path, "a.py", "x = 1\n")
    _write(tmp_path, ".venv/skip.py", "x = 1\n")
    assert [p.as_posix() for p in scanned_files(tmp_path)] == ["a.py"]


# --- the real repository ---------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="TODO: .env.example is intentionally absent; the user must add it manually",
)
def test_real_repository_ships_an_env_example(repo_root: Path):
    assert (repo_root / ".env.example").is_file()


def test_real_repository_is_clean(repo_root: Path):
    scanned = {p.as_posix() for p in scanned_files(repo_root)}
    assert "pyproject.toml" in scanned, "scan is vacuous: pyproject.toml was not scanned"
    assert "src/social_tracker/__init__.py" in scanned
    assert len(scanned) >= 5
    assert find_violations(repo_root) == []
