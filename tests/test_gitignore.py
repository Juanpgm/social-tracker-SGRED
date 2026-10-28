"""The shipped .gitignore must keep secrets, runtime data and heavy artifacts out of git."""

import shutil
import subprocess
from pathlib import Path

import pytest

IGNORED = [
    ".env",
    ".env.local",
    "data/risk.db",
    "data/raw/posts.json",
    "tests/data/risk.db",  # the exception is the folder, not the databases inside it
    "tests/data/local.sqlite3",
    "tests/data/secrets/token.txt",
    "secrets/token.txt",
    "session.cookies",
    "server.pem",
    "deploy/private.key",
    "bundle.p12",
    "id_rsa",
    "id_rsa.pub",
    ".netrc",
    "local.db",
    "risk.sqlite",
    "risk.sqlite3",
    "risk.db-wal",
    "risk.db-shm",
    "risk.db-journal",
    "var/state/risk.db-wal",
    "risk.sqlite-wal",
    "risk.sqlite-shm",
    "risk.sqlite-journal",
    "risk.sqlite3-wal",
    "risk.sqlite3-shm",
    "risk.sqlite3-journal",
    "cookies.txt",
    "cookies_tiktok.json",
    "auth/storage_state.json",
    "my_storage_state_x.json",
    "run.log",
    "logs/app.log",
    "clip.wav",
    "clip.mp3",
    "clip.m4a",
    "clip.opus",
    "clip.ogg",
    "clip.flac",
    "gold_export.yaml",  # a gold export can hold real text from real sources
    "gold_export_2026-03-01.yaml",
    "gold_export.yml",
    "exports/gold.yaml",
    "tmp/exports/anything.txt",
]

NOT_IGNORED = [
    ".env.example",
    "README.md",
    "pyproject.toml",
    "src/social_tracker/data.py",
    "src/social_tracker/domain/keys.py",
    "tests/test_smoke.py",
    "tests/data/gold_examples.yaml",  # synthetic test data is source, unlike runtime data/
    "docs/exports.md",
    "src/social_tracker/gold/yaml_io.py",
    "docs/cookies_policy.md",
    "docs/db-wal.md",
    "notes/db-journal.txt",
    "src/social_tracker/persistence/db.py",
    "src/social_tracker/persistence/migrations/0001_init.sql",
]


@pytest.fixture(scope="module")
def scratch_repo(repo_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available")
    repo = tmp_path_factory.mktemp("ignore-repo")
    subprocess.run([git, "init", "-q"], cwd=repo, check=True)
    shutil.copy(repo_root / ".gitignore", repo / ".gitignore")
    return repo


def _is_ignored(repo: Path, rel: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", rel],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


@pytest.mark.parametrize("path", IGNORED)
def test_sensitive_or_heavy_path_is_ignored(scratch_repo: Path, path: str):
    assert _is_ignored(scratch_repo, path), f"{path} must be git-ignored"


@pytest.mark.parametrize("path", NOT_IGNORED)
def test_regular_source_path_is_not_ignored(scratch_repo: Path, path: str):
    assert not _is_ignored(scratch_repo, path), f"{path} must not be git-ignored"
