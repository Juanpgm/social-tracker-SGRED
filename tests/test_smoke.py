"""Smoke tests: the package imports cleanly and has no import-time side effects."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_package_imports_and_exposes_version():
    import social_tracker

    assert isinstance(social_tracker.__version__, str)
    assert social_tracker.__version__.strip() != ""


def test_version_looks_like_a_release_string():
    import social_tracker

    parts = social_tracker.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_imports_in_isolated_mode(repo_root: Path):
    """`python -I` ignores env vars and cwd; the installed package must still import."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import social_tracker; print(social_tracker.__version__)"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != ""


def test_import_has_no_filesystem_side_effects(tmp_path: Path):
    """Importing the package in a clean cwd must not create files (no DB, no dirs)."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import social_tracker"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["__version__"])
def test_public_attribute_present(name: str):
    import social_tracker

    assert hasattr(social_tracker, name)
