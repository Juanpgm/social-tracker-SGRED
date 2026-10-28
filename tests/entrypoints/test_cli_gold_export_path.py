"""G-1: the gold set can hold real text from real sources, so `gold export --output` will not write
inside a git work tree (where `git add .` would commit it) unless the operator says so."""

from pathlib import Path

import pytest

from social_tracker.entrypoints.inputs import find_work_tree
from tests.entrypoints.conftest import Cli
from tests.entrypoints.test_cli_gold import OFF_TOPIC, PREDICTION, REPORT, add


@pytest.fixture
def labeled(cli: Cli, tmp_path: Path) -> Cli:
    texts = [(PREDICTION, "unintentional_rumor"), (REPORT, "no_risk"), (OFF_TOPIC, "out_of_scope")]
    for n, (text, label) in enumerate(texts):
        assert add(cli, tmp_path, text, label, name=f"t{n}.txt").exit_code == 0
    return cli


# ---- the lookup ----


def test_a_directory_with_a_git_folder_is_a_work_tree(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    assert find_work_tree(tmp_path / "out.yaml") == tmp_path


def test_a_file_named_git_counts_too_as_in_a_linked_work_tree(tmp_path: Path):
    (tmp_path / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    assert find_work_tree(tmp_path / "out.yaml") == tmp_path


def test_a_deep_path_is_found_even_when_its_folders_do_not_exist_yet(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    assert find_work_tree(tmp_path / "a" / "b" / "c" / "out.yaml") == tmp_path


def test_a_path_outside_any_work_tree_is_none(tmp_path: Path):
    assert find_work_tree(tmp_path / "out.yaml") is None


def test_dot_dot_cannot_hide_the_work_tree(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (tmp_path / "other").mkdir()
    sneaky = tmp_path / "other" / ".." / "repo" / "out.yaml"
    assert find_work_tree(sneaky) == repo


def test_a_relative_path_is_resolved_against_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    assert find_work_tree(Path("out.yaml")) == tmp_path


def test_the_home_of_the_files_themselves_is_not_searched_for_other_markers(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("x\n", encoding="utf-8")
    assert find_work_tree(tmp_path / "out.yaml") is None


# ---- the command ----


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    (path / ".git").mkdir(parents=True)
    return path


def test_export_inside_a_work_tree_is_refused_and_nothing_is_written(
    labeled: Cli,
    repo: Path,
):
    target = repo / "gold_export.yaml"
    result = labeled("gold", "export", "-o", str(target))
    assert result.exit_code == 1
    assert "--allow-repo-path" in result.stderr
    assert "git work tree" in result.stderr
    assert not target.exists()
    assert not list(repo.glob(".export-*"))


def test_export_in_a_subfolder_of_the_work_tree_is_refused_too(
    labeled: Cli,
    repo: Path,
):
    (repo / "sub").mkdir()
    result = labeled("gold", "export", "-o", str(repo / "sub" / "gold.yaml"))
    assert result.exit_code == 1
    assert "--allow-repo-path" in result.stderr


def test_the_flag_allows_it_and_says_what_it_did(labeled: Cli, repo: Path):
    target = repo / "gold_export.yaml"
    result = labeled("gold", "export", "-o", str(target), "--allow-repo-path")
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8").count("- id: gld_") == 3
    assert "keep it out of version control" in result.stderr


def test_outside_a_work_tree_nothing_changes(labeled: Cli, tmp_path: Path):
    target = tmp_path / "gold.yaml"
    result = labeled("gold", "export", "-o", str(target))
    assert result.exit_code == 0
    assert "work tree" not in result.stderr


def test_printing_to_the_terminal_is_never_refused(labeled: Cli, repo: Path, monkeypatch):
    monkeypatch.chdir(repo)
    assert labeled.ok("gold", "export").stdout.startswith("# Gold set exported")


def test_the_refusal_does_not_echo_the_path(labeled: Cli, repo: Path):
    result = labeled("gold", "export", "-o", str(repo / "secret-name-x.yaml"))
    assert "secret-name-x" not in result.stderr


def test_the_help_documents_the_flag(cli: Cli):
    out = cli.ok("gold", "export", "--help").stdout
    assert "--allow-repo-path" in out
    assert "work tree" in out
