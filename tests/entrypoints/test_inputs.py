"""Input helpers of the CLI: actor resolution, `--set` parsing, bounded file reading/writing."""

import errno
import os
import stat
from pathlib import Path

import pytest

from social_tracker.entrypoints import inputs
from social_tracker.entrypoints.inputs import (
    ACTOR_ENV,
    parse_fields,
    read_import_text,
    resolve_actor,
    write_export,
)
from social_tracker.shared.errors import ConflictError, ValidationError
from social_tracker.sources.yaml_io import MAX_FILE_BYTES

# ---- actor ----


def test_an_explicit_actor_wins_over_the_environment():
    assert resolve_actor("alice", {ACTOR_ENV: "bob"}) == "alice"


def test_an_explicit_blank_actor_is_passed_on_so_the_service_rejects_it():
    assert resolve_actor("", {ACTOR_ENV: "bob"}) == ""
    assert resolve_actor("   ", {}) == "   "


def test_the_environment_is_used_when_no_actor_is_given():
    assert resolve_actor(None, {ACTOR_ENV: " bob "}) == "bob"


@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_environment_actor_falls_back_to_the_os_user(
    monkeypatch: pytest.MonkeyPatch, blank: str
):
    monkeypatch.setattr("getpass.getuser", lambda: "carol")
    assert resolve_actor(None, {ACTOR_ENV: blank}) == "cli:carol"


def test_the_default_actor_is_cli_plus_the_os_user(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("getpass.getuser", lambda: "dave")
    assert resolve_actor(None, {}) == "cli:dave"


@pytest.mark.parametrize("failure", [OSError("no user"), KeyError("uid")])
def test_an_unknown_os_user_still_yields_a_valid_actor(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
):
    def fail() -> str:
        raise failure

    monkeypatch.setattr("getpass.getuser", fail)
    assert resolve_actor(None, {}) == "cli:unknown"


# ---- --set KEY=VALUE ----


def test_values_are_json_when_they_parse_and_plain_text_otherwise():
    fields = parse_fields(
        [
            "poll_interval_s=600",
            "feed_url=https://example.org/rss.xml",
            'keywords=["a","b"]',
            "include_stories=false",
            "secret_ref=null",
            "credibility=4",
        ]
    )
    assert fields == {
        "poll_interval_s": 600,
        "feed_url": "https://example.org/rss.xml",
        "keywords": ["a", "b"],
        "include_stories": False,
        "secret_ref": None,
        "credibility": 4,
    }


def test_a_quoted_value_stays_a_string_and_only_the_first_equals_splits():
    assert parse_fields(['name="123"', "feed_url=https://x.org/a?b=c"]) == {
        "name": "123",
        "feed_url": "https://x.org/a?b=c",
    }


def test_an_empty_value_is_an_empty_string_and_no_pairs_is_no_fields():
    assert parse_fields(["feed_url="]) == {"feed_url": ""}
    assert parse_fields([]) == {}


def test_a_value_that_is_too_deeply_nested_for_json_is_kept_as_text():
    deep = "[" * 100_000
    assert parse_fields([f"keywords={deep}"]) == {"keywords": deep}


@pytest.mark.parametrize("pair", ["novalue-ghp_leak", "=value", "", "  =x"])
def test_a_pair_without_a_key_or_equals_is_rejected_without_echoing_it(pair: str):
    with pytest.raises(ValidationError) as caught:
        parse_fields([pair])
    assert "KEY=VALUE" in caught.value.message
    assert "ghp_leak" not in caught.value.message


def test_a_repeated_key_is_rejected():
    with pytest.raises(ValidationError) as caught:
        parse_fields(["a_field=1", "a_field=2"])
    assert "more than once" in caught.value.message
    assert "a_field" not in caught.value.message


def test_reserved_keys_are_rejected_when_the_command_has_a_dedicated_option():
    with pytest.raises(ValidationError) as caught:
        parse_fields(["name=x"], reserved=frozenset({"name"}))
    assert "dedicated option" in caught.value.message
    assert parse_fields(["name=x"]) == {"name": "x"}


# ---- reading the import file ----


def test_a_plain_utf8_file_is_read(tmp_path: Path):
    path = tmp_path / "s.yaml"
    path.write_bytes("sources: []  # café\n".encode())
    assert read_import_text(path) == "sources: []  # café\n"


def test_a_utf8_bom_is_stripped(tmp_path: Path):
    path = tmp_path / "bom.yaml"
    path.write_bytes(b"\xef\xbb\xbfsources: []\n")
    assert read_import_text(path) == "sources: []\n"


def test_the_size_limit_is_inclusive_and_one_more_byte_is_refused(tmp_path: Path):
    at_limit = tmp_path / "at.yaml"
    at_limit.write_bytes(b"#" * MAX_FILE_BYTES)
    assert len(read_import_text(at_limit)) == MAX_FILE_BYTES
    over = tmp_path / "over.yaml"
    over.write_bytes(b"#" * (MAX_FILE_BYTES + 1))
    with pytest.raises(ValidationError) as caught:
        read_import_text(over)
    assert str(MAX_FILE_BYTES) in caught.value.message


def test_a_huge_file_is_never_read_beyond_the_limit(tmp_path: Path, monkeypatch):
    path = tmp_path / "huge.yaml"
    path.write_bytes(b"#" * (MAX_FILE_BYTES * 3))
    seen: list[int] = []
    real_fdopen = os.fdopen

    class Spy:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._handle.close()

        def read(self, size=-1):
            seen.append(size)
            return self._handle.read(size)

    monkeypatch.setattr(os, "fdopen", lambda *a, **k: Spy(real_fdopen(*a, **k)))
    with pytest.raises(ValidationError):
        read_import_text(path)
    assert seen == [MAX_FILE_BYTES + 1]


@pytest.mark.parametrize("raw", [b"\xff\xfe\x00", b"sources: \xc3\x28", b"\x80"])
def test_invalid_utf8_is_a_clean_error(tmp_path: Path, raw: bytes):
    path = tmp_path / "bad.yaml"
    path.write_bytes(raw)
    with pytest.raises(ValidationError) as caught:
        read_import_text(path)
    assert "UTF-8" in caught.value.message


def test_an_empty_file_is_returned_as_empty_text(tmp_path: Path):
    path = tmp_path / "empty.yaml"
    path.write_bytes(b"")
    assert read_import_text(path) == ""


def test_a_missing_file_and_a_directory_are_clean_errors(tmp_path: Path):
    for target in (tmp_path / "nope.yaml", tmp_path):
        with pytest.raises(ValidationError) as caught:
            read_import_text(target)
        assert caught.value.message.startswith("cannot read the import file")
        assert str(tmp_path) not in caught.value.message


# ---- writing the export ----


def test_export_writes_utf8_bytes_with_unix_newlines(tmp_path: Path):
    path = tmp_path / "out.yaml"
    write_export(path, "sources:\n- name: café\n", force=False)
    assert path.read_bytes() == "sources:\n- name: café\n".encode()


def test_export_refuses_to_overwrite_an_existing_file_and_leaves_it_untouched(tmp_path: Path):
    path = tmp_path / "out.yaml"
    path.write_text("precious", encoding="utf-8")
    with pytest.raises(ConflictError) as caught:
        write_export(path, "new", force=False)
    assert "--force" in caught.value.message
    assert path.read_text(encoding="utf-8") == "precious"


def test_force_overwrites_and_truncates(tmp_path: Path):
    path = tmp_path / "out.yaml"
    path.write_text("a much longer previous content", encoding="utf-8")
    write_export(path, "new", force=True)
    assert path.read_text(encoding="utf-8") == "new"


def test_export_to_a_missing_directory_or_over_a_directory_is_a_clean_error(tmp_path: Path):
    for target in (tmp_path / "missing" / "out.yaml", tmp_path):
        for force in (False, True):
            with pytest.raises((ValidationError, ConflictError)) as caught:
                write_export(target, "x", force=force)
            assert str(tmp_path) not in caught.value.message


# ---- writing the export: atomic, no partial file, no symlinks ------------------------------

POSIX_ONLY = pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits and FIFOs")


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def _fail_with(errno_code: int = errno.ENOSPC):
    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno_code, os.strerror(errno_code))

    return fail


def test_a_failure_while_writing_with_force_leaves_the_old_file_and_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "out.yaml"
    path.write_text("precious", encoding="utf-8")
    monkeypatch.setattr(os, "fsync", _fail_with())
    with pytest.raises(ValidationError) as caught:
        write_export(path, "replacement", force=True)
    assert caught.value.message.startswith("cannot write the output file")
    assert str(tmp_path) not in caught.value.message
    assert path.read_text(encoding="utf-8") == "precious"
    assert _names(tmp_path) == ["out.yaml"]


def test_a_failure_while_writing_without_force_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(os, "fsync", _fail_with())
    with pytest.raises(ValidationError):
        write_export(tmp_path / "out.yaml", "x", force=False)
    assert _names(tmp_path) == []


@pytest.mark.parametrize("force", [False, True])
def test_a_failure_while_publishing_the_file_leaves_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool
):
    monkeypatch.setattr(os, "replace" if force else "link", _fail_with(errno.EIO))
    with pytest.raises(ValidationError):
        write_export(tmp_path / "out.yaml", "x", force=force)
    assert _names(tmp_path) == []


def test_an_interrupt_while_writing_also_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "out.yaml"
    path.write_text("precious", encoding="utf-8")

    def interrupt(*_args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "fsync", interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_export(path, "replacement", force=True)
    assert path.read_text(encoding="utf-8") == "precious"
    assert _names(tmp_path) == ["out.yaml"]


@pytest.mark.parametrize("force", [False, True])
def test_a_successful_export_leaves_only_the_target(tmp_path: Path, force: bool):
    write_export(tmp_path / "out.yaml", "data", force=force)
    assert _names(tmp_path) == ["out.yaml"]


def test_the_data_is_flushed_to_disk_in_the_target_directory_before_it_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    journal: list[tuple[str, str]] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        journal.append(("fsync", ""))
        real_fsync(fd)

    def replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        journal.append(("replace", str(Path(src).parent)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    write_export(tmp_path / "out.yaml", "data", force=True)
    assert journal == [("fsync", ""), ("replace", str(tmp_path))]


def test_a_file_created_while_the_export_runs_is_never_clobbered_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "out.yaml"
    real_link = os.link

    def racing_link(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        Path(dst).write_text("won the race", encoding="utf-8")
        real_link(src, dst)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(ConflictError) as caught:
        write_export(path, "mine", force=False)
    assert "--force" in caught.value.message
    assert path.read_text(encoding="utf-8") == "won the race"
    assert _names(tmp_path) == ["out.yaml"]


def _fake_lstat(monkeypatch: pytest.MonkeyPatch, target: Path, mode: int) -> None:
    real = os.lstat

    def lstat(path: str | os.PathLike[str], *args: object, **kwargs: object) -> os.stat_result:
        if Path(path) == target:
            return os.stat_result((mode, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        return real(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "lstat", lstat)


@pytest.mark.parametrize("force", [False, True])
def test_a_symbolic_link_target_is_refused_even_where_symlinks_cannot_be_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool
):
    path = tmp_path / "link.yaml"
    _fake_lstat(monkeypatch, path, stat.S_IFLNK | 0o777)
    with pytest.raises(ValidationError) as caught:
        write_export(path, "x", force=force)
    assert "symbolic link" in caught.value.message
    assert _names(tmp_path) == []


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("dangling", [False, True])
def test_a_real_symlink_is_refused_and_what_it_points_to_is_untouched(
    tmp_path: Path, force: bool, dangling: bool
):
    victim = tmp_path / "victim.txt"
    if not dangling:
        victim.write_text("do not touch", encoding="utf-8")
    link = tmp_path / "link.yaml"
    try:
        link.symlink_to(victim)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted on this platform")
    with pytest.raises(ValidationError) as caught:
        write_export(link, "x", force=force)
    assert "symbolic link" in caught.value.message
    assert link.is_symlink()
    assert not victim.exists() or victim.read_text(encoding="utf-8") == "do not touch"
    assert _names(tmp_path) == sorted(["link.yaml"] + ([] if dangling else ["victim.txt"]))


@pytest.mark.parametrize("force", [False, True])
def test_an_existing_device_or_other_non_regular_target_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool
):
    path = tmp_path / "special"
    _fake_lstat(monkeypatch, path, stat.S_IFCHR | 0o666)
    with pytest.raises(ValidationError) as caught:
        write_export(path, "x", force=force)
    assert "not a regular file" in caught.value.message
    assert _names(tmp_path) == []


@POSIX_ONLY
@pytest.mark.parametrize("force", [False, True])
def test_the_export_is_private_to_its_owner(tmp_path: Path, force: bool):
    path = tmp_path / "out.yaml"
    if force:
        path.write_text("old", encoding="utf-8")
        path.chmod(0o644)
    write_export(path, "data", force=force)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("name", ["a" * 200 + ".yaml", "with space.yaml", "ünï.yaml"])
def test_unusual_but_valid_file_names_work(tmp_path: Path, name: str):
    write_export(tmp_path / name, "data", force=False)
    assert (tmp_path / name).read_text(encoding="utf-8") == "data"
    assert _names(tmp_path) == [name]


def test_a_relative_target_is_written_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    write_export(Path("rel.yaml"), "data", force=False)
    assert _names(tmp_path) == ["rel.yaml"]


# ---- reading the import file: regular files only -------------------------------------------


def test_a_device_is_refused_instead_of_read():
    with pytest.raises(ValidationError) as caught:
        read_import_text(Path(os.devnull))
    assert caught.value.message == "cannot read the import file: not a regular file"


@POSIX_ONLY
def test_a_fifo_is_refused_without_blocking_on_a_writer(tmp_path: Path):
    fifo = tmp_path / "pipe.yaml"
    os.mkfifo(fifo)
    with pytest.raises(ValidationError) as caught:
        read_import_text(fifo)
    assert caught.value.message == "cannot read the import file: not a regular file"


@POSIX_ONLY
def test_a_directory_is_refused_as_not_a_regular_file(tmp_path: Path):
    with pytest.raises(ValidationError) as caught:
        read_import_text(tmp_path)
    assert caught.value.message.startswith("cannot read the import file")
    assert str(tmp_path) not in caught.value.message


def test_a_symlink_to_a_regular_file_is_read(tmp_path: Path):
    real = tmp_path / "real.yaml"
    real.write_text("sources: []\n", encoding="utf-8")
    link = tmp_path / "link.yaml"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not permitted on this platform")
    assert read_import_text(link) == "sources: []\n"


# ---- a replace that races another one (Windows reports it as a passing "access denied") ----


def _flaky_replace(monkeypatch: pytest.MonkeyPatch, failures: int) -> tuple[list[int], list[float]]:
    calls: list[int] = []
    real = os.replace

    def replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        calls.append(1)
        if len(calls) <= failures:
            raise PermissionError(errno.EACCES, "Access is denied")
        real(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    pauses: list[float] = []
    monkeypatch.setattr(inputs.time, "sleep", pauses.append)
    return calls, pauses


def test_on_windows_a_replace_denied_for_a_moment_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(inputs, "_IS_WINDOWS", True)
    calls, pauses = _flaky_replace(monkeypatch, failures=3)
    write_export(tmp_path / "out.yaml", "data", force=True)
    assert len(calls) == 4
    assert pauses == [inputs._REPLACE_PAUSE_S] * 3  # it waits between attempts, not after
    assert (tmp_path / "out.yaml").read_text(encoding="utf-8") == "data"
    assert _names(tmp_path) == ["out.yaml"]


def test_on_windows_a_replace_denied_for_good_gives_up_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(inputs, "_IS_WINDOWS", True)
    calls, pauses = _flaky_replace(monkeypatch, failures=10_000)
    path = tmp_path / "out.yaml"
    path.write_text("precious", encoding="utf-8")
    with pytest.raises(ValidationError) as caught:
        write_export(path, "new", force=True)
    assert caught.value.message.startswith("cannot write the output file")
    assert len(calls) == inputs._REPLACE_ATTEMPTS
    assert len(pauses) == inputs._REPLACE_ATTEMPTS - 1
    assert path.read_text(encoding="utf-8") == "precious"
    assert _names(tmp_path) == ["out.yaml"]


def test_elsewhere_a_denied_replace_is_a_real_permission_problem_and_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(inputs, "_IS_WINDOWS", False)
    calls, pauses = _flaky_replace(monkeypatch, failures=1)
    with pytest.raises(ValidationError):
        write_export(tmp_path / "out.yaml", "data", force=True)
    assert len(calls) == 1
    assert pauses == []
    assert _names(tmp_path) == []


def test_an_existing_target_without_force_is_refused_before_any_temporary_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "out.yaml"
    path.write_text("precious", encoding="utf-8")

    def no_temporary_files(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a temporary file was created for a write that is refused up front")

    monkeypatch.setattr(inputs.tempfile, "mkstemp", no_temporary_files)
    with pytest.raises(ConflictError):
        write_export(path, "new", force=False)
