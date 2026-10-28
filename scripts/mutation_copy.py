"""Shared machinery of the mutation sweeps that work on a COPY of the source.

A sweep used to rewrite the production files in place and restore them afterwards, which leaves
the working tree broken if the process is killed (or if a second tool reads it meanwhile). Here
the package is copied once per worker into a temporary directory, every mutant is applied to the
copy, and the tests import the copy (`PYTHONPATH` is put before the editable install). The real
tree is never opened for writing; the sweep checks that it is byte for byte what it was.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "social_tracker"


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str


def m(name: str, path: str, old: str, new: str) -> Mutant:
    return Mutant(name, path, old, new)


def tree_hash(root: Path = SRC) -> str:
    """A hash of every file of the real source tree: it must not change during a sweep."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Workspace:
    """One copy of the package in a temporary directory, with the tests pointed at it."""

    def __init__(self, tests: Sequence[str]) -> None:
        self._holder = tempfile.TemporaryDirectory(prefix="mutation-copy-")
        self.root = Path(self._holder.name)
        shutil.copytree(
            SRC,
            self.root / "src" / "social_tracker",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        self.src = self.root / "src" / "social_tracker"
        self.tests = list(tests)
        self.env = {
            **os.environ,
            "PYTHONPATH": str(self.root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        self.check_imports_the_copy()

    def check_imports_the_copy(self) -> None:
        """Refuse to run if the tests would import the real package instead of the copy."""
        done = subprocess.run(
            [sys.executable, "-c", "import social_tracker; print(social_tracker.__file__)"],
            cwd=ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            check=True,
        )
        imported = Path(done.stdout.strip()).resolve()
        if self.root.resolve() not in imported.parents:
            raise SystemExit(f"the tests would import {imported}, not the copy in {self.root}")

    def tests_pass(self) -> bool:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *self.tests],
            cwd=ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        return done.returncode == 0

    def apply(self, mutant: Mutant) -> tuple[Path, bytes] | None:
        """Change the copy; returns what is needed to restore it, None if the pattern is stale."""
        path = self.src / mutant.path
        original = path.read_bytes()
        text = original.decode("utf-8")
        newline = "\r\n" if "\r\n" in text else "\n"
        normal = text.replace("\r\n", "\n")
        if mutant.old not in normal:
            return None
        mutated = normal.replace(mutant.old, mutant.new, 1).replace("\n", newline)
        path.write_bytes(mutated.encode("utf-8"))
        return path, original

    def close(self) -> None:
        self._holder.cleanup()


def is_stale(mutant: Mutant) -> bool:
    """True when the pattern of the mutant is no longer in the (real) source. Reads only."""
    text = (SRC / mutant.path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return mutant.old not in text


def run_sweep(
    mutants: Sequence[Mutant],
    tests: Sequence[str],
    argv: list[str],
    *,
    describe: Callable[[], str] = lambda: "",
) -> int:
    """Run the sweep. `argv` may hold `--jobs N`, `--check` (only look for stale patterns) and
    words that select mutants by name. Exit status: 0 all killed, 1 some survived or stale, 2 the
    tests fail on the unmutated copy."""
    args = list(argv)
    jobs = 4
    if "--jobs" in args:
        at = args.index("--jobs")
        jobs = int(args[at + 1])
        del args[at : at + 2]
    check_only = "--check" in args
    if check_only:
        args.remove("--check")
    selected = [x for x in mutants if not args or any(a in x.name for a in args)]
    stale = [x for x in selected if is_stale(x)]
    for mutant in stale:
        print(
            f"STALE     {mutant.name}: pattern not found in {mutant.path} "
            "or the mutant does not compile"
        )
    if check_only:
        print(f"{len(selected) - len(stale)} of {len(selected)} patterns found")
        return 1 if stale else 0
    live = [x for x in selected if x not in stale]
    before = tree_hash()
    workspaces = [Workspace(tests) for _ in range(max(1, min(jobs, len(live) or 1)))]
    try:
        if not workspaces[0].tests_pass():
            print("the targeted tests do not pass on the unmutated copy")
            return 2
        survivors = _sweep(live, workspaces)
    finally:
        for workspace in workspaces:
            workspace.close()
    after = tree_hash()
    if before != after:
        print("THE REAL SOURCE TREE CHANGED DURING THE SWEEP")
        return 3
    survivors += stale
    print(f"\n{len(selected) - len(survivors)} of {len(selected)} mutants killed {describe()}")
    for mutant in survivors:
        print(f"  survivor: {mutant.name}")
    return 1 if survivors else 0


def _sweep(mutants: Sequence[Mutant], workspaces: list[Workspace]) -> list[Mutant]:
    free: list[Workspace] = list(workspaces)
    survivors: list[Mutant] = []

    def one(mutant: Mutant) -> None:
        workspace = free.pop()
        try:
            applied = workspace.apply(mutant)
            assert applied is not None
            path, original = applied
            try:
                killed = not workspace.tests_pass()
            finally:
                path.write_bytes(original)
        finally:
            free.append(workspace)
        print(f"{'killed  ' if killed else 'SURVIVED'}  {mutant.name}", flush=True)
        if not killed:
            survivors.append(mutant)

    with ThreadPoolExecutor(max_workers=len(workspaces)) as pool:
        list(pool.map(one, mutants))
    return survivors
