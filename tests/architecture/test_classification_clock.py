"""The classification core never reads the clock (RC-10).

An import rule cannot forbid `datetime.now()`: the core needs the `datetime` types, and `time` is
already banned. So the source of every core module is read here for the calls that ask for "now".
The reader is itself proven on deliberate violations, so that it cannot go quiet.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[2] / "src" / "social_tracker" / "classification"
ALLOWED = {"service.py", "ports.py", "yaml_io.py", "__init__.py"}  # the application layer
CLOCK_CALLS = {"now", "utcnow", "today", "time", "time_ns", "monotonic", "perf_counter"}


def clock_calls(source: str) -> list[str]:
    """The names of the calls in `source` that read the current time or a timer."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in CLOCK_CALLS:
                found.append(name)
    return found


def core_modules() -> list[Path]:
    return sorted(p for p in SRC.glob("*.py") if p.name not in ALLOWED)


def test_the_core_has_modules_to_read():
    names = {p.name for p in core_modules()}
    assert {"domain.py", "rules.py", "text.py", "topics.py", "similarity.py", "links.py"} <= names


@pytest.mark.parametrize("path", core_modules(), ids=lambda p: p.name)
def test_no_core_module_asks_for_the_current_time(path: Path):
    assert clock_calls(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("from datetime import datetime\nx = datetime.now()", ["now"]),
        ("import datetime\nx = datetime.datetime.utcnow()", ["utcnow"]),
        ("from datetime import date\nx = date.today()", ["today"]),
        ("from datetime import datetime as dt\nx = dt.now(tz=None)", ["now"]),
        ("import time\nx = time.time()", ["time"]),
        ("from time import monotonic\nx = monotonic()", ["monotonic"]),
        ("import time\nx = time.perf_counter()", ["perf_counter"]),
        ("def f(clock):\n    return clock.now()", ["now"]),
    ],
)
def test_the_reader_catches_every_way_of_asking_for_the_time(source: str, expected: list[str]):
    assert clock_calls(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "from datetime import datetime\nx = datetime(2026, 3, 1)",
        "from datetime import timedelta\nx = timedelta(days=1)",
        "x = datetime.fromisoformat('2026-03-01T00:00:00+00:00')",
        "x = 'datetime.now() is only text here'",
    ],
)
def test_the_reader_leaves_alone_what_does_not_read_the_clock(source: str):
    assert clock_calls(source) == []
