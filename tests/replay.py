"""Helpers to replay contract test methods by hand (used by the wrong-implementation meta-tests)."""

import itertools
from collections.abc import Callable, Iterator
from typing import Any


def cases(method: Callable[..., Any]) -> Iterator[dict[str, Any]]:
    axes: list[list[dict[str, Any]]] = []
    for mark in getattr(method, "pytestmark", []):
        if mark.name != "parametrize":
            continue
        spec = mark.args[0]
        names = [n.strip() for n in spec.split(",")] if isinstance(spec, str) else list(spec)
        values = mark.args[1]
        axes.append([dict(zip(names, v if len(names) > 1 else (v,), strict=True)) for v in values])
    for combo in itertools.product(*axes):
        merged: dict[str, Any] = {}
        for part in combo:
            merged.update(part)
        yield merged
