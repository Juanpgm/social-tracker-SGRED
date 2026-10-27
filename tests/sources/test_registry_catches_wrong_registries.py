"""Meta-tests: the registry contract must reject each deliberately wrong registry.

The contract tests are plain methods taking an `Env`, so they are replayed here by hand against
an in-memory environment built around the wrong class. `parametrize` marks are expanded with a
tiny cross-product helper (ids and `pytest.param` are not used by the contract).
"""

import itertools
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from social_tracker.sources.service import SourceRegistry
from tests.sources import test_registry as registry_suites
from tests.sources.registry_contract import RegistryContract
from tests.sources.wrong_registries import WRONG_REGISTRIES


def _cases(method: Callable[..., Any]) -> Iterator[dict[str, Any]]:
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


def _failing_tests(registry_cls: type[SourceRegistry]) -> set[str]:
    failed: set[str] = set()
    harness = registry_suites.TestInMemoryRegistry()
    for name in sorted(vars(RegistryContract)):
        if not name.startswith("test_"):
            continue
        method = getattr(RegistryContract, name)
        for kwargs in _cases(method) or [{}]:
            env = harness.make_env(registry_cls)
            try:
                method(harness, env, **kwargs)
            except BaseException:  # noqa: BLE001 - any failure counts, including leaked errors
                failed.add(name)
    return failed


def test_the_reference_registry_passes_every_replayed_test():
    assert _failing_tests(SourceRegistry) == set()


@pytest.mark.parametrize(
    ("label", "registry_cls", "target"), [(k, *v) for k, v in WRONG_REGISTRIES.items()]
)
def test_each_wrong_registry_fails_the_test_aimed_at_it(label, registry_cls, target):
    failed = _failing_tests(registry_cls)
    assert any(name.startswith(target) for name in failed), (label, sorted(failed))
