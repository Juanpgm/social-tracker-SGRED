"""Meta-tests: the registry contract must reject each deliberately wrong registry.

The contract tests are plain methods taking an `Env`, so they are replayed here by hand against
an in-memory environment built around the wrong class. `parametrize` marks are expanded with a
tiny cross-product helper (ids and `pytest.param` are not used by the contract).
"""

import pytest

from social_tracker.sources.service import SourceRegistry
from tests.replay import cases
from tests.sources import test_registry as registry_suites
from tests.sources.registry_contract import RegistryContract
from tests.sources.wrong_registries import WRONG_REGISTRIES


def _failing_tests(registry_cls: type[SourceRegistry]) -> set[str]:
    failed: set[str] = set()

    class Harness(registry_suites.TestInMemoryRegistry):
        """Every environment a test builds, including its own `make_env()`, uses the class."""

        def make_env(self, cls: type[SourceRegistry] = registry_cls):
            return super().make_env(cls)

    harness = Harness()
    for name in sorted(vars(RegistryContract)):
        if not name.startswith("test_"):
            continue
        method = getattr(RegistryContract, name)
        for kwargs in cases(method) or [{}]:
            env = harness.make_env()
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
