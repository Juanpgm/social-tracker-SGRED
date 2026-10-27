"""Meta-tests: the control service contract must reject each deliberately wrong service."""

import pytest

from social_tracker.control.service import ControlService
from tests.control import test_service as service_suites
from tests.control.service_contract import ControlServiceContract
from tests.control.wrong_services import WRONG_SERVICES
from tests.replay import cases


def _failing_tests(service_cls: type[ControlService]) -> set[str]:
    failed: set[str] = set()

    class Harness(service_suites.TestInMemoryControlService):
        """Every environment a test builds, including its own `make_env()`, uses the class."""

        def make_env(self, cls: type[ControlService] = service_cls):
            return super().make_env(cls)

    harness = Harness()
    for name in sorted(vars(ControlServiceContract)):
        if not name.startswith("test_"):
            continue
        method = getattr(ControlServiceContract, name)
        for kwargs in cases(method) or [{}]:
            env = harness.make_env()
            env.add_source("src_one")
            try:
                method(harness, env, **kwargs)
            except BaseException:  # noqa: BLE001 - any failure counts, including leaked errors
                failed.add(name)
    return failed


def test_the_reference_service_passes_every_replayed_test():
    assert _failing_tests(ControlService) == set()


@pytest.mark.parametrize(
    ("label", "service_cls", "target"), [(k, *v) for k, v in WRONG_SERVICES.items()]
)
def test_each_wrong_service_fails_the_test_aimed_at_it(label, service_cls, target):
    failed = _failing_tests(service_cls)
    assert any(name.startswith(target) for name in failed), (label, sorted(failed))
