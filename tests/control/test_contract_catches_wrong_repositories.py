"""Meta-tests: the shared contract must FAIL for every deliberately wrong repository.

Without this, a contract that silently passes broken implementations (empty id as "no filter",
events sorted instead of kept in append order, a public purge, a ring buffer...) would give
unit 1c false confidence. Each wrong repository is paired with the contract tests that must
catch it.
"""

import inspect

import pytest

from social_tracker.shared.errors import DomainError
from tests.control.repository_contract import ControlRepositoryContract
from tests.control.wrong_repositories import WRONG_REPOSITORIES

MUST_BE_CAUGHT_BY = {
    "empty_id_means_no_filter": ["test_an_empty_source_filter_is_not_the_same_as_no_filter"],
    "sorted_by_timestamp": ["test_append_order_wins_over_timestamp_order_and_kind_order"],
    "sorted_by_kind": [
        "test_append_order_wins_over_timestamp_order_and_kind_order",
        "test_same_timestamp_events_keep_append_order",
    ],
    "exposes_purge": ["test_the_implementation_exposes_nothing_beyond_the_port"],
    "exposes_truncate": ["test_the_implementation_exposes_nothing_beyond_the_port"],
    "keeps_only_the_last_30": ["test_a_long_log_is_never_truncated"],
    "rewrites_on_append": ["test_events_never_change_or_disappear_as_the_log_grows"],
    "shares_settings_across_instances": [
        "test_a_fresh_repository_does_not_see_another_ones_settings"
    ],
    "delete_setting_not_idempotent": ["test_delete_setting_is_idempotent"],
    "delete_setting_drops_the_default": [
        "test_deleting_a_setting_that_has_a_default_restores_the_default"
    ],
    "list_settings_by_substring": ["test_list_settings_matches_prefixes_not_substrings"],
    "skips_setting_validation": [
        "test_invalid_value_is_rejected_and_keeps_the_old_one",
        "test_setting_value_over_the_limit_is_rejected_and_keeps_the_old_one",
        "test_global_analysis_rejects_other_values_and_keeps_the_old_one",
    ],
}


def test_every_wrong_repository_has_a_catching_test():
    assert set(MUST_BE_CAUGHT_BY) == set(WRONG_REPOSITORIES)


def _contract_for(repo_cls):
    class Contract(ControlRepositoryContract):
        def make_repo(self):
            return repo_cls()

    return Contract()


def _run(contract: ControlRepositoryContract, test_name: str) -> None:
    test = getattr(contract, test_name)
    if "repo" in inspect.signature(test).parameters:
        test(contract.make_repo())
    else:
        test()


@pytest.mark.parametrize(
    ("wrong", "test_name"),
    [(name, t) for name, tests in MUST_BE_CAUGHT_BY.items() for t in tests],
)
def test_contract_catches_wrong_repositories(wrong: str, test_name: str):
    contract = _contract_for(WRONG_REPOSITORIES[wrong])
    with pytest.raises((AssertionError, DomainError, pytest.fail.Exception)):
        _run(contract, test_name)


@pytest.mark.parametrize(
    "test_name", sorted({t for tests in MUST_BE_CAUGHT_BY.values() for t in tests})
)
def test_the_catching_tests_pass_for_a_correct_repository(test_name: str):
    from tests.fakes.control_repository import InMemoryControlRepository

    _run(_contract_for(InMemoryControlRepository), test_name)
