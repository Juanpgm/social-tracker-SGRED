"""Meta-tests: the shared contracts must FAIL for every deliberately wrong SQLite repository.

A contract that passes a repository without the UNIQUE index, without the lock, without
ORDER BY or with a DELETE that succeeds for a missing id would give false confidence. Each
wrong variant (`wrong_implementations.py`) is paired with the contract tests that must catch
it, and the same tests must pass for the correct adapters.
"""

import inspect
import itertools
from collections.abc import Callable
from pathlib import Path

import pytest

from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.db import Database
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from social_tracker.shared.errors import ConflictError, NotFoundError, StorageError
from tests.control.repository_contract import ControlRepositoryContract
from tests.persistence.wrong_implementations import CONTROL_WRONG, SOURCE_WRONG
from tests.sources.repository_contract import SourceRepositoryContract

SOURCE_MUST_BE_CAUGHT_BY = {
    "no_unique_name_index": [
        "test_add_duplicate_name_differing_by_unicode_composition_conflicts",
        "test_two_threads_creating_the_same_name_exactly_one_wins",
        "test_update_to_another_sources_name_conflicts_and_changes_nothing",
    ],
    "id_is_not_unique": [
        "test_add_duplicate_id_conflicts",
        "test_two_threads_creating_the_same_id_exactly_one_wins",
    ],
    "no_lock": [
        "test_two_threads_creating_the_same_name_exactly_one_wins",
        "test_two_threads_creating_the_same_id_exactly_one_wins",
    ],
    "list_without_order_by": ["test_list_is_sorted_by_id"],
    "list_ordered_by_rowid": ["test_list_is_sorted_by_id"],
    "delete_ignores_missing": [
        "test_delete_missing_raises_not_found",
        "test_delete_twice_raises_not_found_the_second_time",
    ],
    "update_ignores_missing": ["test_update_missing_raises_not_found"],
    "case_sensitive_name_lookup": [
        "test_tampering_with_the_added_source_cannot_change_the_stored_copy"
    ],
    "drops_keywords": ["test_every_source_type_round_trips"],
    "conflict_becomes_storage_error": [
        "test_add_duplicate_id_conflicts",
        "test_add_duplicate_name_differing_by_unicode_composition_conflicts",
    ],
    "update_is_delete_then_add": [
        "test_update_to_another_sources_name_conflicts_and_changes_nothing"
    ],
}

CONTROL_MUST_BE_CAUGHT_BY = {
    "events_ordered_by_timestamp": ["test_append_order_wins_over_timestamp_order_and_kind_order"],
    "events_ordered_by_kind": [
        "test_append_order_wins_over_timestamp_order_and_kind_order",
        "test_same_timestamp_events_keep_append_order",
    ],
    "empty_id_means_no_filter": ["test_an_empty_source_filter_is_not_the_same_as_no_filter"],
    "event_log_capped_at_100": ["test_a_long_log_is_never_truncated"],
    "settings_without_validation": [
        "test_invalid_value_is_rejected_and_keeps_the_old_one",
        "test_global_analysis_rejects_other_values_and_keeps_the_old_one",
        "test_non_string_key_is_rejected",
    ],
    "state_insert_only": ["test_save_is_an_upsert"],
    "delete_state_not_idempotent": ["test_delete_source_state_is_idempotent"],
    "delete_state_wipes_every_state": ["test_delete_source_state_removes_only_that_state"],
    "delete_state_does_nothing": ["test_delete_source_state_removes_only_that_state"],
    "upsert_skips_state": ["test_save_is_an_upsert"],
    "upsert_skips_failures": ["test_save_is_an_upsert"],
    "upsert_skips_next_run_at": ["test_save_is_an_upsert"],
    "state_requires_registered_source": ["test_saving_state_for_an_unregistered_source_is_allowed"],
    "analysis_defaults_to_true": ["test_global_analysis_defaults_to_false"],
    "exposes_purge": ["test_the_implementation_exposes_nothing_beyond_the_port"],
    "forgets_next_run_at": ["test_save_and_get_round_trip_including_backoff_schedule"],
    "events_lose_their_source_id": ["test_events_can_be_filtered_by_source"],
    "no_lock": ["test_concurrent_appends_are_all_recorded"],
    "settings_not_isolated_between_repositories": [
        "test_a_fresh_repository_does_not_see_another_ones_settings"
    ],
}


# How each wrong variant must be caught. Unlisted pairs fail by plain `assert` (AssertionError).
# The listed ones fail differently on purpose: a missing error (`pytest.fail`), or the wrong
# repository itself raising. Pinning type AND message means an unrelated exception (say a
# ValidationError from a broken test fixture) can never pass as "the contract caught it".
DID_NOT_RAISE = pytest.fail.Exception
EXPECTED_FAILURE: dict[tuple[str, str], tuple[type[BaseException], str]] = {
    (
        "no_unique_name_index",
        "test_add_duplicate_name_differing_by_unicode_composition_conflicts",
    ): (
        DID_NOT_RAISE,
        "DID NOT RAISE.*ConflictError",
    ),
    ("no_unique_name_index", "test_update_to_another_sources_name_conflicts_and_changes_nothing"): (
        DID_NOT_RAISE,
        "DID NOT RAISE.*ConflictError",
    ),
    ("id_is_not_unique", "test_add_duplicate_id_conflicts"): (
        DID_NOT_RAISE,
        "DID NOT RAISE.*ConflictError",
    ),
    ("delete_ignores_missing", "test_delete_missing_raises_not_found"): (
        DID_NOT_RAISE,
        "DID NOT RAISE.*NotFoundError",
    ),
    ("delete_ignores_missing", "test_delete_twice_raises_not_found_the_second_time"): (
        DID_NOT_RAISE,
        "DID NOT RAISE.*NotFoundError",
    ),
    ("update_ignores_missing", "test_update_missing_raises_not_found"): (
        DID_NOT_RAISE,
        "DID NOT RAISE.*NotFoundError",
    ),
    ("conflict_becomes_storage_error", "test_add_duplicate_id_conflicts"): (
        StorageError,
        "duplicate",
    ),
    (
        "conflict_becomes_storage_error",
        "test_add_duplicate_name_differing_by_unicode_composition_conflicts",
    ): (StorageError, "duplicate"),
    (
        "update_is_delete_then_add",
        "test_update_to_another_sources_name_conflicts_and_changes_nothing",
    ): (
        NotFoundError,
        "source not found",
    ),
    ("settings_without_validation", "test_invalid_value_is_rejected_and_keeps_the_old_one"): (
        StorageError,
        "SQLITE_CONSTRAINT_NOTNULL",
    ),
    (
        "settings_without_validation",
        "test_global_analysis_rejects_other_values_and_keeps_the_old_one",
    ): (StorageError, "SQLITE_CONSTRAINT_CHECK"),
    ("settings_without_validation", "test_non_string_key_is_rejected"): (
        DID_NOT_RAISE,
        "DID NOT RAISE.*ValidationError",
    ),
    ("delete_state_not_idempotent", "test_delete_source_state_is_idempotent"): (
        NotFoundError,
        "source not found",
    ),
    ("state_insert_only", "test_save_is_an_upsert"): (ConflictError, "unique constraint violated"),
    ("upsert_skips_next_run_at", "test_save_is_an_upsert"): (
        StorageError,
        "SQLITE_CONSTRAINT_CHECK",
    ),
    (
        "state_requires_registered_source",
        "test_saving_state_for_an_unregistered_source_is_allowed",
    ): (
        NotFoundError,
        "source not found",
    ),
    ("forgets_next_run_at", "test_save_and_get_round_trip_including_backoff_schedule"): (
        StorageError,
        "SQLITE_CONSTRAINT_CHECK",
    ),
}


def expected_failure(wrong: str, test_name: str) -> tuple[type[BaseException], str | None]:
    return EXPECTED_FAILURE.get((wrong, test_name), (AssertionError, None))


def _run(contract_cls, build: Callable[[Path], object], root: Path, test_name: str) -> None:
    counter = itertools.count()
    built: list[object] = []

    class Contract(contract_cls):
        def make_repo(self):
            repo = build(root / f"db-{next(counter)}.db")
            built.append(repo)
            return repo

    contract = Contract()
    try:
        test = getattr(contract, test_name)
        if "repo" in inspect.signature(test).parameters:
            test(contract.make_repo())
        else:
            test()
    finally:
        for repo in built:
            repo._db.close()  # type: ignore[attr-defined]


def _correct_source(path: Path) -> object:
    return SqliteSourceRepository(Database(path))


def _correct_control(path: Path) -> object:
    return SqliteControlRepository(Database(path))


def test_every_wrong_variant_has_a_catching_test():
    assert set(SOURCE_MUST_BE_CAUGHT_BY) == set(SOURCE_WRONG)
    assert set(CONTROL_MUST_BE_CAUGHT_BY) == set(CONTROL_WRONG)


def test_every_pinned_failure_refers_to_a_real_wrong_variant_and_test():
    pairs = {
        (wrong, test)
        for mapping in (SOURCE_MUST_BE_CAUGHT_BY, CONTROL_MUST_BE_CAUGHT_BY)
        for wrong, tests in mapping.items()
        for test in tests
    }
    assert set(EXPECTED_FAILURE) <= pairs


def test_the_catching_tests_exist_and_are_not_parametrized():
    # They are invoked directly, so a typo or a parametrized test would fail for the wrong reason.
    for contract_cls, mapping in (
        (SourceRepositoryContract, SOURCE_MUST_BE_CAUGHT_BY),
        (ControlRepositoryContract, CONTROL_MUST_BE_CAUGHT_BY),
    ):
        for test_name in {t for tests in mapping.values() for t in tests}:
            test = getattr(contract_cls, test_name)
            assert not hasattr(test, "pytestmark"), f"{test_name} is parametrized"
            assert set(inspect.signature(test).parameters) <= {"self", "repo"}, test_name


# The wrong variants make the contract's own worker threads blow up with raw errors; that is the
# point, so the resulting "unhandled thread exception" warning is noise here.
@pytest.mark.parametrize(
    ("wrong", "test_name"),
    [(name, t) for name, tests in SOURCE_MUST_BE_CAUGHT_BY.items() for t in tests],
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_source_contract_catches_wrong_sqlite_repositories(tmp_path, wrong, test_name):
    exc_type, match = expected_failure(wrong, test_name)
    with pytest.raises(exc_type, match=match):
        _run(SourceRepositoryContract, SOURCE_WRONG[wrong], tmp_path, test_name)


@pytest.mark.parametrize(
    ("wrong", "test_name"),
    [(name, t) for name, tests in CONTROL_MUST_BE_CAUGHT_BY.items() for t in tests],
)
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_control_contract_catches_wrong_sqlite_repositories(tmp_path, wrong, test_name):
    exc_type, match = expected_failure(wrong, test_name)
    with pytest.raises(exc_type, match=match):
        _run(ControlRepositoryContract, CONTROL_WRONG[wrong], tmp_path, test_name)


@pytest.mark.parametrize(
    "test_name", sorted({t for tests in SOURCE_MUST_BE_CAUGHT_BY.values() for t in tests})
)
def test_the_source_catching_tests_pass_for_the_correct_adapter(tmp_path, test_name):
    _run(SourceRepositoryContract, _correct_source, tmp_path, test_name)


@pytest.mark.parametrize(
    "test_name", sorted({t for tests in CONTROL_MUST_BE_CAUGHT_BY.values() for t in tests})
)
def test_the_control_catching_tests_pass_for_the_correct_adapter(tmp_path, test_name):
    _run(ControlRepositoryContract, _correct_control, tmp_path, test_name)
