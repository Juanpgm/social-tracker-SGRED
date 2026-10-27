"""Wiring check: each repository contract runs against BOTH the in-memory fake and SQLite.

The fake is the reference and the SQLite adapter must be indistinguishable from it, so a
contract that only one of them is subjected to (or a SQLite subclass that overrides contract
tests to make them pass) would silently break that promise.
"""

import pytest

from social_tracker.persistence.control_repo import SqliteControlRepository
from social_tracker.persistence.sources_repo import SqliteSourceRepository
from tests.classification import test_fake_repositories as classification_fake
from tests.classification.repository_contracts import (
    ClassificationRepositoryContract,
    SignatureRepositoryContract,
    TopicConfigRepositoryContract,
)
from tests.control import test_fake_repository as control_fake
from tests.control.repository_contract import ControlRepositoryContract
from tests.fakes.control_repository import InMemoryControlRepository
from tests.fakes.source_repository import InMemorySourceRepository
from tests.persistence import test_sqlite_classification_repositories as classification_sqlite
from tests.persistence import test_sqlite_control_repository as control_sqlite
from tests.persistence import test_sqlite_source_repository as sources_sqlite
from tests.sources import test_fake_repository as sources_fake
from tests.sources.repository_contract import SourceRepositoryContract

# Suites are reached through their modules and bound to non-`Test*` names: otherwise pytest
# would collect (and run) them a second time from this file.
FAKE_SOURCES = sources_fake.TestInMemorySourceRepository
FAKE_CONTROL = control_fake.TestInMemoryControlRepository
SQLITE_SOURCES = sources_sqlite.TestSqliteSourceRepositoryContract
SQLITE_CONTROL = control_sqlite.TestSqliteControlRepositoryContract

CASES = [
    (SourceRepositoryContract, FAKE_SOURCES, SQLITE_SOURCES),
    (ControlRepositoryContract, FAKE_CONTROL, SQLITE_CONTROL),
    (
        ClassificationRepositoryContract,
        classification_fake.TestInMemoryClassificationRepository,
        classification_sqlite.TestSqliteClassificationRepositoryContract,
    ),
    (
        SignatureRepositoryContract,
        classification_fake.TestInMemorySignatureRepository,
        classification_sqlite.TestSqliteSignatureRepositoryContract,
    ),
    (
        TopicConfigRepositoryContract,
        classification_fake.TestInMemoryTopicConfigRepository,
        classification_sqlite.TestSqliteTopicConfigRepositoryContract,
    ),
]


def contract_tests(contract: type) -> set[str]:
    return {name for name in vars(contract) if name.startswith("test_")}


@pytest.mark.parametrize(("contract", "fake_suite", "sqlite_suite"), CASES)
def test_both_implementations_subclass_the_same_contract(contract, fake_suite, sqlite_suite):
    assert issubclass(fake_suite, contract)
    assert issubclass(sqlite_suite, contract)
    assert contract_tests(contract)  # the contract is not empty


@pytest.mark.parametrize(("contract", "fake_suite", "sqlite_suite"), CASES)
def test_no_suite_overrides_or_skips_a_contract_test(contract, fake_suite, sqlite_suite):
    for suite in (fake_suite, sqlite_suite):
        overridden = contract_tests(contract) & set(vars(suite))
        assert overridden == set(), f"{suite.__name__} overrides {sorted(overridden)}"


def test_the_two_suites_build_the_expected_implementations(tmp_path):
    assert isinstance(FAKE_SOURCES().make_repo(), InMemorySourceRepository)
    assert isinstance(FAKE_CONTROL().make_repo(), InMemoryControlRepository)

    sources, control = (
        SQLITE_SOURCES(),
        SQLITE_CONTROL(),
    )
    for suite, expected in ((sources, SqliteSourceRepository), (control, SqliteControlRepository)):
        suite._root = tmp_path
        suite._opened = []
        first, second = suite.make_repo(), suite.make_repo()
        try:
            assert isinstance(first, expected)
            assert first._db is not second._db  # fresh, isolated database per call
            assert first._db.path != second._db.path
        finally:
            for db in suite._opened:
                db.close()


def test_the_sqlite_control_suite_does_not_widen_the_public_surface():
    assert frozenset() == SQLITE_CONTROL.ALLOWED_EXTRA_PUBLIC
    assert frozenset() == FAKE_CONTROL.ALLOWED_EXTRA_PUBLIC
