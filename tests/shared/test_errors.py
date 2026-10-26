"""Shared error taxonomy: stable codes, one catchable base class."""

import pytest

from social_tracker.shared import errors

EXPECTED_CODES = {
    "ValidationError": "validation_error",
    "ConflictError": "conflict",
    "NotFoundError": "not_found",
    "InvalidTransition": "invalid_transition",
    "StorageError": "storage_error",
}


@pytest.mark.parametrize(("name", "code"), EXPECTED_CODES.items())
def test_error_has_stable_code_and_is_a_domain_error(name: str, code: str):
    cls = getattr(errors, name)
    assert issubclass(cls, errors.DomainError)
    assert cls.code == code
    exc = cls("boom")
    assert exc.code == code
    assert str(exc) == "boom"
    assert exc.message == "boom"


def test_codes_are_unique():
    codes = [getattr(errors, n).code for n in EXPECTED_CODES]
    assert len(set(codes)) == len(codes)


@pytest.mark.parametrize("name", EXPECTED_CODES)
def test_every_error_is_catchable_via_domain_error(name: str):
    with pytest.raises(errors.DomainError) as info:
        raise getattr(errors, name)("x")
    assert info.value.code == EXPECTED_CODES[name]


def test_domain_error_base_has_generic_code():
    assert errors.DomainError.code == "domain_error"


def test_sibling_errors_are_distinct_types():
    assert not issubclass(errors.ConflictError, errors.NotFoundError)
    assert not issubclass(errors.NotFoundError, errors.ConflictError)
    assert not issubclass(errors.InvalidTransition, errors.ValidationError)


def test_empty_message_is_allowed_and_code_survives():
    exc = errors.NotFoundError()
    assert exc.message == ""
    assert exc.code == "not_found"


def test_validation_error_carries_known_field_names():
    exc = errors.ValidationError("bad", fields=("id", "name"))
    assert exc.fields == ("id", "name")
    assert str(exc) == "bad"


def test_validation_error_fields_default_to_empty_tuple():
    assert errors.ValidationError("bad").fields == ()
    assert errors.ValidationError().fields == ()


def test_validation_error_fields_are_immutable_even_when_given_a_list():
    exc = errors.ValidationError("bad", fields=["id"])  # type: ignore[arg-type]
    assert exc.fields == ("id",)
    assert isinstance(exc.fields, tuple)
