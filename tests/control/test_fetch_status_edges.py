"""Edge cases of error sanitizing and of the stored validators."""

import pytest

from social_tracker.control.domain import FetchStatus, sanitize_error
from social_tracker.shared.errors import ValidationError


@pytest.mark.parametrize("not_text", [None, 5, b"bytes", ["list"], {"a": 1}])
def test_anything_that_is_not_text_becomes_a_generic_error(not_text):
    assert sanitize_error(not_text) == "unknown error"


@pytest.mark.parametrize("etag", ["tab\there", "café", "new\nline", "\x7f"])
def test_validators_must_be_printable_ascii(etag):
    with pytest.raises(ValidationError):
        FetchStatus("src_one", etag=etag)
    with pytest.raises(ValidationError):
        FetchStatus("src_one", last_modified=etag)


def test_a_typical_validator_is_accepted():
    status = FetchStatus(
        "src_one", etag='W/"abc-123"', last_modified="Sun, 01 Mar 2026 08:00:00 GMT"
    )
    assert status.etag == 'W/"abc-123"'
