import pytest

from social_tracker.control.domain import (
    MAX_SETTING_KEY,
    source_analysis_key,
    validate_setting_key,
)
from social_tracker.shared.errors import ValidationError


def test_the_key_is_the_prefix_plus_the_source_id():
    assert source_analysis_key("src_one") == "source_analysis.src_one"


def test_the_longest_possible_key_is_a_valid_setting_key():
    key = source_analysis_key("a" * 64)
    assert key == "source_analysis." + "a" * 64
    assert len(key) <= MAX_SETTING_KEY
    assert validate_setting_key(key) == key


@pytest.mark.parametrize("bad", ["", "ab", "UPPER", "src one", "src.one", "a" * 65, None, 5])
def test_ids_that_are_not_slugs_are_rejected(bad):
    with pytest.raises(ValidationError):
        source_analysis_key(bad)
