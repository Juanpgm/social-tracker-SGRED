"""Shared text helpers."""

import pytest

from social_tracker.shared.text import has_control_chars


@pytest.mark.parametrize("code", [*range(0, 32), 127, *range(128, 160)])
def test_every_c0_c1_control_and_delete_is_detected(code: int):
    char = chr(code)
    assert has_control_chars(char)
    assert has_control_chars(f"a{char}b")
    assert has_control_chars(f"{char}ab")
    assert has_control_chars(f"ab{char}")


@pytest.mark.parametrize("code", [32, 33, 126, 160, 161, 173, 0x2028, 0x200B, 0xFEFF])
def test_printable_boundaries_are_not_control_characters(code: int):
    # 160 (NBSP) and 161 sit right after the C1 block; the rest are format/separator characters.
    assert not has_control_chars(chr(code))


@pytest.mark.parametrize("text", ["", "plain", "café 新聞 \U0001f4f0", "a b"])
def test_ordinary_text_has_no_control_characters(text: str):
    assert not has_control_chars(text)
