"""Pure text helpers shared by the domain modules."""

import unicodedata

# Characters that change how text is shown or parsed without being visible: lone surrogates
# (not encodable as UTF-8), format characters (zero width, bidi overrides, BOM, soft hyphen,
# tags) and the line/paragraph separators. Combining marks (Mn/Mc/Me), accents and emoji are NOT
# in this set. Trade-off: a ZWJ emoji sequence (family, flags built with joiners) contains Cf
# and is therefore refused.
_HIDDEN_CATEGORIES = frozenset({"Cf", "Cs", "Zl", "Zp"})
_FORMAT_CATEGORY = "Cf"


def has_control_chars(value: str) -> bool:
    """True if `value` contains a C0 control (including NUL), DEL or a C1 control character."""
    return any(ord(ch) < 32 or 127 <= ord(ch) < 160 for ch in value)


def has_unsafe_chars(value: str) -> bool:
    """True if the NFC form of `value` has a lone surrogate, a format character (Cf) or a
    line/paragraph separator (U+2028, U+2029)."""
    normalized = unicodedata.normalize("NFC", value)
    return any(unicodedata.category(ch) in _HIDDEN_CATEGORIES for ch in normalized)


def _escape(char: str) -> str:
    code = ord(char)
    if code <= 0xFF:
        return f"\\x{code:02x}"
    if code <= 0xFFFF:
        return f"\\u{code:04x}"
    return f"\\U{code:08x}"


def escape_unprintable(value: str) -> str:
    """`value` with every control character (C0, DEL, C1) and every hidden character (see
    `has_unsafe_chars`) replaced by a visible `\\xNN`, `\\uNNNN` or `\\UNNNNNNNN` escape, so
    printing it cannot move the cursor, recolour the terminal or reorder what follows. Other
    text, backslashes included, is returned unchanged."""
    return "".join(
        _escape(ch)
        if ord(ch) < 32 or 127 <= ord(ch) < 160 or unicodedata.category(ch) in _HIDDEN_CATEGORIES
        else ch
        for ch in value
    )


def is_blank_text(value: str) -> bool:
    """True if nothing visible is left once format characters and whitespace are removed."""
    visible = "".join(ch for ch in value if unicodedata.category(ch) != _FORMAT_CATEGORY)
    return not visible.strip()
