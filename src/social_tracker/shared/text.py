"""Pure text helpers shared by the domain modules."""


def has_control_chars(value: str) -> bool:
    """True if `value` contains a C0 control (including NUL), DEL or a C1 control character."""
    return any(ord(ch) < 32 or 127 <= ord(ch) < 160 for ch in value)
