"""Slug identifiers: lowercase ASCII letters, digits and underscore, 3-64 characters."""

import re
from typing import Any

from social_tracker.shared.errors import ValidationError

SLUG_MIN = 3
SLUG_MAX = 64
# Linear-time pattern (single bounded character class); used with fullmatch so a
# trailing newline is not accepted the way `$` would accept it.
_SLUG_RE = re.compile(rf"[a-z0-9_]{{{SLUG_MIN},{SLUG_MAX}}}")


def is_valid_slug(value: Any) -> bool:
    return isinstance(value, str) and _SLUG_RE.fullmatch(value) is not None


def validate_slug(value: Any) -> str:
    if not is_valid_slug(value):
        raise ValidationError(
            f"invalid slug: must match [a-z0-9_] and be {SLUG_MIN}-{SLUG_MAX} characters"
        )
    return value
