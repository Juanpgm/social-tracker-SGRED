"""What the forms say about who acts and why."""

from collections.abc import Mapping

ACTOR = "web:local"
DEFAULT_REASON = "via web UI"


def reason_of(form: Mapping[str, str]) -> str:
    """The operator's reason, or the default when the form carries none (row buttons)."""
    return form.get("reason", DEFAULT_REASON)
