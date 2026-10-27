"""YAML codec of the topic configuration: strict, bounded, and the shipped defaults.

Loading goes through `shared.strict_yaml` (no tags, anchors, merge keys, duplicate keys or second
documents; capped in events and depth) and then `TopicConfig.from_mapping`. The text is capped in
bytes. Messages never echo what the file says.
"""

from importlib.resources import files
from typing import Any

import yaml

from social_tracker.classification.topics import TopicConfig
from social_tracker.shared.errors import ValidationError
from social_tracker.shared.strict_yaml import PlainDumper, Rejected, parse_document

MAX_CONFIG_BYTES = 64 * 1024
_MAX_EVENTS = 20_000
_MAX_DEPTH = 6
_DEFAULT_RESOURCE = "default_topics.yaml"
_VERIFY_NOTE = (
    "# CONFIG ONLY - verify before relying: official_domains and official_channel_hints are\n"
    "# whatever the operator put there; nobody has checked them. Only official_domains is used.\n"
)


def load_topic_config(text: str) -> TopicConfig:
    """The configuration a YAML text describes, or `ValidationError`."""
    if not isinstance(text, str):
        raise ValidationError("the topic configuration must be text")
    if len(text.encode("utf-8", "surrogatepass")) > MAX_CONFIG_BYTES:
        raise ValidationError(f"the topic configuration is larger than {MAX_CONFIG_BYTES} bytes")
    try:
        document: Any = parse_document(text, max_events=_MAX_EVENTS, max_depth=_MAX_DEPTH)
    except Rejected as exc:
        raise ValidationError(exc.message) from None
    return TopicConfig.from_mapping(document)


def dump_topic_config(config: TopicConfig) -> str:
    """The configuration as YAML (canonical order, real characters), with a reminder that the
    official domains and channel hints are unverified."""
    body = yaml.dump(
        config.to_mapping(),
        Dumper=PlainDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=MAX_CONFIG_BYTES,
    )
    return _VERIFY_NOTE + body


def default_topic_config() -> TopicConfig:
    """The configuration shipped with the application."""
    resource = files("social_tracker.classification").joinpath(_DEFAULT_RESOURCE)
    return load_topic_config(resource.read_text(encoding="utf-8"))
