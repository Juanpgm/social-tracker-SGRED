"""Hardened YAML loading shared by every file the operator can hand to the application.

Only `yaml.SafeLoader` machinery is used, on top of a parser-event scan made BEFORE anything is
constructed: explicit tags (including `!!python/...`), anchors and aliases (billion-laughs),
merge keys, duplicate mapping keys and multiple documents are refused, and the document is capped
in parser events and nesting depth. Callers cap the size in bytes themselves, before decoding.
Every refusal is a `Rejected` with a fixed message that never echoes a key or a value.
"""

from typing import Any

import yaml


class Rejected(Exception):  # noqa: N818 - control flow of the loader, callers turn it into their error
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate keys.

    Merge keys (`<<`) are rejected too: there is no constructor for the merge tag, so building
    the key node below fails before any merge could happen.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                if key in seen:
                    raise Rejected("duplicate mapping key")
                seen.add(key)
            except TypeError:
                continue  # unhashable key: the base class reports it
        return super().construct_mapping(node, deep)


class PlainDumper(yaml.SafeDumper):
    """Never emits anchors or aliases, even when the same object appears twice."""

    def ignore_aliases(self, data: Any) -> bool:
        return True


def _scan(text: str, max_events: int, max_depth: int) -> None:
    """Walk the parser events (nothing is constructed) and refuse anything unsafe or unbounded."""
    depth = documents = 0
    for count, event in enumerate(yaml.parse(text, Loader=yaml.SafeLoader), start=1):
        if count > max_events:
            raise Rejected("document is too complex")
        if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None) is not None:
            raise Rejected("anchors and aliases are not allowed")
        if getattr(event, "tag", None) is not None:
            raise Rejected("explicit YAML tags are not allowed")
        if isinstance(event, yaml.DocumentStartEvent):
            documents += 1
            if documents > 1:
                raise Rejected("multiple YAML documents are not allowed")
        elif isinstance(event, yaml.MappingStartEvent | yaml.SequenceStartEvent):
            depth += 1
            if depth > max_depth:
                raise Rejected("nesting is too deep")
        elif isinstance(event, yaml.MappingEndEvent | yaml.SequenceEndEvent):
            depth -= 1


def parse_document(text: str, *, max_events: int, max_depth: int) -> Any:
    """The value of a YAML document, or `Rejected`. A blank text (tabs included) is an empty
    document and gives None."""
    if not text.strip():
        return None
    try:
        _scan(text, max_events, max_depth)
        return yaml.load(text, Loader=StrictLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError:
        raise Rejected("malformed or unsupported YAML document") from None
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise Rejected("malformed or unsupported YAML document") from None
