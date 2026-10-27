"""RSS 2.0, RSS 1.0 (RDF) and Atom parsing. The input is untrusted bytes from the network.

Safety (CI-5, CI-9):
- XML goes through `defusedxml` with document type declarations forbidden outright. Entity
  expansion bombs (billion laughs, quadratic blowup), external entities (XXE), external DTDs and
  parameter entities all need a DTD, so none of them is ever evaluated. XInclude is not processed.
- The tree is checked for depth before any recursive helper touches it, and the number of
  entries is bounded (`MAX_ENTRIES`; the rest is counted, not returned).
- Bytes that are not valid for the declared encoding, characters XML forbids and HTML entity
  names such as `&nbsp;` make strict parsing fail; a second, lenient pass replaces undecodable
  bytes with U+FFFD, drops the forbidden characters and numbers the entity names. Anything still
  broken (a truncated document, HTML, JSON) is a `FeedError`: a half-read feed is never
  returned, so the caller cannot store a partial fetch.
- Error messages are fixed text. They never echo the document.

Entry text is returned as it appeared (markup included) in `body_html` and `raw`; turning it
into plain text is `collector.normalize`'s job.
"""

import re
from dataclasses import dataclass
from html.entities import name2codepoint
from typing import Any
from xml.etree.ElementTree import Element, ParseError

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException

from social_tracker.shared.errors import DomainError

MAX_FEED_BYTES = 4 * 1024 * 1024
MAX_ENTRIES = 500
MAX_DEPTH = 100

_RSS_10 = "http://purl.org/rss/1.0/"
_ATOM = "http://www.w3.org/2005/Atom"
_CONTENT_ENCODED = "{http://purl.org/rss/1.0/modules/content/}encoded"
_DC_CREATOR = "{http://purl.org/dc/elements/1.1/}creator"
_DC_DATE = "{http://purl.org/dc/elements/1.1/}date"
_STANDARD_NAMESPACES = frozenset({"", _RSS_10, _ATOM})

_XML_FORBIDDEN_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f" + chr(0xFFFE) + chr(0xFFFF) + "]")
_ENTITY_NAME_RE = re.compile(r"&([A-Za-z][A-Za-z0-9]{1,31});")


class FeedError(DomainError):
    """The bytes are not a usable feed. The message is fixed text and never echoes the input."""

    code = "invalid_feed"


@dataclass(frozen=True)
class FeedEntry:
    title: str
    link: str | None
    guid: str | None
    published: str | None
    body_html: str
    author: str | None
    raw: dict[str, str]


@dataclass(frozen=True)
class ParsedFeed:
    entries: tuple[FeedEntry, ...]
    over_limit: int = 0


# ---- XML ------------------------------------------------------------------------------


def _number_html_entities(match: re.Match[str]) -> str:
    codepoint = name2codepoint.get(match.group(1))
    return match.group(0) if codepoint is None else f"&#{codepoint};"


def _lenient_text(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    text = _XML_FORBIDDEN_RE.sub("", text)
    return _ENTITY_NAME_RE.sub(_number_html_entities, text)


def _parse_tree(data: bytes) -> Element:
    try:
        try:
            return SafeET.fromstring(data, forbid_dtd=True)
        except ParseError:
            return SafeET.fromstring(_lenient_text(data), forbid_dtd=True)
    except DefusedXmlException:
        raise FeedError("the feed has a document type declaration, which is not accepted") from None
    except (ParseError, ValueError, LookupError):
        raise FeedError("the document is not well-formed XML") from None


def _check_depth(root: Element) -> None:
    stack = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise FeedError("the feed is nested too deeply")
        stack.extend((child, depth + 1) for child in element)


# ---- entries --------------------------------------------------------------------------


def _split(tag: Any) -> tuple[str, str]:
    if not isinstance(tag, str):
        return "", ""
    if tag.startswith("{"):
        namespace, _, local = tag[1:].partition("}")
        return namespace, local
    return "", tag


def _text(element: Element | None) -> str:
    return "".join(element.itertext()).strip() if element is not None else ""


def _alternate_link(links: list[Element]) -> str | None:
    for link in links:
        if link.get("rel", "alternate") == "alternate" and link.get("href", "").strip():
            return link.get("href", "").strip()
    return None


def _read_entry(element: Element) -> FeedEntry:
    standard: dict[str, Element] = {}
    extra: dict[str, Element] = {}
    atom_links: list[Element] = []
    for child in element:
        namespace, local = _split(child.tag)
        if namespace in _STANDARD_NAMESPACES:
            standard.setdefault(local, child)
            if local == "link" and namespace == _ATOM:
                atom_links.append(child)
        else:
            extra.setdefault(f"{{{namespace}}}{local}", child)

    link = _alternate_link(atom_links) if atom_links else _text(standard.get("link")) or None
    guid = _text(standard.get("guid")) or _text(standard.get("id")) or None
    published = (
        _text(standard.get("pubDate"))
        or _text(standard.get("published"))
        or _text(extra.get(_DC_DATE))
        or _text(standard.get("updated"))
        or None
    )
    author_element = standard.get("author")
    author_name = author_element.find(f"{{{_ATOM}}}name") if author_element is not None else None
    author = _text(author_name) or _text(author_element) or _text(extra.get(_DC_CREATOR)) or None

    description = _text(standard.get("description")) or _text(standard.get("summary"))
    content = _text(extra.get(_CONTENT_ENCODED)) or _text(standard.get("content"))
    title = _text(standard.get("title"))
    raw = {
        "title": title,
        "link": link or "",
        "guid": guid or "",
        "published": published or "",
        "description": description,
        "content": content,
        "author": author or "",
    }
    return FeedEntry(
        title=title,
        link=link,
        guid=guid,
        published=published,
        body_html=content or description,
        author=author,
        raw={key: value for key, value in raw.items() if value},
    )


def _entry_elements(root: Element) -> list[Element]:
    namespace, local = _split(root.tag)
    if local == "rss":
        channel = next((c for c in root if _split(c.tag)[1] == "channel"), None)
        container, wanted = channel if channel is not None else [], "item"
    elif local == "RDF":
        container, wanted = root, "item"
    elif local == "feed" and namespace == _ATOM:
        container, wanted = root, "entry"
    else:
        raise FeedError("the document is not an RSS or Atom feed")
    return [child for child in container if _split(child.tag)[1] == wanted]


def parse_feed(data: bytes) -> ParsedFeed:
    """The entries of an RSS or Atom document, or `FeedError`."""
    if not isinstance(data, bytes):
        raise FeedError("the feed must be bytes")
    if not data.strip():
        raise FeedError("the feed is empty")
    if len(data) > MAX_FEED_BYTES:
        raise FeedError("the feed is too large")
    root = _parse_tree(data)
    _check_depth(root)
    elements = _entry_elements(root)
    entries = tuple(_read_entry(element) for element in elements[:MAX_ENTRIES])
    return ParsedFeed(entries, over_limit=max(0, len(elements) - MAX_ENTRIES))
