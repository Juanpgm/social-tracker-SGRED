"""From parsed feed entries to normalized `Item`s (CI-2, CI-8, CI-9).

Everything in an entry is untrusted, so the text of an item is what is VISIBLE in the entry:
markup is parsed, `script`/`style` and similar content is dropped, entities are decoded, and
the result is plain text. The entry exactly as received (markup included) is kept in
`raw_payload`; it is data and must be escaped by whatever shows it.

Rules, each with a flag on the item or a reason in the report:
- no visible title and no visible text: discarded, counted as `empty`; entries past the feed
  limit are counted as `over_limit`;
- text over 100 KiB is cut at the byte limit without splitting a character (`truncated`), as
  is a title over 1000 characters;
- no usable publication date: the fetch time is used (`published_at_missing`); a date after the
  fetch time is clamped to it (`published_at_future`);
- no guid: the external id is a hash of link and title (`external_id_synthesized`), and a guid
  longer than the id limit is replaced by its hash;
- a link that is not a plain http(s) address is dropped; the item is kept.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser

from social_tracker.collector.feed import FeedEntry
from social_tracker.items.domain import (
    MAX_AUTHOR_CHARS,
    MAX_EXTERNAL_ID_CHARS,
    MAX_TEXT_BYTES,
    MAX_TITLE_CHARS,
    Item,
    ItemKind,
    ItemStatus,
    bounded_raw_payload,
    content_hash,
    item_id,
    normalize_text,
    optional_url,
    truncate_utf8,
)
from social_tracker.shared.errors import ValidationError

MAX_HTML_CHARS = 500_000  # input to the HTML reader is bounded, so its cost is too
DISCARD_EMPTY = "empty"
DISCARD_OVER_LIMIT = "over_limit"

_DROPPED_CONTENT = frozenset({"script", "style", "noscript", "template", "iframe", "object"})
_PARAGRAPH_TAGS = frozenset(
    {"p", "div", "section", "article", "blockquote", "pre", "table", "ul", "ol"}
    | {"h1", "h2", "h3", "h4", "h5", "h6"}
)
_LINE_START_TAGS = frozenset({"br", "hr"})
_LINE_END_TAGS = frozenset({"li", "tr"})


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._dropping = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROPPED_CONTENT:
            self._dropping += 1
        elif tag in _PARAGRAPH_TAGS:
            self.parts.append("\n\n")
        elif tag in _LINE_START_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROPPED_CONTENT:
            self._dropping = max(0, self._dropping - 1)
        elif tag in _PARAGRAPH_TAGS:
            self.parts.append("\n\n")
        elif tag in _LINE_END_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._dropping:
            self.parts.append(data)


def strip_html(html: str) -> str:
    """The visible text of `html` (or of plain text), normalized. Never executes or follows
    anything; scripts, styles, comments and attributes disappear."""
    reader = _VisibleText()
    reader.feed(html[:MAX_HTML_CHARS])
    reader.close()
    return normalize_text("".join(reader.parts))


# ---- fields ---------------------------------------------------------------------------


def parse_published(raw: str | None) -> datetime | None:
    """RFC 822 (RSS) or ISO 8601 (Atom) text as UTC; a naive time is taken as UTC."""
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError, OverflowError):
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    try:
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    except (OverflowError, ValueError):
        return None


def _hash_id(*parts: str) -> str:
    return "h:" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:40]


def _external_id(entry: FeedEntry, title: str, text: str) -> tuple[str, bool]:
    guid = (entry.guid or "").strip()
    if guid and len(guid) <= MAX_EXTERNAL_ID_CHARS:
        return guid, False
    if guid:
        return _hash_id("guid", guid), False
    link = (entry.link or "").strip()
    return _hash_id(link, title, "" if link or title else text), True


def _safe_url(link: str | None) -> str | None:
    """The link if it is a plain http(s) address (see `Item`), else None."""
    candidate = (link or "").strip()
    try:
        return optional_url(candidate) if candidate else None
    except ValidationError:
        return None


def _one_line(text: str, limit: int) -> str:
    return " ".join(text.split())[:limit]


# ---- entries --------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizeReport:
    items: tuple[Item, ...]
    discarded: dict[str, int]


def _normalize(
    entry: FeedEntry, source_id: str, source_name: str, fetched_at: datetime
) -> Item | None:
    flags: set[str] = set()
    title_full = _one_line(strip_html(entry.title), 10 * MAX_TITLE_CHARS)
    title = title_full[:MAX_TITLE_CHARS]
    text, text_cut = truncate_utf8(strip_html(entry.body_html), MAX_TEXT_BYTES)
    if not title and not text.strip():
        return None
    if len(title_full) > MAX_TITLE_CHARS or text_cut:
        flags.add("truncated")

    external_id, synthesized = _external_id(entry, title, text)
    if synthesized:
        flags.add("external_id_synthesized")

    published = parse_published(entry.published)
    if published is None:
        published = fetched_at
        flags.add("published_at_missing")
    elif published > fetched_at:
        published = fetched_at
        flags.add("published_at_future")

    raw, raw_cut = bounded_raw_payload(entry.raw)
    if raw_cut:
        flags.add("raw_truncated")

    author = _one_line(entry.author or "", MAX_AUTHOR_CHARS) or None
    return Item(
        id=item_id(source_id, external_id),
        source_id=source_id,
        source_name=source_name,
        kind=ItemKind.ARTICLE,
        external_id=external_id,
        parent_id=None,
        url=_safe_url(entry.link),
        title=title,
        author=author,
        text=text,
        language=None,
        published_at=published,
        fetched_at=fetched_at,
        content_hash=content_hash(title, text),
        flags=tuple(sorted(flags)),
        status=ItemStatus.NEW,
        raw_payload=raw,
    )


def normalize_entries(
    entries: Sequence[FeedEntry],
    *,
    source_id: str,
    source_name: str,
    fetched_at: datetime,
    over_limit: int = 0,
) -> NormalizeReport:
    items: list[Item] = []
    discarded: dict[str, int] = {}
    for entry in entries:
        item = _normalize(entry, source_id, source_name, fetched_at)
        if item is None:
            discarded[DISCARD_EMPTY] = discarded.get(DISCARD_EMPTY, 0) + 1
        else:
            items.append(item)
    if over_limit:
        discarded[DISCARD_OVER_LIMIT] = over_limit
    return NormalizeReport(tuple(items), discarded)
