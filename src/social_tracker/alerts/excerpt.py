"""The excerpt an alert keeps of an item, and the snapshot of a source name (AL-11). Pure.

The text of an item is untrusted. What an alert keeps of it is:

- bounded: at most `MAX_EXCERPT_CHARS` characters, cut with an ellipsis when longer, and the
  work is bounded too (a hostile megabyte is read through a window, never in full);
- plain: control characters (NUL, escape sequences...), hidden format characters (zero width,
  right-to-left overrides, soft hyphen, BOM), line and paragraph separators and lone surrogates
  are removed; every other run of white space is one space;
- free of secrets: a link loses its credentials, query and fragment (a query string is where
  tokens live), and cookies, authorization headers, tokens, passwords and the like are replaced by
  `[redacted]`.

It is stored as text and is NOT HTML-escaped: the template that draws it escapes it.
"""

import re

from social_tracker.alerts.domain import MAX_EXCERPT_CHARS, MAX_SOURCE_NAME_CHARS

ELLIPSIS = chr(0x2026)
TITLE_SEPARATOR = " | "
REDACTED = "[redacted]"
_SCAN_CHARS = 2_000_000  # the part of a text that is ever looked at
_WORK_CHARS = 4_000  # what is kept for the redaction, a few times the excerpt

# Every character of the general categories Cc (except the white-space controls turned into
# spaces first), Cf, Cs, Zl and Zp, as code point ranges. tests/alerts/test_excerpt.py compares
# this class with `unicodedata` over the whole code space, so an update of Unicode cannot leave a
# hole.
_HIDDEN_RANGES = (
    (0x00, 0x08), (0x0E, 0x1F), (0x7F, 0x9F),  # Cc
    (0xAD, 0xAD), (0x600, 0x605), (0x61C, 0x61C), (0x6DD, 0x6DD), (0x70F, 0x70F),  # Cf
    (0x890, 0x891), (0x8E2, 0x8E2), (0x180E, 0x180E), (0x200B, 0x200F), (0x202A, 0x202E),
    (0x2060, 0x2064), (0x2066, 0x206F), (0xFEFF, 0xFEFF), (0xFFF9, 0xFFFB), (0x110BD, 0x110BD),
    (0x110CD, 0x110CD), (0x13430, 0x1343F), (0x1BCA0, 0x1BCA3), (0x1D173, 0x1D17A),
    (0xE0001, 0xE0001), (0xE0020, 0xE007F),
    (0x2028, 0x2029),  # Zl, Zp
    (0xD800, 0xDFFF),  # Cs
)  # fmt: skip
_HIDDEN = "".join(re.escape(chr(a)) + "-" + re.escape(chr(b)) for a, b in _HIDDEN_RANGES)
HIDDEN_RE = re.compile(f"[{_HIDDEN}]")
_SPACE_CONTROLS = str.maketrans({"\t": " ", "\x0b": " ", "\x0c": " ", "\r": "\n"})
_SPACES_RE = re.compile(r"\s+")

# A link keeps its scheme, host and path only.
_URL_RE = re.compile(r"(https?://)(?:[^\s/?#@]*@)?([^\s/?#]*)([^\s?#]*)(?:[?#]\S*)?", re.IGNORECASE)
_LINE_SECRET_RE = re.compile(
    r"\b(cookie|set-cookie|authorization|proxy-authorization)\s*[:=][^\n]*", re.IGNORECASE
)
_PAIR_SECRET_RE = re.compile(
    r"\b(x-api-key|api[_-]?key|(?:access|refresh|auth|id|session|csrf)[_-]?token|token|"
    r"(?:client[_-]?)?secret|password|passwd|pwd|session[_-]?id|csrftoken)"
    r"""\s*[:=]\s*("[^"\n]*"|'[^'\n]*'|[^\s;,&]+)""",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*")
_LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")


def _visible(raw: str) -> str:
    """`raw` read through a window, white-space controls as spaces, hidden characters gone."""
    window = raw[:_SCAN_CHARS].translate(_SPACE_CONTROLS)
    return HIDDEN_RE.sub("", window)


def redact(text: str) -> str:
    """`text` without credentials in links and without anything that looks like a secret."""
    text = _URL_RE.sub(lambda m: m.group(1) + m.group(2) + m.group(3), text)
    text = _LINE_SECRET_RE.sub(lambda m: f"{m.group(1)}: {REDACTED}", text)
    text = _PAIR_SECRET_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    text = _BEARER_RE.sub(f"Bearer {REDACTED}", text)
    text = _JWT_RE.sub(REDACTED, text)
    return _LONG_HEX_RE.sub(REDACTED, text)


def make_excerpt(title: str, text: str) -> str:
    """The excerpt of an item: its title, then its text, cleaned, redacted and bounded."""
    parts = [p for p in (_visible(title), _visible(text)) if p.strip()]
    joined = TITLE_SEPARATOR.join(p[:_WORK_CHARS] for p in parts)[:_WORK_CHARS]
    flat = _SPACES_RE.sub(" ", redact(joined)).strip()
    if len(flat) <= MAX_EXCERPT_CHARS:
        return flat
    return flat[: MAX_EXCERPT_CHARS - 1] + ELLIPSIS


def clean_name(name: str, fallback: str) -> str:
    """The name of a source as an alert keeps it: plain, one line, at most 100 characters; the
    source id stands in for a name with nothing visible."""
    flat = _SPACES_RE.sub(" ", _visible(name)).strip()[:MAX_SOURCE_NAME_CHARS].strip()
    return flat or fallback
