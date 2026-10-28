"""The excerpt an alert keeps of an item, and the snapshot of a source name (AL-11). Pure.

The text of an item is untrusted. What an alert keeps of it is:

- bounded: at most `MAX_EXCERPT_CHARS` characters, cut with an ellipsis when longer, and the
  work is bounded too (a hostile megabyte is read through a window, never in full);
- plain: control characters (NUL, escape sequences...), hidden format characters (zero width,
  right-to-left overrides, soft hyphen, BOM), line and paragraph separators and lone surrogates
  are removed; every other run of white space is one space;
- free of secrets: a link loses its credentials (any scheme), query and fragment (a query string
  is where tokens live) and any token-looking segment of its path; cookies, authorization headers,
  vendor keys (AWS, GitHub, OpenAI, Slack, Google, Stripe, SendGrid, JWT), `password`, `secret`,
  `api key`, `token` and `bearer` said in a sentence, and any long random-looking token are replaced
  by `[redacted]`. The rules are tuned against false positives: ordinary Spanish prose, numbers,
  magnitudes, dates, hashtags and plain links come out as they went in.

It is stored as text and is NOT HTML-escaped: the template that draws it escapes it.
"""

import math
import re
from collections import Counter

from social_tracker.alerts.domain import (
    MAX_EXCERPT_CHARS,
    MAX_RATIONALE_CHARS,
    MAX_SOURCE_NAME_CHARS,
)

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

# A link keeps its scheme, host and path only; the path loses what looks like a token.
_URL_RE = re.compile(r"(https?://)(?:[^\s/?#@]*@)?([^\s/?#]*)([^\s?#]*)(?:[?#]\S*)?", re.IGNORECASE)
# Credentials in the address of ANY other scheme (ftp, postgres, redis, amqp, ssh...).
_USERINFO_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9+.\-]*://)[^\s/?#@]+@")
_LINE_SECRET_RE = re.compile(
    r"\b(cookie|set-cookie|authorization|proxy-authorization)\s*[:=][^\n]*", re.IGNORECASE
)
_KEYWORDS = (
    r"x-api-key|api[ _-]?key|(?:access|refresh|auth|id|session|csrf)[_-]?token|token|"
    r"(?:client[_-]?)?secret|password|passwd|pwd|session[_-]?id|csrftoken"
)
_VALUE = r"""("[^"\n]*"|'[^'\n]*'|[^\s;,&]+)"""
# `password: x`, `pwd=x`: said as an assignment, whatever the value.
_PAIR_SECRET_RE = re.compile(rf"\b({_KEYWORDS})\s*[:=]\s*{_VALUE}", re.IGNORECASE)
# `my password is hunter2hunter2`, `api key abcd1234`: said in a sentence. The value must be at
# least 6 characters and look like a secret (a digit, a symbol or mixed case), so that "the
# password manager is down" and "the token expired" stay as they are.
_SPOKEN_SECRET_RE = re.compile(
    rf"\b({_KEYWORDS})\s+(?:(?:is|es)\s+)?(\"[^\"\n]*\"|'[^'\n]*'|[^\s;,&]{{6,}})", re.IGNORECASE
)
# `Bearer abc123`: at least 6 characters, with a digit or symbol in them or 10 or more of them (so
# that "bearer of bad news" and "bearer bonds" stay as they are).
_BEARER_RE = re.compile(r"\bbearer\s+([A-Za-z0-9._~+/=-]{6,})", re.IGNORECASE)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*")
_LONG_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
_VENDOR_RE = re.compile(
    "|".join(
        [
            r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",  # AWS access key ids
            r"\bgh[pousr]_[A-Za-z0-9]{20,}",  # GitHub tokens
            r"\bgithub_pat_[A-Za-z0-9_]{20,}",
            r"\bsk-[A-Za-z0-9_-]{16,}",  # OpenAI-style keys
            r"\bxox[abprs]-[A-Za-z0-9-]{10,}",  # Slack tokens
            r"\bAIza[0-9A-Za-z_-]{30,}",  # Google API keys
            r"\bya29\.[0-9A-Za-z_-]{20,}",  # Google OAuth tokens
            r"\b(?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{10,}",  # Stripe
            r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}",  # SendGrid
            r"\bhooks\.slack\.com/services/[A-Za-z0-9/_-]+",  # Slack webhooks
        ]
    )
)
# A long run of token characters; whether it is a secret is decided by `_random_token`.
# A long run of token characters, or a whole link (which is left to `_link`: its path is not
# a key just because it has capitals and digits).
_TOKENISH_RE = re.compile(
    r"(?P<link>[A-Za-z][A-Za-z0-9+.-]*://\S+)|(?P<run>(?<![\w#@])[A-Za-z0-9+/_=-]{24,}(?!\w))"
)
_CAMEL_WORDS_RE = re.compile(r"(?:[A-Z][a-z]+|[0-9]+)+")  # TerremotoEnCali2026: words, not a key
_HEX_RE = re.compile(r"[0-9a-fA-F]{16,}")
_MIN_PATH_TOKEN = 16
_MIN_RANDOM_TOKEN = 24
_MIN_ENTROPY = 3.8  # bits per character; random base64 is ~4.4, prose in CamelCase ~4.0


def _classes(token: str) -> int:
    return (
        any(c.islower() for c in token)
        + any(c.isupper() for c in token)
        + any(c.isdigit() for c in token)
    )


def _entropy(token: str) -> float:
    counts = Counter(token)
    total = len(token)
    return -sum(n / total * math.log2(n / total) for n in counts.values())


def _random_token(token: str) -> bool:
    """A long run of letters of both cases and digits that is not CamelCase words: a key."""
    return (
        len(token) >= _MIN_RANDOM_TOKEN
        and _classes(token) == 3
        and _CAMEL_WORDS_RE.fullmatch(token) is None
        and _entropy(token) >= _MIN_ENTROPY
    )


def _path_token(segment: str) -> bool:
    """A segment of a link path that is a secret rather than a name: 16 or more hex digits with
    at least a digit and a letter, or letters of both cases mixed with digits."""
    if len(segment) < _MIN_PATH_TOKEN:
        return False
    if _HEX_RE.fullmatch(segment):
        return any(c.isdigit() for c in segment) and any(c.isalpha() for c in segment)
    return (
        re.fullmatch(r"[A-Za-z0-9_-]+", segment) is not None
        and _classes(segment) == 3
        and _CAMEL_WORDS_RE.fullmatch(segment) is None
    )


def _run(match: re.Match[str]) -> str:
    run = match.group("run")
    return REDACTED if run is not None and _random_token(run) else match.group(0)


def _bearer(match: re.Match[str]) -> str:
    value = match.group(1)
    if len(value) >= 10 or any(c.isdigit() or c in "._~+/=-" for c in value):
        return f"Bearer {REDACTED}"
    return match.group(0)


def _link(match: re.Match[str]) -> str:
    path = "/".join(REDACTED if _path_token(s) else s for s in match.group(3).split("/"))
    return match.group(1) + match.group(2) + path


def _spoken(match: re.Match[str]) -> str:
    value = match.group(2)
    quoted = value[:1] in "\"'"
    secret = quoted or (
        any(c.isdigit() for c in value)
        or any(c in "_-+/=@!#$%^&*~" for c in value)
        or (any(c.isupper() for c in value) and any(c.islower() for c in value))
    )
    return f"{match.group(1)}={REDACTED}" if secret else match.group(0)


def _visible(raw: str) -> str:
    """`raw` read through a window, white-space controls as spaces, hidden characters gone."""
    window = raw[:_SCAN_CHARS].translate(_SPACE_CONTROLS)
    return HIDDEN_RE.sub("", window)


def redact(text: str) -> str:
    """`text` without credentials in links and without anything that looks like a secret."""
    text = _URL_RE.sub(_link, text)
    text = _USERINFO_RE.sub(r"\1", text)
    text = _LINE_SECRET_RE.sub(lambda m: f"{m.group(1)}: {REDACTED}", text)
    text = _PAIR_SECRET_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    text = _SPOKEN_SECRET_RE.sub(_spoken, text)
    text = _BEARER_RE.sub(_bearer, text)
    text = _VENDOR_RE.sub(REDACTED, text)
    text = _JWT_RE.sub(REDACTED, text)
    text = _LONG_HEX_RE.sub(REDACTED, text)
    return _TOKENISH_RE.sub(_run, text)


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


def clean_text(value: str, limit: int, fallback: str) -> str:
    """Text a MODEL (or a file it influenced) wrote, as an alert keeps it: plain, one line,
    redacted like an excerpt and at most `limit` characters (cut with an ellipsis). Redacting can
    make a text longer (a short secret becomes `[redacted]`), so the cut comes after it. The
    fallback stands in for a text with nothing left."""
    flat = _SPACES_RE.sub(" ", redact(_visible(value)[:_WORK_CHARS])).strip()
    if len(flat) > limit:
        flat = flat[: limit - 1].rstrip() + ELLIPSIS
    return flat or fallback


def make_rationale(value: str) -> str:
    """The rationale of a verdict as an alert keeps it (A10): the verdict may carry the words of
    a language model, which are untrusted like the item text they were written about."""
    return clean_text(value, MAX_RATIONALE_CHARS, "(no rationale)")
