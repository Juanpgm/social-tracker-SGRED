"""Links in a text and whether they point at an official domain (S3). Pure, no I/O.

An item can claim to be official by quoting a link, so the link is parsed the way a browser reads
it, not searched for as a substring:

- A link starts at a real `http://`, `https://` or `www.` (not in the middle of a word, a path or
  a query string: `https://evil.example/?u=www.official.example` links to evil.example only).
- Its host is the authority up to the first slash, backslash, `?`, `#` or `:`. An authority with
  a login (`user@host`) is a SPOOF: nothing after the `@` is trusted, the link is ignored and
  reported.
- A host is ASCII letters, digits, dots and hyphens; anything else (a non-Latin lookalike) is
  ignored. Case and a trailing dot are irrelevant. Punycode (`xn--`) hosts are just other names.
- A host is official when it IS a configured domain or a subdomain of it, on a label boundary:
  `evil-official.example` and `official.example.evil.com` are not `official.example`.
- A bare mention of a domain with no link start is not a link.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_START = re.compile(
    r"(?<![\w@./:%=&?#+~-])(?:https?://|www\.)",
    re.IGNORECASE,
)
_AUTHORITY_END = re.compile(r"""[\s/\?#<>"'`,;()\[\]{}!|*^]""")
_HOST = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*")
MAX_LINKS = 50  # a text with more links than this is not read further for links


@dataclass(frozen=True)
class Links:
    """The hosts a text links to, and whether any link carried a login (a spoofing attempt)."""

    hosts: frozenset[str]
    spoofed: bool = False


def links_in(text: str) -> Links:
    hosts: set[str] = set()
    spoofed = False
    for count, start in enumerate(_START.finditer(text)):
        if count >= MAX_LINKS:
            break
        begin = start.start() if start.group().lower().startswith("www.") else start.end()
        rest = text[begin : begin + 300]
        end = _AUTHORITY_END.search(rest)
        authority = rest[: end.start()] if end else rest
        if "@" in authority:
            spoofed = True
            continue
        host = authority.split(":", 1)[0].lower().rstrip(".-")
        if _HOST.fullmatch(host):
            hosts.add(host)
    return Links(frozenset(hosts), spoofed)


def is_official_host(host: str, domains: Iterable[str]) -> bool:
    """True when `host` is one of `domains` or a subdomain of one (on a label boundary)."""
    return any(host == d or host.endswith("." + d) for d in domains)


def official_hosts(links: Links, domains: Sequence[str]) -> bool:
    return any(is_official_host(host, domains) for host in links.hosts)
