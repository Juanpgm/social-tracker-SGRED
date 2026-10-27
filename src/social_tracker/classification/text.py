"""How the classifier sees text. Pure, no I/O.

`fold` is the one normalization both the topic filter and the rules use: Unicode NFKC (full-width
letters and ligatures become plain ones), invisible format characters removed (a zero-width space
cannot hide a word), case folded, accents dropped and everything that is not a letter or a digit
turned into a separator. So `¡SÍSMO!`, `sismo` and `ｓｉｓｍｏ` are the same word. The tilde of
`ñ` is an accent like any other (`año` becomes `ano`): the keywords are folded the same way, so
the conflation is harmless for matching and keeps accent typos from defeating a match.

Words are whole tokens: `sismografo` is NOT `sismo`. A term that must also match its plural or a
derived form is listed in the configuration under each form.
"""

import re
import unicodedata
from dataclasses import dataclass

_WORD_RE = re.compile(r"[^\W_]+")
_FORMAT = "Cf"
_MARK = "Mn"


def fold(text: str) -> str:
    """`text` as lowercase accent-free words separated by single spaces. Idempotent."""
    normalized = unicodedata.normalize("NFKC", text)
    visible = "".join(ch for ch in normalized if unicodedata.category(ch) != _FORMAT)
    decomposed = unicodedata.normalize("NFD", visible.casefold())
    plain = "".join(ch for ch in decomposed if unicodedata.category(ch) != _MARK)
    return " ".join(_WORD_RE.findall(plain))


def tokens(text: str) -> tuple[str, ...]:
    return tuple(fold(text).split())


@dataclass(frozen=True)
class PreparedText:
    """An item's title and text ready for the stages: `raw` (NFKC, case and punctuation kept,
    for the rules that read them), `plain` (folded) and its `tokens`."""

    raw: str
    plain: str
    tokens: tuple[str, ...]
    truncated: bool

    @property
    def empty(self) -> bool:
        return not self.tokens


def prepare(title: str, text: str, *, max_chars: int) -> PreparedText:
    """Title and text joined and cut to `max_chars` characters (a character is never split), so a
    huge item costs a bounded amount of work. `truncated` is True when anything was cut."""
    joined = f"{title}\n{text}" if title else text
    truncated = len(joined) > max_chars
    raw = unicodedata.normalize("NFKC", joined[:max_chars])
    plain = fold(raw)
    return PreparedText(raw, plain, tuple(plain.split()), truncated)
