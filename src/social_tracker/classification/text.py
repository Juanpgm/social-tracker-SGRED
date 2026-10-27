"""How the classifier sees text. Pure, no I/O.

`fold` is the one normalization both the topic filter and the rules use: Unicode NFKC (full-width
letters and ligatures become plain ones), invisible format characters and invisible "blank
letters" removed (a zero-width space or a Hangul filler cannot hide a word), case folded, accents
dropped, Cyrillic and Greek lookalikes of Latin letters replaced by those letters (a confusables
skeleton) and everything that is not a letter or a digit turned into a separator. So `¡SÍSMO!`,
`sismo`, `ｓｉｓｍｏ` and `ѕіѕmо` (Cyrillic) are the same word. The tilde of `ñ` is an accent like
any other (`año` becomes `ano`): the keywords are folded the same way, so the conflation is
harmless for matching and keeps accent typos from defeating a match.

Words are whole tokens: `sismografo` is NOT `sismo`. A term that must also match its plural or a
derived form is listed in the configuration under each form.

Evasion by spelling is met by a second view, `decoded`: digits used for letters inside an
otherwise alphabetic word are read as letters (`t3mblor` is `temblor`) and a word spelled letter
by letter (`t.e.m.b.l.o.r`, `t e m b l o r`, up to 15 letters) is joined. The stages look at both
views. The evasion itself is reported in `obfuscation` (mixed scripts in one word, invisible
filler characters, letter-by-letter spelling, digits for letters): evidence that raises suspicion
and never decides a label alone.

A text longer than the bound is not simply cut at the end: the first half of the budget and the
last half are both read (a rumor hidden after a screenful of padding is still seen) and the text is
flagged `truncated`.
"""

import re
import unicodedata
from dataclasses import dataclass

_WORD_RE = re.compile(r"[^\W_]+")
_FORMAT = "Cf"
_MARK = "Mn"

# Characters that render as blank but are letters or symbols to Unicode: Hangul fillers (also what
# U+3164 and U+FFA0 turn into under NFKC), the Braille blank, the Khmer inherent vowels.
_BLANK_LETTERS = frozenset("ᅟᅠㅤﾠ⠀឴឵")

# Cyrillic and Greek letters that look like Latin ones (lowercase: applied after case folding).
_CONFUSABLES = {
    **dict(
        zip(
            "авсеіјкмнорѕтухѵԁһԛԝɡ",
            "abceijkmhopstyxvdhqwg",
            strict=True,
        )
    ),
    **dict(
        zip(
            "αβγεηικμνοπρςστυφχω",
            "abyenikmvonpcstuoxw",
            strict=True,
        )
    ),
}
_NON_LATIN_RANGES = ((0x0370, 0x03FF), (0x1F00, 0x1FFF), (0x0400, 0x052F))

_LEET = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"}
_MIN_SPELLED_RUN = 4
_MAX_SPELLED_RUN = 15
_LEET_MIN_LETTERS = 3


def _is_non_latin(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _NON_LATIN_RANGES)


def _normalized(text: str) -> str:
    """NFKC without format characters or blank letters: what a reader would see."""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in normalized if unicodedata.category(ch) != _FORMAT and ch not in _BLANK_LETTERS
    )


def fold(text: str) -> str:
    """`text` as lowercase accent-free words separated by single spaces. Idempotent."""
    decomposed = unicodedata.normalize("NFD", _normalized(text).casefold())
    plain = "".join(
        _CONFUSABLES.get(ch, ch) for ch in decomposed if unicodedata.category(ch) != _MARK
    )
    return " ".join(_WORD_RE.findall(plain))


def tokens(text: str) -> tuple[str, ...]:
    return tuple(fold(text).split())


# ---- the decoded view and the obfuscation report ----


def _leet(token: str) -> tuple[str, bool]:
    """`token` with isolated digits read as letters, and whether one sat between two letters."""
    if sum(ch.isalpha() for ch in token) < _LEET_MIN_LETTERS or token.isdigit():
        return token, False
    out: list[str] = []
    interior = False
    last = len(token) - 1
    for i, ch in enumerate(token):
        before = token[i - 1] if i else ""
        after = token[i + 1] if i < last else ""
        if (
            ch in _LEET
            and not before.isdigit()
            and not after.isdigit()
            and (before.isalpha() or after.isalpha())
        ):
            out.append(_LEET[ch])
            interior = interior or (before.isalpha() and after.isalpha())
        else:
            out.append(ch)
    return "".join(out), interior


def _decoded_tokens(words: tuple[str, ...]) -> tuple[tuple[str, ...], set[str]]:
    """The tokens with digits-for-letters decoded and spelled-out words joined, and the kinds of
    obfuscation seen (`leet`, `separators`)."""
    kinds: set[str] = set()
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= _MIN_SPELLED_RUN:
            kinds.add("separators")
            if len(run) <= _MAX_SPELLED_RUN:
                out.append("".join(run))
                run.clear()
                return
        out.extend(run)
        run.clear()

    for word in words:
        if len(word) == 1 and word.isalpha():
            run.append(word)
            continue
        flush()
        decoded, interior = _leet(word)
        if interior:
            kinds.add("leet")
        out.append(decoded)
    flush()
    return tuple(out), kinds


def _raw_kinds(raw: str) -> set[str]:
    """Obfuscation visible only in the raw text: mixed scripts and invisible filler."""
    kinds: set[str] = set()
    if any(ch in _BLANK_LETTERS for ch in unicodedata.normalize("NFKC", raw)):
        kinds.add("invisible")
    for word in re.findall(r"[^\W\d_]+", raw):
        foreign = [ch for ch in word if _is_non_latin(ch)]
        if not foreign:
            continue
        if len(foreign) < len(word):
            kinds.add("mixed_script")
        elif len(word) >= _MIN_SPELLED_RUN and all(ch.casefold() in _CONFUSABLES for ch in word):
            kinds.add("mixed_script")  # a whole word made of lookalikes
    return kinds


@dataclass(frozen=True)
class PreparedText:
    """An item's title and text ready for the stages: `raw` (NFKC, case and punctuation kept,
    for the rules that read them), `plain` (folded), its `tokens`, the `decoded` view (see the
    module doc) and the kinds of `obfuscation` seen."""

    raw: str
    plain: str
    tokens: tuple[str, ...]
    truncated: bool
    decoded: str = ""
    obfuscation: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.tokens

    @property
    def decoded_tokens(self) -> tuple[str, ...]:
        return tuple(self.decoded.split())

    @property
    def views(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """`(text, tokens)` of every distinct view the stages must look at."""
        if self.decoded and self.decoded != self.plain:
            return ((self.plain, self.tokens), (self.decoded, self.decoded_tokens))
        return ((self.plain, self.tokens),)


def window(joined: str, max_chars: int) -> tuple[str, bool]:
    """`joined` itself when it fits, else its first and last halves of the budget (one newline
    between them), so padding in the middle or at an end hides nothing at the ends."""
    if len(joined) <= max_chars:
        return joined, False
    head = max_chars // 2
    tail = max_chars - head
    return f"{joined[:head]}\n{joined[len(joined) - tail :]}", True


def prepare(title: str, text: str, *, max_chars: int) -> PreparedText:
    """Title and text joined and bounded to `max_chars` characters (see `window`; a character is
    never split), so a huge item costs a bounded amount of work. `truncated` is True when anything
    was left out."""
    joined = f"{title}\n{text}" if title else text
    scanned, truncated = window(joined, max_chars)
    raw = unicodedata.normalize("NFKC", scanned)
    plain = fold(raw)
    words = tuple(plain.split())
    decoded_words, kinds = _decoded_tokens(words)
    kinds |= _raw_kinds(scanned)
    return PreparedText(
        raw,
        plain,
        words,
        truncated,
        " ".join(decoded_words),
        tuple(sorted(kinds)),
    )
