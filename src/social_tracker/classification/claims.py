"""What an item says about an earthquake: magnitude, time and place (RC-6). Pure, deterministic.

`extract_claims` reads Spanish item text and returns only numbers, an instant and a short place
name; it never keeps or echoes the text, and nothing in the text is interpreted. It exists so that
the catalog cross-check (`catalog_match`) can compare what an item claims with what an official
catalog recorded.

What it reads
- MAGNITUDE: `magnitud 7.5`, `M6.2`, `Mw 6.2`, `temblor de 5,8`, `6.1 grados Richter`, `7.0 Mw`
  (decimal point or comma; one or two decimals), ranges (`entre 5.5 y 6.0`, `5.5-6.0`) of at most
  `MAX_RANGE_SPAN` degrees (a wider one proves nothing and is dropped whole), several magnitudes
  in one text (`sismo de 4.5 y 5.5`, `sismo de 5.5 y luego otro de 4.0`; at most `MAX_MAGNITUDES`,
  each once, in order of appearance; the second number after a plain `y` needs a decimal, so
  `6.0 y 2 réplicas` is one magnitude). A number above 10, with a third digit, with a unit after
  it (`5 km`, `50 m2`, `3 de la tarde`) or in another script's digits is not a magnitude and is
  dropped, never guessed. A text that is hypothetical, about the future or too vague to place
  (`simulacro`, `habrá`, `podría`, `mañana`, `guía`, `qué hacer ante`, `esta madrugada`,
  `este lunes`) claims nothing at all.
- TIME, in this order of precedence. HISTORIC wins: a date other than the publication day, a year
  other than the publication year, or `hace N días/semanas/meses/años` mean the item talks about
  an old event and no instant is claimed. RELATIVE next: `hace un momento`, `hace 5 minutos`,
  `hace una hora` count back from the publication time (the first one in the text). ABSOLUTE last:
  a time of day (`a las 3:15 pm`, `a las 3 de la tarde`, `14:05 UTC`, `ayer a las 11:30 pm`) read
  as local time (the configurable offset, Colombia UTC-5 by default) unless it says UTC/GMT, on
  the publication day, or the day before when that would be later than the publication. A
  12-hour time with no `am`/`pm` and no part of the day (`a las 3:15`) is ambiguous and claims no
  time; so is `12 de la mañana`. A clock only counts when it sits in a sentence that talks about
  an earthquake (a keyword or a magnitude): `sismo de 4.2 en Cali. Reunión a las 5 pm` has no
  time. `ayer`/`anoche` with no time of day cannot be placed in a time window and count as
  HISTORIC.
- PLACE: the capitalised name after `en`, `cerca de`, `frente a`..., folded to plain lowercase
  letters (at most `MAX_PLACE_CHARS`), so it is safe to store and to show.

Only the first `MAX_TEXT_CHARS` characters are read. The text is first normalised with NFKC, which
FOLDS compatibility forms: fullwidth digits (`７.５` reads as 7.5), superscripts (`m²` becomes `m2`,
a unit, not a magnitude) and ligatures. NFKC does NOT fold the digits of other scripts
(Arabic-Indic, Devanagari, Thai...): they are not read, so they are no claim, like look-alike
letters. Zero-width and format characters are removed so that they cannot hide a number. An
extreme publication time (year 1 or 9999) never raises: the time is just not claimed.
"""

import math
import re
import unicodedata
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Final

from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError

MAX_TEXT_CHARS: Final = 20_000
MAX_MAGNITUDES: Final = 5
MAX_PLACE_CHARS: Final = 60
MAX_MAGNITUDE: Final = 10.0
MAX_RANGE_SPAN: Final = 1.0
DEFAULT_UTC_OFFSET_HOURS: Final = -5  # America/Bogota, which has no daylight saving time
FUTURE_TOLERANCE: Final = timedelta(minutes=15)
_FORMAT = "Cf"
_MARK = "Mn"


class TimeKind(StrEnum):
    NONE = "none"
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    HISTORIC = "historic"


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True)
class MagnitudeClaim:
    """A magnitude the text claims: one value (`low == high`) or a range."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if (
            not _is_number(self.low)
            or not _is_number(self.high)
            or not math.isfinite(self.low)
            or not math.isfinite(self.high)
            or not 0.0 <= self.low <= self.high <= MAX_MAGNITUDE
        ):
            raise ValidationError("a magnitude claim is a number or a range from 0 to 10")

    def distance(self, magnitude: float) -> float:
        """How far `magnitude` is from the claim: 0 inside the range."""
        if magnitude < self.low:
            return self.low - magnitude
        return max(0.0, magnitude - self.high)


@dataclass(frozen=True)
class TimeClaim:
    """When the text says the earthquake happened. `at` (UTC) is set for relative and absolute
    claims only."""

    kind: TimeKind = TimeKind.NONE
    at: datetime | None = None

    def __post_init__(self) -> None:
        has_instant = self.kind in (TimeKind.RELATIVE, TimeKind.ABSOLUTE)
        if has_instant != (self.at is not None):
            raise ValidationError("only a relative or absolute time claim has an instant")
        if self.at is not None:
            object.__setattr__(self, "at", ensure_utc(self.at))


@dataclass(frozen=True)
class Claims:
    magnitudes: tuple[MagnitudeClaim, ...] = ()
    when: TimeClaim = TimeClaim()
    place: str | None = None


# ---- normalization ----


def _stripped(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != _FORMAT)


_DOTTED_MERIDIEM = re.compile(r"\b([ap])\.\s?m\b")


def _plain(text: str) -> str:
    """Lowercase, without accents or format characters, `p. m.` written `pm`: the view the
    patterns read."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    bare = "".join(ch for ch in decomposed if unicodedata.category(ch) != _MARK)
    return _DOTTED_MERIDIEM.sub(r"\1m", bare)


# ---- magnitude ----

_NUM = r"([0-9]{1,2}(?:[.,][0-9]{1,2})?)(?![.,]?[0-9])"
_MONTHS = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre"
_NOT_UNIT = (
    r"(?!\s*(?:km\b|kilometros?\b|metros?\b|mts?\b|m\b|m[23]\b|segundos?\b|seg\b|segs\b|minutos?\b|"
    r"min\b|horas?\b|hrs?\b|hs\b|h\b|dias?\b|semanas?\b|meses\b|anos?\b|mil\b|muert|herid|persona|"
    r"victim|desaparec|casas?\b|edificios?\b|viviendas?\b|pisos?\b|%|:|am\b|pm\b|replic|"
    r"sismos\b|temblores\b|veces\b|pesos\b|dolares\b|euros\b|millones\b|"
    rf"de\s+(?:la|las|el|los|{_MONTHS})\b))"
)
_MAGNITUDE_PATTERNS = (
    re.compile(
        r"\b(?:magnitud|magnitude|mag)\b\.?[\s:=]*(?:(?:de|del|of)\s+)?(?:entre\s+)?"
        rf"(?:(?:aprox(?:imadamente)?|unos|casi)\.?\s+)?{_NUM}"
    ),
    # not "50 m2" (square metres): a digit and blanks before the m rule it out
    re.compile(
        r"(?<![a-z0-9])(?:(?<![0-9]\s)(?<![0-9]\s\s)|(?!m[23](?![.,]?[0-9])))"
        rf"m(?:(?=[0-9])|[wlbsd]\s?){_NUM}(?![a-z])"
    ),
    re.compile(
        rf"{_NUM}\s*(?:grados?\s*)?(?:(?:en\s+la\s+|en\s+)?escala\s+(?:de\s+)?|de\s+)?richter\b"
    ),
    re.compile(rf"{_NUM}\s*(?:mw|ml|mb|ms)\b"),
    re.compile(
        r"\b(?:temblor|sismo|terremoto|sacudon|replica|movimiento (?:telurico|sismico))\s+"
        r"(?:de\s+|con\s+)?(?:entre\s+)?(?:(?:una\s+)?magnitud\s+(?:de\s+)?)?"
        rf"{_NUM}{_NOT_UNIT}"
    ),
    re.compile(  # "... y luego otro de 4.0"
        r"\botr[oa]s?\s+(?:(?:sismo|temblor|terremoto|replica|sacudon)s?\s+)?(?:de|con)\s+"
        rf"(?:(?:una\s+)?magnitud\s+(?:de\s+)?)?{_NUM}{_NOT_UNIT}"
    ),
)
_RANGE_TAIL = re.compile(rf"\s*(?:(-|–|—)|\b(a|al|hasta|y|e)\b)\s*{_NUM}{_NOT_UNIT}")
_BEFORE_RANGE_CHARS = 14


def _value(text: str) -> float | None:
    number = float(text.replace(",", "."))
    return number if 0.0 <= number <= MAX_MAGNITUDE else None


def _magnitudes(plain: str) -> tuple[MagnitudeClaim, ...]:
    found: dict[int, tuple[int, str]] = {}
    for pattern in _MAGNITUDE_PATTERNS:
        for match in pattern.finditer(plain):
            found.setdefault(match.start(1), (match.end(1), match.group(1)))
    claims: list[MagnitudeClaim] = []
    consumed = 0
    for start in sorted(found):
        end, text = found[start]
        if start < consumed:
            continue
        low = _value(text)
        if low is None:
            continue
        high, listed = low, None
        tail = _RANGE_TAIL.match(plain, end)
        upper = _value(tail.group(3)) if tail is not None else None
        if tail is not None and upper is not None:
            if _is_range(plain, start, tail):
                if upper >= low:
                    consumed = tail.end(3)
                    if round(upper - low, 2) > MAX_RANGE_SPAN:
                        continue  # "magnitud 4 a 7" proves nothing: dropped whole
                    high = upper
            elif "." in tail.group(3) or "," in tail.group(3):  # "4.5 y 5.5": two magnitudes
                listed, consumed = upper, tail.end(3)
        found_here = [MagnitudeClaim(low, high)]
        if listed is not None:
            found_here.append(MagnitudeClaim(listed, listed))
        claims.extend(c for c in found_here if c not in claims)
        if len(claims) >= MAX_MAGNITUDES:
            break
    return tuple(claims[:MAX_MAGNITUDES])


def _is_range(plain: str, start: int, tail: re.Match[str]) -> bool:
    """A dash, `a`, `al` or `hasta` joins two numbers; `y` and `e` only after `entre`."""
    word = tail.group(2)
    if word not in ("y", "e"):
        return True
    return "entre" in plain[max(0, start - _BEFORE_RANGE_CHARS) : start]


# ---- time ----

_DATE = re.compile(rf"\b([0-9]{{1,2}})\s+de\s+({_MONTHS})\b(?:\s+(?:de|del)\s+([0-9]{{4}}))?")
_YEAR = re.compile(r"(?<![0-9])(19[0-9]{2}|20[0-9]{2})(?![0-9])")
_LONG_AGO = re.compile(
    r"\bhace\s+(?:(?:[0-9]{1,3}|un\s+par\s+de|un|una|unos|unas|dos|tres|cuatro|cinco|varios|"
    r"muchos|algunos)\s+)?(?:anos?|meses|mes|semanas?|dias?)\b"
)
_JUST_NOW = re.compile(
    r"\b(?:hace\s+(?:(?:un|unos|algunos)\s+)?(?:momentos?|instantes?|rato)|hace\s+poco|"
    r"ahora\s+mismo|en\s+este\s+momento|justo\s+ahora)\b"
)
_WORD_NUMBERS = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}
_AGO = re.compile(
    r"\bhace\s+(?:([0-9]{1,3})|(un|una|dos|tres|cuatro|cinco))\s+"
    r"(segundos?|segs?|minutos?|min|horas?|hrs?|hs|h)\b"
)
_AGO_HALF = re.compile(r"\bhace\s+media\s+hora\b")
_AGO_PAIR = re.compile(r"\bhace\s+un\s+par\s+de\s+(horas|minutos)\b")
_AGO_SOME_MINUTES = re.compile(r"\bhace\s+unos\s+minutos\b")
_SOME_MINUTES = timedelta(minutes=5)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_MERIDIEM = r"(?:\s*(?P<mer>am|pm)\b)?"
_HOUR_WORD = r"(?:\s*(?P<hs>h|hs|hrs|horas)\b)?"
_ZONE = r"(?:\s*(?P<utc>utc|gmt)\b)?"
# the space before "de la" belongs to _PART alone: a greedy `\s*` before it would eat the space
# and the part of the day would never match
_PART = r"(?:\s+de\s+la\s+(?P<part>manana|tarde|noche|madrugada)\b)?"
_CLOCKS = (
    re.compile(
        r"\ba\s+las?\s+(?P<h>[0-9]{1,2})(?::(?P<m>[0-9]{2}))?(?::[0-9]{2})?(?![0-9:])"
        rf"{_MERIDIEM}{_HOUR_WORD}{_ZONE}{_PART}"
    ),
    re.compile(
        r"(?<![0-9:.])(?P<h>[0-9]{1,2}):(?P<m>[0-9]{2})(?::[0-9]{2})?(?![0-9:])"
        rf"{_MERIDIEM}{_ZONE}{_PART}"
    ),
    re.compile(r"(?<![0-9:.])(?P<h>[0-9]{1,2})\s*(?P<mer>am|pm)\b" rf"{_ZONE}{_PART}"),
)
_DAY_BEFORE = re.compile(r"\b(?:ayer|anoche)\b")
# a sentence ends at . ! ? followed by a blank (but "3 pm." does not end one) or at a line break
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?<!\bam\.)(?<!\bpm\.)\s+|[\n\r;]+")
_QUAKE_WORD = re.compile(
    r"\b(?:sism|tembl|terremoto|sacud|replica|epicentro|magnitud|telurico|sinti|ocurri)"
)
# hypothetical, future or too vague to place in a window: such a text claims nothing
_WEEKDAY = "lunes|martes|miercoles|jueves|viernes|sabado|domingo"
_SPECULATIVE = re.compile(
    r"\b(?:simulacros?|habra|podrian?|pronostic\w*|predic\w*|guia|que\s+hacer|como\s+actuar|"
    r"en\s+caso\s+de|protocolo|esta\s+madrugada|"
    rf"(?:este|ese|esta|el\s+pasado)\s+(?:{_WEEKDAY})|(?:{_WEEKDAY})\s+pasado)\b"
    r"|(?<!\bla\s)(?<!\besta\s)\bmanana\b"
)


def _check_offset(hours: object) -> timedelta:
    if (
        isinstance(hours, bool) or not isinstance(hours, int) or not -12 <= hours <= 14  # noqa: PLR2004 - the range of the world's UTC offsets
    ):
        raise ValidationError("the local UTC offset must be a whole number of hours (-12..14)")
    return timedelta(hours=hours)


def _is_historic(plain: str, local_day: date) -> bool:
    if _LONG_AGO.search(plain):
        return True
    for match in _DATE.finditer(plain):
        year = int(match.group(3)) if match.group(3) else local_day.year
        month = _MONTHS.split("|").index(match.group(2)) + 1
        try:
            mentioned = date(year, month, int(match.group(1)))
        except ValueError:
            continue  # 31 de febrero: not a date
        if mentioned != local_day:
            return True
    return any(int(match.group(1)) != local_day.year for match in _YEAR.finditer(plain))


def _relative(plain: str, published: datetime) -> datetime | None:
    """The instant of the first relative expression in the text, counted back from publication."""
    found: list[tuple[int, timedelta]] = []
    for match in _JUST_NOW.finditer(plain):
        found.append((match.start(), timedelta(0)))
    for match in _AGO_HALF.finditer(plain):
        found.append((match.start(), timedelta(minutes=30)))
    for match in _AGO_SOME_MINUTES.finditer(plain):
        found.append((match.start(), _SOME_MINUTES))
    for match in _AGO_PAIR.finditer(plain):
        found.append((match.start(), 2 * timedelta(seconds=_unit_seconds(match.group(1)))))
    for match in _AGO.finditer(plain):
        count = int(match.group(1)) if match.group(1) else _WORD_NUMBERS[match.group(2)]
        found.append((match.start(), count * timedelta(seconds=_unit_seconds(match.group(3)))))
    if not found:
        return None
    _, ago = min(found, key=lambda pair: pair[0])
    try:
        return published - ago
    except OverflowError:
        return None


def _unit_seconds(unit: str) -> int:
    return _UNIT_SECONDS[unit[0]]


def _hour(match: re.Match[str]) -> tuple[int, int] | None:
    """The 24-hour clock a match says, or None when it is impossible or ambiguous."""
    groups = match.groupdict()
    hour_text = match.group("h")
    hour = int(hour_text)
    minute = int(groups.get("m") or 0)
    meridiem, part = groups.get("mer"), groups.get("part")
    if minute > 59:  # noqa: PLR2004 - minutes
        return None
    if meridiem:
        if not 1 <= hour <= 12:  # noqa: PLR2004 - a twelve-hour clock
            return None
        return (hour % 12) + (12 if meridiem == "pm" else 0), minute
    if part:
        shifted = _with_part(hour, part)
        return None if shifted is None else (shifted, minute)
    if hour > 23:  # noqa: PLR2004 - hours
        return None
    if groups.get("m") is None and groups.get("hs") is None:
        return None  # "a las 14 personas": a bare number is not a time
    twenty_four = hour == 0 or hour >= 13 or hour_text.startswith("0") or groups.get("utc")  # noqa: PLR2004
    return (hour, minute) if twenty_four else None  # "a las 3:15" could be either half of the day


def _with_part(hour: int, part: str) -> int | None:
    """`hour` of "de la mañana/tarde/noche/madrugada" on a 24-hour clock."""
    if not 1 <= hour <= 12:  # noqa: PLR2004 - a twelve-hour clock
        return None
    if part == "madrugada":
        return hour % 12
    if part == "manana":
        return None if hour == 12 else hour  # noqa: PLR2004 - "12 de la mañana" is noon or not
    if part == "tarde":
        return 12 if hour == 12 else hour + 12  # noqa: PLR2004
    if hour == 12:  # noqa: PLR2004 - "12 de la noche" is midnight
        return 0
    return hour + 12 if hour >= 6 else None  # noqa: PLR2004 - "3 de la noche" is not said


def _sentence_filter(plain: str) -> Callable[[int], bool]:
    """Says whether a position is in a sentence about an earthquake (a keyword or a magnitude).
    A text with no such sentence at all binds nothing: every position counts."""
    cuts = [0] + [m.end() for m in _SENTENCE_BREAK.finditer(plain)]
    anchored = []
    for index, begin in enumerate(cuts):
        end = cuts[index + 1] if index + 1 < len(cuts) else len(plain)
        part = plain[begin:end]
        anchored.append(
            bool(_QUAKE_WORD.search(part) or any(p.search(part) for p in _MAGNITUDE_PATTERNS))
        )
    if not any(anchored):
        return lambda _position: True
    return lambda position: anchored[bisect_right(cuts, position) - 1]


def _absolute(plain: str, published: datetime, offset: timedelta) -> datetime | None:
    candidates: list[tuple[int, re.Match[str]]] = []
    for pattern in _CLOCKS:
        candidates.extend((m.start(), m) for m in pattern.finditer(plain))
    if not candidates:
        return None
    in_quake_sentence = _sentence_filter(plain)
    days_back = 1 if _DAY_BEFORE.search(plain) else 0
    for position, match in sorted(candidates, key=lambda pair: pair[0]):
        clock = _hour(match)
        if clock is None or not in_quake_sentence(position):
            continue
        local_day = (published + offset).date() - timedelta(days=days_back)
        zone = timedelta(0) if match.group("utc") else offset
        at = datetime(local_day.year, local_day.month, local_day.day, *clock, tzinfo=UTC) - zone
        if not days_back and at > published + FUTURE_TOLERANCE:
            at -= timedelta(days=1)  # an earthquake cannot be reported before it happened
        return at
    return None


def _time(plain: str, published: datetime, offset: timedelta) -> TimeClaim:
    local_day = (published + offset).date()
    if _is_historic(plain, local_day):
        return TimeClaim(TimeKind.HISTORIC)
    instant = _relative(plain, published)
    if instant is not None:
        return TimeClaim(TimeKind.RELATIVE, instant)
    instant = _absolute(plain, published, offset)
    if instant is not None:
        return TimeClaim(TimeKind.ABSOLUTE, instant)
    if _DAY_BEFORE.search(plain):  # "ayer" without a time of day cannot be placed in a window
        return TimeClaim(TimeKind.HISTORIC)
    return TimeClaim()


# ---- place ----

_CAP = r"[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü]{0,39}"
_PLACE = re.compile(
    r"(?:\b[Ee]n|\bcerca de|\bfrente a|\bepicentro en|\bsacudi[oó] a)\s+"
    r"(?:(?:las?\s+)?costas?\s+de\s+)?"
    rf"({_CAP}(?:\s+(?:(?:de|del|la|las|los|el|y)\s+)?{_CAP}){{0,3}})"
)


def _place(text: str) -> str | None:
    match = _PLACE.search(text)
    if match is None:
        return None
    return _plain(match.group(1))[:MAX_PLACE_CHARS].strip() or None


# ---- the extractor ----


def extract_claims(
    text: str, published_at: datetime, *, utc_offset_hours: int = DEFAULT_UTC_OFFSET_HOURS
) -> Claims:
    """The magnitudes, the time and the place `text` claims for an earthquake. `published_at`
    anchors relative and absolute times; `utc_offset_hours` is the local time of the audience."""
    if not isinstance(text, str):
        raise ValidationError("the text must be a string")
    published = ensure_utc(published_at)
    offset = _check_offset(utc_offset_hours)
    read = _stripped(unicodedata.normalize("NFKC", text[:MAX_TEXT_CHARS]))
    plain = _plain(read)
    if _SPECULATIVE.search(plain):
        return Claims()
    try:
        when = _time(plain, published, offset)
    except OverflowError:  # a publication time at the edge of the calendar: no time is claimed
        when = TimeClaim()
    return Claims(magnitudes=_magnitudes(plain), when=when, place=_place(read))
