"""What an item says about an earthquake: magnitude, time and place (RC-6). Pure, deterministic.

`extract_claims` reads Spanish item text and returns only numbers, an instant and a short place
name; it never keeps or echoes the text, and nothing in the text is interpreted. It exists so that
the catalog cross-check (`catalog_match`) can compare what an item claims with what an official
catalog recorded.

What it reads
- MAGNITUDE: `magnitud 7.5`, `M6.2`, `Mw 6.2`, `temblor de 5,8`, `6.1 grados Richter`, `7.0 Mw`
  (decimal point or comma; one or two decimals), ranges (`entre 5.5 y 6.0`, `5.5-6.0`), several
  magnitudes in one text (at most `MAX_MAGNITUDES`, each once, in order of appearance). A number
  above 10, with a third digit or decimal, with a unit after it (`5 km`, `3 de la tarde`) or in
  another script's digits is not a magnitude and is dropped, never guessed.
- TIME, in this order of precedence. HISTORIC wins: a date other than the publication day, a year
  other than the publication year, or `hace N años/meses/semanas` mean the item talks about an
  old event and no instant is claimed. RELATIVE next: `hace un momento`, `hace 5 minutos`,
  `hace una hora` count back from the publication time (the first one in the text). ABSOLUTE last:
  a time of day (`a las 3:15 pm`, `14:05 UTC`, `ayer a las 11:30 pm`) read as local time (the
  configurable offset, Colombia UTC-5 by default) unless it says UTC/GMT, on the publication day,
  or the day before when that would be later than the publication.
- PLACE: the capitalised name after `en`, `cerca de`, `frente a`..., folded to plain lowercase
  letters (at most `MAX_PLACE_CHARS`), so it is safe to store and to show.

Only the first `MAX_TEXT_CHARS` characters are read. Digits of other scripts and look-alike
letters are not interpreted (a claim that cannot be read is no claim); zero-width and format
characters are removed first so that they cannot hide a number.
"""

import math
import re
import unicodedata
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
MAX_RANGE_SPAN: Final = 3.0
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


def _plain(text: str) -> str:
    """Lowercase, without accents or format characters: the view the patterns read."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != _MARK)


# ---- magnitude ----

_NUM = r"([0-9]{1,2}(?:[.,][0-9]{1,2})?)(?![.,]?[0-9])"
_MONTHS = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre"
_NOT_UNIT = (
    r"(?!\s*(?:km\b|kilometros?\b|metros?\b|mts?\b|m\b|segundos?\b|seg\b|segs\b|minutos?\b|min\b|"
    r"horas?\b|hrs?\b|hs\b|h\b|dias?\b|semanas?\b|meses\b|anos?\b|mil\b|muert|herid|persona|"
    r"victim|desaparec|casas?\b|edificios?\b|viviendas?\b|pisos?\b|%|:|am\b|pm\b|a\.\s?m|p\.\s?m|"
    rf"de\s+(?:la|las|el|los|{_MONTHS})\b))"
)
_MAGNITUDE_PATTERNS = (
    re.compile(
        r"\b(?:magnitud|magnitude|mag)\b\.?[\s:=]*(?:(?:de|del|of)\s+)?(?:entre\s+)?"
        rf"(?:(?:aprox(?:imadamente)?|unos|casi)\.?\s+)?{_NUM}"
    ),
    re.compile(rf"(?<![a-z0-9])m(?:(?=[0-9])|[wlbsd]\s?){_NUM}"),
    re.compile(
        rf"{_NUM}\s*(?:grados?\s*)?(?:(?:en\s+la\s+|en\s+)?escala\s+(?:de\s+)?|de\s+)?richter\b"
    ),
    re.compile(rf"{_NUM}\s*(?:mw|ml|mb|ms)\b"),
    re.compile(
        r"\b(?:temblor|sismo|terremoto|sacudon|replica|movimiento (?:telurico|sismico))\s+"
        r"(?:de\s+|con\s+)?(?:entre\s+)?(?:(?:una\s+)?magnitud\s+(?:de\s+)?)?"
        rf"{_NUM}{_NOT_UNIT}"
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
        high = low
        tail = _RANGE_TAIL.match(plain, end)
        if tail is not None and _is_range(plain, start, tail):
            upper = _value(tail.group(3))
            if upper is not None and low <= upper <= low + MAX_RANGE_SPAN:
                high = upper
                consumed = tail.end(3)
        claim = MagnitudeClaim(low, high)
        if claim not in claims:
            claims.append(claim)
        if len(claims) == MAX_MAGNITUDES:
            break
    return tuple(claims)


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
    r"\bhace\s+(?:(?:[0-9]{1,3}|un|una|unos|unas|dos|tres|cuatro|cinco|varios|muchos|algunos)\s+)?"
    r"(?:anos?|meses|mes|semanas?)\b"
)
_JUST_NOW = re.compile(
    r"\b(?:hace\s+(?:(?:un|unos|algunos)\s+)?(?:momentos?|instantes?|rato)|hace\s+poco|"
    r"ahora\s+mismo|en\s+este\s+momento|justo\s+ahora)\b"
)
_WORD_NUMBERS = {"un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}
_AGO = re.compile(
    r"\bhace\s+(?:([0-9]{1,3})|(un|una|dos|tres|cuatro|cinco))\s+"
    r"(segundos?|segs?|minutos?|min|horas?|hrs?|hs|h|dias?)\b"
)
_AGO_HALF = re.compile(r"\bhace\s+media\s+hora\b")
_AGO_PAIR = re.compile(r"\bhace\s+un\s+par\s+de\s+(horas|minutos|dias)\b")
_AGO_SOME_MINUTES = re.compile(r"\bhace\s+unos\s+minutos\b")
_SOME_MINUTES = timedelta(minutes=5)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_MERIDIEM = r"(?P<mer>a\.\s?m\.?|p\.\s?m\.?|am\b|pm\b)"
_ZONE = r"(?:\s*(?P<utc>utc|gmt)\b)?"
_PART = r"(?:\s+de\s+la\s+(?P<part>manana|tarde|noche|madrugada))?"
_CLOCKS = (
    re.compile(
        rf"\ba\s+las?\s+(?P<h>[0-9]{{1,2}})(?::(?P<m>[0-9]{{2}}))?(?::[0-9]{{2}})?\s*"
        rf"{_MERIDIEM}?\s*(?:h\b|hs\b|hrs\b|horas\b)?{_ZONE}{_PART}"
    ),
    re.compile(
        rf"(?<![0-9:.])(?P<h>[0-9]{{1,2}}):(?P<m>[0-9]{{2}})(?::[0-9]{{2}})?(?![0-9])\s*"
        rf"{_MERIDIEM}?{_ZONE}{_PART}"
    ),
    re.compile(rf"(?<![0-9:.])(?P<h>[0-9]{{1,2}})\s*{_MERIDIEM}{_ZONE}{_PART}"),
)
_DAY_BEFORE = re.compile(r"\b(?:ayer|anoche)\b")


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
    hour = int(match.group("h"))
    minute = int(match.groupdict().get("m") or 0)
    meridiem = (match.group("mer") or "").replace(".", "").replace(" ", "")
    part = match.group("part")
    if minute > 59:  # noqa: PLR2004 - minutes
        return None
    if meridiem:
        if not 1 <= hour <= 12:  # noqa: PLR2004 - a twelve-hour clock
            return None
        return (hour % 12) + (12 if meridiem == "pm" else 0), minute
    if hour > 23:  # noqa: PLR2004 - hours
        return None
    if part in ("tarde", "noche") and 1 <= hour <= 11:  # noqa: PLR2004
        hour += 12
    elif part in ("noche", "madrugada", "manana") and hour == 12:  # noqa: PLR2004
        hour = 0
    return hour, minute


def _absolute(plain: str, published: datetime, offset: timedelta) -> datetime | None:
    candidates: list[tuple[int, re.Match[str]]] = []
    for pattern in _CLOCKS:
        candidates.extend((m.start(), m) for m in pattern.finditer(plain))
    days_back = 1 if _DAY_BEFORE.search(plain) else 0
    for _, match in sorted(candidates, key=lambda pair: pair[0]):
        clock = _hour(match)
        if clock is None:
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
    return Claims(
        magnitudes=_magnitudes(plain),
        when=_time(plain, published, offset),
        place=_place(read),
    )
