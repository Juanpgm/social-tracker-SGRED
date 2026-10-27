"""Parsers of the USGS and EMSC FDSN event answers (GeoJSON). The payload is untrusted data.

Both services are asked for `format=geojson` / `format=json`, so the only parser is the standard
`json` module (no XML, hence no entity expansion). A payload that is not what was asked for
(not UTF-8, not JSON, not a feature collection, nested too deep, too many features, JSON constants
such as `NaN`, or one in which no feature is readable) is a `CatalogFormatError`: the caller treats
the catalog as unavailable and writes nothing. An EMPTY body (the FDSN services answer `204 No
Content` when nothing matches) is a quiet day: no events. A single damaged feature (no magnitude,
impossible coordinates, an id with odd characters, an origin time in the future or too old, a
number or a time that overflows) is dropped and counted; so is an event that is not an earthquake
(a quarry blast, an explosion), counted apart as `not_earthquake`. An answer that reaches the
requested `limit` may have been cut short and is reported `truncated`. Text fields are bounded
and stripped of control and format characters; the place is data, bound as a parameter where it
is stored and escaped where it is shown.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from social_tracker.catalog.domain import CatalogEvent, CatalogSource
from social_tracker.classification.catalog_match import MAX_PLACE_LENGTH
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import DomainError, ValidationError

MAX_FEATURES: Final = 10_000
MAX_FUTURE: Final = timedelta(minutes=11)
MAX_AGE: Final = timedelta(days=60)
_FORMAT_MESSAGE = "the catalog answer is not a readable event list"


class CatalogFormatError(DomainError):
    """The answer of a catalog could not be read. Carries no part of the payload."""

    code = "catalog_format"

    def __init__(self) -> None:
        super().__init__(_FORMAT_MESSAGE)


@dataclass(frozen=True)
class ParseResult:
    """`discarded` counts every feature dropped, `not_earthquake` those dropped for not being an
    earthquake (a subset), `truncated` an answer that reached the requested limit."""

    events: tuple[CatalogEvent, ...]
    discarded: int
    not_earthquake: int = 0
    truncated: bool = False


def _reject_constant(_name: str) -> Any:
    raise ValueError("JSON constants are not numbers")


def _load(body: bytes) -> dict[str, Any]:
    try:
        text = body.decode("utf-8-sig")
        if not text.strip():
            return {"features": []}  # 204 No Content: nothing matched
        document = json.loads(text, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, RecursionError, TypeError, AttributeError):
        raise CatalogFormatError from None
    if not isinstance(document, dict) or not isinstance(document.get("features"), list):
        raise CatalogFormatError
    if len(document["features"]) > MAX_FEATURES:
        raise CatalogFormatError
    return document


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return float(value)
    except OverflowError:  # an integer of hundreds of digits
        return None


def _clean(value: object) -> str:
    """Printable text of at most MAX_PLACE_LENGTH characters; anything else is empty."""
    if not isinstance(value, str):
        return ""
    kept = (" " if ch.isspace() else ch for ch in value if ch.isspace() or ch.isprintable())
    return " ".join("".join(kept).split())[:MAX_PLACE_LENGTH]


def _identifier(feature: dict[str, Any], source: CatalogSource) -> str | None:
    raw = feature.get("id")
    if source is CatalogSource.EMSC:
        properties = feature.get("properties")
        if isinstance(properties, dict) and properties.get("unid") is not None:
            raw = properties["unid"]
    if isinstance(raw, bool) or not isinstance(raw, int | str):
        return None
    return str(raw)


def _usgs_time(value: object) -> datetime | None:
    millis = _number(value)
    if millis is None:
        return None
    try:
        return datetime.fromtimestamp(millis / 1000, UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _emsc_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.astimezone(UTC) if parsed.tzinfo is not None else None
    except (ValueError, OverflowError):  # "0001-01-01T00:00:00+14:00" has no UTC equivalent
        return None


def _coordinates(feature: dict[str, Any]) -> tuple[float, float] | None:
    geometry = feature.get("geometry")
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coordinates, list) or len(coordinates) < 2:  # noqa: PLR2004
        return None
    longitude, latitude = _number(coordinates[0]), _number(coordinates[1])
    if longitude is None or latitude is None:
        return None
    return latitude, longitude


_NOT_AN_EARTHQUAKE: Final = object()
_EMSC_EARTHQUAKES: Final = frozenset({"ke", "se", "eq", "earthquake"})


def _is_earthquake(properties: dict[str, Any], source: CatalogSource) -> bool | None:
    """True when the event is an earthquake or does not say what it is, False when it says it is
    something else (USGS `type`, EMSC `evtype`: a quarry blast, an explosion...), None when the
    field is not text."""
    key = "type" if source is CatalogSource.USGS else "evtype"
    kind = properties.get(key)
    if kind is None:
        return True
    if not isinstance(kind, str):
        return None
    kind = kind.strip().lower()
    return kind == "earthquake" if source is CatalogSource.USGS else kind in _EMSC_EARTHQUAKES


def _event(feature: Any, source: CatalogSource, now: datetime) -> CatalogEvent | object | None:
    if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict):
        return None
    properties = feature["properties"]
    earthquake = _is_earthquake(properties, source)
    if earthquake is None:
        return None
    if not earthquake:
        return _NOT_AN_EARTHQUAKE
    identifier = _identifier(feature, source)
    when = (_usgs_time if source is CatalogSource.USGS else _emsc_time)(properties.get("time"))
    magnitude = _number(properties.get("mag"))
    place_key = "place" if source is CatalogSource.USGS else "flynn_region"
    where = _coordinates(feature)
    if identifier is None or when is None or magnitude is None or where is None:
        return None
    if when > now + MAX_FUTURE or when < now - MAX_AGE:
        return None
    try:
        return CatalogEvent(
            source,
            identifier,
            when,
            where[0],
            where[1],
            magnitude,
            _clean(properties.get(place_key)),
        )
    except ValidationError:
        return None


def parse_catalog(
    source: CatalogSource, body: bytes, now: datetime, *, limit: int | None = None
) -> ParseResult:
    """The events of one answer. Raises `CatalogFormatError` for an unreadable answer; a feature
    that cannot be read is dropped (and counted) instead, and so is an event that is not an
    earthquake. When the answer has features and none is readable it is a format error too, so a
    changed format is never mistaken for a quiet day (an answer of only non-earthquakes is a
    quiet day: the format was understood). An empty body is an empty answer. With `limit`, an
    answer of that many features or more is reported `truncated`. The same id twice keeps the
    last one."""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("the limit must be a positive whole number")
    current = ensure_utc(now)
    features = _load(body)["features"]
    kept: dict[str, CatalogEvent] = {}
    discarded = not_earthquake = 0
    for feature in features:
        event = _event(feature, CatalogSource(source), current)
        if event is _NOT_AN_EARTHQUAKE:
            discarded += 1
            not_earthquake += 1
        elif event is None:
            discarded += 1
        else:
            kept[event.external_id] = event  # type: ignore[union-attr]
    if features and not kept and not not_earthquake:
        raise CatalogFormatError
    truncated = limit is not None and len(features) >= limit
    return ParseResult(tuple(kept.values()), discarded, not_earthquake, truncated)
