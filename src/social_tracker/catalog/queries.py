"""The requests sent to the two catalogs. Built only from the settings and the clock, never from
anything a response said. Both are public FDSN event web services over https.

Each catalog gets two requests: the REGIONAL one (a circle around the center, from `min_magnitude`)
and the GLOBAL one (anywhere, from `global_min_magnitude`, for earthquakes felt elsewhere). The
refresher treats the two as one unit, so a half-answered catalog is never stored.
"""

from datetime import datetime, timedelta
from typing import Final
from urllib.parse import urlencode

from social_tracker.catalog.domain import CatalogSettings, CatalogSource
from social_tracker.shared.clock import ensure_utc

USGS_URL: Final = "https://earthquake.usgs.gov/fdsnws/event/1/query"
EMSC_URL: Final = "https://www.seismicportal.eu/fdsnws/event/1/query"
KM_PER_DEGREE: Final = 111.195
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def catalog_requests(
    source: CatalogSource, settings: CatalogSettings, now: datetime
) -> tuple[str, ...]:
    """The regional and the global request of `source` for the last `lookback_hours`."""
    end = ensure_utc(now)
    start = end - timedelta(hours=settings.lookback_hours)
    lat, lon = _number(settings.center_latitude), _number(settings.center_longitude)
    if CatalogSource(source) is CatalogSource.USGS:
        base = {
            "format": "geojson",
            "starttime": start.strftime(_TIME_FORMAT),
            "endtime": end.strftime(_TIME_FORMAT),
            "orderby": "time",
            "limit": str(settings.max_events),
        }
        regional = {
            **base,
            "latitude": lat,
            "longitude": lon,
            "maxradiuskm": _number(settings.radius_km),
            "minmagnitude": _number(settings.min_magnitude),
        }
        world = {**base, "minmagnitude": _number(settings.global_min_magnitude)}
        return tuple(f"{USGS_URL}?{urlencode(q)}" for q in (regional, world))
    base = {
        "format": "json",
        "start": start.strftime(_TIME_FORMAT),
        "end": end.strftime(_TIME_FORMAT),
        "orderby": "time",
        "limit": str(settings.max_events),
    }
    regional = {
        **base,
        "lat": lat,
        "lon": lon,
        "maxradius": _number(round(settings.radius_km / KM_PER_DEGREE, 3)),
        "minmag": _number(settings.min_magnitude),
    }
    world = {**base, "minmag": _number(settings.global_min_magnitude)}
    return tuple(f"{EMSC_URL}?{urlencode(q)}" for q in (regional, world))
