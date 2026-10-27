"""Helpers for the catalog tests: payloads as the two services send them and a scripted fetcher."""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from social_tracker.collector.domain import FetchRequest, FetchResponse

NOW = datetime(2026, 10, 27, 18, 0, tzinfo=UTC)
QUAKE = NOW - timedelta(hours=1)


def usgs_payload(*quakes: dict) -> bytes:
    """Each quake: id, mag, time (datetime), lat, lon, place (all optional)."""
    features = []
    for q in quakes:
        when = q.get("time", QUAKE)
        features.append(
            {
                "type": "Feature",
                "id": q.get("id", "us7000abcd"),
                "properties": {
                    "mag": q.get("mag", 5.1),
                    "place": q.get("place", "10 km N of Cali"),
                    "time": int(when.timestamp() * 1000),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [q.get("lon", -76.5), q.get("lat", 3.5), 10.0],
                },
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


def emsc_payload(*quakes: dict) -> bytes:
    features = []
    for q in quakes:
        when = q.get("time", QUAKE)
        lat, lon = q.get("lat", 3.5), q.get("lon", -76.5)
        features.append(
            {
                "type": "Feature",
                "id": q.get("id", "20261027_0000123"),
                "properties": {
                    "unid": q.get("id", "20261027_0000123"),
                    "time": when.strftime("%Y-%m-%dT%H:%M:%S.0Z"),
                    "mag": q.get("mag", 5.3),
                    "flynn_region": q.get("place", "COLOMBIA"),
                    "lat": lat,
                    "lon": lon,
                },
                "geometry": {"type": "Point", "coordinates": [lon, lat, -10.0]},
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


Outcome = bytes | Exception | Callable[[str], bytes]


class ScriptedFetcher:
    """Answers by catalog (`usgs` / `emsc`) and by kind of request (`regional` has a center,
    `global` has none). A rule is bytes, an exception to raise or a callable of the URL."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._rules: list[tuple[str, str | None, Outcome]] = []

    def on(self, source: str, outcome: Outcome, kind: str | None = None) -> "ScriptedFetcher":
        self._rules.insert(0, (source, kind, outcome))
        return self

    def fetch(self, request: FetchRequest) -> FetchResponse:
        url = request.url
        self.calls.append(url)
        source = "usgs" if "usgs.gov" in url else "emsc"
        kind = "regional" if ("latitude=" in url or "lat=" in url) else "global"
        for rule_source, rule_kind, outcome in self._rules:
            if rule_source == source and rule_kind in (None, kind):
                if isinstance(outcome, Exception):
                    raise outcome
                body = outcome(url) if callable(outcome) else outcome
                return FetchResponse(body)
        raise AssertionError(f"no rule for {url}")

    def calls_to(self, source: str) -> list[str]:
        marker = "usgs.gov" if source == "usgs" else "seismicportal"
        return [c for c in self.calls if marker in c]
