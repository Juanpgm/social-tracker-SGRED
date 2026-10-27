"""The USGS and EMSC GeoJSON parsers: the payload is untrusted, a bad one is an error and never a
crash, a bad event is dropped and counted, and nothing is half-read."""

import json
import time
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.catalog.domain import CatalogSource
from social_tracker.catalog.parsers import (
    MAX_FEATURES,
    CatalogFormatError,
    parse_catalog,
)

NOW = datetime(2026, 10, 27, 18, 0, tzinfo=UTC)
NOW_MS = int(NOW.timestamp() * 1000)
EPOCH_MS = int(datetime(2026, 10, 27, 17, 0, tzinfo=UTC).timestamp() * 1000)


def usgs_feature(id="us7000abcd", **props):
    return {
        "type": "Feature",
        "id": id,
        "properties": {"mag": 5.1, "place": "10 km N of Cali", "time": EPOCH_MS, **props},
        "geometry": {"type": "Point", "coordinates": [-76.5, 3.5, 10.0]},
    }


def emsc_feature(**props):
    properties = {
        "unid": "20261027_0000123",
        "time": "2026-10-27T17:00:00.0Z",
        "mag": 5.3,
        "flynn_region": "COLOMBIA",
        "lat": 3.5,
        "lon": -76.5,
        **props,
    }
    return {
        "type": "Feature",
        "id": properties["unid"],
        "properties": properties,
        "geometry": {"type": "Point", "coordinates": [properties["lon"], properties["lat"], -10.0]},
    }


def body(*features, extra=None) -> bytes:
    return json.dumps(
        {"type": "FeatureCollection", "features": list(features), **(extra or {})}
    ).encode()


def usgs(*features):
    return parse_catalog(CatalogSource.USGS, body(*features), NOW)


def emsc(*features):
    return parse_catalog(CatalogSource.EMSC, body(*features), NOW)


# ---- valid payloads ----


def test_a_usgs_feature_becomes_an_event():
    result = usgs(usgs_feature())
    (event,) = result.events
    assert result.discarded == 0
    assert (event.source, event.external_id) == (CatalogSource.USGS, "us7000abcd")
    assert event.origin_time == datetime(2026, 10, 27, 17, 0, tzinfo=UTC)
    assert (event.latitude, event.longitude, event.magnitude) == (3.5, -76.5, 5.1)
    assert event.place == "10 km N of Cali"


@pytest.mark.parametrize(
    "stamp",
    ["2026-10-27T17:00:00.0Z", "2026-10-27T17:00:00Z", "2026-10-27T17:00:00.000+00:00"],
)
def test_an_emsc_feature_becomes_an_event(stamp):
    (event,) = emsc(emsc_feature(time=stamp)).events
    assert (event.source, event.external_id) == (CatalogSource.EMSC, "20261027_0000123")
    assert event.origin_time == datetime(2026, 10, 27, 17, 0, tzinfo=UTC)
    assert (event.latitude, event.longitude, event.magnitude) == (3.5, -76.5, 5.3)
    assert event.place == "COLOMBIA"


def test_an_emsc_time_with_an_offset_is_converted_to_utc():
    (event,) = emsc(emsc_feature(time="2026-10-27T12:00:00-05:00")).events
    assert event.origin_time == datetime(2026, 10, 27, 17, 0, tzinfo=UTC)


def test_an_empty_feed_is_a_valid_answer_with_no_events():
    result = usgs()
    assert result.events == ()
    assert result.discarded == 0


def test_a_utf8_byte_order_mark_is_accepted():
    result = parse_catalog(CatalogSource.USGS, b"\xef\xbb\xbf" + body(usgs_feature()), NOW)
    assert len(result.events) == 1


def test_the_same_id_twice_keeps_the_last_one():
    result = usgs(usgs_feature(mag=5.0), usgs_feature(mag=5.4))
    assert [e.magnitude for e in result.events] == [5.4]


def test_the_boundaries_of_a_valid_event_are_accepted():
    edge = usgs_feature(mag=-2.0)
    edge["geometry"]["coordinates"] = [180.0, -90.0]
    other = usgs_feature(mag=10.0)
    other["id"] = "us2"
    other["geometry"]["coordinates"] = [-180.0, 90.0, 0]
    assert len(usgs(edge, other).events) == 2


# ---- bad events are dropped and counted ----


def damaged(**changes):
    feature = usgs_feature()
    feature["id"] = changes.pop("id", "us1")
    feature["properties"].update(changes.pop("props", {}))
    if "coordinates" in changes:
        feature["geometry"]["coordinates"] = changes["coordinates"]
    if "geometry" in changes:
        feature["geometry"] = changes["geometry"]
    return feature


@pytest.mark.parametrize(
    "feature",
    [
        damaged(props={"mag": None}),
        damaged(props={"mag": "5.1"}),
        damaged(props={"mag": True}),
        damaged(props={"mag": 11}),
        damaged(props={"mag": -3}),
        damaged(props={"time": None}),
        damaged(props={"time": "yesterday"}),
        damaged(props={"time": True}),
        damaged(props={"time": 10**20}),
        damaged(props={"time": -1}),
        damaged(props={"time": NOW_MS + 11 * 60_000 + 1_000}),  # in the future
        damaged(props={"time": NOW_MS - 61 * 86_400_000}),  # too old
        damaged(coordinates=[-76.5]),
        damaged(coordinates=["a", "b"]),
        damaged(coordinates=[-76.5, 91.0]),
        damaged(coordinates=[181.0, 3.5]),
        damaged(coordinates=None),
        damaged(geometry=None),
        damaged(geometry="point"),
        damaged(id=None),
        damaged(id="has space"),
        damaged(id="semi;colon"),
        damaged(id="x" * 65),
        damaged(id=["list"]),
    ],
)
def test_a_damaged_event_is_dropped_and_counted_while_good_ones_survive(feature):
    good = usgs_feature()
    good["id"] = "us-good"
    result = usgs(good, feature)
    assert [e.external_id for e in result.events] == ["us-good"]
    assert result.discarded == 1


def test_an_event_just_inside_the_time_tolerances_is_kept():
    soon = usgs_feature(time=NOW_MS + 10 * 60_000 + 59_000)
    soon["id"] = "us-soon"
    old = usgs_feature(time=NOW_MS - 59 * 86_400_000)
    old["id"] = "us-old"
    assert {e.external_id for e in usgs(soon, old).events} == {"us-soon", "us-old"}


def test_a_numeric_id_is_text():
    feature = usgs_feature()
    feature["id"] = 12345
    assert usgs(feature).events[0].external_id == "12345"


def test_a_payload_where_nothing_is_readable_is_a_format_error_not_an_empty_answer():
    with pytest.raises(CatalogFormatError):
        usgs(damaged(props={"mag": None}), damaged(id="us2", props={"mag": None}))
    with pytest.raises(CatalogFormatError):  # an EMSC document read as USGS
        parse_catalog(CatalogSource.USGS, body(emsc_feature()), NOW)


def test_a_hostile_place_is_cleaned_and_bounded_but_never_executed():
    nasty = "\x07bell\x00 nul <script>alert(1)</script> '; DROP TABLE catalog_events;--\n\tx‮"
    (event,) = usgs(usgs_feature(place=nasty)).events
    assert "\x00" not in event.place
    assert "\x07" not in event.place
    assert "‮" not in event.place
    assert "\n" not in event.place
    assert "<script>" in event.place  # data: escaped where it is shown, bound where it is stored
    long = usgs(usgs_feature(place="p" * 5000)).events[0]
    assert len(long.place) == 200


@pytest.mark.parametrize("place", [None, 5, ["a"], {"a": 1}, True])
def test_a_place_that_is_not_text_becomes_empty(place):
    assert usgs(usgs_feature(place=place)).events[0].place == ""


# ---- bad payloads are errors ----


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[1, 2, 3]",
        b'"text"',
        b"null",
        b"{}",
        b'{"features": null}',
        b'{"features": {"a": 1}}',
        b'{"features": [1, 2, 3]}',
        b'{"features": [{"mag": NaN}]}',
        b'{"features": [{"mag": Infinity}]}',
        b'{"type": "FeatureCollection", "features": [',  # truncated
        b"\xff\xfe\x00bad utf-8",
        b'{"features": []}\xc3\x28',
        b"<?xml version='1.0'?><quakeml/>",
        b"<html><body>503 Service Unavailable</body></html>",
    ],
)
def test_a_malformed_payload_is_a_format_error(payload):
    with pytest.raises(CatalogFormatError):
        parse_catalog(CatalogSource.USGS, payload, NOW)


def test_an_xml_entity_bomb_is_rejected_without_being_expanded():
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
        b'<!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
        b'<!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">]>'
        b"<lolz>&lol2;</lolz>"
    )
    started = time.perf_counter()
    for source in CatalogSource:
        with pytest.raises(CatalogFormatError):
            parse_catalog(source, bomb, NOW)
    assert time.perf_counter() - started < 1.0


def test_a_deeply_nested_document_is_a_format_error_not_a_crash():
    deep = b"[" * 200_000 + b"]" * 200_000
    with pytest.raises(CatalogFormatError):
        parse_catalog(CatalogSource.USGS, deep, NOW)


def test_too_many_features_is_a_format_error():
    many = {"features": [{}] * (MAX_FEATURES + 1)}
    with pytest.raises(CatalogFormatError):
        parse_catalog(CatalogSource.USGS, json.dumps(many).encode(), NOW)


def test_the_error_message_carries_no_payload_text():
    secret = b'{"features": "SECRET-TOKEN-123"}'
    with pytest.raises(CatalogFormatError) as caught:
        parse_catalog(CatalogSource.USGS, secret, NOW)
    assert "SECRET" not in caught.value.message
    assert "SECRET" not in repr(caught.value)


def test_a_naive_clock_is_refused():
    with pytest.raises(Exception, match="timezone"):
        parse_catalog(CatalogSource.USGS, body(), datetime(2026, 10, 27, 18, 0))


def test_the_same_payload_always_gives_the_same_events():
    payload = body(*(damaged(id=f"us{n}", props={"mag": 3 + n / 10}) for n in range(20)))
    assert parse_catalog(CatalogSource.USGS, payload, NOW) == parse_catalog(
        CatalogSource.USGS, payload, NOW + timedelta(seconds=1)
    )
