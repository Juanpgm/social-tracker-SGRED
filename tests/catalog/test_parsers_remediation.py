"""The parsers after the adversarial review: an empty answer is a quiet day (R2), overflowing
numbers and times drop a feature instead of raising (S1), only earthquakes are kept, and a full
page is reported as cut short.
"""

import json
from datetime import UTC, datetime

import pytest

from social_tracker.catalog.domain import CatalogSource
from social_tracker.catalog.parsers import CatalogFormatError, parse_catalog
from tests.catalog.test_parsers import NOW, NOW_MS, body, emsc_feature, usgs_feature

USGS, EMSC = CatalogSource.USGS, CatalogSource.EMSC


# ---- R2: FDSN answers 204 with an empty body when nothing matches ----


@pytest.mark.parametrize(
    "payload", [b"", b"   ", b"\n", b"\r\n\t ", b"\xef\xbb\xbf", b"\xef\xbb\xbf\n"]
)
@pytest.mark.parametrize("source", [USGS, EMSC])
def test_an_empty_body_is_an_empty_answer(source, payload):
    result = parse_catalog(source, payload, NOW)
    assert (result.events, result.discarded, result.truncated) == ((), 0, False)


@pytest.mark.parametrize("payload", [b"null", b"{}", b"[]", b"0", b'""', b"{", b"\x00", b"\xff"])
@pytest.mark.parametrize("source", [USGS, EMSC])
def test_a_body_that_is_not_empty_and_not_a_feature_list_is_still_an_error(source, payload):
    with pytest.raises(CatalogFormatError):
        parse_catalog(source, payload, NOW)


def test_an_empty_feature_list_is_still_an_empty_answer():
    assert parse_catalog(USGS, body(), NOW).events == ()


# ---- S1: numbers and times that overflow drop the feature ----


def raw_usgs(mag: bytes, time: bytes, coordinates: bytes = b"[-76.5, 3.5, 10]") -> bytes:
    feature = (
        b'{"id": "bad", "properties": {"mag": %s, "time": %s}, "geometry": {"coordinates": %s}}'
    )
    good = json.dumps(usgs_feature(id="good")).encode()
    return b'{"features": [' + feature % (mag, time, coordinates) + b"," + good + b"]}"


@pytest.mark.parametrize(
    ("mag", "time", "coordinates"),
    [
        (b"9" * 400, b"%d" % NOW_MS, b"[-76.5, 3.5]"),
        (b"5.1", b"9" * 400, b"[-76.5, 3.5]"),
        (b"5.1", b"%d" % NOW_MS, b"[" + b"9" * 400 + b", 3.5]"),
        (b"5.1", b"%d" % NOW_MS, b"[-76.5, " + b"9" * 400 + b"]"),
        (b"-" + b"9" * 400, b"%d" % NOW_MS, b"[-76.5, 3.5]"),
        (b"1e999", b"%d" % NOW_MS, b"[-76.5, 3.5]"),
        (b"5.1", b"-1e999", b"[-76.5, 3.5]"),
        (b"5.1", b"1e308", b"[-76.5, 3.5]"),
        (b"5.1", b"-1e308", b"[-76.5, 3.5]"),
    ],
)
def test_a_feature_with_an_overflowing_number_is_dropped_and_counted(mag, time, coordinates):
    result = parse_catalog(USGS, raw_usgs(mag, time, coordinates), NOW)
    assert [e.external_id for e in result.events] == ["good"]
    assert result.discarded == 1


@pytest.mark.parametrize(
    "stamp",
    [
        "0001-01-01T00:00:00+14:00",
        "0001-01-01T00:00:00+00:00",
        "0001-01-01T00:00:00Z",
        "9999-12-31T23:59:59-14:00",
        "9999-12-31T23:59:59Z",
        "9999-12-31T23:59:59.999999-23:59",
    ],
)
def test_an_emsc_time_at_the_edge_of_the_calendar_is_dropped_and_counted(stamp):
    result = parse_catalog(
        EMSC, body(emsc_feature(unid="edge", time=stamp), emsc_feature(unid="good")), NOW
    )
    assert [e.external_id for e in result.events] == ["good"]
    assert result.discarded == 1


def test_an_answer_whose_only_feature_overflows_is_a_format_error_not_a_crash():
    only = (
        b'{"features": [{"id": "x", "properties": {"mag": %s, "time": %d}, '
        b'"geometry": {"coordinates": [1, 2]}}]}'
    )
    with pytest.raises(CatalogFormatError):
        parse_catalog(USGS, only % (b"9" * 400, NOW_MS), NOW)


# ---- only earthquakes ----


@pytest.mark.parametrize(
    "kind", ["quarry blast", "explosion", "ice quake", "other event", "mining explosion", "", "  "]
)
def test_a_usgs_event_that_is_not_an_earthquake_is_dropped_and_counted(kind):
    result = parse_catalog(
        USGS, body(usgs_feature(id="blast", type=kind), usgs_feature(id="q")), NOW
    )
    assert [e.external_id for e in result.events] == ["q"]
    assert (result.discarded, result.not_earthquake) == (1, 1)


@pytest.mark.parametrize("kind", ["earthquake", "Earthquake", " earthquake ", None])
def test_a_usgs_earthquake_or_an_event_with_no_type_is_kept(kind):
    props = {} if kind is None else {"type": kind}
    result = parse_catalog(USGS, body(usgs_feature(**props)), NOW)
    assert len(result.events) == 1
    assert result.not_earthquake == 0


def test_a_usgs_type_of_the_wrong_shape_is_a_malformed_feature_not_a_non_earthquake():
    result = parse_catalog(USGS, body(usgs_feature(id="x", type=5), usgs_feature(id="q")), NOW)
    assert [e.external_id for e in result.events] == ["q"]
    assert (result.discarded, result.not_earthquake) == (1, 0)


@pytest.mark.parametrize("kind", ["ex", "qb", "sm", "other", ""])
def test_an_emsc_event_that_is_not_an_earthquake_is_dropped_and_counted(kind):
    result = parse_catalog(
        EMSC, body(emsc_feature(unid="blast", evtype=kind), emsc_feature(unid="q")), NOW
    )
    assert [e.external_id for e in result.events] == ["q"]
    assert (result.discarded, result.not_earthquake) == (1, 1)


@pytest.mark.parametrize("kind", ["ke", "KE", "se", "earthquake", None])
def test_an_emsc_earthquake_or_an_event_with_no_type_is_kept(kind):
    props = {} if kind is None else {"evtype": kind}
    result = parse_catalog(EMSC, body(emsc_feature(**props)), NOW)
    assert len(result.events) == 1


def test_an_answer_of_only_non_earthquakes_is_a_quiet_day_not_a_format_error():
    result = parse_catalog(USGS, body(usgs_feature(type="quarry blast")), NOW)
    assert (result.events, result.discarded, result.not_earthquake) == ((), 1, 1)


def test_an_answer_of_unreadable_features_is_still_a_format_error():
    with pytest.raises(CatalogFormatError):
        parse_catalog(USGS, body(usgs_feature(mag=None)), NOW)


# ---- a page that reached the limit may be cut short ----


@pytest.mark.parametrize(
    ("count", "limit", "truncated"),
    [(2, 3, False), (3, 3, True), (4, 3, True), (0, 3, False), (1, 1, True), (0, 1, False)],
)
def test_an_answer_that_reaches_the_limit_is_reported_truncated(count, limit, truncated):
    features = [usgs_feature(id=f"us{n}") for n in range(count)]
    assert parse_catalog(USGS, body(*features), NOW, limit=limit).truncated is truncated


def test_without_a_limit_nothing_is_truncated():
    assert (
        parse_catalog(USGS, body(*[usgs_feature(id=f"us{n}") for n in range(50)]), NOW).truncated
        is False
    )


def test_the_limit_counts_the_features_of_the_answer_not_the_ones_kept():
    features = [usgs_feature(id=f"b{n}", type="quarry blast") for n in range(3)]
    assert parse_catalog(USGS, body(*features), NOW, limit=3).truncated is True


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "3"])
def test_an_invalid_limit_is_refused(limit):
    with pytest.raises(ValueError, match="limit"):
        parse_catalog(USGS, body(usgs_feature()), NOW, limit=limit)


def test_time_edges_do_not_depend_on_the_clock_zone():
    stamp = datetime(2026, 10, 27, 17, 0, tzinfo=UTC).isoformat()
    result = parse_catalog(EMSC, body(emsc_feature(time=stamp)), NOW)
    assert len(result.events) == 1
