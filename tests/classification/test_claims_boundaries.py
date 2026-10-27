"""Exact edges of the claim extractor, each pinned from both sides (found by a mutation sweep)."""

from datetime import UTC, datetime

import pytest

from social_tracker.classification.claims import (
    MAX_PLACE_CHARS,
    MAX_TEXT_CHARS,
    TimeKind,
    extract_claims,
)
from social_tracker.shared.errors import ValidationError

PUB = datetime(2026, 10, 27, 18, 0, tzinfo=UTC)  # 13:00 in Colombia


def spans(text: str) -> list[tuple[float, float]]:
    return [(m.low, m.high) for m in extract_claims(text, PUB).magnitudes]


def when(text: str):
    return extract_claims(text, PUB).when


def test_a_range_of_exactly_three_degrees_is_a_range_and_a_hair_more_is_not():
    assert spans("magnitud 3 a 6") == [(3.0, 6.0)]
    assert spans("magnitud 3 a 6.1") == [(3.0, 3.0)]


def test_zero_and_ten_are_the_limits_of_a_magnitude():
    assert spans("magnitud 0") == [(0.0, 0.0)]
    assert spans("magnitud 10") == [(10.0, 10.0)]
    assert spans("magnitud 10.1") == []


def test_a_time_of_day_exactly_at_the_future_tolerance_stays_on_the_publication_day():
    assert when("a las 1:15 pm").at == datetime(2026, 10, 27, 18, 15, tzinfo=UTC)
    assert when("a las 1:16 pm").at == datetime(2026, 10, 26, 18, 16, tzinfo=UTC)
    assert when("a las 1:14 pm").at == datetime(2026, 10, 27, 18, 14, tzinfo=UTC)


@pytest.mark.parametrize(
    ("text", "hour"),
    [("a las 1:00 pm", 18), ("a las 12:00 pm", 17), ("a las 1:00 am", 6), ("a las 12:00 am", 5)],
)
def test_the_edges_of_the_twelve_hour_clock_are_read(text, hour):
    assert when(text).at.hour == hour


def test_the_edges_of_the_twenty_four_hour_clock_are_read():
    assert when("a las 23:59").at == datetime(2026, 10, 27, 4, 59, tzinfo=UTC)
    assert when("a las 0:00").at == datetime(2026, 10, 27, 5, 0, tzinfo=UTC)
    assert when("a las 23:60").kind is TimeKind.NONE
    assert when("a las 24:00").kind is TimeKind.NONE
    assert when("a las 12:59 pm").at is not None
    assert when("a las 12:60 pm").kind is TimeKind.NONE


def test_the_cap_of_the_text_read_is_exact():
    pad = "x" * (MAX_TEXT_CHARS - len("magnitud 6") - 1) + " "
    assert spans(pad + "magnitud 6") == [(6.0, 6.0)]
    assert spans(pad + "magnitud 6" + "7") == [(6.0, 6.0)]  # the "7" is past the cap
    assert spans(pad + " magnitud 6") == []  # "6" itself is


@pytest.mark.parametrize("offset", [-12, 14, 0, -5])
def test_the_whole_range_of_utc_offsets_is_accepted(offset):
    assert extract_claims("sismo", PUB, utc_offset_hours=offset).magnitudes == ()


@pytest.mark.parametrize("offset", [-13, 15])
def test_one_hour_beyond_the_range_of_offsets_is_refused(offset):
    with pytest.raises(ValidationError):
        extract_claims("sismo", PUB, utc_offset_hours=offset)


def test_a_long_place_is_cut_at_exactly_the_cap():
    word = "A" + "b" * 38
    place = extract_claims(f"sismo en {word} {word} {word} {word}", PUB).place
    assert len(place) == MAX_PLACE_CHARS
    assert place == " ".join([word.lower()] * 4)[:MAX_PLACE_CHARS]


def test_a_year_other_than_the_publication_year_is_historic_in_either_direction():
    assert when("sismo de 2019").kind is TimeKind.HISTORIC
    assert when("sismo de 2030").kind is TimeKind.HISTORIC
    assert when("sismo de 2026").kind is TimeKind.NONE


def test_the_cap_on_the_number_of_magnitudes_keeps_the_first_in_the_text():
    text = " ".join(f"M{4 + n / 10:.1f}" for n in range(9))
    expected = [float(f"{4 + n / 10:.1f}") for n in range(5)]
    assert spans(text) == [(v, v) for v in expected]
