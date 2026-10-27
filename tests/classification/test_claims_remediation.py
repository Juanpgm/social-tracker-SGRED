"""Remediation of the claim extractor after the adversarial review (R5-R11, R13, S1).

Each test names the finding it pins. Pure and deterministic, like the extractor.
"""

from datetime import UTC, datetime

import pytest

from social_tracker.classification.claims import (
    MAX_MAGNITUDES,
    MAX_RANGE_SPAN,
    TimeKind,
    extract_claims,
)

PUB = datetime(2026, 10, 27, 18, 0, tzinfo=UTC)  # 13:00 in Colombia
EVENING = datetime(2026, 10, 27, 23, 30, tzinfo=UTC)  # 18:30 in Colombia
AFTERNOON = datetime(2026, 10, 27, 20, 30, tzinfo=UTC)  # 15:30 in Colombia


def mags(text: str, published: datetime = PUB) -> list[tuple[float, float]]:
    return [(m.low, m.high) for m in extract_claims(text, published).magnitudes]


def when(text: str, published: datetime = PUB, **options):
    return extract_claims(text, published, **options).when


# ---- R5: the part of the day ("de la tarde") moves the hour ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sismo de 6.0 a las 3 de la tarde", datetime(2026, 10, 27, 20, 0, tzinfo=UTC)),
        ("sismo 6.0 a las 3:15 de la tarde", datetime(2026, 10, 27, 20, 15, tzinfo=UTC)),
        ("sismo 6.0 a las 3:15 de la mañana", datetime(2026, 10, 27, 8, 15, tzinfo=UTC)),
        ("sismo 6.0 a las 3 de la madrugada", datetime(2026, 10, 27, 8, 0, tzinfo=UTC)),
        ("sismo de 6.0 a las 8 de la noche", datetime(2026, 10, 27, 1, 0, tzinfo=UTC)),
        # 23:00 local is 04:00 UTC the next day, after the publication: it was last night
        ("sismo de 6.0 a las 11 de la noche", datetime(2026, 10, 27, 4, 0, tzinfo=UTC)),
        ("sismo de 6.0 a las 12 de la noche", datetime(2026, 10, 27, 5, 0, tzinfo=UTC)),
        ("sismo de 6.0 a las 12 de la tarde", datetime(2026, 10, 27, 17, 0, tzinfo=UTC)),
        ("sismo de 6.0 a las 12 de la madrugada", datetime(2026, 10, 27, 5, 0, tzinfo=UTC)),
        ("sismo de 6.0 a las 2 p. m.", datetime(2026, 10, 27, 19, 0, tzinfo=UTC)),
        ("sismo de 6.0 a las 2 p.m.", datetime(2026, 10, 27, 19, 0, tzinfo=UTC)),
        ("sismo de 6.0 a las 14h", datetime(2026, 10, 27, 19, 0, tzinfo=UTC)),
    ],
)
def test_the_part_of_the_day_decides_the_hour(text, expected):
    claim = when(text, EVENING)
    assert claim.kind is TimeKind.ABSOLUTE
    assert claim.at == expected


def test_twelve_in_the_morning_is_ambiguous_and_claims_no_time():
    assert when("sismo de 6.0 a las 12 de la mañana", EVENING).kind is TimeKind.NONE


def test_a_clock_still_keeps_its_meridiem_when_a_part_of_the_day_follows():
    claim = when("sismo 6.0 a las 3 pm de la tarde", EVENING)
    assert claim.at == datetime(2026, 10, 27, 20, 0, tzinfo=UTC)


# ---- R6: a 12-hour time without meridiem is not guessed ----


@pytest.mark.parametrize(
    "text",
    [
        "sismo 6.0 a las 3:15",
        "sismo 6.0 a las 3",
        "sismo 6.0 a las 12:10",
        "sismo 6.0 11:59",
        "sismo 6.0 a las 3h",
    ],
)
def test_an_ambiguous_twelve_hour_time_is_no_time_claim(text):
    assert when(text, AFTERNOON).kind is TimeKind.NONE


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sismo 6.0 a las 15:15", datetime(2026, 10, 27, 20, 15, tzinfo=UTC)),
        ("sismo 6.0 a las 08:15", datetime(2026, 10, 27, 13, 15, tzinfo=UTC)),
        ("sismo 6.0 a las 0:30", datetime(2026, 10, 27, 5, 30, tzinfo=UTC)),
        ("sismo 6.0 a las 3:15 UTC", datetime(2026, 10, 27, 3, 15, tzinfo=UTC)),
        ("sismo 6.0 a las 3:15 pm", datetime(2026, 10, 27, 20, 15, tzinfo=UTC)),
    ],
)
def test_an_unambiguous_clock_is_still_read(text, expected):
    claim = when(text, AFTERNOON)
    assert claim.kind is TimeKind.ABSOLUTE
    assert claim.at == expected


# ---- R7: "hace N dias" is not an exact instant ----


@pytest.mark.parametrize(
    "text",
    [
        "sismo hace 1 día",
        "sismo hace un día",
        "sismo hace 3 días",
        "sismo hace dos dias",
        "sismo hace un par de días",
        "sismo hace 400 dias",
    ],
)
def test_a_number_of_days_ago_is_historic(text):
    claim = when(text)
    assert claim.kind is TimeKind.HISTORIC
    assert claim.at is None


def test_hours_ago_is_still_relative():
    assert when("sismo hace 23 horas").kind is TimeKind.RELATIVE


# ---- R8: hypothetical, future and vague texts claim nothing ----


@pytest.mark.parametrize(
    "text",
    [
        "Simulacro de sismo de magnitud 7.5 en Cali",
        "El 40% cree que habrá un sismo de magnitud 8.5",
        "Podría ocurrir un terremoto de magnitud 8",
        "podrían ocurrir sismos de magnitud 7",
        "Mañana habrá un sismo de magnitud 7",
        "pasado mañana sismo M7.0",
        "Guía: qué hacer ante un sismo de magnitud 6",
        "Que hacer ante un sismo de magnitud 6",
        "Sismo de magnitud 6.0 esta madrugada",
        "Sismo de magnitud 6.0 este martes",
        "Sismo de magnitud 6.0 ese lunes",
        "sismo de magnitud 6.0 el lunes pasado",
        "Pronóstico: sismo de magnitud 7 en Cali",
    ],
)
def test_hypothetical_future_or_vague_text_claims_no_magnitude(text):
    assert mags(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "Sismo de magnitud 6.0 esta mañana",
        "Sismo de magnitud 6.0 a las 8 de la mañana",
        "sismo de magnitud 6.0 la mañana de hoy",
        "sismo de magnitud 6.0 hace un momento",
    ],
)
def test_a_morning_that_is_today_is_not_tomorrow(text):
    assert mags(text) == [(6.0, 6.0)]


# ---- R9: several magnitudes in one sentence ----


def test_a_second_magnitude_after_another_is_kept():
    assert mags("sismo de 5.5 y luego otro de 4.0") == [(5.5, 5.5), (4.0, 4.0)]


def test_two_magnitudes_joined_by_y_are_both_kept():
    assert mags("sismo de 4.5 y 5.5") == [(4.5, 4.5), (5.5, 5.5)]
    assert mags("sismo de 4.5 e 5.5") == [(4.5, 4.5), (5.5, 5.5)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sismo de magnitud 6.0 y 2 réplicas", 6.0),
        ("sismo de 6.2 y 5 muertos", 6.2),
        ("sismo de 6.2 y 5 réplicas", 6.2),
        ("sismo de 6.2 y 3 sismos más", 6.2),
        ("sismo de 6.2 y 5 pesos", 6.2),
        ("sismo de 6.2 y 10 veces", 6.2),
    ],
)
def test_a_count_after_y_is_not_a_second_magnitude(text, expected):
    assert mags(text) == [(expected, expected)]


def test_the_entre_form_is_still_one_range():
    assert mags("sismo de entre 4 y 5 grados Richter") == [(4.0, 5.0)]


def test_a_long_list_is_still_capped():
    text = (
        "sismo de 4.0 y 4.2, luego otro de 4.4, otro de 4.6, otro de 4.8, otro de 5.0, otro de 5.2"
    )
    found = mags(text)
    assert len(found) == MAX_MAGNITUDES
    assert found[:2] == [(4.0, 4.0), (4.2, 4.2)]


# ---- R10: square and cubic metres are not magnitudes ----


@pytest.mark.parametrize(
    "text",
    [
        "Sismo dejó casas de 50 m2 destruidas",
        "El sismo afectó un lote de 5 m2",
        "El sismo afectó un lote de 5 m²",
        "sismo dejó 120 m3 de escombros",
        "sismo en un lote de 5 m 2",
        "sismo de 5m2",
        "sismo de 5  m3",
    ],
)
def test_square_or_cubic_metres_are_not_a_magnitude(text):
    assert mags(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [("sismo M2.5 en Cali", 2.5), ("sismo m5.1 en Cali", 5.1), ("sismo de M6", 6.0)],
)
def test_a_real_m_magnitude_still_reads(text, expected):
    assert mags(text) == [(expected, expected)]


# ---- R11: the clock must belong to the earthquake sentence ----


def test_an_unrelated_clock_is_not_the_earthquake_time():
    text = "Alcaldía: sismo de 4.2 en Cali. Reunión a las 5 pm"
    claims = extract_claims(text, EVENING)
    assert claims.magnitudes[0].low == 4.2
    assert claims.when.kind is TimeKind.NONE


@pytest.mark.parametrize(
    "text",
    [
        "Sismo de 4.2 en Cali a las 5 pm",
        "Sismo de 4.2 en Cali. A las 5 pm se sintió fuerte",
        "Temblor en Cali. Ocurrió a las 5 pm",
        "A las 5 pm M4.2 en Cali",
    ],
)
def test_a_clock_in_the_earthquake_sentence_is_the_time(text):
    claim = when(text, EVENING)
    assert claim.kind is TimeKind.ABSOLUTE
    assert claim.at == datetime(2026, 10, 27, 22, 0, tzinfo=UTC)


def test_a_clock_in_a_text_with_nothing_to_bind_to_is_read_as_before():
    assert when("a las 11:30 am").kind is TimeKind.ABSOLUTE


def test_a_clock_on_another_line_is_not_bound():
    assert when("Sismo de 4.2 en Cali\nReunión a las 5 pm", EVENING).kind is TimeKind.NONE


# ---- R13: a wide range proves nothing ----


def test_the_range_span_limit_is_one_degree():
    assert MAX_RANGE_SPAN == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("magnitud 5.0-6.0", [(5.0, 6.0)]),
        ("magnitud entre 5.5 y 6.0", [(5.5, 6.0)]),
        ("magnitud 5.0-6.01", []),
        ("magnitud 4 a 7", []),
        ("magnitud entre 4 y 7", []),
        ("magnitud 3 a 8", []),
        ("magnitud 6.0-5.0", [(6.0, 6.0)]),
    ],
)
def test_a_range_wider_than_one_degree_is_dropped_whole(text, expected):
    assert mags(text) == expected


# ---- S1: extreme publication times are contained ----


@pytest.mark.parametrize(
    "published",
    [
        datetime(9999, 12, 31, 23, 59, tzinfo=UTC),
        datetime(1, 1, 1, 0, 30, tzinfo=UTC),
        datetime(1, 1, 1, 0, 0, tzinfo=UTC),
        datetime.max.replace(tzinfo=UTC),
        datetime.min.replace(tzinfo=UTC),
    ],
)
@pytest.mark.parametrize(
    "text",
    ["sismo M5 a las 3 pm", "sismo M5 ayer a las 3 pm", "sismo M5 hace 5 minutos", "sismo M5 hoy"],
)
def test_an_extreme_publication_time_never_raises(text, published):
    claims = extract_claims(text, published)
    assert claims.magnitudes[0].low == 5.0


# ---- the NFKC fold is part of the contract ----


def test_compatibility_forms_are_folded_before_reading():
    assert mags("ｍａｇｎｉｔｕｄ ７．５") == [(7.5, 7.5)]
    assert mags("magnitud 6.5 ﬁrmada") == [(6.5, 6.5)]


@pytest.mark.parametrize("digits", ["٧.٥", "७.५", "৭.৫", "๗.๕"])
def test_digits_of_other_scripts_are_not_folded_by_nfkc(digits):
    assert mags(f"magnitud {digits}") == []


# ---- found by the mutation sweep of the review ----


@pytest.mark.parametrize(
    "text", ["magnitud 1.2 a 2.2", "magnitud 7.3-8.3", "magnitud entre 1.2 y 2.2"]
)
def test_floating_point_noise_does_not_drop_a_range_of_exactly_one_degree(text):
    found = mags(text)
    assert len(found) == 1
    assert round(found[0][1] - found[0][0], 2) == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "sismo de 6.2 y 5 barrios afectados",
        "sismo de 6.2 y 5 en Cali",
        "sismo de 6.2 e 7 personas",
        "sismo de 6.2 y 8",
    ],
)
def test_a_whole_number_after_y_is_never_a_second_magnitude(text):
    assert mags(text) == [(6.2, 6.2)]


def test_a_decimal_comma_after_y_is_a_second_magnitude():
    assert mags("sismo de 4,5 y 5,5") == [(4.5, 4.5), (5.5, 5.5)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sismo 6.0 a las 13:00", datetime(2026, 10, 27, 18, 0, tzinfo=UTC)),
        ("sismo 6.0 a las 13:30", datetime(2026, 10, 27, 18, 30, tzinfo=UTC)),
        ("sismo 6.0 a las 0:00", datetime(2026, 10, 27, 5, 0, tzinfo=UTC)),
        ("sismo 6.0 a las 12:59 pm", datetime(2026, 10, 27, 17, 59, tzinfo=UTC)),
    ],
)
def test_the_first_unambiguous_24_hour_values_are_read_as_such(text, expected):
    claim = when(text, EVENING)
    assert (claim.kind, claim.at) == (TimeKind.ABSOLUTE, expected)


@pytest.mark.parametrize(
    "text",
    ["sismo 6.0 afectó a las 14 familias", "sismo 6.0 a las 15 personas", "sismo 6.0 a las 20"],
)
def test_a_bare_number_after_a_las_is_not_a_time(text):
    assert when(text, EVENING).kind is TimeKind.NONE


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sismo 6.0 a las 6 de la noche", datetime(2026, 10, 27, 23, 0, tzinfo=UTC)),
        ("sismo 6.0 a las 7 de la noche", datetime(2026, 10, 27, 0, 0, tzinfo=UTC)),
    ],
)
def test_the_evening_starts_at_six(text, expected):
    claim = when(text, EVENING)
    assert (claim.kind, claim.at) == (TimeKind.ABSOLUTE, expected)


@pytest.mark.parametrize("hour", [1, 2, 3, 4, 5])
def test_a_small_hour_of_the_night_is_not_said_and_claims_no_time(hour):
    assert when(f"sismo 6.0 a las {hour} de la noche", EVENING).kind is TimeKind.NONE


@pytest.mark.parametrize(
    ("text", "magnitude"), [("sismo M3 en Cali", 3.0), ("sismo M2 en Cali", 2.0)]
)
def test_a_whole_m2_or_m3_magnitude_is_still_a_magnitude(text, magnitude):
    assert mags(text) == [(magnitude, magnitude)]


def test_a_square_metre_after_a_number_with_blanks_is_still_not_a_magnitude():
    assert mags("sismo dejó 120  m2 y 5   m3 de daños") == []


@pytest.mark.parametrize("blanks", [" ", "  ", "   ", " " * 40, "\t", " \t \t"])
def test_square_metres_are_not_a_magnitude_whatever_the_blanks(blanks):
    assert mags(f"sismo dejó 120{blanks}m2 de daños") == []
    assert mags(f"sismo dejó 5{blanks}m3 de escombros") == []


# ---- hostile input stays bounded ----

HOSTILE = {
    "many sentences": "Sismo M5. " * 2000,
    "clock sentences": "sismo. a las 3 pm. " * 1000,
    "many clocks": "a las 3:15 pm " * 1400,
    "many lines": "sismo\na las 3 pm\n" * 1100,
    "many ayer": "ayer " * 4000,
    "blank run": "sismo M5" + " " * 19_000 + "m2",
    "m2 chain": "5 m2 " * 3500,
    "y chain": "sismo de 4.5 y 5.5 y 6.5 " * 700,
}


@pytest.mark.parametrize("text", HOSTILE.values(), ids=HOSTILE.keys())
def test_hostile_text_is_read_in_bounded_time(text):
    import time

    started = time.perf_counter()
    claims = extract_claims(text, EVENING)
    assert len(claims.magnitudes) <= MAX_MAGNITUDES
    assert time.perf_counter() - started < 2.0
