"""The claim extractor: magnitude, time and place from Spanish item text (RC-6).

Pure and deterministic. It reads numbers and times only; the text is untrusted data and nothing in
it is ever interpreted.
"""

import time
from datetime import UTC, datetime, timedelta

import pytest

from social_tracker.classification.claims import (
    MAX_MAGNITUDES,
    MAX_PLACE_CHARS,
    MAX_TEXT_CHARS,
    Claims,
    MagnitudeClaim,
    TimeKind,
    extract_claims,
)
from social_tracker.shared.errors import ValidationError

PUB = datetime(2026, 10, 27, 18, 0, tzinfo=UTC)  # 13:00 in Colombia (UTC-5), a Tuesday


def magnitudes(text: str) -> list[tuple[float, float]]:
    return [(m.low, m.high) for m in extract_claims(text, PUB).magnitudes]


# ---- magnitude: the shapes people write ----


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sismo de magnitud 7.5 hace un momento", 7.5),
        ("Sismo de magnitud 7,5 en el Pacífico", 7.5),
        ("MAGNITUD 6,5 EN LA COSTA", 6.5),
        ("magnitud de 4.8 según el reporte", 4.8),
        ("magnitud: 5.1", 5.1),
        ("mag. 5.3 registrada", 5.3),
        ("magnitude 6.2 reported", 6.2),
        ("Fuerte sismo M6.2 sacude la región", 6.2),
        ("Mw 6.2 frente a la costa", 6.2),
        ("ML 4.5 cerca de Cali", 4.5),
        ("mb 5.1 según el catálogo", 5.1),
        ("temblor de 5,8 sacudió la ciudad", 5.8),
        ("terremoto de 7 sacude el país", 7.0),
        ("sismo de 6.1 grados Richter", 6.1),
        ("sismo de 6,3 grados en la escala de Richter", 6.3),
        ("registraron 5.9 Richter en el occidente", 5.9),
        ("de 7.0 Mw según el reporte preliminar", 7.0),
        ("magnitud 10 es el máximo", 10.0),
        ("magnitud 0.5 imperceptible", 0.5),
    ],
)
def test_a_magnitude_is_read_in_every_usual_shape(text, expected):
    assert magnitudes(text) == [(expected, expected)]


@pytest.mark.parametrize(
    ("text", "low", "high"),
    [
        ("magnitud entre 5.5 y 6.0", 5.5, 6.0),
        ("magnitud 5.5-6.0 según fuentes", 5.5, 6.0),
        ("magnitud 5,5 a 6,0 en la escala", 5.5, 6.0),
        ("M5.5 – 6.0 preliminar", 5.5, 6.0),
        ("sismo de entre 4 y 5 grados Richter", 4.0, 5.0),
    ],
)
def test_a_range_is_one_claim_with_both_ends(text, low, high):
    assert magnitudes(text) == [(low, high)]


@pytest.mark.parametrize(
    "text",
    [
        "magnitud 6.2 a 50 km de la costa",  # the second number is a distance
        "magnitud 6.2 y 5 muertos",  # "y" is a range only after "entre"
        "magnitud 6.2 a 5 horas de Cali",
        "magnitud 6.2 a las 3:15 de la tarde",
    ],
)
def test_a_number_with_a_unit_after_the_magnitude_is_not_the_other_end_of_a_range(text):
    assert magnitudes(text) == [(6.2, 6.2)]


def test_a_range_wider_than_three_degrees_or_backwards_is_not_a_range():
    assert magnitudes("magnitud 3 a 8") == [(3.0, 3.0)]
    assert magnitudes("magnitud 6.0-5.0") == [(6.0, 6.0)]


def test_several_magnitudes_are_all_kept_in_order_of_appearance_without_repeats():
    text = "Preliminar M6.2, revisada a M5.8. Luego el SGC dijo magnitud de 5,8 y M6.2"
    assert magnitudes(text) == [(6.2, 6.2), (5.8, 5.8)]


def test_the_number_of_magnitudes_is_bounded():
    text = " ".join(f"M{3 + n / 10:.1f}" for n in range(40))
    assert len(extract_claims(text, PUB).magnitudes) == MAX_MAGNITUDES


# ---- magnitude: what is not one ----


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "el equipo ganó 3-1 el partido de ayer",
        "se sintió un temblor fuerte en Cali",
        "sismo de 3 de la madrugada",
        "sismo de 5 muertos",
        "temblor de 4 minutos",
        "temblor de 10 segundos",
        "sismo de 5.0 km de profundidad",
        "sismo de 12 de agosto",
        "terremoto de 5 pisos",
        "magnitud",  # the word without a number
        "magnitud -5",
        "magnitud grande",
    ],
)
def test_text_without_a_magnitude_has_none(text):
    assert magnitudes(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "sismo de magnitud 77",
        "sismo de magnitud 11",
        "sismo de magnitud 10.5",
        "sismo de magnitud 99.9",
        "magnitud 777",  # three digits: not a magnitude and not "77" followed by a stray 7
        "magnitud 5.000",  # a thousands separator
        "M6.2.1",  # a version number
        "magnitud 6.234",
    ],
)
def test_an_absurd_or_malformed_magnitude_is_dropped_not_guessed(text):
    assert magnitudes(text) == []


def test_a_valid_magnitude_next_to_an_absurd_one_survives():
    assert magnitudes("magnitud 77 corregida a magnitud 7.7") == [(7.7, 7.7)]


# ---- magnitude: hostile characters ----


def test_fullwidth_digits_are_read_after_normalization():
    assert magnitudes("magnitud ７.５") == [(7.5, 7.5)]


def test_zero_width_characters_inside_a_number_do_not_hide_it():
    assert magnitudes("magnitud 7​.5") == [(7.5, 7.5)]
    assert magnitudes("mag​nitud 6.1") == [(6.1, 6.1)]


@pytest.mark.parametrize(
    "text",
    [
        "magnitud ٧.٥",  # Arabic-Indic digits: not read, never crashes
        "magnitud १.५",  # Devanagari digits
        "mаgnitud 7.5",  # a Cyrillic "a" inside the word
        "magnitud ٣",  # a lone Arabic-Indic three
    ],
)
def test_digits_of_other_scripts_and_homoglyph_words_are_not_guessed(text):
    assert magnitudes(text) == []


def test_text_beyond_the_cap_is_not_read():
    far = "x" * (MAX_TEXT_CHARS + 10) + " magnitud 6.0"
    near = "magnitud 6.0 " + "x" * (MAX_TEXT_CHARS * 5)
    assert magnitudes(far) == []
    assert magnitudes(near) == [(6.0, 6.0)]


HOSTILE = {
    "repeated keyword": "magnitud " * 50_000,
    "only m": "M" * 100_000,
    "only digits": "1" * 100_000,
    "repeated hace": "hace " * 20_000,
    "repeated place": "En " + "Abc " * 20_000,
    "repeated clock": "a las " * 20_000,
    "repeated number": "5," * 50_000,
}


@pytest.mark.parametrize("text", HOSTILE.values(), ids=HOSTILE.keys())
def test_hostile_long_input_is_handled_in_bounded_time(text):
    started = time.perf_counter()
    claims = extract_claims(text, PUB)
    assert isinstance(claims, Claims)
    assert time.perf_counter() - started < 2.0


# ---- time ----


def when(text: str, published: datetime = PUB, **options):
    return extract_claims(text, published, **options).when


@pytest.mark.parametrize(
    "text",
    [
        "hace un momento",
        "hace un instante",
        "hace instantes",
        "hace poco",
        "ahora mismo",
        "en este momento",
        "justo ahora",
        "Sismo de magnitud 7.5 hace un momento",
    ],
)
def test_a_moment_ago_is_the_publication_time(text):
    claim = when(text)
    assert claim.kind is TimeKind.RELATIVE
    assert claim.at == PUB


@pytest.mark.parametrize(
    ("text", "ago"),
    [
        ("hace 5 minutos", timedelta(minutes=5)),
        ("hace 1 minuto", timedelta(minutes=1)),
        ("hace 90 minutos", timedelta(minutes=90)),
        ("hace 2 horas", timedelta(hours=2)),
        ("hace 30 seg", timedelta(seconds=30)),
        ("hace una hora", timedelta(hours=1)),
        ("hace media hora", timedelta(minutes=30)),
        ("hace dos horas", timedelta(hours=2)),
        ("hace un par de horas", timedelta(hours=2)),
        ("hace unos minutos", timedelta(minutes=5)),
        ("hace 3 días", timedelta(days=3)),
        ("hace un día", timedelta(days=1)),
    ],
)
def test_a_relative_time_counts_back_from_publication(text, ago):
    claim = when(text)
    assert claim.kind is TimeKind.RELATIVE
    assert claim.at == PUB - ago


@pytest.mark.parametrize(
    "text",
    ["hace 5 años", "hace 2 meses", "hace 3 semanas", "hace un año", "hace años", "en 1999 hubo"],
)
def test_a_long_ago_is_historic_and_has_no_instant(text):
    claim = when(text)
    assert claim.kind is TimeKind.HISTORIC
    assert claim.at is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a las 11:30 am", datetime(2026, 10, 27, 16, 30, tzinfo=UTC)),
        ("a las 11:30 a.m.", datetime(2026, 10, 27, 16, 30, tzinfo=UTC)),
        ("a las 12:05 pm", datetime(2026, 10, 27, 17, 5, tzinfo=UTC)),
        ("a las 12:05 am", datetime(2026, 10, 27, 5, 5, tzinfo=UTC)),
        ("a las 08:15", datetime(2026, 10, 27, 13, 15, tzinfo=UTC)),
        ("a las 8 de la mañana", datetime(2026, 10, 27, 13, 0, tzinfo=UTC)),
        ("a las 1 de la madrugada", datetime(2026, 10, 27, 6, 0, tzinfo=UTC)),
        ("a las 14:05 UTC", datetime(2026, 10, 27, 14, 5, tzinfo=UTC)),
        ("a las 14:05 GMT", datetime(2026, 10, 27, 14, 5, tzinfo=UTC)),
        ("ayer a las 11:30 pm", datetime(2026, 10, 27, 4, 30, tzinfo=UTC)),
        ("anoche a las 9:00 pm", datetime(2026, 10, 27, 2, 0, tzinfo=UTC)),
        ("a las 3:15 pm", datetime(2026, 10, 26, 20, 15, tzinfo=UTC)),  # later than now: yesterday
        ("temblor 11:59", datetime(2026, 10, 27, 16, 59, tzinfo=UTC)),
    ],
)
def test_a_time_of_day_is_local_unless_it_says_utc_and_never_in_the_future(text, expected):
    claim = when(text)
    assert claim.kind is TimeKind.ABSOLUTE
    assert claim.at == expected


@pytest.mark.parametrize(
    "text", ["a las 25:00", "a las 7:61", "a las 13 pm", "a las 0 pm", "99:99"]
)
def test_an_impossible_time_of_day_is_ignored(text):
    claim = when(text)
    assert claim.kind is TimeKind.NONE
    assert claim.at is None


def test_the_local_offset_is_configurable():
    utc = when("a las 11:30 am", utc_offset_hours=0)
    assert utc.at == datetime(2026, 10, 27, 11, 30, tzinfo=UTC)
    assert when("a las 11:30 am", utc_offset_hours=-5).at == datetime(
        2026, 10, 27, 16, 30, tzinfo=UTC
    )


@pytest.mark.parametrize("offset", [-13, 15, 99, 1.5, True, None])
def test_an_impossible_local_offset_is_refused(offset):
    with pytest.raises(ValidationError):
        extract_claims("sismo", PUB, utc_offset_hours=offset)


def test_a_date_other_than_the_publication_day_is_historic():
    assert when("el 12 de agosto de 2019 a las 3:00 pm").kind is TimeKind.HISTORIC
    assert when("sismo del 12 de agosto").kind is TimeKind.HISTORIC
    assert when("sismo del 26 de octubre").kind is TimeKind.HISTORIC


def test_the_date_of_the_publication_day_is_not_historic():
    claim = when("el 27 de octubre a las 11:30 am")
    assert claim.kind is TimeKind.ABSOLUTE
    assert claim.at == datetime(2026, 10, 27, 16, 30, tzinfo=UTC)
    assert when("sismo del 27 de octubre de 2026").kind is TimeKind.NONE
    assert when("sismo en 2026").kind is TimeKind.NONE


def test_an_impossible_date_is_ignored_not_historic():
    assert when("el 31 de febrero").kind is TimeKind.NONE


def test_a_relative_time_wins_over_a_time_of_day_and_history_wins_over_both():
    both = when("hace 5 minutos, a las 3:15 pm")
    assert both.kind is TimeKind.RELATIVE
    assert both.at == PUB - timedelta(minutes=5)
    assert when("hace un momento, como en 1999").kind is TimeKind.HISTORIC


def test_the_first_relative_time_in_the_text_is_the_one_used():
    claim = when("hace 10 minutos ... y hace 3 horas")
    assert claim.at == PUB - timedelta(minutes=10)


def test_no_time_is_none():
    claim = when("se sintió un temblor fuerte")
    assert claim.kind is TimeKind.NONE
    assert claim.at is None


# ---- place ----


@pytest.mark.parametrize(
    ("text", "place"),
    [
        ("Fuerte sismo en Cali esta tarde", "cali"),
        ("En Cali se sintió un temblor", "cali"),
        ("temblor cerca de Popayán", "popayan"),
        ("sismo con epicentro en Buenaventura", "buenaventura"),
        ("sismo frente a las costas de Chocó", "choco"),
        ("temblor en Santiago de Cali ayer", "santiago de cali"),
        ("sismo en el Valle del Cauca", None),  # lowercase article: no capitalised name follows
    ],
)
def test_the_place_is_the_capitalised_name_after_a_locating_word(text, place):
    assert extract_claims(text, PUB).place == place


def test_no_place_is_none():
    assert extract_claims("tembló fuerte", PUB).place is None
    assert extract_claims("en la ciudad", PUB).place is None


def test_a_place_is_short_plain_lowercase_letters_whatever_the_text_carries():
    hostile = [
        "sismo en Cali<script>alert(1)</script>",
        "sismo en Cali'; DROP TABLE catalog_events;--",
        "sismo en " + "Abcdefghij" * 30,
        "sismo en Ñandú Ünter Çñ",
        "sismo en Cali‮ evil",
    ]
    for text in hostile:
        place = extract_claims(text, PUB).place
        assert place is not None
        assert len(place) <= MAX_PLACE_CHARS
        assert all(ch == " " or "a" <= ch <= "z" for ch in place), place


# ---- shape and determinism ----


def test_the_same_input_always_gives_the_same_claims():
    text = "Sismo de magnitud 6,2 hace 5 minutos en Cali, M5.9 revisado"
    first = extract_claims(text, PUB)
    assert extract_claims(text, PUB) == first
    assert first == Claims(
        magnitudes=(MagnitudeClaim(6.2, 6.2), MagnitudeClaim(5.9, 5.9)),
        when=when(text),
        place="cali",
    )


def test_a_naive_publication_time_is_refused():
    with pytest.raises(ValidationError):
        extract_claims("sismo", datetime(2026, 10, 27, 18, 0))


def test_a_non_string_text_is_refused():
    with pytest.raises(ValidationError):
        extract_claims(None, PUB)  # type: ignore[arg-type]


def test_a_magnitude_claim_validates_its_numbers():
    with pytest.raises(ValidationError):
        MagnitudeClaim(float("nan"), 5.0)
    with pytest.raises(ValidationError):
        MagnitudeClaim(6.0, 5.0)
    with pytest.raises(ValidationError):
        MagnitudeClaim(-0.5, 5.0)
    with pytest.raises(ValidationError):
        MagnitudeClaim(5.0, 11.0)
    with pytest.raises(ValidationError):
        MagnitudeClaim(True, 5.0)  # type: ignore[arg-type]
