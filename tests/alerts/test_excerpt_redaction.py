"""A9: what the excerpt of an item must not carry. Credentials of vendors, of any URL scheme, tokens
inside a link path, secrets said in a sentence and anything that is plainly a random token are
replaced by `[redacted]`; ordinary Spanish prose, numbers, magnitudes, dates, hashtags and plain
links come out untouched."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from social_tracker.alerts.excerpt import REDACTED, make_excerpt, redact

KEYS = {
    "aws": "AKI" + "AIOSFODNN7EXAMPLE",
    "aws_session": "ASI" + "AIOSFODNN7EXAMPLE",
    "gh_p": "ghp" + "_abcdefghijklmnopqrstuvwxyz0123456789",
    "gh_o": "gho" + "_abcdefghijklmnopqrstuvwxyz0123456789",
    "gh_u": "ghu" + "_abcdefghijklmnopqrstuvwxyz0123456789",
    "gh_s": "ghs" + "_abcdefghijklmnopqrstuvwxyz0123456789",
    "gh_pat": "git" + "hub_pat_11ABCDEFG0abcdefghijklmnop_1234567890abcdefghij",
    "openai": "sk-" + "abcdefghijklmnopqrstuvwxyz123456",
    "openai_proj": "sk-" + "proj-abcdefghijklmnopqrstuvwx",
    "slack_b": "xox" + "b-123456789012-1234567890123-AbCdEfGhIjKlMnOpQrStUvWx",
    "slack_a": "xox" + "a-123456789012-abcdefghijklmnop",
    "slack_p": "xox" + "p-123456789012-abcdefghijklmnop",
    "slack_r": "xox" + "r-123456789012-abcdefghijklmnop",
    "slack_s": "xox" + "s-123456789012-abcdefghijklmnop",
    "google": "AIz" + "aSyA1234567890abcdefghijklmnopqrstu",
    "google_plain": "AIza" + "abcdefghijklmnopqrstuvwxyzabcdefghij",  # two classes: no other rule
    "google_oauth": "ya2" + "9.a0AfH6SMBx1234567890abcdefghijklmnop",
    "jwt": ".".join(
        [
            "eyJ" + "hbGciOiJIUzI1NiJ9",
            "eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0",
            "dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1g",
        ]
    ),
    "stripe_live": "sk_" + "live_4eC39HqLyjWDarjtT1zdp7dc",
    "stripe_test": "rk_" + "test_4eC39HqLyjWDarjtT1zdp7dc",
    "sendgrid": "SG." + "abcdefghijklmnop1234.abcdefghijklmnopqrstuvwxyz",
}


@pytest.mark.parametrize("name", sorted(KEYS))
def test_a_vendor_key_is_redacted_wherever_it_is(name):
    key = KEYS[name]
    for text in (f"key {key} done", f"{key}", f"clave:{key}.", f'("{key}")'):
        out = make_excerpt("", text)
        assert key not in out
        assert REDACTED in out


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("see https://x.example/reset/9f8a7b6c5d4e3f2a1b0c?x=1", "9f8a7b6c5d4e3f2a1b0c"),
        ("https://x.example/verify/Kj8x2VqP9mN4tR7zA1bC/now", "Kj8x2VqP9mN4tR7zA1bC"),
        (
            "https://x.example/a/0123456789abcdef0123456789abcdef",
            "0123456789abcdef0123456789abcdef",
        ),
        (
            "https://hooks."
            + "slack.com/services/T000"
            + "00000/B00000000/XX"
            + "XXXXXXXXXXXXXXXXXXXXXX",
            "T00000000",
        ),
        ("https://user" + ":pw@host.example/p", "user:pw"),
        ("ftp://user" + ":pass@host/x", "user:pass"),
        ("postgres://u" + ":p@h/db", "u:p@"),
        ("mysql://root" + ":toor@db:3306/x", "root:toor"),
        ("redis://" + ":secretpass@cache:6379", "secretpass"),
        ("amqp://guest" + ":guest@broker.example/", "guest:guest"),
        ("ssh://deploy@host.example/repo", "deploy@"),
    ],
)
def test_a_secret_in_a_link_goes(text, secret):
    assert secret not in make_excerpt("", text)


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("my password is hunter2hunter2", "hunter2hunter2"),
        ("mi password es Hunter2Hunter2", "Hunter2Hunter2"),
        ("the passwd is s3cr3t-value", "s3cr3t-value"),
        ("pwd=abcdef1", "abcdef1"),
        ("api key: abcdef123456", "abcdef123456"),
        ("api_key=abc123def", "abc123def"),
        ("API KEY abcdef123456", "abcdef123456"),
        ("x-api-key abcd1234", "abcd1234"),
        ("x-api-key: abcd1234", "abcd1234"),
        ("client_secret abc123def456", "abc123def456"),
        ("the secret is s3cr3tv4lue", "s3cr3tv4lue"),
        ("token es abc123xyz", "abc123xyz"),
        ("token: abc", "abc"),
        ("sessionid: abcdef123", "abcdef123"),
        ("bearer abc123", "abc123"),
        ("Bearer\nabcdefghijklmnop", "abcdefghijklmnop"),
        ("Authorization Bearer abcdefghij", "abcdefghij"),
        ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("password: 'two words here'", "two words here"),
        ('secret = "a b c d"', "a b c d"),
    ],
)
def test_a_secret_said_in_a_sentence_goes(text, secret):
    assert secret not in make_excerpt("", text)


@pytest.mark.parametrize(
    "token",
    [
        "Zm9vYmFyOnNlY3JldHBhc3N3b3JkMTIz",
        "a1B2c3D4e5F6g7H8i9J0k1L2m3N4",
        "kJ8x2VqP9mN4tR7zA1bC3dE5",
        "XyZ123abcDEF456ghiJKL789mno",
    ],
)
def test_a_random_looking_token_goes_even_with_no_label(token):
    out = make_excerpt("", f"el codigo {token} sirve")
    assert token not in out
    assert out.startswith("el codigo")
    assert out.endswith("sirve")


@pytest.mark.parametrize(
    "text",
    [
        "Mañana habrá un temblor fuerte en Cali, compartan este mensaje urgente",
        "El secretario informó que la clave del éxito es la prevención",
        "El token de acceso del sistema de buses ya no funciona hoy",
        "the token expired yesterday and the password manager is down",
        "secret service agents arrived; the secret is out",
        "Sismo de magnitud 6.5 a las 14:32 con réplicas de 4.2 y 3.9",
        "coordenadas 3.4516,-76.5320 profundidad 10.2 km",
        "123456789012345678901234567890",
        "Terremoto 2026030112345678 registrado",
        "fecha 2026-03-01T12:00:00+00:00 hora local",
        "#TerremotoEnCaliUrgenteHoy2026 #SismoCali @GestionRiesgoCali",
        "TerremotoCali2026Informe ElTerremotoDeCali2026Magnitud",
        "https://www.eltiempo.com/colombia/cali/temblor-hoy-en-cali-magnitud-4-5-articulo-12345678",
        "https://www.cali.gov.co/gestiondelriesgo/publicaciones/2026/03/boletin",
        "electroencefalografistas anticonstitucionalmente",
        "contacto: prensa.gestionriesgo@cali.gov.co",
        "el 3 de marzo de 2026 a las 3 pm",
        "sk-learn y desk-top son palabras",
        "ask-me-anything about earthquakes",
        "ids 123e4567-e89b-12d3-a456-426614174000",
        "bearer of bad news",
    ],
)
def test_ordinary_text_comes_out_unchanged(text):
    assert make_excerpt("", text) == text


def test_a_plain_link_keeps_its_scheme_host_and_path():
    text = "https://www.cali.gov.co/gestiondelriesgo/boletin-123"
    assert make_excerpt("", text) == text


def test_a_link_loses_its_query_and_fragment_but_keeps_a_normal_path():
    out = make_excerpt("", "ver https://x.example/noticia/sismo-en-cali?utm=1&t=abc#frag fin")
    assert out == "ver https://x.example/noticia/sismo-en-cali fin"


def test_redact_is_idempotent():
    text = (
        "pwd=abcdef1 AKI"
        + "AIOSFODNN7EXAMPLE https://u"
        + ":p@h.example/x/9f8a7b6c5d4e3f2a1b0c?q=1"
    )
    once = redact(text)
    assert redact(once) == once


def test_redacting_a_hostile_megabyte_stays_fast():
    import time

    text = ("token " + "a" * 40 + " " + "A1" * 30 + " ") * 4000
    started = time.perf_counter()
    make_excerpt("", text)
    assert time.perf_counter() - started < 2.0


def test_nothing_in_a_redacted_text_leaks_the_secret_of_the_pair_rule():
    out = make_excerpt("", "login: pwd=Sup3rS3cret!!")
    assert "Sup3rS3cret" not in out
    assert out.startswith("login: pwd=" + REDACTED)


# ---- the edges of the rules ----


@pytest.mark.parametrize(
    ("path", "gone"),
    [
        ("0123456789abcdef", True),  # 16 hex digits with a digit and a letter
        ("0123456789abcde", False),  # 15
        ("aBcDeFgHiJkLm1nO", True),  # 16, three classes
        ("aBcDeFgHiJkLm1n", False),  # 15
        ("0123456789012345", False),  # 16 digits: a number, not a token
        ("abcdefabcdefabcd", False),  # 16 hex letters and no digit: a word
    ],
)
def test_a_path_segment_is_a_token_from_16_characters_with_digits_and_letters(path, gone):
    out = make_excerpt("", f"https://x.example/r/{path}/done")
    assert (path not in out) is gone


@pytest.mark.parametrize(
    ("value", "gone"),
    [("abc123", True), ("abc12", False), ("abcde1x", True), ("simple", False)],
)
def test_a_spoken_secret_needs_six_characters_that_look_like_one(value, gone):
    out = make_excerpt("", f"my password is {value} ok")
    assert (value not in out) is gone


def test_a_repetitive_run_of_three_classes_is_not_a_random_token():
    text = "Ab1" * 9
    assert make_excerpt("", text) == text


def test_a_random_run_of_exactly_24_characters_goes_and_one_of_23_stays():
    token = "kJ8x2VqP9mN4tR7zA1bC3dE5"
    assert len(token) == 24
    assert token not in make_excerpt("", token)
    assert make_excerpt("", token[:23]) == token[:23]


@pytest.mark.parametrize(
    "url",
    [
        "https://www.cali.gov.co/gestiondelriesgo/Publicaciones/Boletin2026Informe",
        "https://example.org/Noticias/Colombia/Cali/Terremoto2026EnCaliUrgenteHoy",
        "https://x.example/a/BcD/eF12/gH34/iJ56/kL78/mN90/oPqRsT",
        "ftp://files.example/Reportes/2026/Marzo/Boletin01Final",
    ],
)
def test_a_link_with_capitals_and_digits_in_its_path_is_not_taken_for_a_key(url):
    assert make_excerpt("", f"ver {url} fin") == f"ver {url} fin"


def test_a_secret_next_to_a_link_still_goes_and_the_link_stays():
    token = "kJ8x2VqP9mN4tR7zA1bC3dE5"
    out = make_excerpt("", f"https://x.example/Noticia2026/Cali {token} y mas")
    assert token not in out
    assert "https://x.example/Noticia2026/Cali" in out


def test_a_base64_key_with_slashes_and_plus_signs_goes_when_it_is_not_in_a_link():
    key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLE7Key"
    assert key not in make_excerpt("", f"secret aws {key} fin")


# ---- prose is left alone, whatever the order of its words ----

WORDS = [
    "mañana", "habrá", "un", "temblor", "fuerte", "en", "Cali", "compartan", "este", "mensaje",
    "URGENTE", "sismo", "de", "magnitud", "6.5", "a", "las", "14:32", "horas", "según", "la",
    "Secretaría", "de", "Gestión", "del", "Riesgo", "evacuación", "inmediata", "#TerremotoCali",
    "@GestionRiesgo", "2026-03-01", "https://www.cali.gov.co/gestiondelriesgo/Boletin2026",
    "electroencefalografistas", "contraseña", "clave", "token", "secret", "bearer", "password",
]  # fmt: skip


@settings(max_examples=300, deadline=None)
@given(st.lists(st.sampled_from(WORDS), min_size=1, max_size=18))
def test_ordinary_sentences_come_out_as_they_went_in_unless_a_secret_word_has_a_value(words):
    text = " ".join(words)
    secret_words = {"token", "secret", "bearer", "password"}
    risky = any(
        words[i] in secret_words and i + 1 < len(words) and len(words[i + 1]) >= 6
        for i in range(len(words))
    )
    if not risky:
        assert make_excerpt("", text) == text
