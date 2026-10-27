"""The vocabulary: inflections, ambiguous words that need context, English, and the validation of
official domains (C5, S7)."""

import pytest

from social_tracker.classification.domain import OutOfScopeReason, Topic
from social_tracker.classification.text import prepare
from social_tracker.classification.topics import (
    CONTEXT_WINDOW,
    PUBLIC_SUFFIXES,
    TopicConfig,
    TopicMatcher,
)
from social_tracker.classification.yaml_io import default_topic_config
from social_tracker.shared.errors import ValidationError
from tests.classification.support import default


def verdict(text: str):
    _, matcher = default()
    return matcher.evaluate_prepared(prepare("", text, max_chars=20_000))


# ---- inflected forms of real quake reports ----


@pytest.mark.parametrize(
    "text",
    [
        "Esta temblando muy fuerte en Cali ahora mismo",
        "Está temblando en el norte de la ciudad",
        "Están temblando los edificios del centro",
        "Estaba temblando cuando llegué a casa",
        "La tierra tiembla otra vez",
        "Tiemblan las ventanas de mi casa",
        "Todo temblaba en el piso diez",
        "Temblaron las lámparas",
        "Tembló fuerte hace un momento",
        "Algo sacudió la ciudad hace un minuto",
        "Nos sacudieron dos veces seguidas",
        "Sentimos una sacudida larga",
        "Se sintió un movimiento fuerte en Cali",
        "Qué estremecimiento hubo en el edificio",
        "Se estremeció todo el piso",
        "Hubo un remezón en el suroccidente",
    ],
)
def test_the_inflected_forms_of_a_real_quake_report_are_in_scope(text):
    result = verdict(text)
    assert result.in_scope is True
    assert Topic.EARTHQUAKE in result.topics


# ---- ambiguous words need context ----


@pytest.mark.parametrize(
    "text",
    [
        "El alcalde replica a las críticas",
        "La oposición replicó con dureza y las réplicas políticas siguen",
        "Réplicas del cuadro de Botero se venden en el centro de Cali",
        "La magnitud del problema de movilidad en Cali es enorme",
        "Magnitud de la crisis económica según el ministro",
    ],
)
def test_replica_and_magnitud_alone_are_not_a_quake(text):
    result = verdict(text)
    assert result.in_scope is False
    assert result.reason is OutOfScopeReason.NO_TOPIC_MATCH


@pytest.mark.parametrize(
    "text",
    [
        "Hubo una réplica del sismo de anoche",
        "Se esperan réplicas después del temblor",
        "Réplicas sísmicas en el Valle del Cauca",
        "El terremoto dejó una réplica de magnitud 4",
        "Sismo de magnitud 4.2 en Cali",
        "Magnitud 5 en el Pacífico",
        "La magnitud fue de 4 grados",
        "Magnitud: 3.9",
        "Con magnitud 6 se sintió fuerte",
    ],
)
def test_replica_and_magnitud_count_with_a_quake_term_or_a_number(text):
    assert verdict(text).in_scope is True


def test_the_context_window_is_bounded_on_both_sides():
    filler = " ".join(["palabra"] * CONTEXT_WINDOW)
    near = "el sismo " + " ".join(["palabra"] * (CONTEXT_WINDOW - 2)) + " replica"
    assert "replica" in verdict(near).matched_terms
    far = f"el sismo {filler} {filler} replica fuerte"
    assert verdict(far).matched_terms == ("sismo",)  # the distant "replica" is not counted
    before = f"replica {filler} {filler} el sismo"
    assert verdict(before).matched_terms == ("sismo",)


def test_a_number_must_follow_magnitud_closely():
    far = "la magnitud de la crisis de movilidad que vive Cali en 2026"
    assert verdict(far).in_scope is False


def test_a_custom_vocabulary_without_the_ambiguous_words_is_unaffected():
    config = TopicConfig.from_mapping({"version": 1, "earthquake": ["sismo"]})
    matcher = TopicMatcher(config)
    result = matcher.evaluate_prepared(prepare("", "el sismo y la magnitud 4", max_chars=100))
    assert result.matched_terms == ("sismo",)


# ---- English ----


@pytest.mark.parametrize(
    "text",
    [
        "The mayor said that the city will be ready for the weekend",
        "An earthquake of magnitude 5 hit the coast and people are afraid",
        "This is what we know about the storm that hit them",
    ],
)
def test_english_text_without_a_topic_is_counted_apart(text):
    result = verdict(text)
    assert (result.in_scope, result.reason) == (False, OutOfScopeReason.NON_SPANISH)


@pytest.mark.parametrize(
    "text",
    [
        "El equipo ganó el partido de ayer en el estadio de la ciudad",
        "Hubo una fiesta en la casa de mi abuela el sábado",
        "the",  # one English word is not a language
        "ok ok ok",
        "Cali Cali Cali",
    ],
)
def test_spanish_or_ambiguous_text_without_a_topic_is_no_topic_match(text):
    assert verdict(text).reason is OutOfScopeReason.NO_TOPIC_MATCH


def test_english_that_names_a_topic_term_is_in_scope_and_crime_wins_over_language():
    assert verdict("the temblor was strong and the people ran").in_scope is True
    crime = verdict("The robo at the bank was reported by the police and the people")
    assert crime.reason is OutOfScopeReason.CRIME_PUBLIC_ORDER


# ---- decoded view ----


@pytest.mark.parametrize(
    "text",
    [
        "ayer hubo un t3mblor en Cali",
        "ayer hubo un t e m b l o r en Cali",
        "ayer hubo un t.e.m.b.l.o.r en Cali",
        "ayer hubo un ѕіѕmо en Cali",
    ],
)
def test_a_word_spelled_around_the_vocabulary_still_puts_the_text_in_scope(text):
    assert verdict(text).in_scope is True


# ---- official domains ----


def config_with(domains) -> dict:
    return {**default_topic_config().to_mapping(), "official_domains": domains}


@pytest.mark.parametrize(
    "domain",
    ["cali.gov.co", "sgc.gov.co", "official-domain.example", "a.b.example", "xn--cal-9ma.example"],
)
def test_a_registrable_domain_is_accepted(domain):
    assert domain in TopicConfig.from_mapping(config_with([domain])).official_domains


@pytest.mark.parametrize(
    "domain",
    [
        "com",
        "gov",
        "co",
        "localhost",
        "gov.co",
        "com.co",
        "edu.co",
        "org.co",
        "net.co",
        "co.uk",
        "github.io",
        "COM.CO",  # case is ignored
        " gov.co ",
        "1.2.3.4",
        "10.0.0",
        ".gov.co",
        "gov.co.",
        "a..b.co",
        "-a.example.com",
        "a-.example.com",
        "exa mple.com",
        "ex@mple.com",
        "https://cali.gov.co",
        "cali.gov.co/path",
        "cali.gov.co:443",
        "",
        "a" * 64 + ".com",
        ("a." * 130) + "com",
    ],
)
def test_a_public_suffix_or_a_malformed_name_is_refused(domain):
    with pytest.raises(ValidationError, match="registrable"):
        TopicConfig.from_mapping(config_with([domain]))


def test_the_suffix_list_is_lowercase_and_has_the_colombian_ones():
    assert all(s == s.lower() for s in PUBLIC_SUFFIXES)
    assert {
        "com",
        "org",
        "net",
        "gov",
        "co",
        "com.co",
        "gov.co",
        "edu",
        "edu.co",
    } <= PUBLIC_SUFFIXES


def test_the_shipped_defaults_still_load():
    assert default_topic_config().official_domains == ("official-domain.example",)
