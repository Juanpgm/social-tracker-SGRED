"""Topic filter: configuration validation and matching (RC-1 stage 1, RC-2)."""

import pytest

from social_tracker.classification.domain import OutOfScopeReason, Topic
from social_tracker.classification.text import tokens
from social_tracker.classification.topics import TopicConfig, TopicMatcher
from social_tracker.shared.errors import ValidationError

BASE = {
    "version": 1,
    "earthquake": ["sismo", "sismos", "temblor", "movimiento telúrico", "escala Richter"],
    "secretariat": [
        "Secretaría de Gestión del Riesgo de Emergencias y Desastres",
        "Secretaría de Gestión del Riesgo",
        "DAGRD",
    ],
    "other_disaster": ["inundación", "volcán", "incendio forestal"],
    "places": ["Cali", "Valle del Cauca"],
    "exclude": ["robo", "asesinato", "orden público"],
}


def config(**changes) -> TopicConfig:
    return TopicConfig.from_mapping({**BASE, **changes})


def verdict(text: str, **changes):
    return TopicMatcher(config(**changes)).evaluate(tokens(text))


# ---- matching ----


@pytest.mark.parametrize(
    "text",
    [
        "Sismo",
        "SISMO!!",
        "un sísmo",
        "¡SÍSMO en la ciudad!",
        "sismo,temblor",
        "ＳＩＳＭＯ",
        "sis​mo",
    ],
)
def test_a_term_matches_ignoring_case_accents_punctuation_and_width(text):
    found = verdict(text)
    assert found.in_scope is True
    assert Topic.EARTHQUAKE in found.topics
    assert "sismo" in found.matched_terms


@pytest.mark.parametrize("text", ["sismografo", "sismologico", "presismo", "sismoo", "sis mo"])
def test_a_term_is_a_whole_word_never_a_substring(text):
    found = verdict(text)
    assert (found.in_scope, found.matched_terms) == (False, ())
    assert found.reason is OutOfScopeReason.NO_TOPIC_MATCH


def test_a_plural_is_matched_only_when_the_configuration_lists_it():
    assert verdict("hubo sismos").matched_terms == ("sismos",)
    assert verdict("hubo temblores").in_scope is False  # "temblores" is not listed


def test_a_phrase_matches_its_words_in_order_and_only_together():
    assert verdict("fue un movimiento telúrico").matched_terms == ("movimiento telurico",)
    assert verdict("escala RICHTER de 5").matched_terms == ("escala richter",)
    assert verdict("telúrico movimiento").in_scope is False
    assert verdict("un movimiento fuerte").in_scope is False


@pytest.mark.parametrize(
    "text",
    [
        "Secretaría de Gestión del Riesgo de Emergencias y Desastres de Cali",
        "secretaria de gestion del riesgo",
        "El DAGRD informa",
        "dagrd",
    ],
)
def test_the_secretariat_is_recognised_under_its_name_variants(text):
    found = verdict(text)
    assert Topic.SECRETARIAT in found.topics


@pytest.mark.parametrize(
    "text",
    [
        "Secretaría de Gestión Ambiental",
        "Secretaría de Salud de Cali",
        "gestión del riesgo financiero",
        "dagrdx anuncia",
        "secretaría del riesgo",
    ],
)
def test_look_alikes_of_the_secretariat_are_not_the_secretariat(text):
    found = verdict(text)
    assert Topic.SECRETARIAT not in found.topics


def test_topics_come_in_a_fixed_order_whatever_the_order_in_the_text():
    found = verdict("el DAGRD dice que el volcán y el sismo")
    assert found.topics == (Topic.EARTHQUAKE, Topic.SECRETARIAT, Topic.OTHER_DISASTER)
    assert found.matched_terms == ("dagrd", "sismo", "volcan")


def test_other_disasters_are_in_scope():
    found = verdict("incendio forestal cerca de Cali")
    assert (found.in_scope, found.topics) == (True, (Topic.OTHER_DISASTER,))


def test_a_place_alone_does_not_put_an_item_in_scope_but_adds_to_the_score():
    assert verdict("Cali amaneció soleada").in_scope is False
    with_place = verdict("sismo en Cali")
    without = verdict("sismo en otra parte")
    assert "cali" in with_place.matched_terms
    assert with_place.score > without.score


def test_score_grows_with_distinct_terms_and_is_capped():
    one = verdict("sismo").score
    two = verdict("sismo temblor").score
    three = verdict("sismo temblor volcán").score
    everything = verdict("sismo temblor volcán dagrd inundación Cali Valle del Cauca").score
    assert one == 0.5
    assert two == 0.65
    assert three == 0.8
    assert everything == 1.0
    assert verdict("sismo sismo sismo").score == one  # repeating a term adds nothing


def test_out_of_scope_items_carry_their_reason():
    assert verdict("hoy hay sol").reason is OutOfScopeReason.NO_TOPIC_MATCH
    assert verdict("").reason is OutOfScopeReason.EMPTY_TEXT
    assert verdict("hubo un robo en Cali").reason is OutOfScopeReason.CRIME_PUBLIC_ORDER
    assert verdict("hoy hay sol").score == 0.0


def test_a_disaster_term_beats_a_crime_term():
    found = verdict("robo durante el sismo")
    assert found.in_scope is True
    assert found.reason is None


def test_mixed_languages_and_emoji_do_not_get_in_the_way():
    assert verdict("BREAKING: sismo felt in Cali 🌋🔥").in_scope is True
    assert verdict("BREAKING: earthquake felt in Tokyo").in_scope is False


def test_matching_is_deterministic_and_independent_of_the_order_of_the_terms():
    shuffled = config(earthquake=list(reversed(BASE["earthquake"])))
    assert shuffled == config()
    text = tokens("sismo y temblor en Cali")
    assert TopicMatcher(shuffled).evaluate(text) == TopicMatcher(config()).evaluate(text)


def test_official_bodies_are_found_by_position_including_the_secretariat():
    cfg = config(official_bodies=["UNGRD", "Servicio Geológico Colombiano"])
    matcher = TopicMatcher(cfg)
    text = tokens("segun la UNGRD y el servicio geologico colombiano y el dagrd")
    assert matcher.official_bodies(text) == [(2, 3), (5, 8), (10, 11)]
    assert matcher.official_bodies(tokens("nadie oficial")) == []


# ---- configuration ----


def test_the_mapping_round_trips_and_is_canonical():
    cfg = config()
    assert TopicConfig.from_mapping(cfg.to_mapping()) == cfg
    assert cfg.to_mapping()["version"] == 1


def test_weights_are_validated_and_kept_sorted():
    cfg = config(weights={"prediction_claim": 0.6, "all_caps": 0})
    assert cfg.weights == (("all_caps", 0.0), ("prediction_claim", 0.6))
    assert cfg.to_mapping()["weights"] == {"all_caps": 0.0, "prediction_claim": 0.6}


@pytest.mark.parametrize(
    "bad",
    [
        {"unknown_key": []},
        {"version": 2},
        {"version": "1"},
        {"version": True},
        {"earthquake": "sismo"},  # a string, not a list
        {"earthquake": ["sismo", 5]},
        {"earthquake": ["sismo", "SÍSMO"]},  # duplicate once folded
        {"earthquake": ["sismo", ""]},
        {"earthquake": ["sismo", "  ¡! "]},  # no word left
        {"earthquake": ["sismo", "a\x00b"]},
        {"earthquake": ["sismo", "x" * 81]},
        {"earthquake": ["sismo", "a b c d e f g h i j k l m"]},  # thirteen words
        {"earthquake": [f"term{n}" for n in range(501)]},
        {"places": ["sismo"]},  # same term in two groups
        {"exclude": ["Temblor"]},
        {"weights": {"nonsense": 0.5}},
        {"weights": {"prediction_claim": 1.5}},
        {"weights": {"prediction_claim": -0.1}},
        {"weights": {"prediction_claim": True}},
        {"weights": {"prediction_claim": "0.5"}},
        {"weights": [0.5]},
        {"official_domains": ["https://example.org/x"]},
        {"official_domains": ["a b"]},
        {"official_domains": ["x.org", "X.ORG"]},
        {"official_channel_hints": ["@a", "@A"]},
        {"official_channel_hints": [" "]},
    ],
)
def test_an_invalid_configuration_is_refused(bad):
    with pytest.raises(ValidationError):
        config(**bad)


def test_a_missing_version_or_a_non_mapping_is_refused():
    with pytest.raises(ValidationError):
        TopicConfig.from_mapping({k: v for k, v in BASE.items() if k != "version"})
    for bad in (None, [], "text", 5):
        with pytest.raises(ValidationError):
            TopicConfig.from_mapping(bad)  # type: ignore[arg-type]


def test_a_configuration_with_no_topic_term_at_all_is_refused():
    with pytest.raises(ValidationError):
        TopicConfig.from_mapping({"version": 1, "places": ["Cali"]})
    assert TopicConfig.from_mapping({"version": 1, "secretariat": ["DAGRD"]}).secretariat == (
        "DAGRD",
    )


def test_the_limits_are_exact_at_the_boundary():
    assert len(config(earthquake=[f"term{n}" for n in range(500)]).earthquake) == 500
    assert config(earthquake=["x" * 80]).earthquake == ("x" * 80,)
    assert config(earthquake=["a b c d e f g h i j k l"]).earthquake == ("a b c d e f g h i j k l",)
