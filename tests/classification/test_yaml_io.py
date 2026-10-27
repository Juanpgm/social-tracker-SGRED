"""The topic configuration file: strict loading, dumping and the shipped defaults."""

from importlib.resources import files

import pytest

from social_tracker.classification.domain import Topic
from social_tracker.classification.text import tokens
from social_tracker.classification.topics import TopicConfig, TopicMatcher
from social_tracker.classification.yaml_io import (
    MAX_CONFIG_BYTES,
    default_topic_config,
    dump_topic_config,
    load_topic_config,
)
from social_tracker.shared.errors import ValidationError

MINIMAL = "version: 1\nearthquake: [sismo]\n"


@pytest.fixture(scope="module")
def default() -> TopicConfig:
    return default_topic_config()


def folded(terms) -> set[str]:
    return {" ".join(tokens(t)) for t in terms}


# ---- the shipped defaults ----


@pytest.mark.parametrize(
    "term",
    [
        "sismo",
        "temblor",
        "terremoto",
        "réplica",
        "magnitud",
        "epicentro",
        "movimiento telúrico",
        "Richter",
        "escala Richter",
        "alerta sísmica",
    ],
)
def test_the_defaults_carry_the_earthquake_vocabulary(default, term):
    assert " ".join(tokens(term)) in folded(default.earthquake)


@pytest.mark.parametrize(
    "term", ["derrumbe", "evacuación", "inundación", "volcán", "incendio forestal"]
)
def test_the_defaults_carry_the_secondary_disasters(default, term):
    assert " ".join(tokens(term)) in folded(default.other_disaster)


def test_the_defaults_know_the_secretariat_under_its_name_variants(default):
    matcher = TopicMatcher(default)
    for text in (
        "Secretaría de Gestión del Riesgo de Emergencias y Desastres de Cali",
        "la secretaria de gestion del riesgo",
        "el DAGRD de Cali",
        "Departamento Administrativo de Gestión del Riesgo de Desastres",
        "Secretario de Gestión del Riesgo",
    ):
        assert Topic.SECRETARIAT in matcher.evaluate(tokens(text)).topics, text
    for text in ("Secretaría de Gestión Ambiental", "Secretaría de Salud", "dagrdx"):
        assert Topic.SECRETARIAT not in matcher.evaluate(tokens(text)).topics, text


def test_the_defaults_decide_the_documented_substring_case(default):
    matcher = TopicMatcher(default)
    assert matcher.evaluate(tokens("el sismógrafo registró")).in_scope is False
    assert matcher.evaluate(tokens("el sismo de ayer")).in_scope is True


def test_the_defaults_do_not_invent_addresses_and_say_to_verify_them():
    text = files("social_tracker.classification").joinpath("default_topics.yaml").read_text("utf-8")
    assert "http" not in text.lower()
    assert text.lower().count("verify before relying") >= 2  # domains and channel hints


def test_the_default_domains_and_hints_are_placeholders(default):
    assert default.official_domains == ("official-domain.example",)
    assert default.official_channel_hints == ("@official_channel_placeholder",)


# ---- loading ----


def test_a_file_loads_into_a_configuration():
    assert load_topic_config(MINIMAL).earthquake == ("sismo",)


def test_a_dump_loads_back_to_the_same_configuration(default):
    assert load_topic_config(dump_topic_config(default)) == default


def test_a_dump_says_to_verify_the_official_hints_before_relying_on_them(default):
    assert "verify before relying" in dump_topic_config(default)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n",
        "- a\n- b\n",
        "just text",
        "version: 1\nearthquake: [sismo\n",  # malformed
        "version: 1\nversion: 1\nearthquake: [sismo]\n",  # duplicate key
        "version: 1\nunknown: [x]\nearthquake: [sismo]\n",
        "version: 1\nearthquake: &a [sismo]\nplaces: *a\n",  # anchors and aliases
        "version: 1\nearthquake: !!python/object:os.system [sismo]\n",  # explicit tag
        "version: 1\nearthquake: [sismo]\n---\nversion: 1\n",  # two documents
        "version: 1\n<<: {earthquake: [sismo]}\n",  # merge key
        "version: 1\nearthquake: [[[[[[[[[[sismo]]]]]]]]]]\n",  # too deep
        "version: 1\nearthquake: [sismo, 5]\n",
    ],
)
def test_an_invalid_file_is_refused(text):
    with pytest.raises(ValidationError):
        load_topic_config(text)


def test_the_size_limit_is_in_bytes_and_exact():
    padding = "#" + "ñ" * 10  # two bytes per character
    body = MINIMAL
    lines = []
    size = len(body.encode("utf-8"))
    while size + len((padding + "\n").encode("utf-8")) <= MAX_CONFIG_BYTES:
        lines.append(padding)
        size += len((padding + "\n").encode("utf-8"))
    exact = body + "\n".join(lines) + "\n" + "#" * (MAX_CONFIG_BYTES - size - 1) + "\n"
    assert len(exact.encode("utf-8")) == MAX_CONFIG_BYTES
    assert load_topic_config(exact).earthquake == ("sismo",)
    with pytest.raises(ValidationError):
        load_topic_config(exact + "#")


def test_the_error_does_not_echo_the_file_content():
    marker = "hunter2-do-not-print"
    with pytest.raises(ValidationError) as caught:
        load_topic_config(f"version: 1\nearthquake: [sismo]\n{marker}: x\n")
    assert marker not in str(caught.value)
    with pytest.raises(ValidationError) as caught:
        load_topic_config(f"version: 1\nearthquake: [sismo, {marker}, {marker}]\n")
    assert marker not in str(caught.value)
