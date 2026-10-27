"""Calibration and hardening of the rules: benign evidence, risk signals, obfuscation (S4, S5, C3,
C4, W3 and the vocabulary suggestions)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from social_tracker.classification.domain import (
    COMPANION_SIGNALS,
    SIGNALS,
    WEAK_SIGNALS,
    ClassifierSettings,
    Label,
    Signal,
)
from social_tracker.classification.rules import (
    apply_prior,
    combine,
    needs_escalation,
    undo_prior,
)
from tests.classification.support import SETTINGS, run_rules

LINK = "https://official-domain.example/sismos"
REPORT = "El Servicio Geológico Colombiano informó un sismo de magnitud 4.2 en Cali a las 10:30"


def sig(name: str, weight: float | None = None) -> Signal:
    spec = SIGNALS[name]
    return Signal(name, spec.label, spec.weight if weight is None else weight)


# ---- C3: benign evidence can settle ----


def test_an_official_report_with_an_official_link_settles_as_no_risk_at_the_alert_grade():
    result = run_rules(f"{REPORT}. Más información en {LINK}", credibility=3)
    assert result.names == {"news_report", "official_link"}
    assert result.label is Label.NO_RISK
    assert result.confidence >= 0.8
    assert result.escalated is False


@pytest.mark.parametrize(("credibility", "minimum"), [(1, 0.74), (3, 0.8), (5, 0.86)])
def test_the_credibility_of_the_source_moves_a_settled_report_either_way(credibility, minimum):
    result = run_rules(f"{REPORT}. Más información en {LINK}", credibility=credibility)
    assert result.confidence == pytest.approx(minimum)
    assert result.escalated is False


def test_a_plain_news_report_without_any_attribution_is_settled():
    result = run_rules("Un sismo de magnitud 4.2 se registró anoche en Cali sin reporte de daños")
    assert result.names == {"news_report"}
    assert (result.label, result.confidence, result.escalated) == (Label.NO_RISK, 0.6, False)


@pytest.mark.parametrize(
    "text",
    [
        "Qué mal trabaja la Secretaría de Gestión del Riesgo, nunca limpian los caños en Cali",
        "Otra vez la Secretaría de Gestión del Riesgo de Cali sin respuesta a los vecinos",
        "Ayer sentimos el temblor en el barrio y los vecinos salieron a la calle a conversar",
        "Me parece que el DAGRD debería explicar mejor cómo se prepara la ciudad para un sismo",
    ],
)
def test_neutral_chatter_with_a_topic_is_no_risk_and_is_not_escalated(text):
    result = run_rules(text)
    assert result.names == set()
    assert (result.label, result.confidence) == (Label.NO_RISK, 0.6)
    assert result.escalated is False


@pytest.mark.parametrize(
    "text",
    [
        "Sismo en Cali",  # too short to call harmless
        "sismo xkqzv wpfhj zzzq qqqx lkjhg",  # a pile of tokens, no Spanish function word
    ],
)
def test_text_that_is_not_a_sentence_stays_inconclusive_and_escalated(text):
    result = run_rules(text)
    assert (result.label, result.outcome.inconclusive, result.escalated) == (
        Label.UNCLASSIFIED,
        True,
        True,
    )


def test_a_neutral_verdict_is_configurable():
    strict = ClassifierSettings(neutral_confidence=0.9, neutral_min_tokens=20)
    text = "Qué mal trabaja la Secretaría de Gestión del Riesgo, nunca limpian los caños en Cali"
    assert run_rules(text, settings=strict).label is Label.UNCLASSIFIED  # fewer than 20 words
    confident = ClassifierSettings(neutral_confidence=0.9)
    assert run_rules(text, settings=confident).confidence == 0.9


def test_thin_benign_evidence_alone_is_not_settled():
    thin = combine([sig("official_link", 0.35)], credibility=3, settings=SETTINGS)
    assert (thin.label, thin.inconclusive) == (Label.UNCLASSIFIED, True)
    assert thin.confidence == 0.35


def test_a_settled_no_risk_is_never_queued_but_a_risk_label_in_the_band_is():
    kwargs = {
        "in_scope": True,
        "inconclusive": False,
        "conflicting": False,
        "duplicate_of_judged": False,
        "settings": SETTINGS,
    }
    assert needs_escalation(confidence=0.6, label=Label.NO_RISK, **kwargs) is False
    assert needs_escalation(confidence=0.6, label=Label.UNINTENTIONAL_RUMOR, **kwargs) is True
    assert needs_escalation(confidence=0.6, **kwargs) is True  # no label: the band applies
    assert needs_escalation(confidence=0.6, label=Label.NO_RISK, **{**kwargs, "conflicting": True})


# ---- S4: a benign sentence cannot launder a panic message ----

PANIC = "temblor: 300 muertos, edificios colapsaron, evacuen"
APPENDED = "se registro un sismo de magnitud 5"


def test_the_appended_benign_sentence_does_not_turn_a_panic_message_into_no_risk():
    alone = run_rules(PANIC)
    mixed = run_rules(f"{PANIC} {APPENDED}")
    assert "news_report" in mixed.names
    assert {"casualty_claim", "evacuation_claim"} <= mixed.names
    assert mixed.label is not Label.NO_RISK
    assert mixed.escalated is True
    assert mixed.outcome.conflicting is True
    assert mixed.outcome.mixed is True
    assert mixed.outcome.risk_strength >= 0.5
    assert alone.outcome.mixed is False


def test_the_mixed_message_is_queued_before_the_same_message_without_the_benign_sentence():
    from social_tracker.classification.rules import priority

    mixed = run_rules(f"{PANIC} {APPENDED}").outcome
    plain = run_rules(PANIC).outcome
    assert priority(mixed.risk_strength, 0.5, secretariat=False) > priority(
        plain.risk_strength, 0.5, secretariat=False
    )


BLOCKING = sorted(
    name
    for name, spec in SIGNALS.items()
    if spec.label
    in (Label.UNINTENTIONAL_RUMOR, Label.DELIBERATE_DISINFORMATION, Label.RECYCLED_CONTENT)
    and name not in WEAK_SIGNALS
)


@pytest.mark.parametrize("name", BLOCKING)
@pytest.mark.parametrize("benign", [("news_report", 0.6), ("official_link", 0.9)])
def test_no_risk_never_wins_next_to_a_blocking_risk_signal(name, benign):
    outcome = combine(
        [sig(name), sig(benign[0], benign[1]), sig("news_report", 0.95)],
        credibility=5,
        settings=SETTINGS,
    )
    assert outcome.label is not Label.NO_RISK
    assert outcome.mixed is True
    assert outcome.conflicting is True


@given(
    risky=st.sets(st.sampled_from(BLOCKING), min_size=1),
    benign=st.floats(min_value=0.05, max_value=0.99),
    credibility=st.integers(min_value=1, max_value=5),
)
def test_property_benign_evidence_never_settles_an_item_with_a_blocking_risk_signal(
    risky, benign, credibility
):
    signals = [sig(n) for n in risky] + [sig("news_report", benign)]
    outcome = combine(signals, credibility=credibility, settings=SETTINGS)
    assert outcome.label is not Label.NO_RISK
    assert outcome.conflicting is True


def test_stylistic_signals_do_not_stop_a_clean_report_from_settling():
    outcome = combine(
        [sig("news_report"), sig("official_link"), sig("all_caps"), sig("old_year_reference")],
        credibility=3,
        settings=SETTINGS,
    )
    assert (outcome.label, outcome.mixed) == (Label.NO_RISK, False)
    assert outcome.confidence >= 0.8


def test_an_obfuscated_report_cannot_be_settled_either():
    outcome = combine(
        [sig("news_report"), sig("official_link"), sig("obfuscated_text")],
        credibility=5,
        settings=SETTINGS,
    )
    assert (outcome.label, outcome.mixed, outcome.conflicting) == (Label.UNCLASSIFIED, True, True)


# ---- S5: obfuscation raises suspicion, never a label ----


@pytest.mark.parametrize(
    "text",
    [
        "Mañana habrá un tеmblor fuerte en Cali",  # Cyrillic e
        "Mañana habrá un t.e.m.b.l.o.r fuerte en Cali",
        "Mañana habrá un t e m b l o r fuerte en Cali",
        "Mañana habrá un t3mblor fuerte en Cali",
        "Mañana habrá un tem\u3164blor fuerte en Cali",  # Hangul filler
        "ΜΑÑΑΝΑ HABRÁ UN TΕΜBLΟR",  # Greek capitals
    ],
)
def test_an_evasion_does_not_hide_a_prediction_and_is_itself_a_signal(text):
    result = run_rules(text)
    assert {"prediction_claim", "obfuscated_text"} <= result.names
    assert result.escalated is True


def test_obfuscation_alone_decides_no_label():
    outcome = combine([sig("obfuscated_text")], credibility=3, settings=SETTINGS)
    assert (outcome.label, outcome.inconclusive) == (Label.UNCLASSIFIED, True)
    assert outcome.risk_strength == pytest.approx(0.3)
    text = run_rules("Mañana te cuento del tеmblor que sentimos todos en el barrio")
    assert text.label is Label.UNCLASSIFIED
    assert text.escalated is True


def test_a_clean_text_has_no_obfuscation_signal():
    assert "obfuscated_text" not in run_rules("Mañana habrá un temblor fuerte en Cali").names


def test_obfuscation_signal_can_be_switched_off_by_weight():
    from social_tracker.classification.topics import TopicConfig
    from social_tracker.classification.yaml_io import default_topic_config

    mapping = {**default_topic_config().to_mapping(), "weights": {"obfuscated_text": 0}}
    config = TopicConfig.from_mapping(mapping)
    assert "obfuscated_text" not in run_rules("un tеmblor en Cali", config=config).names


def test_a_megabyte_of_adversarial_text_is_classified_quickly():
    import time

    chunk = "Mañana habrá un t.e.m.b.l.o.r ѕіѕmо t3mblor \u3164 http://a@b.c " + "x " * 200
    hostile = "sismo " + chunk * (1_000_000 // len(chunk) + 1)
    started = time.perf_counter()
    result = run_rules(hostile)
    assert time.perf_counter() - started < 0.5
    assert result.escalated is True


# ---- C4: no dead zone, and predictions with new verbs ----


@pytest.mark.parametrize(
    "text",
    [
        "El SGC confirma que un sismo de magnitud 7 sacudirá Cali mañana a las 5 am. Compartan",
        "Un sismo de magnitud 8 sacudirá Cali el sábado a las 3 pm, compártelo",
        "Un terremoto azotará Cali esta noche, avisen a todos",
        "Un temblor golpeará la ciudad el domingo",
        "Un sismo fuerte ocurrirá en Cali el lunes a las 8 am",
        "Un sismo en Cali habrá otro más fuerte el viernes",
        "Prepárense para el temblor que se viene esta semana en Cali",
        "Alcaldía de Cali informa: suspendidas las clases mañana por el temblor que se viene",
    ],
)
def test_predictions_with_the_future_of_a_quake_verb_are_rumors_and_escalated(text):
    result = run_rules(text)
    assert "prediction_claim" in result.names
    assert result.label in (Label.UNINTENTIONAL_RUMOR, Label.DELIBERATE_DISINFORMATION)
    assert result.confidence >= 0.4
    assert result.escalated is True


@pytest.mark.parametrize(
    "text",
    [
        "El sismo sacudió Cali ayer a las 5 am",  # past tense
        "No es cierto que un sismo sacudirá Cali mañana, es falso",
        "El temblor de ayer dejó daños; habrá reunión en la alcaldía",
        "Tras el sismo habrá más trabajo para los bomberos de Cali",
    ],
)
def test_the_new_verbs_do_not_turn_reports_and_debunks_into_predictions(text):
    assert "prediction_claim" not in run_rules(text).names


def test_the_time_of_a_prediction_is_not_a_second_opinion():
    result = run_rules("Mañana habrá un temblor fuerte en Cali")
    assert {"prediction_claim", "prediction_specific_time"} <= result.names
    assert result.label is Label.UNINTENTIONAL_RUMOR
    assert result.outcome.conflicting is False


def test_companion_signals_are_known_signals():
    for companion, parent in COMPANION_SIGNALS.items():
        assert companion in SIGNALS
        assert parent in SIGNALS


@given(
    weights=st.lists(st.floats(min_value=0.01, max_value=0.99), min_size=1, max_size=6),
    credibility=st.integers(min_value=1, max_value=5),
)
def test_property_escalation_is_monotonic_in_the_strength_of_the_evidence(weights, credibility):
    """Adding rumor evidence one signal at a time: once the evidence is strong enough to settle
    it stays settled, and until then it is always escalated (no dead zone)."""
    names = ["prediction_claim", "pseudo_authority", "virality_phrases", "unverifiable_sources"]
    states = []
    chosen: list[Signal] = []
    for index, weight in enumerate(sorted(weights)):
        chosen.append(
            Signal(
                names[index % len(names)] if index < 4 else "all_caps",
                Label.UNINTENTIONAL_RUMOR,
                weight,
            )
        )
        # distinct names are not needed by `combine`, only the weights matter
        outcome = combine(chosen, credibility=credibility, settings=SETTINGS)
        states.append(
            needs_escalation(
                in_scope=True,
                confidence=outcome.confidence,
                inconclusive=outcome.inconclusive,
                conflicting=outcome.conflicting,
                duplicate_of_judged=False,
                settings=SETTINGS,
                label=outcome.label,
            )
        )
    first_settled = next((i for i, escalated in enumerate(states) if not escalated), len(states))
    assert all(states[:first_settled])
    assert not any(states[first_settled:])


@pytest.mark.parametrize(
    ("weight", "escalated"),
    [
        (0.05, True),
        (0.2, True),
        (0.39, True),
        (0.4, True),
        (0.79, True),
        (0.8, False),
        (1.0, False),
    ],
)
def test_there_is_no_dead_zone_between_inconclusive_and_the_llm_band(weight, escalated):
    outcome = combine([sig("prediction_claim", weight)], credibility=3, settings=SETTINGS)
    assert (
        needs_escalation(
            in_scope=True,
            confidence=outcome.confidence,
            inconclusive=outcome.inconclusive,
            conflicting=outcome.conflicting,
            duplicate_of_judged=False,
            settings=SETTINGS,
            label=outcome.label,
        )
        is escalated
    )


def test_a_credible_source_cannot_push_weak_evidence_into_the_dead_zone():
    # evidence 0.42 is a decided rumor; a credibility of 5 lowers its confidence to 0.36, which is
    # under the band: it must still go to the LLM
    outcome = combine([sig("prediction_claim", 0.42)], credibility=5, settings=SETTINGS)
    assert (outcome.label, outcome.confidence) == (Label.UNINTENTIONAL_RUMOR, 0.36)
    assert needs_escalation(
        in_scope=True,
        confidence=outcome.confidence,
        inconclusive=False,
        conflicting=False,
        duplicate_of_judged=False,
        settings=SETTINGS,
        label=outcome.label,
    )


def test_the_prior_can_be_undone():
    for label in (Label.UNINTENTIONAL_RUMOR, Label.NO_RISK, Label.UNCLASSIFIED):
        for credibility in range(1, 6):
            moved = apply_prior(label, 0.5, credibility)
            assert undo_prior(label, moved, credibility) == pytest.approx(0.5)


# ---- W3: false attribution ----


def test_the_secretary_announcing_a_quake_is_a_deliberate_disinformation_candidate():
    result = run_rules("La Secretaria anunció que el sábado habrá un terremoto")
    assert {"official_attribution_unlinked", "prediction_claim"} <= result.names
    assert result.label is Label.DELIBERATE_DISINFORMATION
    assert result.confidence >= 0.5
    assert result.escalated is True


def test_an_attribution_without_an_alarm_word_counts_for_less_than_one_with_it():
    plain = run_rules("Según la Secretaría de Gestión del Riesgo, el sismo de ayer fue leve")
    alarmed = run_rules("Según la Secretaría de Gestión del Riesgo, evacuen la zona de inmediato")
    weight = {s.name: s.weight for s in plain.outcome.signals}["official_attribution_unlinked"]
    strong = {s.name: s.weight for s in alarmed.outcome.signals}["official_attribution_unlinked"]
    assert 0 < weight < strong == 0.45


@pytest.mark.parametrize(
    "text",
    [
        "La Alcaldía de Cali informó que habrá cortes de luz por el sismo",
        "Bomberos Cali confirmó que el temblor no dejó heridos",
        "Según la UNGRD el terremoto de ayer fue de 5 grados",
    ],
)
def test_any_claim_attributed_to_an_official_body_without_a_link_is_flagged(text):
    assert "official_attribution_unlinked" in run_rules(text).names


def test_a_source_that_is_itself_official_makes_the_attribution_genuine():
    text = "La Secretaría de Gestión del Riesgo informó que se registró un sismo de magnitud 4.2"
    assert "official_attribution_unlinked" in run_rules(text).names
    genuine = run_rules(text, source_official=True)
    assert "official_attribution_unlinked" not in genuine.names
    assert "official_link" in genuine.names
    assert genuine.label is Label.NO_RISK


def test_an_item_served_from_an_official_address_is_genuine_too():
    text = "La Secretaría de Gestión del Riesgo ordena evacuar la zona"
    served = run_rules(text, url="https://www.official-domain.example/avisos/3")
    assert "official_attribution_unlinked" not in served.names


# ---- suggestions ----


def test_a_casualty_claim_argues_less_when_a_report_or_an_official_link_stands_with_it():
    bare = run_rules("Hay 300 muertos por el temblor en Cali")
    reported = run_rules(f"Se registró un sismo de magnitud 5 y hay 300 muertos. {LINK}")
    weight = lambda r: {s.name: s.weight for s in r.outcome.signals}["casualty_claim"]  # noqa: E731
    assert weight(reported) < weight(bare) == 0.25


@pytest.mark.parametrize(
    "text",
    [
        "El temblor de 4.5 grados se sintió en Cali, informó el SGC",
        "El Servicio Geológico Colombiano informó un temblor en el Valle del Cauca",
        "La Red Sismológica Nacional registró un sismo en Cali",
        "La Red Sismológica Nacional informó un temblor al suroccidente",
    ],
)
def test_more_report_wordings_are_news_reports(text):
    assert "news_report" in run_rules(text).names


@pytest.mark.parametrize(
    ("text", "viral"),
    [
        ("Compartan este mensaje, el temblor de hoy", True),
        ("No compartan rumores sobre el temblor", False),
        ("Por favor no difundan videos falsos del temblor", False),
        ("Nunca reenvíen cadenas sobre sismos", False),
        ("Es urgente: temblor en Cali", True),
        ("No es urgente, el temblor fue leve", False),
        ("No compartan esto. Pero compartan lo oficial del temblor", True),
    ],
)
def test_a_negated_viral_word_asks_for_the_opposite_and_is_not_a_signal(text, viral):
    assert ("virality_phrases" in run_rules(text).names) is viral


@pytest.mark.parametrize(
    "text",
    [
        "Fotos del archivo del temblor de Armenia circulan hoy",
        "Imágenes de archivo del terremoto, no es de hoy",
        "Video del archivo de un sismo en Cali",
        "Este temblor ocurrió de hace 20 años, no es nuevo",
        "El video es de hace 10 años y lo comparten como si fuera de hoy, temblor",
    ],
)
def test_more_recycled_markers(text):
    assert "recycled_marker" in run_rules(text).names


# ---- the exact edges of the calibration ----


@pytest.mark.parametrize(
    ("top", "rival", "conflicting"),
    [(0.40, 0.30, True), (0.40, 0.29, False), (0.50, 0.30, False), (0.44, 0.30, True)],
)
def test_two_labels_conflict_from_the_minimum_strength_of_the_rival(top, rival, conflicting):
    outcome = combine(
        [sig("prediction_claim", top), sig("censorship_bait", rival)],
        credibility=3,
        settings=SETTINGS,
    )
    assert outcome.conflicting is conflicting


@pytest.mark.parametrize(
    ("benign", "label"),
    [(0.5, Label.NO_RISK), (0.49, Label.UNCLASSIFIED), (0.51, Label.NO_RISK)],
)
def test_benign_evidence_alone_settles_from_the_minimum_confidence(benign, label):
    outcome = combine([sig("news_report", benign)], credibility=3, settings=SETTINGS)
    assert outcome.label is label


def casualty_weight(text: str, **options) -> float:
    found = run_rules(text, **options).outcome.signals
    return {s.name: s.weight for s in found}["casualty_claim"]


def test_a_casualty_claim_argues_less_next_to_a_report_alone_and_next_to_a_link_alone():
    claim = "hay 300 muertos por el temblor en Cali"
    assert casualty_weight(claim) == 0.25
    assert casualty_weight("Se registró un sismo de magnitud 5 y " + claim) == 0.15  # a report
    assert casualty_weight(claim + " " + LINK) == 0.15  # a link
    assert casualty_weight(claim, url="https://official-domain.example/x") == 0.15
