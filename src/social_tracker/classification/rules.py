"""The rules stage of the cascade (RC-1 stage 2): heuristic signals, their combination into a
preliminary label with a confidence (RC-4), the credibility prior (RC-8) and the escalation
decision (RC-7). Pure and deterministic (RC-10): the same text and configuration always give the
same result, whatever the order of anything.

How a decision is reached
- Detectors look for signals in the item's text (see `SIGNALS` for the list). Each signal argues
  for one label with a weight; a weight of 0 in the configuration switches a signal off.
- The evidence for a label is the noisy-or of its signals' weights, 1 - prod(1 - w): independent
  clues add up but never pass 1. The label with the most evidence wins; an exact tie goes to the
  more serious label. No signal at all, or a top evidence under `inconclusive_below`, leaves the
  item `unclassified` (inconclusive).
- Two labels that are both strong and close are CONFLICTING signals.
- The credibility of the source (1..5) is a prior: it moves the confidence of a label the
  evidence already chose, down for a risk label and up for `no_risk` as credibility grows, by 0.03
  per step from 3. It never picks a label and never touches an unclassified item.
- An in-scope item that is inconclusive, conflicting or whose confidence is in [0.4, 0.8) is
  marked for the LLM, unless it duplicates an item already judged.

Detectors read the folded text (`text.fold`) except those that need case or punctuation, which
read the NFKC text. Item text is data: nothing in it is ever interpreted as an instruction, and a
rationale never repeats it.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from social_tracker.classification.domain import (
    MAX_RATIONALE_CHARS,
    RISK_LABELS,
    SIGNALS,
    ClassifierSettings,
    Label,
    Signal,
    TopicVerdict,
)
from social_tracker.classification.text import PreparedText
from social_tracker.classification.topics import TopicConfig, TopicMatcher
from social_tracker.shared.errors import ValidationError

PRIOR_STEP: Final = 0.03  # confidence moved per credibility step away from 3
NEUTRAL_CREDIBILITY: Final = 3
_LABEL_ORDER = (
    Label.DELIBERATE_DISINFORMATION,
    Label.UNINTENTIONAL_RUMOR,
    Label.RECYCLED_CONTENT,
    Label.NO_RISK,
)


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# ---- patterns (on folded text: lowercase, no accents, no punctuation) ----

_QUAKE = r"(?:temblor|sismo|terremoto|replica|sacudon|movimiento telurico)"
_FUTURE = r"(?:habra|va a haber|vendra|ocurrira|se viene|se aproxima|se acerca|llegara|tendremos)"
_FORECAST = r"(?:pronostican|pronostico|pronosticaron|predicen|predijo|prediccion|profecia)"
_TOMORROW = r"(?<!\besta )(?<!\bla )manana"
_PREDICTIONS = tuple(
    re.compile(p)
    for p in (
        rf"\b{_FUTURE}\b(?: \w+){{0,4}} {_QUAKE}\b",
        r"\b(?:va a|van a|vamos a|ira a) (?:temblar|sacudir)\b",
        rf"\b{_FORECAST}\b(?: \w+){{0,6}} {_QUAKE}\b",
        rf"\b{_QUAKE}\b(?: \w+){{0,4}} (?:{_TOMORROW}|pasado manana|esta noche|esta madrugada)\b",
    )
)
_NEGATION = re.compile(
    r"\b(?:no|ni|sin|falso|falsa|desmiente|desmienten|descarta|descartan|imposible|nunca|jamas)\b"
)
_NEGATION_WINDOW = 40
_MONTHS = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre"
_WEEKDAYS = "lunes|martes|miercoles|jueves|viernes|sabado|domingo"
_WHEN = re.compile(
    rf"\b(?:\d{{1,2}} de (?:{_MONTHS})|\d{{1,2}}(?: \d{{2}})? ?(?:am|pm|a m|p m|horas|hrs)"
    rf"|a las \d{{1,2}}|{_WEEKDAYS}|{_TOMORROW}|pasado manana|esta noche|esta madrugada|hoy)\b"
)
_WHEN_WINDOW = 80
_AUTHORITY = (
    r"(?:nasa|un cientifico|un experto|un sismologo|un astronomo|un geologo|un especialista)"
)
_SAYING = r"(?:lo dijo|dijo|dice|segun|asegura|aseguran|advierte|advirtio|confirma|confirmo|afirma)"
_PSEUDO_AUTHORITY = (
    re.compile(rf"\b{_SAYING}\b(?: \w+){{0,3}} {_AUTHORITY}\b"),
    re.compile(
        rf"\b{_AUTHORITY}\b(?: \w+){{0,3}} (?:dijo|dice|asegura|advierte|advirtio|confirma|"
        r"confirmo|predijo|pronostico|afirma)\b"
    ),
)
_VIRAL_WORDS = _words(
    "compartan comparte compartelo comparta reenvia reenvien reenvialo difundan difunde "
    "difundelo pasalo pasenlo viralicen viralizalo urgente"
)
_VIRAL_PHRASES = ("corre la voz", "comparte esto")
_CENSORSHIP = (
    "antes de que lo borren",
    "antes de que lo eliminen",
    "antes de que lo quiten",
    "no quieren que sepas",
    "no quieren que se sepa",
    "los medios no dicen",
    "los medios no te lo cuentan",
    "lo estan ocultando",
    "nos ocultan",
    "nos estan ocultando",
    "lo que no te dicen",
    "el gobierno oculta",
)
_UNVERIFIABLE = (
    "fuentes confiables",
    "fuente confiable",
    "fuentes cercanas",
    "fuente cercana",
    "fuentes extraoficiales",
    "dicen que",
    "se dice que",
    "se comenta que",
    "me contaron",
    "me dijo un amigo",
    "un amigo me",
    "segun rumores",
    "un contacto",
)
_ATTRIBUTION_BEFORE = _words(
    "segun acuerdo informa informo comunica comunico alerta alerto ordena ordeno anuncia anuncio "
    "confirma confirmo emite emitio comunicado boletin pide solicita advierte"
)
_ATTRIBUTION_AFTER = _words(
    "informa informo ordena ordeno alerta alerto pide pidio solicita advierte advirtio confirma "
    "confirmo anuncia anuncio comunica comunico dice dijo ordenan emite emitio"
)
_ATTRIBUTION_REACH = 6  # tokens before a body name; after it only the next four
_ATTRIBUTION_AFTER_REACH = 4
_ALARM_WORDS = _words(
    "evacuen evacuar salgan abandonen refugiense desalojen huyan corran resguardense alejense "
    "inmediatamente urgente peligro alarma"
)
_EVACUATION_WORDS = _words("evacuen evacuar desalojen abandonen refugiense resguardense")
_EVACUATION_PHRASES = (
    "salgan de",
    "evacuacion inmediata",
    "ordenan la evacuacion",
    "orden de evacuacion",
    "habilitan albergues",
    "albergues temporales",
    "albergue temporal",
    "cierre de vias",
    "vias cerradas",
    "cierran las vias",
    "suspendidas las clases",
    "suspenden las clases",
    "clases suspendidas",
    "suspension de clases",
)
_VICTIMS = "muertos?|fallecidos?|victimas?|cadaveres|heridos?|desaparecidos?"
_STRUCTURES = "edificios?|casas?|puentes?|torres?|hospital(?:es)?|viviendas?"
_COLLAPSE = "colapso|colapsaron|se cayo|se cayeron|se derrumbo|se derrumbaron|derrumbo|derrumbaron"
_CASUALTIES = (
    re.compile(rf"\b\d{{1,6}} (?:{_VICTIMS})\b"),
    re.compile(rf"\b(?:cientos|miles|decenas|centenares) de (?:{_VICTIMS})\b"),
    re.compile(rf"\b(?:{_STRUCTURES})\b(?: \w+){{0,3}} (?:{_COLLAPSE})\b"),
    re.compile(rf"\b(?:{_COLLAPSE})\b(?: \w+){{0,3}} (?:{_STRUCTURES})\b"),
)
_MEDIA = "video|videos|imagenes|imagen|foto|fotos|fotografia|audio"
_RECYCLED = (
    re.compile(rf"\b(?:{_MEDIA})\b(?: \w+){{0,2}} (?:de|del) (?:19|20)\d\d\b"),
    re.compile(
        r"\b(?:hace anos|de hace anos|video antiguo|video viejo|imagen antigua|imagen vieja|"
        r"foto antigua|foto vieja|imagenes de archivo|video de archivo|foto de archivo|"
        r"no es de hoy|no es reciente|es de otro temblor|es de otro terremoto)\b"
    ),
)
_YEAR = re.compile(r"\b(19[5-9]\d|20\d\d)\b")
_REPORT_VERB = re.compile(
    r"\b(?:reporto|registro|informo|confirmo|detecto|midio|ocurrio|se presento|se sintio)\b"
)
_MAGNITUDE = re.compile(r"\bmagnitud \d")
_URL = re.compile(r"(?:https?://|www\.)([a-z0-9.-]+)")
_CAPS_MIN_LETTERS = 20
_CAPS_RATIO = 0.7
_EXCLAMATIONS = 3


def _padded(plain: str) -> str:
    return f" {plain} "


def _any_phrase(padded: str, phrases: Sequence[str]) -> bool:
    return any(f" {phrase} " in padded for phrase in phrases)


def _predictions(plain: str) -> list[re.Match[str]]:
    """Prediction matches that are not negated shortly before (a debunk is not a prediction)."""
    found = []
    for pattern in _PREDICTIONS:
        for match in pattern.finditer(plain):
            before = plain[max(0, match.start() - _NEGATION_WINDOW) : match.start()]
            if not _NEGATION.search(before):
                found.append(match)
    return found


def _prediction_with_time(plain: str, matches: Sequence[re.Match[str]]) -> bool:
    for match in matches:
        start, end = max(0, match.start() - _WHEN_WINDOW), match.end() + _WHEN_WINDOW
        if _WHEN.search(plain, start, end):
            return True
    return False


def _shouting(raw: str) -> bool:
    letters = [ch for ch in raw if ch.isalpha()]
    if len(letters) < _CAPS_MIN_LETTERS:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return upper / len(letters) >= _CAPS_RATIO


def _exclaiming(raw: str) -> bool:
    return raw.count("!") >= _EXCLAMATIONS or "!!" in raw or "¡¡" in raw


def _hosts(text: str) -> set[str]:
    return {host.strip(".-") for host in _URL.findall(text.lower())}


def _official_host(host: str, domains: Sequence[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def _official_link(raw: str, item_url: str | None, domains: Sequence[str]) -> bool:
    hosts = _hosts(raw) | (_hosts(item_url) if item_url else set())
    return any(_official_host(host, domains) for host in hosts)


def _attributed(words: Sequence[str], spans: Sequence[tuple[int, int]]) -> bool:
    """True when an official body is named as the speaker: an attribution word shortly before
    its name or right after it."""
    for start, end in spans:
        before = words[max(0, start - _ATTRIBUTION_REACH) : start]
        after = words[end : end + _ATTRIBUTION_AFTER_REACH]
        if _ATTRIBUTION_BEFORE.intersection(before) or _ATTRIBUTION_AFTER.intersection(after):
            return True
    return False


def _news_report(plain: str) -> bool:
    return bool(
        _REPORT_VERB.search(plain)
        and _MAGNITUDE.search(plain)
        and re.search(rf"\b{_QUAKE}\b", plain)
    )


def _old_years(plain: str, reference_year: int, gap: int) -> bool:
    return any(int(y) <= reference_year - gap for y in _YEAR.findall(plain))


def weights_for(config: TopicConfig) -> dict[str, float]:
    """The weight of every signal: the default, or the configured override."""
    weights = {name: spec.weight for name, spec in SIGNALS.items()}
    weights.update(dict(config.weights))
    return weights


def detect_signals(
    prepared: PreparedText,
    matcher: TopicMatcher,
    config: TopicConfig,
    *,
    reference_year: int,
    item_url: str | None,
    settings: ClassifierSettings,
) -> tuple[Signal, ...]:
    """Every signal the text shows, sorted by name, each once, with its weight. Signals whose
    configured weight is 0 are left out."""
    plain, raw, words = prepared.plain, prepared.raw, prepared.tokens
    padded = _padded(plain)
    predictions = _predictions(plain)
    linked = _official_link(raw, item_url, config.official_domains)
    spans = matcher.official_bodies(words)
    alarmed = bool(_ALARM_WORDS.intersection(words))
    present = {
        "prediction_claim": bool(predictions),
        "prediction_specific_time": _prediction_with_time(plain, predictions),
        "pseudo_authority": any(p.search(plain) for p in _PSEUDO_AUTHORITY),
        "virality_phrases": bool(_VIRAL_WORDS.intersection(words))
        or _any_phrase(padded, _VIRAL_PHRASES),
        "censorship_bait": _any_phrase(padded, _CENSORSHIP),
        "all_caps": _shouting(raw),
        "repeated_exclamation": _exclaiming(raw),
        "unverifiable_sources": _any_phrase(padded, _UNVERIFIABLE),
        "official_attribution_unlinked": not linked and alarmed and _attributed(words, spans),
        "official_link": linked,
        "evacuation_claim": bool(_EVACUATION_WORDS.intersection(words))
        or _any_phrase(padded, _EVACUATION_PHRASES),
        "casualty_claim": any(p.search(plain) for p in _CASUALTIES),
        "recycled_marker": any(p.search(plain) for p in _RECYCLED),
        "old_year_reference": _old_years(plain, reference_year, settings.old_year_gap),
        "news_report": _news_report(plain),
    }
    weights = weights_for(config)
    return tuple(
        Signal(name, SIGNALS[name].label, weights[name])
        for name in sorted(present)
        if present[name] and weights[name] > 0
    )


# ---- combination ----


@dataclass(frozen=True)
class RuleOutcome:
    """The preliminary result of the rules. `risk_strength` is the evidence for the strongest
    risk label (before the prior), used for the LLM queue priority."""

    label: Label
    confidence: float
    signals: tuple[Signal, ...]
    inconclusive: bool
    conflicting: bool
    risk_strength: float


def apply_prior(label: Label, confidence: float, credibility: int) -> float:
    """`confidence` moved by the credibility of the source: a risk label is believed less the more
    credible the source, `no_risk` more; other labels are not touched. In [0, 1], rounded."""
    if isinstance(credibility, bool) or not isinstance(credibility, int):
        raise ValidationError("credibility must be a whole number from 1 to 5")
    if not 1 <= credibility <= 5:
        raise ValidationError("credibility must be a whole number from 1 to 5")
    shift = PRIOR_STEP * (credibility - NEUTRAL_CREDIBILITY)
    if label in RISK_LABELS:
        confidence -= shift
    elif label is Label.NO_RISK:
        confidence += shift
    return round(min(1.0, max(0.0, confidence)), 4)


def _evidence(signals: Sequence[Signal]) -> dict[Label, float]:
    remaining: dict[Label, float] = {}
    for signal in signals:
        remaining[signal.label] = remaining.get(signal.label, 1.0) * (1.0 - signal.weight)
    return {label: round(1.0 - left, 4) for label, left in remaining.items()}


def combine(
    signals: Sequence[Signal], *, credibility: int, settings: ClassifierSettings
) -> RuleOutcome:
    """Turn signals into a label, a confidence and the inconclusive/conflicting flags."""
    apply_prior(Label.UNCLASSIFIED, 0.0, credibility)  # validates the credibility
    ordered = tuple(sorted(signals, key=lambda s: s.name))
    evidence = _evidence(ordered)
    if not evidence:
        return RuleOutcome(Label.UNCLASSIFIED, 0.0, ordered, True, False, 0.0)
    ranked = sorted(evidence.items(), key=lambda item: (-item[1], _LABEL_ORDER.index(item[0])))
    top_label, top = ranked[0]
    risk = max((score for label, score in evidence.items() if label in RISK_LABELS), default=0.0)
    if top < settings.inconclusive_below:
        return RuleOutcome(Label.UNCLASSIFIED, top, ordered, True, False, risk)
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    conflicting = (
        second >= settings.conflict_min and round(top - second, 4) < settings.conflict_margin
    )
    confidence = apply_prior(top_label, top, credibility)
    return RuleOutcome(top_label, confidence, ordered, False, conflicting, risk)


def evaluate_rules(
    prepared: PreparedText,
    matcher: TopicMatcher,
    config: TopicConfig,
    *,
    credibility: int,
    reference_year: int,
    item_url: str | None,
    settings: ClassifierSettings,
) -> RuleOutcome:
    """The whole stage: detect the signals, then combine them."""
    signals = detect_signals(
        prepared,
        matcher,
        config,
        reference_year=reference_year,
        item_url=item_url,
        settings=settings,
    )
    return combine(signals, credibility=credibility, settings=settings)


# ---- decisions on a result ----


def needs_escalation(
    *,
    in_scope: bool,
    confidence: float,
    inconclusive: bool,
    conflicting: bool,
    duplicate_of_judged: bool,
    settings: ClassifierSettings,
) -> bool:
    """RC-7: send to the LLM an in-scope item that the earlier stages could not settle
    (inconclusive, conflicting, or a confidence in [escalate_low, escalate_high)) unless it
    duplicates an item already judged. A confidence of 0.8 or more is settled; 0.4 is not."""
    if not in_scope or duplicate_of_judged:
        return False
    in_band = settings.escalate_low <= confidence < settings.escalate_high
    return inconclusive or conflicting or in_band


def is_low_confidence(confidence: float, alert_min_confidence: float) -> bool:
    """RC-11: strictly below the alert threshold."""
    return confidence < alert_min_confidence


def priority(risk_strength: float, topic_score: float, *, secretariat: bool) -> float:
    """Rank in the LLM queue, in [0, 1]: strong risk evidence first, then how clearly the item is
    on topic, with a bump for items that are about the secretariat."""
    value = 0.6 * risk_strength + 0.3 * topic_score + (0.1 if secretariat else 0.0)
    return round(min(1.0, max(0.0, value)), 4)


_MATCHED_SHOWN = 5
_TERM_SHOWN_CHARS = 40
_SIGNALS_SHOWN = 4


def rationale_for_rules(outcome: RuleOutcome, topic: TopicVerdict) -> str:
    """A short explanation of a rules verdict (RC-4): the label, the strongest signals that argue
    for it, the topics and a few matched configuration terms, in quotes. Item text is never
    quoted, and the result is at most `MAX_RATIONALE_CHARS` long."""
    if outcome.inconclusive:
        names = ", ".join(s.name for s in outcome.signals) or "no signal"
        head = f"inconclusive: nothing decisive ({names})"
    else:
        ranked = sorted(
            (s for s in outcome.signals if s.label is outcome.label),
            key=lambda s: (-s.weight, s.name),
        )
        names = ", ".join(s.name for s in ranked[:_SIGNALS_SHOWN])
        head = f"{outcome.label.value} candidate from rules: {names}"
        if outcome.conflicting:
            head += "; conflicting signals"
    topics = ", ".join(t.value for t in topic.topics)
    matched = ", ".join(f'"{t[:_TERM_SHOWN_CHARS]}"' for t in topic.matched_terms[:_MATCHED_SHOWN])
    return f"{head}; topics: {topics}; matched: {matched}"[:MAX_RATIONALE_CHARS]
