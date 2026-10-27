"""The rules stage of the cascade (RC-1 stage 2): heuristic signals, their combination into a
preliminary label with a confidence (RC-4), the credibility prior (RC-8) and the escalation
decision (RC-7). Pure and deterministic (RC-10): the same text and configuration always give the
same result, whatever the order of anything.

How a decision is reached
- Detectors look for signals in the item's text (see `SIGNALS` for the list), in the plain view
  and, when the text spells words around the detectors, in the decoded one (`text.PreparedText`).
  Each signal argues for one label with a weight; a weight of 0 in the configuration switches a
  signal off.
- The evidence for a label is the noisy-or of its signals' weights, 1 - prod(1 - w): independent
  clues add up but never pass 1. The label with the most evidence wins; an exact tie goes to the
  more serious label. Evidence under `inconclusive_below` (0.4) decides nothing: the item is
  `unclassified` and inconclusive.
- Nothing argues for risk (no risk signal, no obfuscation): an on-topic text with enough words is
  `no_risk` at `neutral_confidence`, or at the strength of its benign evidence if that is higher;
  such a verdict is settled from `no_risk_settled_min` on. A text too short for that, or with
  only thin benign evidence, is inconclusive.
- A risk signal (except the stylistic ones, `WEAK_SIGNALS`) or an obfuscation next to benign
  evidence means `no_risk` cannot win, however much benign evidence there is: a panic message with
  a harmless sentence appended stays suspect. The item is then marked conflicting and its queue
  priority is raised.
- Two labels that are both strong and close are CONFLICTING signals. The time of a prediction
  restates the prediction (`COMPANION_SIGNALS`) and is not counted as a second opinion for that.
- The credibility of the source (1..5) is a prior: it moves the confidence of a label the
  evidence already chose, down for a risk label and up for `no_risk` as credibility grows, by 0.03
  per step from 3. It never picks a label and never touches an unclassified item.
- An in-scope item is marked for the LLM when it is inconclusive, conflicting or, for a risk
  label, when its confidence is in [0.4, 0.8), unless it duplicates an item already judged. A
  settled `no_risk` is never queued: only a risk signal, a conflict or too little evidence queues.

Detectors read the folded text (`text.fold`) except those that need case or punctuation, which
read the NFKC text. Item text is data: nothing in it is ever interpreted as an instruction, and a
rationale never repeats it.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from social_tracker.classification.domain import (
    COMPANION_SIGNALS,
    MAX_RATIONALE_CHARS,
    PLAIN_ATTRIBUTION_FACTOR,
    RISK_LABELS,
    SIGNALS,
    WEAK_SIGNALS,
    ClassifierSettings,
    Label,
    Signal,
    TopicVerdict,
)
from social_tracker.classification.links import links_in, official_hosts
from social_tracker.classification.text import PreparedText
from social_tracker.classification.topics import TopicConfig, TopicMatcher, reads_spanish
from social_tracker.shared.errors import ValidationError

PRIOR_STEP: Final = 0.03  # confidence moved per credibility step away from 3
NEUTRAL_CREDIBILITY: Final = 3
MIXED_RISK_FLOOR: Final = 0.5  # the queue strength of an item whose benign evidence cannot win
CORROBORATED_CASUALTY_FACTOR: Final = 0.6  # a casualty claim next to a report or an official link
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
_QUAKE_VERB = r"(?:sacudira|azotara|golpeara|ocurrira|remecera|estremecera|afectara)"
_PREDICTIONS = tuple(
    re.compile(p)
    for p in (
        rf"\b{_FUTURE}\b(?: \w+){{0,4}} {_QUAKE}\b",
        r"\b(?:va a|van a|vamos a|ira a) (?:temblar|sacudir)\b",
        rf"\b{_FORECAST}\b(?: \w+){{0,6}} {_QUAKE}\b",
        rf"\b{_QUAKE}\b(?: \w+){{0,4}} (?:{_TOMORROW}|pasado manana|esta noche|esta madrugada)\b",
        rf"\b{_QUAKE}\b(?: \w+){{0,4}} {_QUAKE_VERB}\b",
        rf"\b{_QUAKE}\b(?: \w+){{0,3}} habra\b(?: \w+){{0,3}} (?:replicas?|otro|otra|magnitud)\b",
    )
)
_NEGATION = re.compile(
    r"\b(?:no|ni|sin|falso|falsa|desmiente|desmienten|descarta|descartan|imposible|nunca|jamas)\b"
)
_NEGATION_WORDS = _words(
    "no ni sin falso falsa desmiente desmienten descarta descartan imposible nunca jamas"
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
_VIRAL_RE = re.compile("(?<= )(?:" + "|".join([*sorted(_VIRAL_WORDS), *_VIRAL_PHRASES]) + ")(?= )")
_VIRAL_NEGATION_REACH = 2  # words before a viral word that can negate it ("no es urgente")
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
    re.compile(rf"\b(?:{_MEDIA})\b(?: \w+){{0,2}} (?:de|del) archivo\b"),
    re.compile(r"\b(?:de )?hace \d{1,2} anos\b"),
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
_MAGNITUDE = re.compile(r"\bmagnitud \d|\bde \d{1,2}(?: \d)? grados\b")
_AGENCY_REPORT = re.compile(
    r"\b(?:servicio geologico colombiano|sgc|red sismologica(?: nacional)?|ungrd)"
    r" (?:informo|reporto|registro|detecto|confirmo)\b"
)
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


def _viral(plain: str) -> bool:
    """A viral word or phrase that is not negated right before it ("no compartan" asks for the
    opposite)."""
    padded = _padded(plain)
    return any(
        not _NEGATION_WORDS.intersection(padded[: m.start()].split()[-_VIRAL_NEGATION_REACH:])
        for m in _VIRAL_RE.finditer(padded)
    )


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
    if not re.search(rf"\b{_QUAKE}\b", plain):
        return False
    return bool(
        (_REPORT_VERB.search(plain) and _MAGNITUDE.search(plain)) or _AGENCY_REPORT.search(plain)
    )


def _old_years(plain: str, reference_year: int, gap: int) -> bool:
    return any(int(y) <= reference_year - gap for y in _YEAR.findall(plain))


def weights_for(config: TopicConfig) -> dict[str, float]:
    """The weight of every signal: the default, or the configured override."""
    weights = {name: spec.weight for name, spec in SIGNALS.items()}
    weights.update(dict(config.weights))
    return weights


_TEXT_SIGNALS = (
    "prediction_claim",
    "prediction_specific_time",
    "pseudo_authority",
    "virality_phrases",
    "censorship_bait",
    "unverifiable_sources",
    "evacuation_claim",
    "casualty_claim",
    "recycled_marker",
    "old_year_reference",
    "news_report",
)


def _view_flags(
    plain: str, words: Sequence[str], matcher: TopicMatcher, reference_year: int, gap: int
) -> dict[str, bool]:
    """The text-reading detectors on one view of the text."""
    padded = _padded(plain)
    predictions = _predictions(plain)
    return {
        "prediction_claim": bool(predictions),
        "prediction_specific_time": _prediction_with_time(plain, predictions),
        "pseudo_authority": any(p.search(plain) for p in _PSEUDO_AUTHORITY),
        "virality_phrases": _viral(plain),
        "censorship_bait": _any_phrase(padded, _CENSORSHIP),
        "unverifiable_sources": _any_phrase(padded, _UNVERIFIABLE),
        "attributed": _attributed(words, matcher.official_bodies(words)),
        "alarmed": bool(_ALARM_WORDS.intersection(words)),
        "evacuation_claim": bool(_EVACUATION_WORDS.intersection(words))
        or _any_phrase(padded, _EVACUATION_PHRASES),
        "casualty_claim": any(p.search(plain) for p in _CASUALTIES),
        "recycled_marker": any(p.search(plain) for p in _RECYCLED),
        "old_year_reference": _old_years(plain, reference_year, gap),
        "news_report": _news_report(plain),
    }


def is_official_item(item_url: str | None, domains: Sequence[str]) -> bool:
    """True when the item itself is served from an official domain (its address, or the address
    of its source)."""
    return item_url is not None and official_hosts(links_in(item_url), domains)


def detect_signals(
    prepared: PreparedText,
    matcher: TopicMatcher,
    config: TopicConfig,
    *,
    reference_year: int,
    item_url: str | None,
    settings: ClassifierSettings,
    source_official: bool = False,
) -> tuple[Signal, ...]:
    """Every signal the text shows, sorted by name, each once, with its weight. Signals whose
    configured weight is 0 are left out.

    `source_official` says that the source of the item is itself an official domain: its
    attributions are genuine and it counts as served from one."""
    domains = config.official_domains
    linked = (
        official_hosts(links_in(prepared.raw), domains)
        or is_official_item(item_url, domains)
        or source_official
    )
    flags: dict[str, bool] = {}
    for plain, words in prepared.views:
        found = _view_flags(plain, words, matcher, reference_year, settings.old_year_gap)
        for name, value in found.items():
            flags[name] = flags.get(name, False) or value
    weights = weights_for(config)
    if not flags["alarmed"]:  # no order or alarm word: a plain attribution counts for less
        weights["official_attribution_unlinked"] = round(
            weights["official_attribution_unlinked"] * PLAIN_ATTRIBUTION_FACTOR, 4
        )
    if flags["news_report"] or linked:
        weights["casualty_claim"] = round(
            weights["casualty_claim"] * CORROBORATED_CASUALTY_FACTOR, 4
        )
    present = {
        **{name: flags[name] for name in _TEXT_SIGNALS},
        "all_caps": _shouting(prepared.raw),
        "repeated_exclamation": _exclaiming(prepared.raw),
        "official_attribution_unlinked": not linked and flags["attributed"],
        "official_link": linked,
        "obfuscated_text": bool(prepared.obfuscation),
    }
    return tuple(
        Signal(name, SIGNALS[name].label, weights[name])
        for name in sorted(present)
        if present[name] and weights[name] > 0
    )


# ---- combination ----


@dataclass(frozen=True)
class RuleOutcome:
    """The preliminary result of the rules. `risk_strength` is the evidence for the strongest
    risk label (before the prior), used for the LLM queue priority. `mixed` says that benign
    evidence was present next to a risk signal or an obfuscation and was not allowed to win."""

    label: Label
    confidence: float
    signals: tuple[Signal, ...]
    inconclusive: bool
    conflicting: bool
    risk_strength: float
    mixed: bool = False


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


def undo_prior(label: Label, confidence: float, credibility: int) -> float:
    """The inverse of `apply_prior`: the confidence before the credibility moved it (exact except
    where the clamp to [0, 1] had cut it)."""
    apply_prior(Label.UNCLASSIFIED, 0.0, credibility)  # validates the credibility
    shift = PRIOR_STEP * (credibility - NEUTRAL_CREDIBILITY)
    if label in RISK_LABELS:
        confidence += shift
    elif label is Label.NO_RISK:
        confidence -= shift
    return round(min(1.0, max(0.0, confidence)), 4)


def _evidence(signals: Sequence[Signal]) -> dict[Label, float]:
    remaining: dict[Label, float] = {}
    for signal in signals:
        if signal.label is Label.UNCLASSIFIED:
            continue  # an obfuscation argues for no label
        remaining[signal.label] = remaining.get(signal.label, 1.0) * (1.0 - signal.weight)
    return {label: round(1.0 - left, 4) for label, left in remaining.items()}


def _independent(signals: Sequence[Signal]) -> list[Signal]:
    """The signals without those that only restate another one present (see COMPANION_SIGNALS)."""
    names = {s.name for s in signals}
    return [s for s in signals if COMPANION_SIGNALS.get(s.name) not in names]


def _conflicts(signals: Sequence[Signal], top_label: Label, settings: ClassifierSettings) -> bool:
    """True when the label that won and its strongest rival are both strong and close, counting
    every claim once."""
    evidence = _evidence(_independent(signals))
    mine = evidence.get(top_label, 0.0)
    rival = max((v for label, v in evidence.items() if label is not top_label), default=0.0)
    return rival >= settings.conflict_min and round(abs(mine - rival), 4) < settings.conflict_margin


def _unsettled(
    confidence: float,
    signals: tuple[Signal, ...],
    risk: float,
    *,
    conflicting: bool = False,
    mixed: bool = False,
) -> RuleOutcome:
    return RuleOutcome(Label.UNCLASSIFIED, confidence, signals, True, conflicting, risk, mixed)


def combine(
    signals: Sequence[Signal],
    *,
    credibility: int,
    settings: ClassifierSettings,
    neutral: bool = False,
) -> RuleOutcome:
    """Turn signals into a label, a confidence and the inconclusive/conflicting flags.

    `neutral` says the text is on topic and long enough to be called harmless when nothing at all
    argues for risk (see the module doc)."""
    apply_prior(Label.UNCLASSIFIED, 0.0, credibility)  # validates the credibility
    ordered = tuple(sorted(signals, key=lambda s: s.name))
    evidence = _evidence(ordered)
    risk_signals = [s for s in ordered if s.label in RISK_LABELS]
    obfuscation = [s.weight for s in ordered if s.name == "obfuscated_text"]
    risk = max(
        [v for label, v in evidence.items() if label in RISK_LABELS] + obfuscation, default=0.0
    )
    benign = evidence.get(Label.NO_RISK, 0.0)
    if not risk_signals and not obfuscation:
        base = max(benign, settings.neutral_confidence) if neutral else benign
        if base < settings.no_risk_settled_min:
            return _unsettled(benign, ordered, 0.0)
        confidence = apply_prior(Label.NO_RISK, base, credibility)
        return RuleOutcome(Label.NO_RISK, confidence, ordered, False, False, 0.0)
    blocking = bool(obfuscation) or any(s.name not in WEAK_SIGNALS for s in risk_signals)
    mixed = blocking and benign > 0
    if mixed:
        evidence = {label: v for label, v in evidence.items() if label is not Label.NO_RISK}
        risk = max(risk, MIXED_RISK_FLOOR)
    if not evidence:
        return _unsettled(0.0, ordered, risk, conflicting=mixed, mixed=mixed)
    ranked = sorted(evidence.items(), key=lambda item: (-item[1], _LABEL_ORDER.index(item[0])))
    top_label, top = ranked[0]
    floor = (
        settings.no_risk_settled_min if top_label is Label.NO_RISK else settings.inconclusive_below
    )
    if top < floor:
        return _unsettled(top, ordered, risk, conflicting=mixed, mixed=mixed)
    conflicting = mixed or _conflicts(ordered, top_label, settings)
    confidence = apply_prior(top_label, top, credibility)
    return RuleOutcome(top_label, confidence, ordered, False, conflicting, risk, mixed)


def evaluate_rules(
    prepared: PreparedText,
    matcher: TopicMatcher,
    config: TopicConfig,
    *,
    credibility: int,
    reference_year: int,
    item_url: str | None,
    settings: ClassifierSettings,
    source_official: bool = False,
) -> RuleOutcome:
    """The whole stage: detect the signals, then combine them."""
    signals = detect_signals(
        prepared,
        matcher,
        config,
        reference_year=reference_year,
        item_url=item_url,
        settings=settings,
        source_official=source_official,
    )
    neutral = len(prepared.tokens) >= settings.neutral_min_tokens and reads_spanish(prepared.tokens)
    return combine(signals, credibility=credibility, settings=settings, neutral=neutral)


# ---- decisions on a result ----


def needs_escalation(
    *,
    in_scope: bool,
    confidence: float,
    inconclusive: bool,
    conflicting: bool,
    duplicate_of_judged: bool,
    settings: ClassifierSettings,
    label: Label | None = None,
) -> bool:
    """RC-7: send to the LLM an in-scope item that the earlier stages could not settle
    (inconclusive, conflicting, or a risk label with a confidence under escalate_high; for an
    unlabelled confidence, in [escalate_low, escalate_high)) unless it duplicates an item already
    judged. A confidence of 0.8 or more is settled; 0.4 is not. A `no_risk` verdict has no band:
    the rules only give it when nothing argues for risk, so it is settled."""
    if not in_scope or duplicate_of_judged:
        return False
    in_band = settings.escalate_low <= confidence < settings.escalate_high
    if label is Label.NO_RISK:
        in_band = False
    elif label in RISK_LABELS:
        # A risk label always rests on evidence from `inconclusive_below` up. The credibility prior
        # may move its confidence a little under `escalate_low`; that must not let it slip between
        # "inconclusive" and the band (no dead zone), so only the upper bound applies.
        in_band = confidence < settings.escalate_high
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
        if outcome.conflicting and not outcome.mixed:
            head += "; conflicting signals"
    if outcome.mixed:
        head += "; benign evidence next to a risk signal"
    topics = ", ".join(t.value for t in topic.topics)
    matched = ", ".join(f'"{t[:_TERM_SHOWN_CHARS]}"' for t in topic.matched_terms[:_MATCHED_SHOWN])
    return f"{head}; topics: {topics}; matched: {matched}"[:MAX_RATIONALE_CHARS]
