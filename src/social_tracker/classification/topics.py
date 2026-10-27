"""The topic configuration and the topic filter (RC-1 stage 1, RC-2). Pure, no I/O.

`TopicConfig` is the editable vocabulary: keyword sets per topic (earthquake, secretariat, other
natural disasters), place terms, crime terms that mark an item as out of scope, the official
bodies and domains used by the attribution rule, and optional signal weight overrides. It is
validated strictly (`from_mapping`) and canonical: terms are kept in their folded form (lowercase,
no accents or punctuation) and sorted, so two configurations that differ only in order, case or
accents are equal.

Matching works on folded tokens (see `text.fold`): whole words and whole phrases only, ignoring
case, accents, punctuation and invisible characters. A place term alone never puts an item in
scope; it only raises the score of an item that is. A crime term puts an item out of scope only
when no disaster term is present.

Two earthquake words are ambiguous alone and count only with context: `magnitud` (a number right
after it, or an earthquake term nearby: "la magnitud del problema" is not a quake) and `replica` /
`replicas` (an earthquake term nearby: "el alcalde replica a las criticas" is not an aftershock).
"Nearby" is `CONTEXT_WINDOW` words on either side. An item with no topic is counted as
`non_spanish` instead of `no_topic_match` when it reads as English.
"""

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from social_tracker.classification.domain import SIGNALS, OutOfScopeReason, Topic, TopicVerdict
from social_tracker.classification.text import PreparedText, fold
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError

FORMAT_VERSION = 1
MAX_TERMS_PER_GROUP = 500
MAX_TERM_CHARS = 80
MAX_TERM_WORDS = 12
MAX_STORED_BYTES = 256 * 1024
MAX_ACTOR_CHARS = 100
MAX_REASON_CHARS = 500

TOPIC_GROUPS = ("earthquake", "secretariat", "other_disaster")
_TERM_GROUPS = (*TOPIC_GROUPS, "places", "exclude", "official_bodies")
_FREE_GROUPS = ("official_domains", "official_channel_hints")
_KEYS = frozenset({"version", "weights", *_TERM_GROUPS, *_FREE_GROUPS})
_EXCLUSIVE_GROUPS = (*TOPIC_GROUPS, "places", "exclude")  # a term lives in at most one of these
_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
CONTEXT_WINDOW = 8
MAGNITUDE_REACH = 3  # words after "magnitud" where its number may stand
CONTEXTUAL_TERMS = frozenset({"magnitud", "replica", "replicas"})


def _word_set(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Names that are a whole registry, not an organisation: nobody owns them, so a link "to" one of them
# proves nothing. Not exhaustive; a single label (a bare top-level domain) is refused anyway.
PUBLIC_SUFFIXES = _word_set(
    "com org net gov edu mil int info biz name pro co io app dev xyz online site top club "
    "news live com.co org.co net.co gov.co edu.co mil.co nom.co info.co co.uk org.uk gov.uk "
    "ac.uk com.br gov.br com.mx gob.mx com.ar gob.ar com.pe gob.pe com.ec gob.ec com.ve "
    "gob.ve com.es gob.es co.za co.jp com.au gov.au co.in gov.in github.io blogspot.com "
    "wordpress.com herokuapp.com vercel.app netlify.app pages.dev web.app firebaseapp.com"
)
_ENGLISH_WORDS = _word_set(
    "the and is are was were of to in that it for on with as this from at be by have has not "
    "you they will an or but we their there been what when who how which about into than "
    "then these those would could should"
)
_SPANISH_WORDS = _word_set(
    "el la los las de del en y que se un una por con para es son al lo como mas pero sus le "
    "ya o este esta si porque muy sin sobre entre cuando tambien hasta hay donde quien desde "
    "todo nos"
)
_SECRETARY = "secretaria"
_OF = frozenset({"de", "del"})
_BASE_SCORE = 0.5
_PER_EXTRA_TERM = 0.15
_PER_PLACE = 0.1
_MAX_PLACES_COUNTED = 2


def _bad(message: str, field: str | None = None) -> ValidationError:
    return ValidationError(message, (field,) if field else ())


def _text_items(data: Mapping[str, Any], group: str) -> list[str]:
    value = data.get(group, [])
    if not isinstance(value, list | tuple) or not all(isinstance(t, str) for t in value):
        raise _bad(f"'{group}' must be a list of text entries", group)
    if len(value) > MAX_TERMS_PER_GROUP:
        raise _bad(f"'{group}' has more than {MAX_TERMS_PER_GROUP} entries", group)
    return list(value)


def _check_term(term: str, group: str) -> str:
    """The folded form of a keyword, after the checks every keyword must pass."""
    if len(term) > MAX_TERM_CHARS or any(ord(ch) < 32 or 127 <= ord(ch) < 160 for ch in term):
        raise _bad(f"an entry of '{group}' is too long or has control characters", group)
    folded = fold(term)
    if not folded:
        raise _bad(f"an entry of '{group}' has no word in it", group)
    if len(folded.split()) > MAX_TERM_WORDS:
        raise _bad(f"an entry of '{group}' has more than {MAX_TERM_WORDS} words", group)
    return folded


def _terms(data: Mapping[str, Any], group: str) -> tuple[tuple[str, ...], set[str]]:
    """The terms of a group in their folded form (that is how they match), sorted."""
    seen: set[str] = set()
    for term in _text_items(data, group):
        folded = _check_term(term, group)
        if folded in seen:
            raise _bad(f"'{group}' lists the same term twice (ignoring case and accents)", group)
        seen.add(folded)
    return tuple(sorted(seen)), seen


def _valid_domain(host: str) -> bool:
    """A registrable name: at least two labels, valid labels, not an address, not a public suffix
    (a link to `com` or `gov.co` proves nothing about who wrote it)."""
    labels = host.split(".")
    return (
        len(host) <= 253
        and len(labels) >= 2
        and all(_LABEL_RE.fullmatch(label) for label in labels)
        and not all(label.isdigit() for label in labels)
        and host not in PUBLIC_SUFFIXES
    )


def _domains(data: Mapping[str, Any]) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for domain in _text_items(data, "official_domains"):
        host = domain.strip().lower()
        if not _valid_domain(host) or host in seen:
            raise _bad(
                "'official_domains' must list distinct registrable domain names with at least "
                "two labels, not a public suffix such as com or gov.co",
                "official_domains",
            )
        seen[host] = host
    return tuple(sorted(seen))


def _hints(data: Mapping[str, Any]) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for hint in _text_items(data, "official_channel_hints"):
        text = hint.strip()
        key = text.casefold()
        printable = all(ord(ch) >= 32 and not 127 <= ord(ch) < 160 for ch in text)
        if not text or len(text) > MAX_TERM_CHARS or not printable or key in seen:
            raise _bad(
                "'official_channel_hints' must list distinct short texts", "official_channel_hints"
            )
        seen[key] = text
    return tuple(seen[k] for k in sorted(seen))


def _weights(data: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    raw = data.get("weights", {})
    if not isinstance(raw, Mapping):
        raise _bad("'weights' must be a mapping of signal name to weight", "weights")
    for name, weight in raw.items():
        known = isinstance(name, str) and name in SIGNALS
        number = isinstance(weight, int | float) and not isinstance(weight, bool)
        if not known or not number or not 0.0 <= weight <= 1.0:
            raise _bad("'weights' maps known signal names to numbers from 0 to 1", "weights")
    return tuple(sorted((name, float(weight)) for name, weight in raw.items()))


@dataclass(frozen=True)
class TopicConfig:
    earthquake: tuple[str, ...] = ()
    secretariat: tuple[str, ...] = ()
    other_disaster: tuple[str, ...] = ()
    places: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    official_bodies: tuple[str, ...] = ()
    official_domains: tuple[str, ...] = ()
    official_channel_hints: tuple[str, ...] = ()
    weights: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_mapping(cls, data: Any) -> "TopicConfig":
        """The configuration a mapping describes, or `ValidationError`. Strict: unknown keys,
        a wrong format version, duplicates (ignoring case and accents), a term in two groups,
        oversize lists or terms, and unknown or out-of-range weights are all refused."""
        if not isinstance(data, Mapping):
            raise _bad("the topic configuration must be a mapping")
        unknown = set(data) - _KEYS
        if unknown or not all(isinstance(k, str) for k in data):
            raise _bad("the topic configuration has an unknown key")
        version = data.get("version")
        if isinstance(version, bool) or version != FORMAT_VERSION:
            raise _bad(f"'version' must be {FORMAT_VERSION}", "version")
        groups: dict[str, tuple[str, ...]] = {}
        owner: dict[str, str] = {}
        for group in _TERM_GROUPS:
            groups[group], keyed = _terms(data, group)
            if group in _EXCLUSIVE_GROUPS:
                for folded in keyed:
                    if owner.setdefault(folded, group) != group:
                        raise _bad(
                            f"a term is listed in both '{owner[folded]}' and '{group}'", group
                        )
        if not any(groups[g] for g in TOPIC_GROUPS):
            raise _bad("at least one earthquake, secretariat or other_disaster term is required")
        return cls(
            **groups,
            official_domains=_domains(data),
            official_channel_hints=_hints(data),
            weights=_weights(data),
        )

    def to_mapping(self) -> dict[str, Any]:
        """The canonical mapping (what `from_mapping` accepts; the inverse of it)."""
        out: dict[str, Any] = {"version": FORMAT_VERSION}
        for group in (*_TERM_GROUPS, *_FREE_GROUPS):
            out[group] = list(getattr(self, group))
        out["weights"] = dict(self.weights)
        return out

    def to_json(self) -> str:
        """The canonical JSON stored with each version (compact, key-sorted, real characters)."""
        text = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        if len(text.encode("utf-8")) > MAX_STORED_BYTES:
            raise _bad(f"the topic configuration is larger than {MAX_STORED_BYTES} bytes")
        return text

    @classmethod
    def from_json(cls, text: str) -> "TopicConfig":
        try:
            return cls.from_mapping(json.loads(text))
        except (ValueError, RecursionError):
            raise _bad("the stored topic configuration is not valid JSON") from None


@dataclass(frozen=True)
class StoredTopicConfig:
    """One version of the configuration in the history: who set it, when and why. `version` is
    assigned by the store (0 until then)."""

    config: TopicConfig
    created_at: datetime
    actor: str
    reason: str
    version: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.config, TopicConfig):
            raise _bad("a stored configuration needs a TopicConfig")
        object.__setattr__(self, "created_at", ensure_utc(self.created_at))
        for value, limit in ((self.actor, MAX_ACTOR_CHARS), (self.reason, MAX_REASON_CHARS)):
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise _bad("actor and reason must be short non-blank text")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 0:
            raise _bad("version must be a whole number")


class _PhraseIndex:
    """Phrases looked up by their first word: a scan costs one dict lookup per token."""

    def __init__(self, terms: Sequence[str]) -> None:
        by_first: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for term in {fold(t) for t in terms}:
            words = tuple(term.split())
            by_first[words[0]].append(words)
        # sorted once here, so a scan never sorts and its order does not depend on set order
        self._by_first = {first: sorted(phrases) for first, phrases in by_first.items()}

    def find(self, words: Sequence[str]) -> list[tuple[int, int, str]]:
        """Every `(start, end, folded term)` occurrence, in text order."""
        found: list[tuple[int, int, str]] = []
        for start, word in enumerate(words):
            for phrase in self._by_first.get(word, ()):
                end = start + len(phrase)
                if tuple(words[start:end]) == phrase:
                    found.append((start, end, " ".join(phrase)))
        return found


def _has_number(words: Sequence[str]) -> bool:
    return any(any(ch.isdigit() for ch in word) for word in words)


def _reads_english(words: Sequence[str]) -> bool:
    english = sum(word in _ENGLISH_WORDS for word in words)
    spanish = sum(word in _SPANISH_WORDS for word in words)
    return english >= 2 and english > 2 * spanish


def reads_spanish(words: Sequence[str]) -> bool:
    """True when enough of the words are common Spanish function words for the text to be
    sentences rather than a pile of tokens."""
    spanish = sum(word in _SPANISH_WORDS for word in words)
    return spanish >= 1 and spanish * 10 >= len(words)


class TopicMatcher:
    def __init__(self, config: TopicConfig) -> None:
        self._topics = {
            Topic(group): _PhraseIndex(getattr(config, group)) for group in TOPIC_GROUPS
        }
        self._places = _PhraseIndex(config.places)
        self._exclude = _PhraseIndex(config.exclude)
        self._bodies = _PhraseIndex((*config.official_bodies, *config.secretariat))

    def _topic_hits(self, topic: Topic, words: Sequence[str]) -> set[str]:
        found = self._topics[topic].find(words)
        if topic is not Topic.EARTHQUAKE:
            return {term for _, _, term in found}
        plain = [(start, end) for start, end, term in found if term not in CONTEXTUAL_TERMS]
        hits = {term for _, _, term in found if term not in CONTEXTUAL_TERMS}
        for start, end, term in found:
            if term not in CONTEXTUAL_TERMS:
                continue
            nearby = any(
                other_start < end + CONTEXT_WINDOW and other_end > start - CONTEXT_WINDOW
                for other_start, other_end in plain
            )
            numbered = term == "magnitud" and _has_number(words[end : end + MAGNITUDE_REACH])
            if nearby or numbered:
                hits.add(term)
        return hits

    def evaluate(self, words: Sequence[str]) -> TopicVerdict:
        """Decide the scope of folded `words`."""
        if not words:
            return TopicVerdict(False, (), (), 0.0, OutOfScopeReason.EMPTY_TEXT)
        hits = {topic: self._topic_hits(topic, words) for topic in self._topics}
        topics = tuple(topic for topic in Topic if hits[topic])
        if not topics:
            if self._exclude.find(words):
                reason = OutOfScopeReason.CRIME_PUBLIC_ORDER
            elif _reads_english(words):
                reason = OutOfScopeReason.NON_SPANISH
            else:
                reason = OutOfScopeReason.NO_TOPIC_MATCH
            return TopicVerdict(False, (), (), 0.0, reason)
        strong = set().union(*hits.values())
        places = {t for _, _, t in self._places.find(words)}
        score = _BASE_SCORE + _PER_EXTRA_TERM * (len(strong) - 1)
        score += _PER_PLACE * min(len(places), _MAX_PLACES_COUNTED)
        return TopicVerdict(True, topics, tuple(sorted(strong | places)), round(min(score, 1.0), 4))

    def evaluate_prepared(self, prepared: PreparedText) -> TopicVerdict:
        """The scope of a prepared text: as written, or else as decoded (a word spelled around the
        vocabulary, such as `t3mblor`, is still the word)."""
        verdict = self.evaluate(prepared.tokens)
        if verdict.in_scope or prepared.decoded == prepared.plain:
            return verdict
        decoded = self.evaluate(prepared.decoded_tokens)
        return decoded if decoded.in_scope else verdict

    def official_bodies(self, words: Sequence[str]) -> list[tuple[int, int]]:
        """Token spans `(start, end)` where an official body (or the secretariat) is named. A
        bare "la secretaria" followed by "de" starts the name of some other secretariat ("la
        secretaria de gestion ambiental"): it is not the risk secretariat unless the full name is
        listed, and then that longer span is found on its own."""
        spans = []
        for start, end, _ in self._bodies.find(words):
            bare = words[end - 1] == _SECRETARY and end < len(words) and words[end] in _OF
            if not bare:
                spans.append((start, end))
        return spans
