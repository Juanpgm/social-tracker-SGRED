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
"""

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from social_tracker.classification.domain import SIGNALS, OutOfScopeReason, Topic, TopicVerdict
from social_tracker.classification.text import fold
from social_tracker.shared.clock import ensure_utc
from social_tracker.shared.errors import ValidationError

FORMAT_VERSION = 1
MAX_TERMS_PER_GROUP = 500
MAX_TERM_CHARS = 80
MAX_TERM_WORDS = 12
MAX_TOTAL_TERMS = 3000
MAX_STORED_BYTES = 256 * 1024
MAX_ACTOR_CHARS = 100
MAX_REASON_CHARS = 500

TOPIC_GROUPS = ("earthquake", "secretariat", "other_disaster")
_TERM_GROUPS = (*TOPIC_GROUPS, "places", "exclude", "official_bodies")
_FREE_GROUPS = ("official_domains", "official_channel_hints")
_KEYS = frozenset({"version", "weights", *_TERM_GROUPS, *_FREE_GROUPS})
_EXCLUSIVE_GROUPS = (*TOPIC_GROUPS, "places", "exclude")  # a term lives in at most one of these
_DOMAIN_RE = re.compile(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?")
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


def _domains(data: Mapping[str, Any]) -> tuple[str, ...]:
    seen: dict[str, str] = {}
    for domain in _text_items(data, "official_domains"):
        host = domain.strip().lower()
        if not _DOMAIN_RE.fullmatch(host) or host in seen:
            raise _bad(
                "'official_domains' must list distinct bare domain names", "official_domains"
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
        if sum(map(len, groups.values())) > MAX_TOTAL_TERMS:
            raise _bad(f"the configuration has more than {MAX_TOTAL_TERMS} terms")
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
        self._by_first: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for term in {fold(t) for t in terms}:
            words = tuple(term.split())
            self._by_first[words[0]].append(words)

    def find(self, words: Sequence[str]) -> list[tuple[int, int, str]]:
        """Every `(start, end, folded term)` occurrence, in text order."""
        found: list[tuple[int, int, str]] = []
        for start, word in enumerate(words):
            for phrase in sorted(self._by_first.get(word, ())):
                end = start + len(phrase)
                if tuple(words[start:end]) == phrase:
                    found.append((start, end, " ".join(phrase)))
        return found


class TopicMatcher:
    def __init__(self, config: TopicConfig) -> None:
        self._topics = {
            Topic(group): _PhraseIndex(getattr(config, group)) for group in TOPIC_GROUPS
        }
        self._places = _PhraseIndex(config.places)
        self._exclude = _PhraseIndex(config.exclude)
        self._bodies = _PhraseIndex((*config.official_bodies, *config.secretariat))

    def evaluate(self, words: Sequence[str]) -> TopicVerdict:
        """Decide the scope of folded `words`."""
        if not words:
            return TopicVerdict(False, (), (), 0.0, OutOfScopeReason.EMPTY_TEXT)
        hits = {
            topic: {t for _, _, t in index.find(words)} for topic, index in self._topics.items()
        }
        topics = tuple(topic for topic in Topic if hits[topic])
        if not topics:
            crime = bool(self._exclude.find(words))
            reason = (
                OutOfScopeReason.CRIME_PUBLIC_ORDER if crime else OutOfScopeReason.NO_TOPIC_MATCH
            )
            return TopicVerdict(False, (), (), 0.0, reason)
        strong = set().union(*hits.values())
        places = {t for _, _, t in self._places.find(words)}
        score = _BASE_SCORE + _PER_EXTRA_TERM * (len(strong) - 1)
        score += _PER_PLACE * min(len(places), _MAX_PLACES_COUNTED)
        return TopicVerdict(True, topics, tuple(sorted(strong | places)), round(min(score, 1.0), 4))

    def official_bodies(self, words: Sequence[str]) -> list[tuple[int, int]]:
        """Token spans `(start, end)` where an official body (or the secretariat) is named."""
        return [(start, end) for start, end, _ in self._bodies.find(words)]
