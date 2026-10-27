"""The classifier and the topic configuration service (application layer).

`ClassifierService.classify_pending` runs the CPU stages of the cascade (RC-1) over the items that
have not been classified yet, oldest published first:

1. TOPIC: an item with no earthquake, secretariat or other-disaster term is `out_of_scope` (and
   counted by reason); one with no words at all is `unclassified`.
2. RULES: weighted heuristic signals, the credibility prior, the escalation decision.
3. DEDUP: content seen before. A match first seen/published STRICTLY more than `recycle_days`
   earlier is `recycled_content` (it references the original); a match within the window is a
   duplicate that reuses the earlier verdict and is not escalated again.
4. CATALOG: the cross-check port (`unavailable` until a real one is wired).

The LLM stage is a later unit: an item the cascade cannot settle is only MARKED (`needs_llm` and a
priority) here.

Rules of the road
- PC-2: an item is processed only while global analysis is on, its source exists and is enabled,
  and the source's analysis switch is on. That is re-checked inside the item's own transaction, so
  a source disabled in the middle of a run stops being classified at the next item and the items it
  left behind stay untouched.
- One item is one atomic unit: its verdict, its signature and its new status are written together
  or not at all. A verdict is stored once per item, so a repeated or racing run (two processes)
  never stores two: the one that loses the race counts the item as skipped. A storage failure
  ends the run (counted as an error) and the next run resumes.
- Processing is deterministic and independent of arrival order: items are taken in publication
  order (then by id), and every decision is a pure function of the item, the stored history and
  the configuration.
- Item text is untrusted data. It is analysed, never interpreted, and a rationale only ever
  contains signal names, topics and matched configuration terms.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta

from social_tracker.classification.domain import (
    Classification,
    ClassificationStats,
    ClassifierSettings,
    ItemSignature,
    Label,
    OutOfScopeReason,
    Stage,
    Topic,
    TopicVerdict,
)
from social_tracker.classification.ports import (
    CatalogCheck,
    ClassificationRepository,
    SignatureRepository,
    TopicConfigRepository,
    UnavailableCatalog,
)
from social_tracker.classification.rules import (
    RuleOutcome,
    apply_prior,
    evaluate_rules,
    is_low_confidence,
    needs_escalation,
    priority,
    rationale_for_rules,
)
from social_tracker.classification.similarity import band_keys, signature, similarity
from social_tracker.classification.text import PreparedText, prepare
from social_tracker.classification.topics import StoredTopicConfig, TopicConfig, TopicMatcher
from social_tracker.classification.yaml_io import dump_topic_config, load_topic_config
from social_tracker.control.domain import (
    ControlEvent,
    event_detail_json,
    validate_actor,
    validate_reason,
)
from social_tracker.control.ports import ControlRepository
from social_tracker.control.service import ControlService
from social_tracker.items.domain import Item, ItemStatus, validate_limit
from social_tracker.items.ports import ItemRepository
from social_tracker.shared.clock import Clock
from social_tracker.shared.errors import (
    DomainError,
    NotFoundError,
    StorageError,
    ValidationError,
)
from social_tracker.shared.ids import validate_slug
from social_tracker.shared.uow import UnitOfWork
from social_tracker.sources.ports import SourceRepository

CLASSIFIER_VERSION = "cascade-1"
EVENT_TOPIC_CONFIG = "topic_config_changed"
DEFAULT_BATCH = 200
CANDIDATE_LIMIT = 200  # signatures compared per item: the oldest that share a band or the hash
EXACT_RECYCLED_CONFIDENCE = 0.9  # identical content; a near copy scales it by its similarity
NOT_CHECKED = "not_checked"
UNAVAILABLE = "unavailable"
_LOG = logging.getLogger("social_tracker.classifier")


# ---- the topic configuration ----


@dataclass(frozen=True)
class CurrentConfig:
    version: int  # 0: the defaults shipped with the application
    config: TopicConfig


@dataclass(frozen=True)
class ConfigChange:
    version: int
    changed: bool


class TopicConfigService:
    """The editable topic vocabulary: versioned, audited, atomic.

    Every real change appends a version AND one control event (actor, reason, from, to) in the
    same transaction. A request that changes nothing (the same configuration again, or a reset
    while the defaults are in force) writes neither. With no stored version the defaults given at
    construction are in force (version 0)."""

    def __init__(
        self,
        repo: TopicConfigRepository,
        control: ControlRepository,
        uow: UnitOfWork,
        clock: Clock,
        default: TopicConfig,
    ) -> None:
        self._repo = repo
        self._control = control
        self._uow = uow
        self._clock = clock
        self._default = default

    def current(self) -> CurrentConfig:
        latest = self._repo.latest()
        if latest is None:
            return CurrentConfig(0, self._default)
        return CurrentConfig(latest.version, latest.config)

    def show_text(self) -> str:
        """The configuration in force as YAML (what `set_text` accepts back)."""
        return dump_topic_config(self.current().config)

    def set_text(self, text: str, *, actor: str, reason: str) -> ConfigChange:
        return self.set_config(load_topic_config(text), actor=actor, reason=reason)

    def reset(self, *, actor: str, reason: str) -> ConfigChange:
        return self.set_config(self._default, actor=actor, reason=reason)

    def set_config(self, config: TopicConfig, *, actor: str, reason: str) -> ConfigChange:
        actor, reason = validate_actor(actor), validate_reason(reason)
        if not isinstance(config, TopicConfig):
            raise ValidationError("a topic configuration is required")
        with self._uow.transaction():
            now = self._clock.now()
            current = self.current()
            if current.config == config:
                return ConfigChange(current.version, changed=False)
            version = self._repo.append(StoredTopicConfig(config, now, actor, reason))
            detail = event_detail_json(
                {
                    "action": "set",
                    "actor": actor,
                    "reason": reason,
                    "from": current.version,
                    "to": version,
                }
            )
            self._control.append_control_event(ControlEvent(now, EVENT_TOPIC_CONFIG, None, detail))
        return ConfigChange(version, changed=True)


# ---- the classifier ----


@dataclass(frozen=True)
class ClassifyReport:
    """What one `classify_pending` run did. `skipped` are items that stopped being eligible (or
    were taken by another run) between listing and processing; `errors` are storage failures
    (the run stops at the first and the next run resumes)."""

    classified: int = 0
    out_of_scope: int = 0
    skipped: int = 0
    errors: int = 0
    needs_llm: int = 0
    eligible_sources: int = 0


@dataclass(frozen=True)
class _Original:
    signature: ItemSignature
    verdict: Classification
    similarity: float
    exact: bool


def reference_time(item: Item):
    """The instant an item's content counts from: when it was published or first seen here,
    whichever is earlier (a feed cannot claim a later date than we saw it)."""
    return min(item.published_at, item.fetched_at)


def _status_for(label: Label) -> ItemStatus:
    return ItemStatus.OUT_OF_SCOPE if label is Label.OUT_OF_SCOPE else ItemStatus.CLASSIFIED


class ClassifierService:
    def __init__(
        self,
        items: ItemRepository,
        sources: SourceRepository,
        control: ControlService,
        classifications: ClassificationRepository,
        signatures: SignatureRepository,
        topics: TopicConfigService,
        uow: UnitOfWork,
        clock: Clock,
        *,
        catalog: CatalogCheck | None = None,
        settings: ClassifierSettings | None = None,
    ) -> None:
        self._items = items
        self._sources = sources
        self._control = control
        self._classifications = classifications
        self._signatures = signatures
        self._topics = topics
        self._uow = uow
        self._clock = clock
        self._catalog = catalog if catalog is not None else UnavailableCatalog()
        self._settings = settings if settings is not None else ClassifierSettings()
        self._matcher: tuple[TopicConfig, TopicMatcher] | None = None

    # ---- reading ----

    def get(self, item_id: str) -> Classification:
        """The verdict of an item, or `NotFoundError`."""
        verdict = self._classifications.get(validate_slug(item_id))
        if verdict is None:
            raise NotFoundError("this item has no classification")
        return verdict

    def stats(self) -> ClassificationStats:
        return self._classifications.stats()

    # ---- running ----

    def classify_pending(self, limit: int = DEFAULT_BATCH) -> ClassifyReport:
        """Classify up to `limit` pending items of the eligible sources, oldest published first."""
        validate_limit(limit)
        overview = self._control.overview()
        eligible = sorted(o.source.id for o in overview.sources if o.effective)
        if not eligible:
            return ClassifyReport()
        current = self._topics.current()
        matcher = self._matcher_for(current.config)
        version = f"{CLASSIFIER_VERSION}/topics-{current.version}"
        pending = self._items.list_by_status(ItemStatus.NEW, limit=limit, source_ids=eligible)
        counts = {"classified": 0, "out_of_scope": 0, "skipped": 0, "errors": 0, "needs_llm": 0}
        for item in pending:
            try:
                verdict = self._classify_one(
                    item.id, current.config, matcher, version, overview.alert_min_confidence
                )
            except StorageError as exc:
                _LOG.warning("classification stopped by a storage error: %s", exc.code)
                counts["errors"] += 1
                break
            except DomainError as exc:
                _LOG.warning("an item could not be classified: %s", exc.code)
                counts["errors"] += 1
                continue
            if verdict is None:
                counts["skipped"] += 1
            elif verdict.label is Label.OUT_OF_SCOPE:
                counts["out_of_scope"] += 1
            else:
                counts["classified"] += 1
                counts["needs_llm"] += verdict.needs_llm
        self._forget_old_signatures()
        return ClassifyReport(eligible_sources=len(eligible), **counts)

    def _matcher_for(self, config: TopicConfig) -> TopicMatcher:
        if self._matcher is None or self._matcher[0] != config:
            self._matcher = (config, TopicMatcher(config))
        return self._matcher[1]

    def _forget_old_signatures(self) -> None:
        cutoff = self._clock.now() - timedelta(days=self._settings.signature_retention_days)
        try:
            self._signatures.prune_before(cutoff)
        except DomainError as exc:  # housekeeping must never lose the report of the run
            _LOG.warning("old signatures could not be forgotten: %s", exc.code)

    def _credibility(self, source_id: str) -> int | None:
        """The credibility of the source if it is still eligible, else None (PC-2)."""
        try:
            if not self._control.effective_analysis(source_id):
                return None
            return self._sources.get(source_id).credibility
        except NotFoundError:
            return None

    def _classify_one(
        self,
        item_id: str,
        config: TopicConfig,
        matcher: TopicMatcher,
        version: str,
        alert_min: float,
    ) -> Classification | None:
        """Classify one item atomically; None when it is no longer eligible or was taken."""
        with self._uow.transaction():
            item = self._items.get(item_id)
            if item is None or item.status is not ItemStatus.NEW:
                return None
            credibility = self._credibility(item.source_id)
            if credibility is None:
                return None
            verdict, sign = self._decide(item, credibility, config, matcher, version, alert_min)
            if not self._classifications.add(verdict):
                existing = self._classifications.get(item_id)
                if existing is not None:  # a verdict without its status: finish that write
                    self._items.set_status(item_id, _status_for(existing.label))
                return None
            if sign is not None:
                self._signatures.add(sign)
            self._items.set_status(item_id, _status_for(verdict.label))
            return verdict

    # ---- the cascade ----

    def _decide(
        self,
        item: Item,
        credibility: int,
        config: TopicConfig,
        matcher: TopicMatcher,
        version: str,
        alert_min: float,
    ) -> tuple[Classification, ItemSignature | None]:
        settings = self._settings
        prepared = prepare(item.title, item.text, max_chars=settings.max_text_chars)
        common = {
            "item_id": item.id,
            "classified_at": self._clock.now(),
            "classifier_version": version,
            "truncated": prepared.truncated,
            "source_credibility_used": credibility,
        }
        topic = matcher.evaluate(prepared.tokens)
        if not topic.in_scope:
            return self._without_topic(topic, common), None
        outcome = evaluate_rules(
            prepared,
            matcher,
            config,
            credibility=credibility,
            reference_year=reference_time(item).year,
            item_url=item.url,
            settings=settings,
        )
        sign = signature(prepared.tokens)
        original = self._find_original(item, sign)
        age = reference_time(item) - original.signature.ref_at if original else None
        if original is not None and age <= timedelta(days=settings.recycle_days):
            return self._duplicate(topic, original, alert_min, common), None
        catalog = self._catalog_status(topic, prepared, item)
        if original is None or age is None:
            verdict = self._from_rules(outcome, topic, catalog, alert_min, common)
            return verdict, ItemSignature(item.id, item.content_hash, reference_time(item), sign)
        return self._recycled(outcome, topic, original, age, catalog, alert_min, common), None

    def _without_topic(self, topic: TopicVerdict, common: dict) -> Classification:
        if topic.reason is OutOfScopeReason.EMPTY_TEXT:
            return Classification(
                label=Label.UNCLASSIFIED,
                confidence=0.0,
                stage=Stage.TOPIC,
                rationale="no words to classify",
                low_confidence=True,
                catalog_status=NOT_CHECKED,
                **common,
            )
        crime = topic.reason is OutOfScopeReason.CRIME_PUBLIC_ORDER
        rationale = (
            "out of scope: crime or public order, no disaster term"
            if crime
            else "out of scope: no earthquake, secretariat or disaster term"
        )
        return Classification(
            label=Label.OUT_OF_SCOPE,
            confidence=1.0,
            stage=Stage.TOPIC,
            rationale=rationale,
            catalog_status=NOT_CHECKED,
            scope_reason=topic.reason,
            **common,
        )

    def _from_rules(
        self,
        outcome: RuleOutcome,
        topic: TopicVerdict,
        catalog: str,
        alert_min: float,
        common: dict,
    ) -> Classification:
        escalate = needs_escalation(
            in_scope=True,
            confidence=outcome.confidence,
            inconclusive=outcome.inconclusive,
            conflicting=outcome.conflicting,
            duplicate_of_judged=False,
            settings=self._settings,
        )
        return Classification(
            label=outcome.label,
            confidence=outcome.confidence,
            stage=Stage.RULES,
            rationale=rationale_for_rules(outcome, topic),
            signals=outcome.signals,
            topics=topic.topics,
            low_confidence=is_low_confidence(outcome.confidence, alert_min),
            needs_llm=escalate,
            catalog_status=catalog,
            priority=self._priority(escalate, outcome.risk_strength, topic),
            **common,
        )

    def _recycled(
        self,
        outcome: RuleOutcome,
        topic: TopicVerdict,
        original: _Original,
        age: timedelta,
        catalog: str,
        alert_min: float,
        common: dict,
    ) -> Classification:
        base = (
            EXACT_RECYCLED_CONFIDENCE
            if original.exact
            else EXACT_RECYCLED_CONFIDENCE * original.similarity
        )
        confidence = apply_prior(
            Label.RECYCLED_CONTENT, round(base, 4), common["source_credibility_used"]
        )
        judged = original.verdict.stage is Stage.LLM
        escalate = needs_escalation(
            in_scope=True,
            confidence=confidence,
            inconclusive=False,
            conflicting=False,
            duplicate_of_judged=judged,
            settings=self._settings,
        )
        how = "identical" if original.exact else f"{original.similarity:.0%} similar"
        return Classification(
            label=Label.RECYCLED_CONTENT,
            confidence=confidence,
            stage=Stage.DEDUP,
            rationale=(
                f"recycled content: {how} to {original.verdict.item_id}, "
                f"seen {age.days} days earlier"
            ),
            signals=outcome.signals,
            topics=topic.topics,
            low_confidence=is_low_confidence(confidence, alert_min),
            needs_llm=escalate,
            recycled_of=original.verdict.item_id,
            catalog_status=catalog,
            priority=self._priority(escalate, confidence, topic),
            **common,
        )

    def _duplicate(
        self,
        topic: TopicVerdict,
        original: _Original,
        alert_min: float,
        common: dict,
    ) -> Classification:
        earlier = original.verdict
        return Classification(
            label=earlier.label,
            confidence=earlier.confidence,
            stage=Stage.DEDUP,
            rationale=f"duplicate of {earlier.item_id}: reused its {earlier.label.value} verdict",
            signals=earlier.signals,
            topics=topic.topics,
            low_confidence=is_low_confidence(earlier.confidence, alert_min),
            duplicate_of=earlier.item_id,
            catalog_status=earlier.catalog_status,
            **common,
        )

    @staticmethod
    def _priority(escalate: bool, strength: float, topic: TopicVerdict) -> float:
        if not escalate:
            return 0.0
        return priority(strength, topic.score, secretariat=Topic.SECRETARIAT in topic.topics)

    def _find_original(self, item: Item, sign: tuple[int, ...]) -> _Original | None:
        """The oldest earlier item with the same content or a near copy of it, with its verdict."""
        candidates = self._signatures.find_candidates(
            item_id=item.id,
            content_hash=item.content_hash,
            band_keys=band_keys(sign),
            limit=CANDIDATE_LIMIT,
        )
        for candidate in candidates:  # oldest first
            exact = candidate.content_hash == item.content_hash
            score = 1.0 if exact else similarity(sign, candidate.minhash)
            if not exact and score < self._settings.near_duplicate_threshold:
                continue
            verdict = self._classifications.get(candidate.item_id)
            if verdict is not None:
                return _Original(candidate, verdict, score, exact)
        return None

    def _catalog_status(self, topic: TopicVerdict, prepared: PreparedText, item: Item) -> str:
        """The outcome of the catalog cross-check; only earthquake items are checked (RC-6)."""
        if Topic.EARTHQUAKE not in topic.topics:
            return NOT_CHECKED
        try:
            status = self._catalog.check(prepared.raw, item.published_at).status
        except Exception:  # noqa: BLE001 - a catalog that fails must never block a classification
            return UNAVAILABLE
        return status if _valid_status(status) else UNAVAILABLE


def _valid_status(status: object) -> bool:
    return (
        isinstance(status, str)
        and 1 <= len(status) <= 32
        and all(ch == "_" or "a" <= ch <= "z" for ch in status)
    )
