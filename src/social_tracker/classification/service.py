"""The classifier and the topic configuration service (application layer).

`ClassifierService.classify_pending` runs the CPU stages of the cascade (RC-1) over the items that
have not been classified yet, oldest published first:

1. TOPIC: an item with no earthquake, secretariat or other-disaster term is `out_of_scope` (and
   counted by reason); one with no words at all is `unclassified`. A truncated text with no topic
   in the parts that were read is not called off topic: it is kept for review.
2. RULES: weighted heuristic signals, the credibility prior, the escalation decision.
3. DEDUP: content seen before, measured from when this application FIRST SAW each item (never
   from the publication date, which the feed writes). Content identical to an earlier item's:
   within `recycle_days` it is a duplicate that reuses the earlier verdict (with this source's own
   credibility prior, and flagged when the original still waits for the LLM); further apart than
   that, in either direction, it is `recycled_content`. A NEAR copy (similar, same numbers, long
   enough) never inherits a verdict: its rules run on its own text; within the window it only
   links to the original (`near_of`), beyond it it is recycled. Short texts match only identical
   content inside the window.
4. CATALOG: the cross-check port (`unavailable` until a real one is wired), resolved before the
   item's write transaction opens.

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
  ends the run (counted as a storage error) and the next run resumes. Any other error belongs to
  one item: it is counted, the run goes on, and after `max_item_failures` consecutive errors the
  item is parked as `classification_failed` so that it cannot stall the queue (`reclassify`
  retries it).
- Processing is deterministic given the history: items are taken in publication order (then by
  id), and every decision is a pure function of the item, the stored history and the
  configuration.
- Item text is untrusted data. It is analysed, never interpreted, and a rationale only ever
  contains signal names, topics and matched configuration terms.
"""

import logging
from dataclasses import dataclass, replace
from datetime import timedelta

from social_tracker.classification.domain import (
    MAX_RATIONALE_CHARS,
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
    is_official_item,
    needs_escalation,
    priority,
    rationale_for_rules,
    undo_prior,
)
from social_tracker.classification.similarity import (
    band_keys,
    numbers_digest,
    signature,
    similarity,
    text_digest,
)
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

CLASSIFIER_VERSION = "cascade-2"
EVENT_TOPIC_CONFIG = "topic_config_changed"
DEFAULT_BATCH = 200
EXACT_RECYCLED_CONFIDENCE = 0.9  # identical content; a near copy scales it by its similarity
NOT_CHECKED = "not_checked"
REVIEW_PRIORITY = 0.1  # a truncated text with no topic in what was read: last in the queue
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
    were taken by another run) between listing and processing; `errors` are items that could not
    be classified this time (they are retried, and parked as `failed` after
    `max_item_failures` consecutive errors); `storage_errors` is 1 when a storage failure ended
    the run (the next run resumes); `candidate_cap_hits` counts near-duplicate lookups that hit
    the `candidate_limit` (a sign it is too low for this corpus)."""

    classified: int = 0
    out_of_scope: int = 0
    skipped: int = 0
    errors: int = 0
    needs_llm: int = 0
    eligible_sources: int = 0
    storage_errors: int = 0
    failed: int = 0
    candidate_cap_hits: int = 0


@dataclass(frozen=True)
class _Match:
    """The earlier item a new one matches: its signature, its verdict, how alike they are, whether
    the content is identical and how far apart in time they were first seen."""

    signature: ItemSignature
    verdict: Classification
    similarity: float
    exact: bool
    gap: timedelta


@dataclass(frozen=True)
class _Source:
    credibility: int
    official: bool


def reference_time(item: Item):
    """The instant an item's content counts from: when THIS application first saw it. The
    publication date is written by the feed, which can claim anything, so it never decides
    whether content is old."""
    return item.fetched_at


def _status_for(label: Label) -> ItemStatus:
    return ItemStatus.OUT_OF_SCOPE if label is Label.OUT_OF_SCOPE else ItemStatus.CLASSIFIED


# Statuses a person can send back to the classifier with `reclassify`.
RECLASSIFIABLE = (ItemStatus.OUT_OF_SCOPE, ItemStatus.CLASSIFICATION_FAILED)
PRUNE_PAGE = 500
PRUNE_MAX_PAGES = 20


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
        self._cap_hits = 0

    # ---- reading ----

    def get(self, item_id: str) -> Classification:
        """The verdict of an item, or `NotFoundError`."""
        verdict = self._classifications.get(validate_slug(item_id))
        if verdict is None:
            raise NotFoundError("this item has no classification")
        return verdict

    def stats(self) -> ClassificationStats:
        failed = self._items.count_by_status(ItemStatus.CLASSIFICATION_FAILED)
        return replace(self._classifications.stats(), failed=failed)

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
        listed = self._items.list_by_status(ItemStatus.NEW, limit=limit, source_ids=eligible)
        # Within the batch, the one seen first here is classified first and becomes the original of
        # its copies; the publication date, which the feed writes, never decides that.
        pending = sorted(listed, key=lambda item: (item.fetched_at, item.id))
        counts = {
            "classified": 0,
            "out_of_scope": 0,
            "skipped": 0,
            "errors": 0,
            "needs_llm": 0,
            "storage_errors": 0,
            "failed": 0,
        }
        self._cap_hits = 0
        for item in pending:
            try:
                verdict = self._classify_one(
                    item.id, current.config, matcher, version, overview.alert_min_confidence
                )
            except StorageError as exc:
                _LOG.warning("classification stopped by a storage error: %s", exc.code)
                counts["storage_errors"] += 1
                break
            except Exception as exc:  # noqa: BLE001 - one poison item must not stall the queue
                _LOG.warning("an item could not be classified: %s", _reason(exc))
                counts["errors"] += 1
                try:
                    counts["failed"] += self._record_failure(item.id)
                except StorageError as storage:
                    _LOG.warning("a failure could not be recorded: %s", storage.code)
                    counts["storage_errors"] += 1
                    break
                continue
            if verdict is None:
                counts["skipped"] += 1
            elif verdict.label is Label.OUT_OF_SCOPE:
                counts["out_of_scope"] += 1
            else:
                counts["classified"] += 1
                counts["needs_llm"] += verdict.needs_llm
        self._forget_old_signatures()
        return ClassifyReport(
            eligible_sources=len(eligible), candidate_cap_hits=self._cap_hits, **counts
        )

    def reclassify(self, status: ItemStatus, limit: int = DEFAULT_BATCH) -> int:
        """Send up to `limit` items at `status` (`out_of_scope` or `classification_failed`) back
        to the queue: their verdict is forgotten and they are `new` again, so the next run
        classifies them with the vocabulary and the code in force. Returns how many."""
        if status not in RECLASSIFIABLE:
            raise ValidationError("only out_of_scope and classification_failed items can be redone")
        validate_limit(limit)
        redone = 0
        for item in self._items.list_by_status(status, limit=limit):
            with self._uow.transaction():
                current = self._items.get(item.id)
                if current is None or current.status is not status:
                    continue
                self._classifications.delete(item.id)  # also forgets its error counter
                self._classifications.clear_failure(item.id)
                self._items.set_status(item.id, ItemStatus.NEW)
                redone += 1
        return redone

    def prune_history(self) -> int:
        """Forget the verdicts older than `classification_retention_days` whose item no longer
        exists (the retention purge removed it). Verdicts of items still stored are never
        touched. Returns how many were forgotten."""
        cutoff = self._clock.now() - timedelta(days=self._settings.classification_retention_days)
        gone: list[str] = []
        for page in range(PRUNE_MAX_PAGES):
            ids = self._classifications.older_than(cutoff, PRUNE_PAGE, page * PRUNE_PAGE)
            gone.extend(item_id for item_id in ids if self._items.get(item_id) is None)
            if len(ids) < PRUNE_PAGE:
                break
        removed = 0
        for start in range(0, len(gone), PRUNE_PAGE):
            removed += self._classifications.delete_many(gone[start : start + PRUNE_PAGE])
        return removed

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

    def _record_failure(self, item_id: str) -> int:
        """Count one more consecutive error for the item; park it after the configured number.
        Returns 1 when it was parked."""
        with self._uow.transaction():
            failures = self._classifications.note_failure(item_id, self._clock.now())
            if failures < self._settings.max_item_failures:
                return 0
            self._items.set_status(item_id, ItemStatus.CLASSIFICATION_FAILED)
            self._classifications.clear_failure(item_id)
            return 1

    def _source(self, source_id: str, config: TopicConfig) -> _Source | None:
        """The credibility of the source and whether it is an official address, if the source is
        still eligible, else None (PC-2)."""
        try:
            if not self._control.effective_analysis(source_id):
                return None
            source = self._sources.get(source_id)
        except NotFoundError:
            return None
        url = getattr(source, "feed_url", None) or getattr(source, "stream_url", None)
        return _Source(source.credibility, is_official_item(url, config.official_domains))

    def _classify_one(
        self,
        item_id: str,
        config: TopicConfig,
        matcher: TopicMatcher,
        version: str,
        alert_min: float,
    ) -> Classification | None:
        """Classify one item atomically; None when it is no longer eligible or was taken.

        Everything that can be slow or fail on its own is done BEFORE the write transaction opens:
        reading the item, preparing its text, the topic filter and the catalog cross-check. The
        transaction then only re-checks eligibility, looks up the history and writes the verdict,
        the signature and the status together, so a slow catalog never holds the write lock. An
        exact copy inside the window inherits the catalog outcome of its original, so the catalog
        is not asked for it; should that copy turn out not to be one by the time the transaction
        runs (a race with a purge), its catalog outcome is `unavailable`."""
        item = self._items.get(item_id)
        if item is None or item.status is not ItemStatus.NEW:
            return None
        prepared = prepare(item.title, item.text, max_chars=self._settings.max_text_chars)
        topic = matcher.evaluate_prepared(prepared)
        sign = self._signature_of(item, prepared) if topic.in_scope else None
        inherits = sign is not None and self._is_copy(self._find_match(item, sign, track=False))
        catalog = None if inherits else self._catalog_status(topic, prepared, item)
        with self._uow.transaction():
            item = self._items.get(item_id)
            if item is None or item.status is not ItemStatus.NEW:
                return None
            source = self._source(item.source_id, config)
            if source is None:
                return None
            verdict, stored = self._decide(
                item, prepared, topic, sign, catalog, source, config, matcher, version, alert_min
            )
            if not self._classifications.add(verdict):
                existing = self._classifications.get(item_id)
                if existing is not None:  # a verdict without its status: finish that write
                    self._items.set_status(item_id, _status_for(existing.label))
                return None
            if stored is not None:
                self._signatures.add(stored)
            self._classifications.clear_failure(item_id)
            self._items.set_status(item_id, _status_for(verdict.label))
            return verdict

    # ---- the cascade ----

    def _decide(
        self,
        item: Item,
        prepared: PreparedText,
        topic: TopicVerdict,
        sign: ItemSignature | None,
        catalog: str | None,
        source: _Source,
        config: TopicConfig,
        matcher: TopicMatcher,
        version: str,
        alert_min: float,
    ) -> tuple[Classification, ItemSignature | None]:
        settings = self._settings
        common = {
            "item_id": item.id,
            "classified_at": self._clock.now(),
            "classifier_version": version,
            "truncated": prepared.truncated,
            "source_credibility_used": source.credibility,
        }
        if not topic.in_scope or sign is None:
            return self._without_topic(topic, common), None
        outcome = evaluate_rules(
            prepared,
            matcher,
            config,
            credibility=source.credibility,
            reference_year=reference_time(item).year,
            item_url=item.url,
            settings=settings,
            source_official=source.official,
        )
        match = self._find_match(item, sign)
        catalog = catalog if catalog is not None else UNAVAILABLE
        if match is None:
            verdict = self._from_rules(outcome, topic, catalog, alert_min, common)
            return verdict, sign
        if match.gap > timedelta(days=settings.recycle_days):
            return self._recycled(outcome, topic, match, catalog, alert_min, common), None
        if match.exact:
            return self._duplicate(topic, match, alert_min, common), None
        verdict = self._from_rules(outcome, topic, catalog, alert_min, common, near=match)
        return verdict, sign

    @staticmethod
    def _signature_of(item: Item, prepared: PreparedText) -> ItemSignature:
        words = prepared.tokens
        return ItemSignature(
            item.id,
            item.content_hash,
            reference_time(item),
            signature(words),
            text_digest(words),
            numbers_digest(words),
            len(words),
        )

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
        if common["truncated"]:  # part of the text was never read: it cannot be called off topic
            return Classification(
                label=Label.UNCLASSIFIED,
                confidence=0.0,
                stage=Stage.TOPIC,
                rationale="truncated text with no topic in the parts read: needs review",
                low_confidence=True,
                needs_llm=True,
                priority=REVIEW_PRIORITY,
                catalog_status=NOT_CHECKED,
                **common,
            )
        rationale = _OUT_OF_SCOPE_RATIONALE[topic.reason]
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
        near: _Match | None = None,
    ) -> Classification:
        escalate = needs_escalation(
            in_scope=True,
            confidence=outcome.confidence,
            inconclusive=outcome.inconclusive,
            conflicting=outcome.conflicting,
            duplicate_of_judged=False,
            settings=self._settings,
            label=outcome.label,
        )
        rationale = rationale_for_rules(outcome, topic)
        if near is not None:
            link = f"; near copy of {near.verdict.item_id} ({near.similarity:.0%})"
            rationale = rationale[: MAX_RATIONALE_CHARS - len(link)] + link
        return Classification(
            label=outcome.label,
            confidence=outcome.confidence,
            stage=Stage.RULES,
            rationale=rationale,
            signals=outcome.signals,
            topics=topic.topics,
            low_confidence=is_low_confidence(outcome.confidence, alert_min),
            needs_llm=escalate,
            near_of=near.verdict.item_id if near is not None else None,
            catalog_status=catalog,
            priority=self._priority(escalate, outcome.risk_strength, topic),
            **common,
        )

    def _recycled(
        self,
        outcome: RuleOutcome,
        topic: TopicVerdict,
        match: _Match,
        catalog: str,
        alert_min: float,
        common: dict,
    ) -> Classification:
        base = (
            EXACT_RECYCLED_CONFIDENCE
            if match.exact
            else EXACT_RECYCLED_CONFIDENCE * match.similarity
        )
        confidence = apply_prior(
            Label.RECYCLED_CONTENT, round(base, 4), common["source_credibility_used"]
        )
        judged = match.verdict.stage is Stage.LLM
        escalate = needs_escalation(
            in_scope=True,
            confidence=confidence,
            inconclusive=False,
            conflicting=False,
            duplicate_of_judged=judged,
            settings=self._settings,
            label=Label.RECYCLED_CONTENT,
        )
        how = "identical" if match.exact else f"{match.similarity:.0%} similar"
        return Classification(
            label=Label.RECYCLED_CONTENT,
            confidence=confidence,
            stage=Stage.DEDUP,
            rationale=(
                f"recycled content: {how} to {match.verdict.item_id}, "
                f"seen {match.gap.days} days apart"
            ),
            signals=outcome.signals,
            topics=topic.topics,
            low_confidence=is_low_confidence(confidence, alert_min),
            needs_llm=escalate,
            recycled_of=match.verdict.item_id,
            catalog_status=catalog,
            priority=self._priority(escalate, confidence, topic),
            **common,
        )

    def _duplicate(
        self,
        topic: TopicVerdict,
        match: _Match,
        alert_min: float,
        common: dict,
    ) -> Classification:
        """An exact copy within the window: the label and the evidence of the earlier verdict, with
        THIS source's credibility prior on its own record (the earlier one's prior is taken off
        first). An original still waiting for the LLM leaves the copy provisional: it is linked and
        flagged `awaiting_original`, to be settled when the original is judged."""
        earlier = match.verdict
        credibility = common["source_credibility_used"]
        confidence = earlier.confidence
        if earlier.stage is not Stage.LLM:  # a judged confidence is not a rules confidence
            base = undo_prior(earlier.label, confidence, earlier.source_credibility_used)
            confidence = apply_prior(earlier.label, base, credibility)
        waiting = earlier.needs_llm and earlier.stage is not Stage.LLM
        note = " (its original still waits for the LLM)" if waiting else ""
        return Classification(
            label=earlier.label,
            confidence=confidence,
            stage=Stage.DEDUP,
            rationale=(
                f"duplicate of {earlier.item_id}: reused its {earlier.label.value} verdict{note}"
            ),
            signals=earlier.signals,
            topics=topic.topics,
            low_confidence=is_low_confidence(confidence, alert_min),
            duplicate_of=earlier.item_id,
            awaiting_original=waiting,
            catalog_status=earlier.catalog_status,
            **common,
        )

    @staticmethod
    def _priority(escalate: bool, strength: float, topic: TopicVerdict) -> float:
        if not escalate:
            return 0.0
        return priority(strength, topic.score, secretariat=Topic.SECRETARIAT in topic.topics)

    def _is_copy(self, match: _Match | None) -> bool:
        """True for an exact copy inside the recycle window: a duplicate."""
        return (
            match is not None
            and match.exact
            and match.gap <= timedelta(days=self._settings.recycle_days)
        )

    def _find_match(self, item: Item, sign: ItemSignature, *, track: bool = True) -> _Match | None:
        """The earlier item with the same content, or else a near copy of it, with its verdict.

        1. Identical content (same content hash or same folded text): looked up on its own, never
           limited by the candidate cap, the oldest first. A SHORT text (fewer than
           `min_near_tokens` words) matches only inside the recycle window: a headline repeated
           by another source weeks later is not recycled content.
        2. Otherwise, for a text long enough, the candidates that share bands, best first, up to
           `candidate_limit`: each must have long enough text, the same numbers (a changed
           magnitude or time is a different report) and a similarity of at least the threshold.
           The most similar wins, the best-ranked on a tie.
        The gap is the absolute time between the two first sightings, whichever came first."""
        settings = self._settings
        when = sign.ref_at
        window = timedelta(days=settings.recycle_days)
        short = sign.n_tokens < settings.min_near_tokens
        exact = self._signatures.find_exact(
            item_id=item.id,
            content_hash=item.content_hash,
            text_hash=sign.text_hash,
            ref_from=when - window if short else None,
            ref_to=when + window if short else None,
        )
        for candidate in exact:  # oldest first
            verdict = self._classifications.get(candidate.item_id)
            if verdict is not None:
                return _Match(candidate, verdict, 1.0, True, abs(when - candidate.ref_at))
        if short:
            return None
        limit = settings.candidate_limit
        candidates = self._signatures.find_candidates(
            item_id=item.id,
            content_hash=item.content_hash,
            band_keys=band_keys(sign.minhash),
            limit=limit + 1,
        )
        if len(candidates) > limit:
            self._cap_hits += track
            candidates = candidates[:limit]
        best: _Match | None = None
        for candidate in candidates:
            if candidate.n_tokens < settings.min_near_tokens or (
                candidate.digits_hash != sign.digits_hash
            ):
                continue
            score = similarity(sign.minhash, candidate.minhash)
            if score < settings.near_duplicate_threshold or (best and score <= best.similarity):
                continue
            verdict = self._classifications.get(candidate.item_id)
            if verdict is not None:
                best = _Match(candidate, verdict, score, False, abs(when - candidate.ref_at))
        return best

    def _catalog_status(self, topic: TopicVerdict, prepared: PreparedText, item: Item) -> str:
        """The outcome of the catalog cross-check; only in-scope earthquake items are checked
        (RC-6). It runs before the write transaction, and a catalog that fails or answers badly
        is `unavailable`."""
        if not topic.in_scope or Topic.EARTHQUAKE not in topic.topics:
            return NOT_CHECKED
        try:
            status = self._catalog.check(prepared.raw, item.published_at).status
        except Exception:  # noqa: BLE001 - a catalog that fails must never block a classification
            return UNAVAILABLE
        return status if _valid_status(status) else UNAVAILABLE


_OUT_OF_SCOPE_RATIONALE = {
    OutOfScopeReason.NO_TOPIC_MATCH: "out of scope: no earthquake, secretariat or disaster term",
    OutOfScopeReason.CRIME_PUBLIC_ORDER: "out of scope: crime or public order, no disaster term",
    OutOfScopeReason.NON_SPANISH: "out of scope: not Spanish and no disaster term",
}


def _reason(exc: Exception) -> str:
    """A log-safe name for an error: its stable code if it has one, else only its type."""
    return exc.code if isinstance(exc, DomainError) else type(exc).__name__


def _valid_status(status: object) -> bool:
    return (
        isinstance(status, str)
        and 1 <= len(status) <= 32
        and all(ch == "_" or "a" <= ch <= "z" for ch in status)
    )
