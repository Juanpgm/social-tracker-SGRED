"""A verdict the language model gave is shared only with a copy of the EXACT same text.

The duplicate lookup also finds copies that differ only at the level of the folded text
(punctuation, case, spacing, look-alike letters, another address). Those must not inherit a
model-judged verdict: they are judged for themselves. A verdict of the CPU stages is still shared
with them (it is a pure function of the folded text, which they have in common).
"""

from dataclasses import replace

import pytest

from social_tracker.classification.domain import Label, Stage
from tests.classification.harness import PREDICTION, memory_world, sqlite_world

JUDGED_LABEL = Label.NO_RISK
JUDGED_CONFIDENCE = 0.93


@pytest.fixture(params=["memory", "sqlite"])
def world(request, tmp_path):
    made = memory_world() if request.param == "memory" else sqlite_world(tmp_path / "fold.db")
    made.ready()
    made.add_source(
        "src_other"
    )  # a copy comes from another source: one source never repeats itself
    yield made
    made.close()


def _judged_original(world, text: str = PREDICTION):
    """The first item, classified and then settled as if the model had judged it."""
    first = world.add("guid-first", text)
    world.service.classify_pending()
    cascade = world.verdict(first)
    queued = replace(cascade, needs_llm=True, priority=0.5)
    # put it in the state the escalation leaves a judged item in
    with world.uow.transaction():
        world.classifications.delete(first.id)
        world.classifications.add(queued)
        settled = replace(
            queued,
            label=JUDGED_LABEL,
            confidence=JUDGED_CONFIDENCE,
            stage=Stage.LLM,
            needs_llm=False,
            priority=0.0,
            settled_at=world.clock.now(),
            settle_reason="llm",
        )
        assert world.classifications.settle(settled)
    return first


def _next(world, external_id: str, text: str, **kwargs):
    item = world.add(external_id, text, source_id="src_other", **kwargs)
    world.service.classify_pending()
    return item, world.verdict(item)


def test_an_identical_text_takes_the_judged_verdict(world):
    first = _judged_original(world)
    _, copy = _next(world, "guid-copy", PREDICTION)
    assert copy.duplicate_of == first.id
    assert (copy.label, copy.stage) == (JUDGED_LABEL, Stage.DEDUP)
    assert copy.confidence == pytest.approx(JUDGED_CONFIDENCE)
    assert copy.needs_llm is False


@pytest.mark.parametrize(
    "variant",
    [
        PREDICTION + ".",  # punctuation
        PREDICTION.upper(),  # case
        PREDICTION.replace(" ", "  "),  # spacing (the stored text is normalised, the bytes differ)
        PREDICTION.replace("temblor", "temblоr"),  # a Cyrillic o
        PREDICTION + " https://sgc.gov.co/aviso",  # a link the original did not have
    ],
    ids=["punctuation", "case", "spacing", "homoglyph", "url"],
)
def test_a_text_that_only_folds_to_the_same_words_does_not_inherit_a_judged_verdict(world, variant):
    first = _judged_original(world)
    _, copy = _next(world, "guid-variant", variant)
    assert copy.duplicate_of is None
    assert copy.stage is not Stage.LLM
    assert (copy.label, copy.confidence) != (JUDGED_LABEL, pytest.approx(JUDGED_CONFIDENCE))
    assert copy.near_of in (first.id, None)  # linked as a near copy at most
    assert copy.awaiting_original is False


def test_a_look_alike_address_is_judged_for_itself(world):
    original_text = "Aviso oficial del SGC sobre el sismo de ayer en Cali: https://sgc.gov.co/aviso"
    first = _judged_original(world, original_text)
    fake = original_text.replace("sgc.gov.co", "sgc-gov.co")
    _, copy = _next(world, "guid-fake", fake)
    assert copy.duplicate_of is None
    assert copy.stage is not Stage.LLM
    assert copy.near_of in (first.id, None)


def test_a_variant_of_a_text_the_cpu_settled_still_shares_the_cpu_verdict(world):
    first = world.add("guid-first", PREDICTION)
    world.service.classify_pending()
    cpu = world.verdict(first)
    assert cpu.stage is not Stage.LLM
    _, copy = _next(world, "guid-variant", PREDICTION + ".")
    assert copy.duplicate_of == first.id  # the CPU stages are a function of the folded text
    assert copy.label is cpu.label


def test_an_original_that_is_gone_cannot_vouch_for_the_text(world):
    first = _judged_original(world)
    world.items.purge_fetched_before(world.clock.now() + world.clock.now().resolution * 10**9)
    assert world.items.get(first.id) is None
    _, copy = _next(world, "guid-after", PREDICTION)
    assert copy.stage is not Stage.LLM
    assert copy.duplicate_of is None
