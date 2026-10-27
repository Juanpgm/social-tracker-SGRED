"""Near-duplicate detection: word-shingle MinHash with LSH bands, pure and deterministic (RC-5)."""

import random
import time

import pytest

from social_tracker.classification.similarity import (
    BANDS,
    NUM_PERM,
    band_keys,
    is_near_duplicate,
    pack,
    signature,
    similarity,
    unpack,
)
from social_tracker.classification.text import tokens
from social_tracker.shared.errors import ValidationError

VOCABULARY = [f"palabra{n}" for n in range(400)]


def words(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [rng.choice(VOCABULARY) for _ in range(count)]


def sig_of(ws: list[str]) -> tuple[int, ...]:
    return signature(tuple(ws))


def test_the_signature_has_the_fixed_shape():
    sig = sig_of(words(60, 1))
    assert len(sig) == NUM_PERM == 80
    assert all(isinstance(v, int) and 0 <= v < 2**32 for v in sig)
    assert BANDS * 5 == NUM_PERM


def test_the_signature_is_deterministic_across_processes():
    # Pinned values: they must not depend on Python's per-process hash randomization.
    sig = signature(tokens("el sismo de ayer fue sentido en toda la ciudad de cali"))
    assert sig[:3] == (776558042, 202218136, 51220112)
    assert sig == signature(tokens("el sismo de ayer fue sentido en toda la ciudad de cali"))


def test_identical_texts_are_identical_and_different_texts_are_not():
    a, b = sig_of(words(60, 1)), sig_of(words(60, 2))
    assert similarity(a, a) == 1.0
    assert similarity(a, b) < 0.1


def test_case_accents_and_punctuation_do_not_change_the_signature():
    assert signature(tokens("¡SÍSMO en Cali, hoy!")) == signature(tokens("sismo EN cali hoy"))


def test_one_changed_word_in_sixty_stays_above_the_default_threshold():
    base = words(60, 3)
    changed = list(base)
    changed[30] = "otra_palabra"
    assert is_near_duplicate(sig_of(base), sig_of(changed), 0.8)


def test_half_of_the_words_changed_is_far_below_the_default_threshold():
    base = words(60, 4)
    changed = base[:30] + words(30, 99)
    assert not is_near_duplicate(sig_of(base), sig_of(changed), 0.8)
    assert similarity(sig_of(base), sig_of(changed)) < 0.6


def test_a_text_shorter_than_a_shingle_still_has_a_signature():
    one, same, other = sig_of(["sismo"]), sig_of(["sismo"]), sig_of(["temblor"])
    two = sig_of(["sismo", "fuerte"])
    assert similarity(one, same) == 1.0
    assert similarity(one, other) < 0.1
    assert similarity(one, two) < 0.1


def test_a_text_without_words_has_no_signature():
    with pytest.raises(ValidationError):
        signature(())


@pytest.mark.parametrize(
    ("differing", "threshold", "near"),
    [
        (16, 0.8, True),  # 64 of 80 positions agree: exactly the threshold
        (15, 0.8, True),
        (17, 0.8, False),  # 63 of 80
        (0, 1.0, True),
        (1, 1.0, False),
        (40, 0.5, True),
        (41, 0.5, False),
    ],
)
def test_the_threshold_is_inclusive(differing, threshold, near):
    a = tuple(range(NUM_PERM))
    b = tuple(v + 1000 if i < differing else v for i, v in enumerate(a))
    assert similarity(a, b) == (NUM_PERM - differing) / NUM_PERM
    assert is_near_duplicate(a, b, threshold) is near


def test_similarity_needs_two_signatures_of_the_same_length():
    with pytest.raises(ValidationError):
        similarity((1, 2, 3), (1, 2, 3))


def test_band_keys_are_stable_signed_64_bit_integers_one_per_band():
    keys = band_keys(sig_of(words(60, 5)))
    assert len(keys) == BANDS == 16
    assert all(-(2**63) <= k < 2**63 for k in keys)
    assert keys == band_keys(sig_of(words(60, 5)))


def test_signatures_agreeing_on_a_whole_band_share_that_band_key_and_no_other():
    a = tuple(range(NUM_PERM))
    b = tuple(v if i < 5 else v + 7000 for i, v in enumerate(a))  # only band 0 agrees
    shared = [i for i, (x, y) in enumerate(zip(band_keys(a), band_keys(b), strict=True)) if x == y]
    assert shared == [0]
    almost = tuple(v if i != 4 else 9999 for i, v in enumerate(a))  # band 0 differs in one row
    assert band_keys(almost)[0] != band_keys(a)[0]
    assert band_keys(almost)[1:] == band_keys(a)[1:]


def test_the_bands_find_near_duplicates_and_almost_never_unrelated_texts():
    found = unrelated = 0
    for seed in range(60):
        base = words(60, seed)
        changed = list(base)
        changed[seed % 60] = "cambiada"
        found += bool(set(band_keys(sig_of(base))) & set(band_keys(sig_of(changed))))
        other = words(60, seed + 1000)
        unrelated += bool(set(band_keys(sig_of(base))) & set(band_keys(sig_of(other))))
    assert found == 60  # a candidate is never missed at this similarity
    assert unrelated == 0


def test_pack_and_unpack_round_trip_and_reject_bad_input():
    sig = sig_of(words(60, 6))
    data = pack(sig)
    assert len(data) == NUM_PERM * 4
    assert unpack(data) == sig
    for bad in (b"", data[:-1], data + b"\x00", None, "text"):
        with pytest.raises(ValidationError):
            unpack(bad)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        pack(sig[:-1])
    with pytest.raises(ValidationError):
        pack(tuple([2**32] + list(sig[1:])))


def test_a_very_long_text_is_signed_in_bounded_time():
    started = time.perf_counter()
    signature(tuple(words(4000, 7)))
    assert time.perf_counter() - started < 3.0
