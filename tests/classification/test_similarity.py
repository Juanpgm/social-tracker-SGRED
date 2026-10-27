"""Near-duplicate detection: character-shingle MinHash with LSH bands, pure and deterministic
(RC-5)."""

import hashlib
import random
import time

import pytest

from social_tracker.classification.similarity import (
    BANDS,
    NUM_PERM,
    band_keys,
    is_near_duplicate,
    numbers_digest,
    pack,
    signature,
    similarity,
    text_digest,
    unpack,
)
from social_tracker.classification.text import tokens
from social_tracker.shared.errors import ValidationError

_LETTERS = "abcdefghilmnopqrstuvz"
_RNG = random.Random(20260301)
# Words of random letters: unrelated texts then share about what natural text does, not the long
# common stem that a numbered vocabulary would give every word.
VOCABULARY = ["".join(_RNG.choice(_LETTERS) for _ in range(_RNG.randint(4, 9))) for _ in range(400)]


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
    assert sig[:3] == (4133714, 339434827, 135910339)
    assert sig == signature(tokens("el sismo de ayer fue sentido en toda la ciudad de cali"))


def test_the_whole_signature_is_pinned_so_that_no_hash_function_can_drift():
    sig = signature(tokens("el sismo de ayer fue sentido en toda la ciudad de cali"))
    digest = hashlib.sha256(pack(sig)).hexdigest()
    assert digest == "54804c25c1a8b9381f13651a769c6a6110844ca06279028e8d9a1e5049799613"


def test_identical_texts_are_identical_and_different_texts_are_not():
    a, b = sig_of(words(60, 1)), sig_of(words(60, 2))
    assert similarity(a, a) == 1.0
    assert similarity(a, b) < 0.25


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
    one, same, other = sig_of(["sis"]), sig_of(["sis"]), sig_of(["tem"])
    two = sig_of(["sis", "fuerte"])
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


# ---- numbers and digests ----


def test_the_numbers_digest_sees_every_number_and_ignores_their_order():
    assert numbers_digest(tokens("sismo de magnitud 3.2 a las 10:30")) == numbers_digest(
        tokens("a las 10:30 magnitud 3.2 sismo de")
    )
    one_digit = tokens("sismo de magnitud 3.2 a las 10:30")
    for changed in (
        "sismo de magnitud 7.9 a las 10:30",  # the magnitude
        "sismo de magnitud 3.2 a las 11:30",  # the time
        "sismo de magnitud 3.3 a las 10:30",  # one digit
        "sismo de magnitud 3.2 a las 10:30 y 5 heridos",  # one more number
        "sismo de magnitud 3.2 a las 10",  # one number fewer
    ):
        assert numbers_digest(tokens(changed)) != numbers_digest(one_digit), changed


def test_the_numbers_digest_counts_a_repeated_number_twice():
    assert numbers_digest(("a", "5", "5")) != numbers_digest(("a", "5"))


def test_texts_without_numbers_share_the_empty_digest():
    assert numbers_digest(tokens("sismo fuerte en cali")) == numbers_digest(tokens("otro texto"))
    assert numbers_digest(()) == numbers_digest(("sin", "numeros"))


def test_the_text_digest_ignores_case_accents_and_punctuation_only():
    assert text_digest(tokens("¡SÍSMO en Cali!")) == text_digest(tokens("sismo en cali"))
    assert text_digest(tokens("sismo en cali")) != text_digest(tokens("sismo en yumbo"))
    assert len(text_digest(("a",))) == 64


# ---- how well a near copy is found: the measured recall (W4) ----


def _shingles(words: list[str]) -> set[str]:
    text = " ".join(words)
    return {text[i : i + 4] for i in range(len(text) - 3)}


def test_the_recall_by_true_similarity_is_the_documented_one():
    """Pairs of random texts with 1 to 25 % of their words replaced, binned by the TRUE Jaccard
    similarity of their shingles: how often the pair is a candidate (shares a band) and how often
    the estimate reaches the default threshold of 0.8. The README quotes this table."""
    rng = random.Random(7)
    bins: dict[float, list[tuple[bool, bool]]] = {0.65: [], 0.75: [], 0.85: [], 0.95: []}
    for _ in range(700):
        n = rng.choice([15, 20, 30, 50])
        base = [rng.choice(VOCABULARY) for _ in range(n)]
        changed = list(base)
        for pos in rng.sample(range(n), rng.choice([1, 1, 2, 3, max(1, n // 8), max(1, n // 4)])):
            changed[pos] = rng.choice(VOCABULARY)
        a, b = _shingles(base), _shingles(changed)
        truth = len(a & b) / len(a | b)
        sa, sb = sig_of(base), sig_of(changed)
        candidate = bool(set(band_keys(sa)) & set(band_keys(sb)))
        for low, rows in bins.items():
            if low <= truth < low + 0.1:
                rows.append((candidate, is_near_duplicate(sa, sb, 0.8)))

    def rate(low: float, index: int) -> float:
        rows = bins[low]
        return sum(row[index] for row in rows) / len(rows)

    assert all(len(rows) >= 15 for rows in bins.values()), {k: len(v) for k, v in bins.items()}
    assert rate(0.65, 0) >= 0.85  # about 0.7: usually a candidate...
    assert rate(0.65, 1) <= 0.20  # ...and almost never accepted: below the threshold
    assert rate(0.75, 0) >= 0.98
    assert 0.30 <= rate(0.75, 1) <= 0.80  # at the threshold the estimate noise decides
    assert rate(0.85, 0) == 1.0
    assert rate(0.85, 1) >= 0.95  # about 0.9: found
    assert rate(0.95, 1) == 1.0


def test_a_one_word_edit_of_a_twenty_word_caption_is_still_a_near_duplicate():
    rng = random.Random(11)
    caught = 0
    for _ in range(200):
        base = [rng.choice(VOCABULARY) for _ in range(20)]
        changed = list(base)
        changed[rng.randrange(20)] = rng.choice(VOCABULARY)
        caught += is_near_duplicate(sig_of(base), sig_of(changed), 0.8)
    assert caught >= 190  # at least 95 %: the estimator's noise (0.04) is the only way out
