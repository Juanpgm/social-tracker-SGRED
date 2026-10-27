"""Near-duplicate detection without any model: character-shingle MinHash with LSH bands (RC-5).
Pure, deterministic and dependency free.

A text becomes the set of its 4-character shingles (the folded words joined by single spaces, see
`text.fold`; a text of fewer than four characters is one shingle). Characters, not words: a
one-word edit of a 20-word caption keeps about 0.87 of its shingles (0.74 with word triples,
which a threshold of 0.8 would call a different text) while unrelated news items of the same
language stay under 0.25. The MinHash signature keeps, for each of
80 fixed hash functions, the smallest value over the shingles; the share of equal positions of two
signatures estimates the Jaccard similarity of the shingle sets, with a standard error of
sqrt(J (1 - J) / 80): 0.045 at J = 0.8, 0.034 at J = 0.9. The 80 positions are cut into 16 bands
of 5: two signatures that agree on a whole band share its key, so a store indexed by
`(band, key)` finds the candidates of a text without comparing it to every other one. The chance
that a pair of similarity J shares at least one band is 1 - (1 - J^5)^16: 0.998 at 0.8, 0.9998 at
0.9, 0.82 at 0.6. Candidates are then verified with the full signature.

A similar text is not a copy: a magnitude of 3.2 changed to 7.9 leaves the shingles nearly
untouched. `numbers_digest` therefore digests the numbers of a text apart (as a multiset), and two
texts are near copies only when the digests are equal and the estimate reaches the threshold.

Nothing depends on `hash()` (randomized per process) or on `random`: shingles are hashed with
BLAKE2b and the hash functions come from fixed BLAKE2b-derived constants, so a signature is the
same in every process and every release.
"""

import hashlib
import struct
from collections.abc import Sequence

from social_tracker.shared.errors import ValidationError

NUM_PERM = 80
BANDS = 16
ROWS = NUM_PERM // BANDS
SHINGLE_CHARS = 4
_MASK64 = (1 << 64) - 1
_MAX_VALUE = 1 << 32


def _constant(label: str, index: int) -> int:
    digest = hashlib.blake2b(f"minhash-{label}-{index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


_MULTIPLIERS = tuple(_constant("a", i) | 1 for i in range(NUM_PERM))  # odd: a bijection mod 2^64
_OFFSETS = tuple(_constant("b", i) for i in range(NUM_PERM))
_PACK = struct.Struct(f">{NUM_PERM}I")


def _shingles(words: Sequence[str]) -> set[int]:
    text = " ".join(words)
    if len(text) < SHINGLE_CHARS:
        grams = [text]
    else:
        grams = [text[i : i + SHINGLE_CHARS] for i in range(len(text) - SHINGLE_CHARS + 1)]
    return {
        int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams
    }


def signature(words: Sequence[str]) -> tuple[int, ...]:
    """The MinHash signature of folded `words`: `NUM_PERM` unsigned 32-bit values."""
    if not words:
        raise ValidationError("a text without words has no signature")
    hashed = _shingles(words)
    return tuple(
        min(((a * x + b) & _MASK64) >> 32 for x in hashed)
        for a, b in zip(_MULTIPLIERS, _OFFSETS, strict=True)
    )


def _check(sig: Sequence[int]) -> None:
    if len(sig) != NUM_PERM or not all(
        isinstance(v, int) and not isinstance(v, bool) and 0 <= v < _MAX_VALUE for v in sig
    ):
        raise ValidationError("a signature is 80 unsigned 32-bit values")


def similarity(a: Sequence[int], b: Sequence[int]) -> float:
    """The estimated Jaccard similarity: the share of positions where the signatures agree."""
    _check(a)
    _check(b)
    return sum(x == y for x, y in zip(a, b, strict=True)) / NUM_PERM


def is_near_duplicate(a: Sequence[int], b: Sequence[int], threshold: float) -> bool:
    """True when the similarity is `threshold` or more (inclusive)."""
    return similarity(a, b) >= threshold


def band_keys(sig: Sequence[int]) -> tuple[int, ...]:
    """One signed 64-bit key per band, the lookup keys of the candidate index."""
    _check(sig)
    keys = []
    for band in range(BANDS):
        rows = sig[band * ROWS : (band + 1) * ROWS]
        digest = hashlib.blake2b(struct.pack(f">{ROWS}I", *rows), digest_size=8).digest()
        keys.append(int.from_bytes(digest, "big", signed=True))
    return tuple(keys)


def _digest(items: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def text_digest(words: Sequence[str]) -> str:
    """SHA-256 of the folded words: equal for texts that differ only in case, accents, spacing or
    punctuation."""
    return _digest([" ".join(words)])


def numbers_digest(words: Sequence[str]) -> str:
    """SHA-256 of the numbers of a text (every word with a digit in it), as a multiset: the same
    numbers in any order and nothing else give the same digest."""
    return _digest(sorted(word for word in words if any(ch.isdigit() for ch in word)))


def pack(sig: Sequence[int]) -> bytes:
    _check(sig)
    return _PACK.pack(*sig)


def unpack(data: bytes) -> tuple[int, ...]:
    if not isinstance(data, bytes | bytearray | memoryview) or len(data) != _PACK.size:
        raise ValidationError("a stored signature has the wrong size")
    return _PACK.unpack(bytes(data))
