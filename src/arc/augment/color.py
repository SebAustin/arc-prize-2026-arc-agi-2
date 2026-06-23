"""Colour (symbol) permutations on grids, each with an exact inverse.

A permutation is a length-10 tuple `perm` mapping symbol s -> perm[s]. Applying
`perm` then its inverse recovers the original grid. Optionally symbol 0 is held
fixed (treat 0 as background); by default all 10 symbols are permuted, since
ARC-AGI-2's harder tasks often do not use 0 as background.
"""

from __future__ import annotations

import random

from ..io.grid import Grid, NUM_COLORS

Perm = tuple[int, ...]

IDENTITY_PERM: Perm = tuple(range(NUM_COLORS))


def random_perm(seed: int, keep_zero: bool = False) -> Perm:
    """A random symbol permutation, deterministic in `seed`.

    If `keep_zero`, symbol 0 maps to itself and only 1-9 are shuffled.
    """
    rng = random.Random(seed)
    if keep_zero:
        movable = list(range(1, NUM_COLORS))
        shuffled = movable[:]
        rng.shuffle(shuffled)
        perm = [0] + [0] * (NUM_COLORS - 1)
        for src, dst in zip(movable, shuffled):
            perm[src] = dst
        return tuple(perm)
    symbols = list(range(NUM_COLORS))
    shuffled = symbols[:]
    rng.shuffle(shuffled)
    return tuple(shuffled)


def invert_perm(perm: Perm) -> Perm:
    """The inverse permutation, such that invert_perm(perm)[perm[s]] == s."""
    inv = [0] * NUM_COLORS
    for src, dst in enumerate(perm):
        inv[dst] = src
    return tuple(inv)


def apply(perm: Perm, grid: Grid) -> Grid:
    """Recolour a grid by mapping every symbol s -> perm[s]."""
    return tuple(tuple(perm[c] for c in row) for row in grid)


def invert(perm: Perm, grid: Grid) -> Grid:
    """Undo a recolouring (apply the inverse permutation)."""
    return apply(invert_perm(perm), grid)
