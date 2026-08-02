"""The canonical grid type and conversions.

A grid is an immutable, hashable `tuple[tuple[int, ...], ...]` of symbols 0-9.
Immutability matches the project's no-mutation rule; hashability is required by
the candidate-voting selection stage (grids are used as dict keys).

Solvers may compute in numpy for convenience and convert back with `from_numpy`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Grid = tuple[tuple[int, ...], ...]

NUM_COLORS = 10       # symbols 0-9
MAX_DIM = 30          # competition grids are at most 30x30
MIN_DIM = 1


def from_lists(rows: Sequence[Sequence[int]]) -> Grid:
    """Convert a JSON list-of-lists into the canonical immutable grid."""
    return tuple(tuple(int(c) for c in row) for row in rows)


def to_lists(grid: Grid) -> list[list[int]]:
    """Convert a canonical grid back to JSON-serialisable list-of-lists."""
    return [list(row) for row in grid]


def from_numpy(arr: np.ndarray) -> Grid:
    """Convert a 2-D integer numpy array into a canonical grid."""
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {arr.shape}")
    return tuple(tuple(int(c) for c in row) for row in arr.tolist())


def to_numpy(grid: Grid) -> np.ndarray:
    """Convert a canonical grid into a 2-D int8 numpy array."""
    return np.array(grid, dtype=np.int8)


def shape(grid: Grid) -> tuple[int, int]:
    """Return (height, width). Assumes a rectangular grid."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def is_valid_grid(grid: Grid) -> bool:
    """True iff `grid` is rectangular, within size limits, and all cells 0-9."""
    if not isinstance(grid, tuple) or len(grid) < MIN_DIM or len(grid) > MAX_DIM:
        return False
    width = None
    for row in grid:
        if not isinstance(row, tuple):
            return False
        if width is None:
            width = len(row)
            if width < MIN_DIM or width > MAX_DIM:
                return False
        elif len(row) != width:  # ragged → invalid
            return False
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell >= NUM_COLORS:
                return False
    return True


def grids_equal(a: Grid, b: Grid) -> bool:
    """Exact cell-for-cell equality (the competition's correctness criterion)."""
    return a == b


def color_counts(grid: Grid) -> dict[int, int]:
    """Histogram of symbol -> count across all cells."""
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return counts


def background_color(grid: Grid) -> int:
    """Heuristic background = most frequent symbol (ties → lowest symbol)."""
    counts = color_counts(grid)
    if not counts:
        return 0
    return max(sorted(counts), key=lambda c: counts[c])
