"""D4 dihedral symmetries on grids, each with an exact inverse.

The 8 elements of the symmetry group of the square. Augmentation transforms a
task's inputs; at inference we transform the input, predict, then apply the
INVERSE transform to bring the prediction back to the canonical frame before
voting. Exact invertibility is required — it is enforced by round-trip tests.
"""

from __future__ import annotations

import numpy as np

from ..io.grid import Grid, from_numpy, to_numpy

# Forward transforms on numpy arrays. np.rot90 rotates counter-clockwise.
_FORWARD = {
    "identity": lambda a: a,
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "flip_h": lambda a: np.fliplr(a),
    "flip_v": lambda a: np.flipud(a),
    "transpose": lambda a: a.T,
    "anti_transpose": lambda a: np.rot90(np.fliplr(a), 1),
}

# Inverse element of each transform within D4.
_INVERSE_NAME = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_h": "flip_h",
    "flip_v": "flip_v",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}

D4_NAMES = tuple(_FORWARD.keys())


def apply(name: str, grid: Grid) -> Grid:
    """Apply the named D4 transform to a grid."""
    return from_numpy(np.ascontiguousarray(_FORWARD[name](to_numpy(grid))))


def invert(name: str, grid: Grid) -> Grid:
    """Apply the inverse of the named D4 transform to a grid."""
    return apply(_INVERSE_NAME[name], grid)


def inverse_name(name: str) -> str:
    return _INVERSE_NAME[name]
