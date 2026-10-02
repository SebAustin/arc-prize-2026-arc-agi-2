"""Grid->grid primitives for the DSL micro-solver.

Two kinds:
  * Parameter-free geometric ops (rotations, flips, crop-to-content).
  * Parameter ops whose parameters are INFERRED from the task's train pairs
    (integer scaling, tiling, and a learned cellwise colour map). Inferring
    parameters from the demonstrations keeps the search space tiny while still
    covering a large slice of common ARC transformations.
"""

from __future__ import annotations

import numpy as np

from ...augment import symmetry
from ...io.grid import Grid, background_color, from_numpy, shape, to_numpy
from ...io.loader import Task

# ARC grids are at most 30x30; a program whose output would exceed this can never
# be a correct answer, so scale/tile refuse to allocate beyond it (bounds memory
# on a pathological test input and lets the oversized program fail verification).
MAX_GRID_DIM = 30


# ---- Parameter-free geometric ops -----------------------------------------
def crop_to_content(grid: Grid) -> Grid:
    """Crop to the bounding box of non-background cells (background = mode)."""
    bg = background_color(grid)
    arr = to_numpy(grid)
    mask = arr != bg
    if not mask.any():
        return grid
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return from_numpy(arr[r0 : r1 + 1, c0 : c1 + 1])


PARAM_FREE: dict[str, callable] = {
    "identity": lambda g: g,
    "rot90": lambda g: symmetry.apply("rot90", g),
    "rot180": lambda g: symmetry.apply("rot180", g),
    "rot270": lambda g: symmetry.apply("rot270", g),
    "flip_h": lambda g: symmetry.apply("flip_h", g),
    "flip_v": lambda g: symmetry.apply("flip_v", g),
    "transpose": lambda g: symmetry.apply("transpose", g),
    "anti_transpose": lambda g: symmetry.apply("anti_transpose", g),
    "crop_to_content": crop_to_content,
}


# ---- Inferred parameter ops -----------------------------------------------
def _shape_ratio(task: Task) -> tuple[int, int] | None:
    """Consistent integer (out/in) shape ratio across train pairs, or None."""
    ratios: set[tuple[int, int]] = set()
    for pair in task.train:
        if pair.output is None:
            return None
        ih, iw = shape(pair.input)
        oh, ow = shape(pair.output)
        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            return None
        ratios.add((oh // ih, ow // iw))
    if len(ratios) != 1:
        return None
    fy, fx = next(iter(ratios))
    return (fy, fx) if (fy, fx) != (1, 1) else None


def scale_program(fy: int, fx: int):
    """Each cell becomes an fy x fx block (np.repeat)."""

    def f(g: Grid) -> Grid:
        arr = to_numpy(g)
        if arr.shape[0] * fy > MAX_GRID_DIM or arr.shape[1] * fx > MAX_GRID_DIM:
            return g  # oversized -> pass through so the program fails to verify
        return from_numpy(np.repeat(np.repeat(arr, fy, axis=0), fx, axis=1))

    return f


def tile_program(ny: int, nx: int):
    """Repeat the whole grid ny x nx times (np.tile)."""

    def f(g: Grid) -> Grid:
        arr = to_numpy(g)
        if arr.shape[0] * ny > MAX_GRID_DIM or arr.shape[1] * nx > MAX_GRID_DIM:
            return g  # oversized -> pass through so the program fails to verify
        return from_numpy(np.tile(arr, (ny, nx)))

    return f


def learn_colormap(task: Task) -> dict[int, int] | None:
    """Learn a consistent cellwise symbol map if every train pair is a pure
    same-shape recolour; else None."""
    mapping: dict[int, int] = {}
    for pair in task.train:
        if pair.output is None or shape(pair.input) != shape(pair.output):
            return None
        for in_row, out_row in zip(pair.input, pair.output, strict=False):
            for s, d in zip(in_row, out_row, strict=False):
                if s in mapping and mapping[s] != d:
                    return None
                mapping[s] = d
    return mapping or None


def colormap_program(mapping: dict[int, int]):
    """Apply a learned colour map; unknown symbols pass through unchanged."""

    def f(g: Grid) -> Grid:
        return tuple(tuple(mapping.get(c, c) for c in row) for row in g)

    return f
