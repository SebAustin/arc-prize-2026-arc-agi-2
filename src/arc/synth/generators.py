"""Procedural ARC-like task generators.

Each generator samples ONE grid->grid transformation plus a set of input grids;
applying the transform to every input yields a self-consistent task. Building
from a single transform means rule-consistency is guaranteed by construction —
the test suite just re-checks `transform(input) == output` for every pair.

A large, diverse synthetic corpus from these generators is what lets base
fine-tuning give the model a strong ARC prior, so per-task test-time training
starts from a good place rather than cold (the edge behind the 2025 winner).
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..augment import symmetry
from ..io.grid import Grid, from_numpy, is_valid_grid, to_numpy
from ..io.loader import Pair, Task
from ..solvers.dsl.primitives import (
    colormap_program,
    crop_to_content,
    scale_program,
    tile_program,
)

Transform = Callable[[Grid], Grid]
BG = 0


# ---- random helpers --------------------------------------------------------
def _palette(rng: np.random.Generator, k: int | None = None) -> list[int]:
    """A task palette: background 0 plus a few distinct non-zero symbols."""
    k = k or int(rng.integers(2, 5))
    nonzero = list(rng.choice(range(1, 10), size=min(k, 9), replace=False))
    return [BG] + [int(c) for c in nonzero]


def _random_grid(
    rng: np.random.Generator,
    h: int,
    w: int,
    palette: list[int],
    weights: list[float] | None = None,
) -> Grid:
    arr = rng.choice(palette, size=(h, w), p=weights)
    return from_numpy(np.asarray(arr, dtype=np.int8))


def _size(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


# ---- non-DSL transforms (numpy) -------------------------------------------
def _gravity_down(g: Grid) -> Grid:
    arr = to_numpy(g)
    h, w = arr.shape
    out = np.full_like(arr, BG)
    for c in range(w):
        col = arr[:, c]
        non = col[col != BG]
        if len(non):
            out[h - len(non) :, c] = non
    return from_numpy(out)


def _mirror_concat_h(g: Grid) -> Grid:
    arr = to_numpy(g)
    return from_numpy(np.hstack([arr, np.fliplr(arr)]))


def _border(color: int, width: int = 1) -> Transform:
    def f(g: Grid) -> Grid:
        return from_numpy(np.pad(to_numpy(g), width, constant_values=color))

    return f


# ---- shared helpers for the "hard" (DSL-unsolvable) generators below -------
def _connected_components(arr: np.ndarray, bg: int) -> list[np.ndarray]:
    """4-connected components of non-background cells, as boolean masks.

    Plain flood fill over a numpy array; no scipy dependency needed for the
    small (<=30x30) grids these generators produce.
    """
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if arr[r0, c0] == bg or seen[r0, c0]:
                continue
            mask = np.zeros((h, w), dtype=bool)
            stack = [(r0, c0)]
            seen[r0, c0] = True
            while stack:
                y, x = stack.pop()
                mask[y, x] = True
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and arr[ny, nx] != bg:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(mask)
    return comps


def _scattered_objects(
    rng: np.random.Generator, colors: list[int], h: int, w: int, n_objs: int
) -> Grid:
    """A canvas of `n_objs` separated solid rectangular blocks of varied size.

    Blocks get a 1-cell gap on placement so each stays its own connected
    component -- shared by the object-level generators below.
    """
    arr = np.full((h, w), BG, dtype=np.int8)
    for _ in range(n_objs):
        oh, ow = _size(rng, 1, min(4, h)), _size(rng, 1, min(4, w))
        for _try in range(12):
            r0, c0 = int(rng.integers(0, h - oh + 1)), int(rng.integers(0, w - ow + 1))
            pad = arr[max(0, r0 - 1) : r0 + oh + 1, max(0, c0 - 1) : c0 + ow + 1]
            if (pad != BG).any():
                continue
            arr[r0 : r0 + oh, c0 : c0 + ow] = int(rng.choice(colors))
            break
    return from_numpy(arr)


def _recolor_by_size(big_color: int, small_color: int) -> Transform:
    """Recolour the largest connected object `big_color`, every other object
    `small_color`. Needs component-size comparison -- outside the DSL's fixed
    value->value colormap, which cannot single out "whichever object happens
    to be biggest" (that identity changes from grid to grid).
    """

    def f(g: Grid) -> Grid:
        arr = to_numpy(g)
        comps = _connected_components(arr, BG)
        if not comps:
            return from_numpy(arr)
        sizes = [int(m.sum()) for m in comps]
        largest = int(np.argmax(sizes))
        out = arr.copy()
        for i, m in enumerate(comps):
            out[m] = big_color if i == largest else small_color
        return from_numpy(out)

    return f


def _slide_to_wall(direction: str) -> Transform:
    """Translate every connected object independently until it touches one
    grid wall (up/down/left/right). Each object moves by a DIFFERENT offset
    (however far it started from the wall), so this is not a single rigid
    whole-grid transform the way a D4 op is.
    """

    def f(g: Grid) -> Grid:
        arr = to_numpy(g)
        h, w = arr.shape
        out = np.full_like(arr, BG)
        for mask in _connected_components(arr, BG):
            rows, cols = np.where(mask)
            r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
            block = arr[r0 : r1 + 1, c0 : c1 + 1]
            bh, bw = block.shape
            nr0 = 0 if direction == "up" else (h - bh if direction == "down" else r0)
            nc0 = 0 if direction == "left" else (w - bw if direction == "right" else c0)
            keep = block != BG
            out[nr0 : nr0 + bh, nc0 : nc0 + bw][keep] = block[keep]
        return from_numpy(out)

    return f


def _neighbor_halo(trigger: int, halo: int) -> Transform:
    """Recolour background cells orthogonally adjacent to `trigger` to `halo`.

    A cellwise rule that depends on NEIGHBOURS: two background (0) cells can
    map to different output symbols depending on local context, which breaks
    the DSL colormap's "same input value always maps to the same output
    value" assumption.
    """

    def f(g: Grid) -> Grid:
        arr = to_numpy(g)
        trig = arr == trigger
        adjacent = np.zeros_like(trig)
        adjacent[1:, :] |= trig[:-1, :]
        adjacent[:-1, :] |= trig[1:, :]
        adjacent[:, 1:] |= trig[:, :-1]
        adjacent[:, :-1] |= trig[:, 1:]
        out = arr.copy()
        out[(arr == BG) & adjacent] = halo
        return from_numpy(out)

    return f


def _mirror_grid(rng: np.random.Generator, h: int, w: int, palette: list[int], axis: str) -> Grid:
    """A grid that is exactly mirror-symmetric across `axis` ("h" or "v")."""
    if axis == "h":
        half = (w + 1) // 2
        left = np.asarray(rng.choice(palette, size=(h, half)), dtype=np.int8)
        arr = np.empty((h, w), dtype=np.int8)
        arr[:, :half] = left
        arr[:, w - half :] = np.fliplr(left)
    else:
        half = (h + 1) // 2
        top = np.asarray(rng.choice(palette, size=(half, w)), dtype=np.int8)
        arr = np.empty((h, w), dtype=np.int8)
        arr[:half, :] = top
        arr[h - half :, :] = np.flipud(top)
    return from_numpy(arr)


def _repair_symmetry(axis: str, noise: int) -> Transform:
    """Fill cells marked `noise` from their mirror counterpart across `axis`.

    The DSL only ever applies a whole-grid rigid transform; it never inspects
    a grid's OWN internal symmetry to patch a subset of cells from their
    reflection while leaving the rest untouched.
    """

    def f(g: Grid) -> Grid:
        arr = to_numpy(g)
        mirror = np.fliplr(arr) if axis == "h" else np.flipud(arr)
        out = arr.copy()
        mask = out == noise
        out[mask] = mirror[mask]
        return from_numpy(out)

    return f


def _reflect_upper_triangle(g: Grid) -> Grid:
    """Overwrite the half ABOVE the main diagonal with its transpose value;
    the lower half (incl. the diagonal) is left untouched -- a partial, not
    whole-grid, reflection that the DSL's global `transpose` cannot express.
    """
    arr = to_numpy(g)
    n = arr.shape[0]
    out = arr.copy()
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = arr[j, i]
    return from_numpy(out)


# ---- generator interface ---------------------------------------------------
@dataclass(frozen=True)
class GeneratedTask:
    task: Task
    rule: str
    transform: Transform


class Generator(abc.ABC):
    name: str = "generator"

    @abc.abstractmethod
    def sample(self, rng: np.random.Generator, n_inputs: int) -> tuple[Transform, list[Grid]]:
        """Return (transform, inputs) — inputs sized so outputs stay <= 30x30."""
        raise NotImplementedError


class RecolorGenerator(Generator):
    name = "recolor"

    def sample(self, rng, n_inputs):
        palette = _palette(rng)
        shuffled = list(rng.permutation(palette))
        mapping = {int(s): int(d) for s, d in zip(palette, shuffled, strict=False)}
        transform = colormap_program(mapping)
        inputs = [
            _random_grid(rng, _size(rng, 3, 12), _size(rng, 3, 12), palette)
            for _ in range(n_inputs)
        ]
        return transform, inputs


class ColorSwapGenerator(Generator):
    name = "color_swap"

    def sample(self, rng, n_inputs):
        palette = _palette(rng, k=3)
        a, b = (int(x) for x in rng.choice(palette[1:], size=2, replace=False))
        mapping = {a: b, b: a}
        transform = colormap_program(mapping)
        inputs = [
            _random_grid(rng, _size(rng, 3, 12), _size(rng, 3, 12), palette)
            for _ in range(n_inputs)
        ]
        return transform, inputs


class SymmetryGenerator(Generator):
    name = "symmetry"

    def sample(self, rng, n_inputs):
        op = str(rng.choice([n for n in symmetry.D4_NAMES if n != "identity"]))
        transform = (lambda name: (lambda g: symmetry.apply(name, g)))(op)
        palette = _palette(rng)
        inputs = [
            _random_grid(rng, _size(rng, 3, 14), _size(rng, 3, 14), palette)
            for _ in range(n_inputs)
        ]
        return transform, inputs


class ScaleGenerator(Generator):
    name = "scale"

    def sample(self, rng, n_inputs):
        fy, fx = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        if (fy, fx) == (1, 1):
            fx = 2
        transform = scale_program(fy, fx)
        palette = _palette(rng)
        max_h, max_w = 30 // fy, 30 // fx
        inputs = [
            _random_grid(rng, _size(rng, 2, min(max_h, 10)), _size(rng, 2, min(max_w, 10)), palette)
            for _ in range(n_inputs)
        ]
        return transform, inputs


class TileGenerator(Generator):
    name = "tile"

    def sample(self, rng, n_inputs):
        ny, nx = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        if (ny, nx) == (1, 1):
            nx = 2
        transform = tile_program(ny, nx)
        palette = _palette(rng)
        max_h, max_w = 30 // ny, 30 // nx
        inputs = [
            _random_grid(rng, _size(rng, 2, min(max_h, 10)), _size(rng, 2, min(max_w, 10)), palette)
            for _ in range(n_inputs)
        ]
        return transform, inputs


class MirrorConcatGenerator(Generator):
    name = "mirror_concat"

    def sample(self, rng, n_inputs):
        palette = _palette(rng)
        inputs = [
            _random_grid(rng, _size(rng, 3, 12), _size(rng, 2, 14), palette)
            for _ in range(n_inputs)
        ]
        return _mirror_concat_h, inputs


class BorderGenerator(Generator):
    name = "border"

    def sample(self, rng, n_inputs):
        palette = _palette(rng)
        color = int(rng.choice(palette[1:]))
        transform = _border(color, width=1)
        inputs = [
            _random_grid(rng, _size(rng, 3, 26), _size(rng, 3, 26), palette)
            for _ in range(n_inputs)
        ]
        return transform, inputs


class CropToContentGenerator(Generator):
    name = "crop_to_content"

    def sample(self, rng, n_inputs):
        palette = _palette(rng)
        inputs = [self._canvas(rng, palette) for _ in range(n_inputs)]
        return crop_to_content, inputs

    def _canvas(self, rng, palette) -> Grid:
        H, W = _size(rng, 8, 16), _size(rng, 8, 16)
        arr = np.full((H, W), BG, dtype=np.int8)
        oh, ow = _size(rng, 2, max(2, H - 2)), _size(rng, 2, max(2, W - 2))
        r0, c0 = int(rng.integers(0, H - oh + 1)), int(rng.integers(0, W - ow + 1))
        obj = np.asarray(
            rng.choice(palette[1:], size=(oh, ow)), dtype=np.int8
        )  # non-bg object
        arr[r0 : r0 + oh, c0 : c0 + ow] = obj
        return from_numpy(arr)


class GravityGenerator(Generator):
    name = "gravity"

    def sample(self, rng, n_inputs):
        palette = _palette(rng)
        # sparse grids so falling is visible (bias toward background)
        weights = self._weights(palette)
        inputs = [
            _random_grid(rng, _size(rng, 4, 14), _size(rng, 3, 12), palette, weights)
            for _ in range(n_inputs)
        ]
        return _gravity_down, inputs

    @staticmethod
    def _weights(palette: list[int]) -> list[float]:
        n = len(palette)
        w = [0.6] + [0.4 / (n - 1)] * (n - 1)  # 60% background
        return w


class ObjectRecolorBySizeGenerator(Generator):
    """Concept: recolour the LARGEST object one colour, every smaller object
    another. Requires comparing connected-component sizes -- not expressible
    as a fixed per-value colormap (see `_recolor_by_size`)."""

    name = "object_recolor_by_size"

    def sample(self, rng, n_inputs):
        palette = _palette(rng, k=4)
        big_color, small_color = (
            int(x) for x in rng.choice(palette[1:], size=2, replace=False)
        )
        transform = _recolor_by_size(big_color, small_color)
        inputs = [
            _scattered_objects(
                rng, palette[1:], _size(rng, 10, 20), _size(rng, 10, 20), int(rng.integers(2, 5))
            )
            for _ in range(n_inputs)
        ]
        return transform, inputs


class ObjectWallGenerator(Generator):
    """Concept: slide every object flush against one wall (fixed per task).
    Objects start at different distances from the wall, so this is a
    per-object translation, not one rigid whole-grid transform."""

    name = "object_wall"
    _DIRS = ("up", "down", "left", "right")

    def sample(self, rng, n_inputs):
        direction = str(rng.choice(self._DIRS))
        palette = _palette(rng, k=4)
        transform = _slide_to_wall(direction)
        inputs = [self._canvas(rng, palette, direction) for _ in range(n_inputs)]
        return transform, inputs

    def _canvas(self, rng, palette, direction) -> Grid:
        h, w = _size(rng, 12, 22), _size(rng, 12, 22)
        arr = np.full((h, w), BG, dtype=np.int8)
        n_objs = int(rng.integers(2, 4))
        vertical = direction in ("up", "down")
        perp = w if vertical else h
        band = perp // n_objs
        for i in range(n_objs):
            dim = _size(rng, 1, max(1, min(3, band - 2)))
            offset = int(rng.integers(0, max(1, band - dim)))
            pos = i * band + offset
            other = _size(rng, 1, 3)
            color = int(rng.choice(palette[1:]))
            if vertical:
                c0, oh, ow = pos, other, dim
                r0 = int(rng.integers(0, h - oh + 1))
            else:
                r0, oh, ow = pos, dim, other
                c0 = int(rng.integers(0, w - ow + 1))
            arr[r0 : r0 + oh, c0 : c0 + ow] = color
        return from_numpy(arr)


class CountEncodeGenerator(Generator):
    """Concept: output a 1xN bar whose filled prefix encodes the COUNT of
    objects in the input. The DSL never derives a new small grid from an
    aggregate statistic of the input -- it only reshapes/recolours what is
    already there."""

    name = "count_encode"
    _MAX_COUNT = 9

    def sample(self, rng, n_inputs):
        palette = _palette(rng, k=3)
        mark = int(rng.choice(palette[1:]))
        obj_color = int(rng.choice([c for c in palette[1:] if c != mark]))

        def transform(g: Grid) -> Grid:
            arr = to_numpy(g)
            count = min(len(_connected_components(arr, BG)), self._MAX_COUNT)
            row = [mark] * count + [BG] * (self._MAX_COUNT - count)
            return (tuple(row),)

        inputs = [self._canvas(rng, obj_color) for _ in range(n_inputs)]
        return transform, inputs

    def _canvas(self, rng, obj_color) -> Grid:
        h, w = _size(rng, 8, 16), _size(rng, 8, 16)
        arr = np.full((h, w), BG, dtype=np.int8)
        n_objs = int(rng.integers(1, self._MAX_COUNT + 1))
        placed, attempts = 0, 0
        while placed < n_objs and attempts < 40:
            attempts += 1
            r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
            if arr[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2].any():
                continue
            arr[r, c] = obj_color
            placed += 1
        return from_numpy(arr)


class SymmetryRepairGenerator(Generator):
    """Concept: a mirror-symmetric grid has one region occluded by a noise
    colour; restore it from the mirrored (still-visible) half. The DSL never
    exploits a grid's OWN internal symmetry -- it only applies whole-grid
    rigid transforms."""

    name = "symmetry_repair"
    NOISE = 5  # reserved occlusion marker, excluded from the task's palette

    def sample(self, rng, n_inputs):
        axis = str(rng.choice(["h", "v"]))
        nonzero = [c for c in range(1, 10) if c != self.NOISE]
        chosen = rng.choice(nonzero, size=int(rng.integers(2, 4)), replace=False)
        palette = [BG] + [int(c) for c in chosen]
        transform = _repair_symmetry(axis, self.NOISE)
        inputs = [self._occluded(rng, palette, axis) for _ in range(n_inputs)]
        return transform, inputs

    def _occluded(self, rng, palette, axis) -> Grid:
        h, w = _size(rng, 6, 16), _size(rng, 6, 16)
        arr = to_numpy(_mirror_grid(rng, h, w, palette, axis)).copy()
        if axis == "h":
            half = w // 2 or 1
            ow, oh = _size(rng, 1, half), _size(rng, 1, h)
            r0, c0 = int(rng.integers(0, h - oh + 1)), int(rng.integers(0, half - ow + 1))
        else:
            half = h // 2 or 1
            oh, ow = _size(rng, 1, half), _size(rng, 1, w)
            r0, c0 = int(rng.integers(0, half - oh + 1)), int(rng.integers(0, w - ow + 1))
        arr[r0 : r0 + oh, c0 : c0 + ow] = self.NOISE
        return from_numpy(arr)


class ColorByNeighborRuleGenerator(Generator):
    """Concept: background cells touching a trigger colour turn into a halo
    colour; other cells are untouched. A cellwise rule keyed on NEIGHBOURS,
    not the DSL's fixed per-value colormap."""

    name = "color_by_neighbor"

    def sample(self, rng, n_inputs):
        palette = _palette(rng, k=3)
        trigger, halo = (int(x) for x in rng.choice(palette[1:], size=2, replace=False))
        transform = _neighbor_halo(trigger, halo)
        weights = [0.7] + [0.3 / (len(palette) - 1)] * (len(palette) - 1)
        inputs = [
            _random_grid(rng, _size(rng, 6, 18), _size(rng, 6, 18), palette, weights)
            for _ in range(n_inputs)
        ]
        return transform, inputs


class CompositionGenerator(Generator):
    """Concept: chain TWO already-hard, non-DSL ops -- recolour the largest
    object, then let everything fall under gravity. Depth-2 DSL search only
    composes from its own ~10-op vocabulary, so a pipeline built from two ops
    already outside that vocabulary is a fortiori outside every depth-2 slot
    it enumerates."""

    name = "composition"

    def sample(self, rng, n_inputs):
        palette = _palette(rng, k=4)
        big_color, small_color = (
            int(x) for x in rng.choice(palette[1:], size=2, replace=False)
        )
        recolor = _recolor_by_size(big_color, small_color)

        def transform(g: Grid) -> Grid:
            return _gravity_down(recolor(g))

        inputs = [
            _scattered_objects(
                rng, palette[1:], _size(rng, 10, 18), _size(rng, 8, 14), int(rng.integers(2, 5))
            )
            for _ in range(n_inputs)
        ]
        return transform, inputs


class DiagonalReflectSelectiveGenerator(Generator):
    """Concept: mirror only the half above the main diagonal onto itself; the
    lower half (incl. diagonal) is left untouched. A partial, not whole-grid,
    reflection -- the DSL's `transpose` always reflects everything."""

    name = "diagonal_reflect_selective"

    def sample(self, rng, n_inputs):
        palette = _palette(rng)
        sizes = [_size(rng, 4, 20) for _ in range(n_inputs)]
        inputs = [_random_grid(rng, n, n, palette) for n in sizes]
        return _reflect_upper_triangle, inputs


GENERATORS: tuple[Generator, ...] = (
    RecolorGenerator(),
    ColorSwapGenerator(),
    SymmetryGenerator(),
    ScaleGenerator(),
    TileGenerator(),
    MirrorConcatGenerator(),
    BorderGenerator(),
    CropToContentGenerator(),
    GravityGenerator(),
    ObjectRecolorBySizeGenerator(),
    ObjectWallGenerator(),
    CountEncodeGenerator(),
    SymmetryRepairGenerator(),
    ColorByNeighborRuleGenerator(),
    CompositionGenerator(),
    DiagonalReflectSelectiveGenerator(),
)

# The subset of GENERATORS whose concept lies provably outside the DSL micro-
# solver's vocabulary (single/depth-2 compositions of D4 ops, crop-to-content,
# a learned same-shape colormap, and integer scale/tile) -- see
# `tests/test_synth_hard.py` for the per-generator, per-seed proof.
HARD_GENERATORS: tuple[Generator, ...] = (
    ObjectRecolorBySizeGenerator(),
    ObjectWallGenerator(),
    CountEncodeGenerator(),
    SymmetryRepairGenerator(),
    ColorByNeighborRuleGenerator(),
    CompositionGenerator(),
    DiagonalReflectSelectiveGenerator(),
)


def build_task(gen: Generator, seed: int, num_pairs: int = 3) -> GeneratedTask:
    """Generate one self-consistent task from `gen`, seeded for determinism."""
    rng = np.random.default_rng(seed)
    transform, inputs = gen.sample(rng, num_pairs + 1)
    pairs = [Pair(input=g, output=transform(g)) for g in inputs]
    train = tuple(pairs[:num_pairs])
    test = pairs[num_pairs]
    task = Task(
        task_id=f"synth-{gen.name}-{seed}",
        train=train,
        test=(Pair(input=test.input, output=test.output),),
    )
    return GeneratedTask(task=task, rule=gen.name, transform=transform)


def is_well_formed(gt: GeneratedTask) -> bool:
    """All grids valid (<=30x30, symbols 0-9) and the rule reproduces outputs."""
    for pair in (*gt.task.train, *gt.task.test):
        if not is_valid_grid(pair.input) or not is_valid_grid(pair.output):
            return False
        if gt.transform(pair.input) != pair.output:
            return False
    return True
