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
