"""Shared test fixtures and grid/task factories."""

from __future__ import annotations

import numpy as np
import pytest

from arc.io.grid import Grid, from_numpy
from arc.io.loader import Pair, Task


@pytest.fixture
def grid_factory():
    """Returns make(h, w, ncolors=6, seed=0) -> Grid of random symbols."""

    def make(h: int, w: int, ncolors: int = 6, seed: int = 0) -> Grid:
        rng = np.random.default_rng(seed)
        return from_numpy(rng.integers(0, ncolors, size=(h, w)))

    return make


@pytest.fixture
def task_factory():
    """Returns make(train_pairs, test_inputs, task_id="synthetic") -> Task.

    `train_pairs` is a list of (input_grid, output_grid); `test_inputs` a list of
    input grids.
    """

    def make(train_pairs, test_inputs, task_id: str = "synthetic") -> Task:
        train = tuple(Pair(input=i, output=o) for i, o in train_pairs)
        test = tuple(Pair(input=i) for i in test_inputs)
        return Task(task_id=task_id, train=train, test=test)

    return make
