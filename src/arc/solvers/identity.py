"""Cheap, always-available heuristic solvers (microsecond cost).

These exist as an ensemble floor and a fallback when expensive solvers time out
or OOM. Each still respects the verify-on-train contract where applicable.
"""

from __future__ import annotations

from collections import Counter

from ..io.grid import Grid
from ..io.loader import Task
from .base import Candidates, Solver, apply_to_tests, verify_program


class IdentitySolver(Solver):
    """Predict output == input. Wins the (rare) tasks where the grid is unchanged."""

    name = "identity"

    def solve(self, task: Task, budget_s: float) -> Candidates:
        if not verify_program(lambda g: g, task):
            return self._empty(task)
        preds = apply_to_tests(lambda g: g, task)
        return [[p] if p is not None else [] for p in preds]


class ConstantOutputSolver(Solver):
    """If all train outputs are identical, predict that constant grid.

    Captures tasks whose answer ignores the input (e.g. a fixed legend/key).
    """

    name = "constant_output"

    def solve(self, task: Task, budget_s: float) -> Candidates:
        outputs = [p.output for p in task.train if p.output is not None]
        if not outputs:
            return self._empty(task)
        first = outputs[0]
        if all(o == first for o in outputs):
            return [[first] for _ in task.test]
        return self._empty(task)


class MajorityShapeSolver(Solver):
    """Weak prior: emit a background-filled grid at the most common output shape.

    Never verified against train, so the pipeline must rank it below verified
    candidates. Useful only as a last-resort, non-trivial fallback shape.
    """

    name = "majority_shape"

    def solve(self, task: Task, budget_s: float) -> Candidates:
        shapes = [
            (len(p.output), len(p.output[0]))
            for p in task.train
            if p.output is not None and p.output
        ]
        if not shapes:
            return self._empty(task)
        (h, w), _ = Counter(shapes).most_common(1)[0]
        # Most frequent symbol across train outputs as the fill colour.
        fill = _most_common_symbol(task)
        grid: Grid = tuple(tuple(fill for _ in range(w)) for _ in range(h))
        return [[grid] for _ in task.test]


def _most_common_symbol(task: Task) -> int:
    counter: Counter[int] = Counter()
    for pair in task.train:
        if pair.output is None:
            continue
        for row in pair.output:
            counter.update(row)
    return counter.most_common(1)[0][0] if counter else 0


CHEAP_SOLVERS: tuple[Solver, ...] = (
    IdentitySolver(),
    ConstantOutputSolver(),
    MajorityShapeSolver(),
)
