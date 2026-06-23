"""Solver interface and the train-verification contract.

A Solver examines a Task and proposes, for each test input (in order), a ranked
list of candidate output grids (best first, possibly empty). The pipeline merges
candidates from several solvers into the final two attempts.

The shared `verify_program` helper encodes the central safety principle: a
grid->grid transformation is only trustworthy if it reproduces EVERY
demonstration output exactly. On a novel, label-free test task the train pairs
are the only ground truth available, so they double as the acceptance test.
"""

from __future__ import annotations

import abc
from collections.abc import Callable

from ..io.grid import Grid, grids_equal
from ..io.loader import Task

# A program is any grid -> grid function (may raise; callers guard).
Program = Callable[[Grid], Grid]

# Per test input, a ranked list of candidate grids.
Candidates = list[list[Grid]]


def verify_program(program: Program, task: Task) -> bool:
    """True iff `program` maps every train input to its exact train output."""
    if not task.train:
        return False
    for pair in task.train:
        if pair.output is None:
            return False
        try:
            predicted = program(pair.input)
        except Exception:
            return False
        if not grids_equal(predicted, pair.output):
            return False
    return True


def apply_to_tests(program: Program, task: Task) -> list[Grid | None]:
    """Apply a (verified) program to each test input; None on failure."""
    out: list[Grid | None] = []
    for pair in task.test:
        try:
            out.append(program(pair.input))
        except Exception:
            out.append(None)
    return out


class Solver(abc.ABC):
    """Base class for all solvers."""

    name: str = "solver"

    @abc.abstractmethod
    def solve(self, task: Task, budget_s: float) -> Candidates:
        """Return ranked candidate grids per test input (outer index = test i)."""
        raise NotImplementedError

    def _empty(self, task: Task) -> Candidates:
        return [[] for _ in task.test]
