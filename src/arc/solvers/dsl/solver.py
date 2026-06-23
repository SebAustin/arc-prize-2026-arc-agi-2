"""DSLSolver — wraps the program search behind the Solver interface."""

from __future__ import annotations

from ...io.grid import Grid
from ...io.loader import Task
from ..base import Candidates, Solver, apply_to_tests
from .search import search


class DSLSolver(Solver):
    """Search for grid programs that reproduce every train pair, then apply the
    simplest survivors to each test input as ranked candidates."""

    name = "dsl"

    def __init__(self, max_candidates: int = 4):
        self.max_candidates = max_candidates

    def solve(self, task: Task, budget_s: float) -> Candidates:
        programs = search(task, budget_s=budget_s)
        if not programs:
            return self._empty(task)

        per_test: Candidates = [[] for _ in task.test]
        for _label, program in programs:
            preds = apply_to_tests(program, task)
            for i, pred in enumerate(preds):
                if pred is None:
                    continue
                bucket = per_test[i]
                if pred not in bucket and len(bucket) < self.max_candidates:
                    bucket.append(pred)
        return per_test
