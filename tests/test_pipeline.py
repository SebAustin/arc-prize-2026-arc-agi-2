"""Pipeline orchestration tests: voting, attempts, schema validity, watchdog."""

from __future__ import annotations

from arc.augment import symmetry
from arc.io.submission import build_submission, validate_submission
from arc.pipeline import default_solvers, run, solve_task
from arc.solvers.base import Candidates, Solver


def _rot90(g):
    return symmetry.apply("rot90", g)


def test_solve_task_picks_verified_answer(grid_factory, task_factory):
    inputs = [grid_factory(3, 3, ncolors=6, seed=s) for s in range(3)]
    test_in = grid_factory(3, 3, ncolors=6, seed=44)
    task = task_factory([(g, _rot90(g)) for g in inputs], [test_in])
    attempts = solve_task(task, default_solvers(), budget_s=5.0)
    assert attempts[0].attempt_1 == _rot90(test_in)


def test_run_produces_valid_submission(grid_factory, task_factory):
    inputs = [grid_factory(3, 3, seed=s) for s in range(3)]
    tasks = {
        "rot": task_factory(
            [(g, _rot90(g)) for g in inputs], [grid_factory(3, 3, seed=5)], task_id="rot"
        ),
        "hard": task_factory(
            [(grid_factory(3, 3, seed=s), grid_factory(3, 3, seed=99 + s)) for s in range(3)],
            [grid_factory(3, 3, seed=7)],
            task_id="hard",
        ),
    }
    preds = run(tasks, per_task_budget_s=5.0)
    # every task has attempts; submission is schema-valid even for the hard task.
    assert set(preds) == set(tasks)
    assert validate_submission(build_submission(preds), tasks) == []


def test_watchdog_zero_budget_still_valid(grid_factory, task_factory):
    tasks = {
        "t": task_factory(
            [(grid_factory(2, 2, seed=1), grid_factory(2, 2, seed=2))],
            [grid_factory(2, 2, seed=3)],
            task_id="t",
        )
    }
    preds = run(tasks, total_budget_s=0.0)  # exhausted immediately
    # fallback predictions still present and valid
    assert validate_submission(build_submission(preds), tasks) == []


class _RecordingSolver(Solver):
    name = "recording"

    def __init__(self):
        self.budgets: list[float] = []

    def solve(self, task, budget_s: float) -> Candidates:
        self.budgets.append(budget_s)
        return self._empty(task)


def test_solvers_share_one_budget(grid_factory, task_factory):
    g = grid_factory(2, 2, seed=1)
    task = task_factory([(g, g)], [g])
    s1, s2 = _RecordingSolver(), _RecordingSolver()
    solve_task(task, [s1, s2], budget_s=10.0)
    assert s1.budgets and s2.budgets
    assert s1.budgets[0] <= 10.0 + 1e-6           # first gets (almost) the full budget
    assert s2.budgets[0] <= s1.budgets[0] + 1e-6  # time only moves forward
