"""Ensemble orchestration: run solvers, vote candidates into two attempts per
test output, and guarantee a complete valid submission under a hard time budget.

Design notes:
  * A complete FALLBACK submission is written before any solving begins, then
    overwritten as real answers arrive and re-checkpointed periodically. If the
    process is killed mid-run, a valid file is already on disk.
  * Candidates from all solvers are merged by weighted voting; the same merge
    will absorb the LLM-TTT solver's candidates in later milestones.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from .config import DEFAULT_PER_TASK_BUDGET_S, TOTAL_RUNTIME_BUDGET_S
from .io.grid import Grid
from .io.loader import Task
from .io.submission import (
    FALLBACK_GRID,
    Attempt,
    Predictions,
    build_submission,
    empty_predictions,
    write_submission,
)
from .solvers.base import Candidates, Solver
from .solvers.dsl.solver import DSLSolver
from .solvers.identity import CHEAP_SOLVERS

# Base vote weight per solver (verified solvers dominate heuristic priors).
# identity must stay BELOW llm_ttt/2: votes decay as weight/(rank+1), so at 8.0
# identity's rank-0 vote (8.0) outbid llm_ttt's rank-1 vote (4.5) and attempt_2
# was routinely wasted on "output = input" — almost never right on ARC-AGI-2.
SOLVER_WEIGHTS: dict[str, float] = {
    "dsl": 10.0,
    "identity": 2.0,
    "constant_output": 1.5,
    "llm_ttt": 9.0,
    "llm": 4.0,
    "majority_shape": 0.5,
}
_DEFAULT_WEIGHT = 1.0
_CHECKPOINT_EVERY = 25  # rewrite submission at least every N tasks
_CHECKPOINT_SECONDS = 300.0  # ...and at least every N seconds of wall-clock

_log = logging.getLogger(__name__)


def default_solvers(llm_model=None, llm_kwargs: dict | None = None) -> list[Solver]:
    """The CPU ensemble (DSL + cheap heuristics). If `llm_model` is supplied, an
    `LLMSolver` is appended — works with the real HFModel on Kaggle or a MockModel
    locally, so the full ensemble is testable without a GPU."""
    solvers: list[Solver] = [DSLSolver(), *CHEAP_SOLVERS]
    if llm_model is not None:
        from .solvers.llm.solver import LLMSolver

        solvers.append(LLMSolver(llm_model, **(llm_kwargs or {})))
    return solvers


def solve_task(task: Task, solvers: list[Solver], budget_s: float) -> list[Attempt]:
    """Run all solvers on a task and vote candidates into two attempts per test
    output. Solvers share one per-task budget: each receives the time remaining,
    so cheap solvers (listed first) run free and the expensive LLM/TTT gets the
    rest."""
    deadline = time.monotonic() + budget_s
    votes: list[defaultdict[Grid, float]] = [defaultdict(float) for _ in task.test]
    for solver in solvers:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            candidates: Candidates = solver.solve(task, remaining)
        except Exception:
            # Never let one solver sink the task: log for diagnosis (silent
            # degradation to the remaining solvers is otherwise invisible in the
            # Kaggle run log) and fall through to whatever else voted.
            _log.exception("solver %s failed on task %s", solver.name, task.task_id)
            continue
        weight = SOLVER_WEIGHTS.get(solver.name, _DEFAULT_WEIGHT)
        for i, ranked in enumerate(candidates):
            if i >= len(votes):
                break
            for rank, grid in enumerate(ranked):
                votes[i][grid] += weight / (rank + 1)

    return [_top2(v) for v in votes]


def _top2(vote: defaultdict[Grid, float]) -> Attempt:
    """Pick the two highest-weighted distinct grids (with safe fallbacks)."""
    ranked = sorted(vote.items(), key=lambda kv: (-kv[1], _grid_size(kv[0])))
    a1 = ranked[0][0] if ranked else FALLBACK_GRID
    a2 = ranked[1][0] if len(ranked) > 1 else a1
    return Attempt(a1, a2)


def _grid_size(grid: Grid) -> int:
    return len(grid) * (len(grid[0]) if grid else 0)


def run(
    tasks: dict[str, Task],
    solvers: list[Solver] | None = None,
    output_path=None,
    total_budget_s: float = TOTAL_RUNTIME_BUDGET_S,
    per_task_budget_s: float = DEFAULT_PER_TASK_BUDGET_S,
    verbose: bool = False,
) -> Predictions:
    """Solve all tasks under a global time budget, checkpointing the submission."""
    solvers = solvers if solvers is not None else default_solvers()
    for solver in solvers:
        if solver.name not in SOLVER_WEIGHTS:
            _log.warning(
                "solver %r has no registered vote weight; using default %.1f",
                solver.name,
                _DEFAULT_WEIGHT,
            )
    predictions = empty_predictions(tasks)  # complete valid fallback up front
    if output_path is not None:
        write_submission(build_submission(predictions), output_path)

    start = time.monotonic()
    last_checkpoint = start
    for n, (task_id, task) in enumerate(tasks.items(), 1):
        if time.monotonic() - start > total_budget_s:
            if verbose:
                print(f"[watchdog] budget exhausted after {n - 1} tasks")
            break
        predictions[task_id] = solve_task(task, solvers, per_task_budget_s)
        now = time.monotonic()
        # Checkpoint on task count OR elapsed time — a slow run (few tasks, long
        # each) still flushes real answers instead of losing up to an hour to a
        # crash between the count-based checkpoints.
        due = n % _CHECKPOINT_EVERY == 0 or now - last_checkpoint >= _CHECKPOINT_SECONDS
        if output_path is not None and due:
            write_submission(build_submission(predictions), output_path)
            last_checkpoint = now
        if verbose and n % _CHECKPOINT_EVERY == 0:
            print(f"[pipeline] solved {n}/{len(tasks)} tasks")

    if output_path is not None:
        write_submission(build_submission(predictions), output_path)
    return predictions
