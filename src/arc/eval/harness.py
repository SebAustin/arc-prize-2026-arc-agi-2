"""Evaluation harness: load a split, run the pipeline, score exact-match top-2.

Used both for local development (smoke runs over the public eval split) and as
the per-milestone regression check. Scoring requires a solutions file, so it
works on 'training' and 'evaluation' but not 'test' (solutions hidden).
"""

from __future__ import annotations

from ..config import Config, get_config
from ..io.loader import Task, load_challenges, load_solutions
from ..pipeline import run as run_pipeline
from ..solvers.base import Solver
from .metrics import score_predictions


def load_split(split: str, cfg: Config | None = None, limit: int | None = None):
    """Load (tasks, solutions|None) for a split, optionally truncated to `limit`."""
    cfg = cfg or get_config()
    tasks = load_challenges(cfg.challenges_path(split))
    if limit is not None:
        tasks = dict(list(tasks.items())[:limit])
    solutions = None
    if split in ("training", "evaluation"):
        sol = load_solutions(cfg.solutions_path(split))
        solutions = {k: v for k, v in sol.items() if k in tasks}
    return tasks, solutions


def evaluate(
    split: str = "evaluation",
    solvers: list[Solver] | None = None,
    limit: int | None = None,
    per_task_budget_s: float = 10.0,
    verbose: bool = False,
) -> dict:
    """Run the pipeline over a split and return a score summary."""
    tasks, solutions = load_split(split, limit=limit)
    predictions = run_pipeline(
        tasks,
        solvers=solvers,
        output_path=None,
        per_task_budget_s=per_task_budget_s,
        verbose=verbose,
    )
    if solutions is None:
        return {"split": split, "num_tasks": len(tasks), "scored": False}

    summary = score_predictions(predictions, solutions)
    summary.update({"split": split, "scored": True})
    return summary
