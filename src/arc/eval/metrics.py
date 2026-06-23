"""The competition metric: top-2 exact-match, averaged over test outputs.

For each test output there is one ground-truth grid. A task output scores 1 if
EITHER of the two attempts matches the truth exactly, else 0. The final score is
the mean over all test outputs across all tasks.
"""

from __future__ import annotations

from ..io.grid import Grid, grids_equal
from ..io.submission import Attempt, Predictions


def score_output(attempt: Attempt, truth: Grid) -> int:
    """1 if either guess equals the ground truth exactly, else 0."""
    return int(grids_equal(attempt.attempt_1, truth) or grids_equal(attempt.attempt_2, truth))


def score_predictions(
    predictions: Predictions,
    solutions: dict[str, list[Grid]],
) -> dict:
    """Score predictions against ground-truth solutions.

    Returns a summary dict with the overall mean plus raw counts. Only task
    outputs that have a ground-truth entry are scored (so this works on any
    subset, e.g. a smoke-test slice of the eval split).
    """
    total = 0
    correct = 0
    per_task: dict[str, float] = {}
    for task_id, truths in solutions.items():
        attempts = predictions.get(task_id)
        if attempts is None:
            continue
        task_correct = 0
        n = min(len(attempts), len(truths))
        for i in range(n):
            s = score_output(attempts[i], truths[i])
            task_correct += s
            correct += s
            total += 1
        per_task[task_id] = task_correct / n if n else 0.0

    return {
        "score": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "num_tasks": len(per_task),
        "per_task": per_task,
    }
