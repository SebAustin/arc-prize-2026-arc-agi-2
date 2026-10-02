"""The competition metric (top-2 exact-match) plus partial-credit signals.

The competition scores a test output 1 if EITHER of the two attempts matches the
truth exactly, else 0; the final score is the mean over all test outputs. That is
the only number that counts for the prize — but at low ability it is 0 for every
candidate, so it gives no gradient to iterate on. `score_predictions` therefore
also reports two sub-exact-match signals that move continuously below the floor:

  * shape_correct_rate — did either attempt at least get the output DIMENSIONS
    right? (a coarse but meaningful first milestone on any ARC task)
  * cell_accuracy — best per-cell match fraction across the two attempts, counted
    only when the shape matches (a wrong-sized grid is wholly wrong, so 0.0).

These are diagnostics for steering, never a substitute for exact-match.
"""

from __future__ import annotations

from ..io.grid import Grid, grids_equal
from ..io.submission import Attempt, Predictions


def score_output(attempt: Attempt, truth: Grid) -> int:
    """1 if either guess equals the ground truth exactly, else 0."""
    return int(grids_equal(attempt.attempt_1, truth) or grids_equal(attempt.attempt_2, truth))


def _shape(grid: Grid) -> tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def _cell_accuracy(grid: Grid, truth: Grid) -> float:
    """Per-cell match fraction; 0.0 when the shape differs (wrong size = wrong)."""
    if _shape(grid) != _shape(truth):
        return 0.0
    total = 0
    match = 0
    # strict=False: this is a lenient diagnostic — a ragged/malformed grid should
    # score low, never raise and break the whole eval summary. Row/col counts are
    # already shape-guarded above, so well-formed grids iterate fully regardless.
    for grid_row, truth_row in zip(grid, truth, strict=False):
        for gc, tc in zip(grid_row, truth_row, strict=False):
            total += 1
            match += gc == tc
    return match / total if total else 0.0


def best_cell_accuracy(attempt: Attempt, truth: Grid) -> float:
    """Best per-cell accuracy across the two attempts."""
    return max(
        _cell_accuracy(attempt.attempt_1, truth),
        _cell_accuracy(attempt.attempt_2, truth),
    )


def shape_correct(attempt: Attempt, truth: Grid) -> int:
    """1 if either attempt has the exact output dimensions of the truth."""
    ts = _shape(truth)
    return int(_shape(attempt.attempt_1) == ts or _shape(attempt.attempt_2) == ts)


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
    cell_acc_sum = 0.0
    shape_sum = 0
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
            cell_acc_sum += best_cell_accuracy(attempts[i], truths[i])
            shape_sum += shape_correct(attempts[i], truths[i])
            total += 1
        per_task[task_id] = task_correct / n if n else 0.0

    return {
        "score": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "num_tasks": len(per_task),
        # Partial-credit gradient below the exact-match floor (diagnostic only).
        "cell_accuracy": cell_acc_sum / total if total else 0.0,
        "shape_correct_rate": shape_sum / total if total else 0.0,
        "per_task": per_task,
    }
