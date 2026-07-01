"""Build, validate and write the competition submission.json.

Required schema (validated against the test challenges):
    {task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}
  - one dict per test input, in the SAME order as the task's test inputs;
  - BOTH attempt_1 and attempt_2 must be present for every test output;
  - EVERY task_id in the challenges file must appear in the submission.

We model a single test output's two guesses as an `Attempt(attempt_1, attempt_2)`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .grid import Grid, is_valid_grid, to_lists
from .loader import Task

# Fallback grid used whenever a solver produced nothing — guarantees the schema
# is always satisfiable. A 1x1 zero grid is the cheapest valid placeholder.
FALLBACK_GRID: Grid = ((0,),)


@dataclass(frozen=True)
class Attempt:
    """The two guesses for one test output."""

    attempt_1: Grid
    attempt_2: Grid


# A full prediction set: {task_id: [Attempt per test input, in order]}
Predictions = dict[str, list[Attempt]]


def build_submission(predictions: Predictions) -> dict:
    """Serialise predictions into the competition's JSON structure."""
    submission: dict[str, list[dict]] = {}
    for task_id, attempts in predictions.items():
        submission[task_id] = [
            {
                "attempt_1": to_lists(a.attempt_1),
                "attempt_2": to_lists(a.attempt_2),
            }
            for a in attempts
        ]
    return submission


def empty_predictions(tasks: dict[str, Task]) -> Predictions:
    """A complete, schema-valid fallback prediction (all 1x1 zeros).

    The pipeline writes this first so a valid submission always exists, then
    overwrites entries as real answers arrive (time-watchdog safety net).
    """
    return {
        task_id: [Attempt(FALLBACK_GRID, FALLBACK_GRID) for _ in task.test]
        for task_id, task in tasks.items()
    }


def fallback_from_raw(raw: object) -> Predictions:
    """Best-effort schema-valid fallback from *raw* (possibly malformed) challenge
    JSON: one 1x1-zero Attempt per test input, defaulting to a single output when
    a task's test count cannot be determined.

    Used by the Kaggle entrypoint to guarantee a scoreable submission exists on
    disk *before* any parsing/model work, so a crash or OOM anywhere downstream
    cannot leave an empty `/kaggle/working`.
    """
    preds: Predictions = {}
    if not isinstance(raw, dict):
        return preds
    for task_id, body in raw.items():
        n = 1
        if isinstance(body, dict):
            test = body.get("test")
            if isinstance(test, list) and test:
                n = len(test)
        preds[str(task_id)] = [Attempt(FALLBACK_GRID, FALLBACK_GRID) for _ in range(n)]
    return preds


def validate_submission(submission: dict, tasks: dict[str, Task]) -> list[str]:
    """Return a list of schema problems; empty list means the submission is valid."""
    problems: list[str] = []

    missing = set(tasks) - set(submission)
    if missing:
        problems.append(f"missing {len(missing)} task_ids, e.g. {sorted(missing)[:3]}")
    extra = set(submission) - set(tasks)
    if extra:
        problems.append(f"unexpected {len(extra)} task_ids, e.g. {sorted(extra)[:3]}")

    for task_id, task in tasks.items():
        entry = submission.get(task_id)
        if entry is None:
            continue
        if not isinstance(entry, list) or len(entry) != task.num_test:
            problems.append(
                f"{task_id}: expected {task.num_test} outputs, got "
                f"{len(entry) if isinstance(entry, list) else type(entry).__name__}"
            )
            continue
        for i, out in enumerate(entry):
            for key in ("attempt_1", "attempt_2"):
                if key not in out:
                    problems.append(f"{task_id}[{i}]: missing {key}")
                    continue
                grid = _coerce_grid(out[key])
                if grid is None or not is_valid_grid(grid):
                    problems.append(f"{task_id}[{i}].{key}: invalid grid")
    return problems


def _coerce_grid(value) -> Grid | None:
    """Best-effort convert a JSON list-of-lists into a Grid for validation."""
    if not isinstance(value, list) or not value:
        return None
    try:
        return tuple(tuple(int(c) for c in row) for row in value)
    except (TypeError, ValueError):
        return None


def write_submission(submission: dict, path: str | Path) -> Path:
    """Write submission JSON to `path`, returning the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(submission, f)
    return path
