"""Load ARC-AGI-2 challenge/solution JSON into typed objects.

JSON shape (confirmed from the competition data):
    challenges: {task_id: {"train": [{"input": grid, "output": grid}, ...],
                            "test":  [{"input": grid}, ...]}}
    solutions:  {task_id: [grid, ...]}   # one output grid per test input, in order

Test challenges carry train pairs + test inputs only (no outputs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .grid import Grid, from_lists

_log = logging.getLogger(__name__)


class MalformedTaskError(ValueError):
    """A challenges file does not match the expected schema.

    Raised with the offending ``task_id`` and a concrete reason so a bad rerun
    file surfaces as an actionable message instead of a bare ``KeyError`` deep in
    parsing.
    """


@dataclass(frozen=True)
class Pair:
    """A single demonstration or test pair. `output` is None for test inputs."""

    input: Grid
    output: Grid | None = None


@dataclass(frozen=True)
class Task:
    """One ARC task: demonstration pairs plus test input(s) to solve."""

    task_id: str
    train: tuple[Pair, ...]
    test: tuple[Pair, ...]

    @property
    def num_test(self) -> int:
        return len(self.test)


def _parse_grid(value, where: str) -> Grid:
    if not isinstance(value, list) or not value:
        raise MalformedTaskError(f"{where}: expected a non-empty grid (list of rows)")
    try:
        return from_lists(value)
    except (TypeError, ValueError) as exc:
        raise MalformedTaskError(f"{where}: invalid grid ({exc})") from exc


def _parse_pair(raw: dict, where: str) -> Pair:
    if not isinstance(raw, dict) or "input" not in raw:
        raise MalformedTaskError(f"{where}: missing 'input'")
    out = raw.get("output")
    return Pair(
        input=_parse_grid(raw["input"], f"{where}.input"),
        output=_parse_grid(out, f"{where}.output") if out is not None else None,
    )


def _parse_task(task_id: str, body) -> Task:
    if not isinstance(body, dict) or "train" not in body or "test" not in body:
        raise MalformedTaskError(f"task {task_id}: missing 'train'/'test'")
    train, test = body["train"], body["test"]
    if not isinstance(train, list) or not isinstance(test, list):
        raise MalformedTaskError(f"task {task_id}: 'train'/'test' must be lists")
    if not test:
        raise MalformedTaskError(f"task {task_id}: 'test' must be non-empty")
    return Task(
        task_id=task_id,
        train=tuple(
            _parse_pair(p, f"task {task_id} train[{i}]") for i, p in enumerate(train)
        ),
        test=tuple(
            _parse_pair(p, f"task {task_id} test[{i}]") for i, p in enumerate(test)
        ),
    )


def load_challenges(path: str | Path, *, skip_invalid: bool = False) -> dict[str, Task]:
    """Load a *_challenges.json file into {task_id: Task}.

    Validates structure and raises a clear :class:`MalformedTaskError` (naming the
    task and reason) on a bad file, instead of a bare ``KeyError``. With
    ``skip_invalid=True`` malformed tasks are logged and dropped rather than
    aborting the load — callers that require every task_id present (a submission)
    should keep the default and rely on the entrypoint's fallback safety net.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise MalformedTaskError(f"{path}: top-level JSON must be an object of tasks")
    tasks: dict[str, Task] = {}
    dropped: list[str] = []
    for task_id, body in raw.items():
        try:
            tasks[task_id] = _parse_task(task_id, body)
        except MalformedTaskError as exc:
            if not skip_invalid:
                raise
            dropped.append(str(exc))
    if dropped:
        _log.warning(
            "skipped %d malformed task(s): %s", len(dropped), "; ".join(dropped[:5])
        )
    return tasks


def load_solutions(path: str | Path) -> dict[str, list[Grid]]:
    """Load a *_solutions.json file into {task_id: [output_grid, ...]}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        task_id: [from_lists(g) for g in grids] for task_id, grids in raw.items()
    }
