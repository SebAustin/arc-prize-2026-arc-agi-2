"""Load ARC-AGI-2 challenge/solution JSON into typed objects.

JSON shape (confirmed from the competition data):
    challenges: {task_id: {"train": [{"input": grid, "output": grid}, ...],
                            "test":  [{"input": grid}, ...]}}
    solutions:  {task_id: [grid, ...]}   # one output grid per test input, in order

Test challenges carry train pairs + test inputs only (no outputs).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .grid import Grid, from_lists


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


def _parse_pair(raw: dict) -> Pair:
    out = raw.get("output")
    return Pair(
        input=from_lists(raw["input"]),
        output=from_lists(out) if out is not None else None,
    )


def load_challenges(path: str | Path) -> dict[str, Task]:
    """Load a *_challenges.json file into {task_id: Task}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tasks: dict[str, Task] = {}
    for task_id, body in raw.items():
        tasks[task_id] = Task(
            task_id=task_id,
            train=tuple(_parse_pair(p) for p in body["train"]),
            test=tuple(_parse_pair(p) for p in body["test"]),
        )
    return tasks


def load_solutions(path: str | Path) -> dict[str, list[Grid]]:
    """Load a *_solutions.json file into {task_id: [output_grid, ...]}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        task_id: [from_lists(g) for g in grids] for task_id, grids in raw.items()
    }
