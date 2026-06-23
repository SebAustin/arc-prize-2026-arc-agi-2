"""Assemble synthetic tasks into a base-fine-tuning corpus.

Each generated task becomes one supervised example: prompt = demonstration pairs
+ test input (ending at the open `Output:`), completion = the test output grid.
The corpus is what base fine-tuning trains on so the model learns the ARC I/O
format and a broad library of transformations before per-task TTT.

`save/load_examples_jsonl` persist the corpus so it can be generated once and
staged as a Kaggle Dataset for offline fine-tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..io.loader import Task
from ..serialize.prompt import build_prompt
from ..serialize.tokenizer import grid_to_str
from ..solvers.llm.ttt_data import COMPLETION_PREFIX, TrainExample
from .generators import GENERATORS, Generator, build_task, is_well_formed

_MAX_RETRIES = 5


def build_synthetic_tasks(
    n: int,
    seed: int = 0,
    generators: tuple[Generator, ...] = GENERATORS,
    num_pairs: int = 3,
) -> list[Task]:
    """Generate `n` well-formed synthetic tasks, round-robin over generators."""
    tasks: list[Task] = []
    i = 0
    while len(tasks) < n:
        gen = generators[i % len(generators)]
        gt = None
        for r in range(_MAX_RETRIES):
            candidate = build_task(gen, seed=seed * 1_000_003 + i * 7 + r, num_pairs=num_pairs)
            if is_well_formed(candidate):
                gt = candidate
                break
        i += 1
        if gt is not None:
            tasks.append(gt.task)
    return tasks


def task_to_example(task: Task) -> TrainExample:
    """One (prompt, completion) example from a task's (single) test pair."""
    test = task.test[0]
    prompt = build_prompt(task.train, test.input)
    completion = COMPLETION_PREFIX + grid_to_str(test.output)
    return TrainExample(prompt=prompt, completion=completion)


def tasks_to_examples(tasks: list[Task]) -> list[TrainExample]:
    return [task_to_example(t) for t in tasks]


def generate_examples(n: int, seed: int = 0) -> list[TrainExample]:
    """Convenience: build n synthetic tasks and return their fine-tuning examples."""
    return tasks_to_examples(build_synthetic_tasks(n, seed=seed))


def save_examples_jsonl(examples: list[TrainExample], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"prompt": ex.prompt, "completion": ex.completion}) + "\n")
    return path


def load_examples_jsonl(path: str | Path) -> list[TrainExample]:
    examples: list[TrainExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(TrainExample(prompt=row["prompt"], completion=row["completion"]))
    return examples
