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
import logging
from pathlib import Path

from ..io.loader import Task
from ..serialize.prompt import build_prompt
from ..serialize.tokenizer import grid_to_str
from ..solvers.dsl.solver import DSLSolver
from ..solvers.llm.ttt_data import COMPLETION_PREFIX, TrainExample
from .generators import GENERATORS, HARD_GENERATORS, Generator, build_task, is_well_formed

_log = logging.getLogger(__name__)
_MAX_RETRIES = 5


def is_dsl_solvable(task: Task, budget_s: float = 2.0) -> bool:
    """True iff the DSL micro-solver's own candidates for test 0 include the
    task's true output.

    A generated task carries its ground truth, so "solvable" here means exact
    grid equality against `DSLSolver`'s ranked candidates -- the same bar the
    real pipeline uses at inference. Cheap enough for a handful of ad hoc
    checks (used per-generator in tests), but O(task) overall: corpus-scale
    filtering (`exclude_dsl_solvable=True` over thousands of tasks) is meant to
    run OFFLINE once, at corpus-build time (`scripts/build_synthetic.py
    --hard-only`), not inside the training loop.
    """
    test = task.test[0]
    if test.output is None:
        return False
    candidates = DSLSolver().solve(task, budget_s=budget_s)
    return bool(candidates) and test.output in candidates[0]


def build_synthetic_tasks(
    n: int,
    seed: int = 0,
    generators: tuple[Generator, ...] = GENERATORS,
    num_pairs: int = 3,
    exclude_dsl_solvable: bool = False,
    hard_only: bool = False,
) -> list[Task]:
    """Generate `n` well-formed synthetic tasks, round-robin over generators.

    `exclude_dsl_solvable=True` re-rolls (bounded retries) any candidate task
    the DSL micro-solver already solves, so the corpus teaches the model
    transformations beyond the DSL's own reach; the drop rate is logged.
    `hard_only=True` restricts `generators` to `HARD_GENERATORS` (the
    provably-DSL-unsolvable set) and forces `exclude_dsl_solvable=True`,
    overriding an explicit `generators=` argument.
    """
    if hard_only:
        generators = HARD_GENERATORS
        exclude_dsl_solvable = True

    tasks: list[Task] = []
    considered = 0
    dropped = 0
    i = 0
    while len(tasks) < n:
        gen = generators[i % len(generators)]
        gt = None
        for r in range(_MAX_RETRIES):
            candidate = build_task(gen, seed=seed * 1_000_003 + i * 7 + r, num_pairs=num_pairs)
            if not is_well_formed(candidate):
                continue
            if exclude_dsl_solvable:
                considered += 1
                if is_dsl_solvable(candidate.task):
                    dropped += 1
                    continue
            gt = candidate
            break
        i += 1
        if gt is not None:
            tasks.append(gt.task)
    if exclude_dsl_solvable and considered:
        _log.info(
            "exclude_dsl_solvable: dropped %d/%d well-formed candidate(s) (%.1f%%)",
            dropped,
            considered,
            100.0 * dropped / considered,
        )
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
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            examples.append(TrainExample(prompt=row["prompt"], completion=row["completion"]))
    return examples
