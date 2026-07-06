"""Public-eval canary: score the ensemble on the PUBLIC evaluation split.

The measurement instrument for the score-climb plan: runs on Kaggle GPU against
`arc-agi_evaluation_challenges.json` (+solutions), printing a per-task PASS/FAIL
line (for paired comparisons between rungs) and the exact-match summary. Spends
GPU hours but ZERO competition submissions.

Usage (Kaggle notebook, after the bootstrap cell):
    from kaggle_eval import main
    main(model_path=MODEL_DS, limit=40)            # fixed first-40 canary slice
    main(model_path=MODEL_DS, limit=80, offset=40) # confirmation slice
"""

from __future__ import annotations

import argparse
import itertools
import os
import time

from arc.config import get_config
from arc.eval.harness import load_split
from arc.eval.metrics import score_predictions
from arc.io.grid import grids_equal
from arc.pipeline import solve_task
from kaggle_submit import (
    DEFAULT_LLM_KWARGS,
    _build_solvers,
)


def _task_solved(attempts, expected) -> bool:
    """Competition rule: every test output matched by either of its 2 attempts."""
    if len(attempts) != len(expected):
        return False
    return all(
        grids_equal(att.attempt_1, exp) or grids_equal(att.attempt_2, exp)
        for att, exp in zip(attempts, expected, strict=True)
    )


def main(
    model_path: str | None = None,
    adapter_path: str | None = None,
    per_task_budget_s: float = 150.0,
    llm_kwargs: dict | None = None,
    use_ttt: bool = True,
    ttt_config: dict | None = None,
    limit: int = 40,
    offset: int = 0,
    split: str = "evaluation",
) -> dict:
    cfg = get_config()
    model_path = model_path or os.environ.get("ARC_MODEL_PATH")
    adapter_path = adapter_path or os.environ.get("ARC_ADAPTER_PATH")
    print(f"eval: split={split} offset={offset} limit={limit} model={model_path}")

    tasks, solutions = load_split(split, cfg=cfg)
    if solutions is None:
        raise ValueError(f"split {split!r} has no solutions; cannot score")
    items = itertools.islice(tasks.items(), offset, offset + limit)
    tasks = dict(items)
    solutions = {k: v for k, v in solutions.items() if k in tasks}
    print(f"scoring {len(tasks)} tasks")

    solvers = None
    if model_path:
        from arc.solvers.llm import HFModel  # noqa: PLC0415 — lazy: imports torch

        model = HFModel(model_path, adapter_path=adapter_path)
        solvers = _build_solvers(model, use_ttt, llm_kwargs or DEFAULT_LLM_KWARGS, ttt_config)
    else:
        from arc.pipeline import default_solvers  # noqa: PLC0415

        solvers = default_solvers()
        print("WARNING: no model_path — DSL/heuristic ensemble only")

    predictions = {}
    solved = 0
    t0 = time.monotonic()
    for n, (task_id, task) in enumerate(tasks.items(), 1):
        t_task = time.monotonic()
        predictions[task_id] = solve_task(task, solvers, per_task_budget_s)
        ok = _task_solved(predictions[task_id], solutions[task_id])
        solved += ok
        # Per-task line: the unit of paired comparison between rungs.
        print(
            f"[{n:3d}/{len(tasks)}] {task_id}  {'PASS' if ok else 'fail'}  "
            f"({time.monotonic() - t_task:.1f}s, running {solved}/{n})"
        )

    elapsed = time.monotonic() - t0
    summary = score_predictions(predictions, solutions)
    summary.update(
        {
            "split": split,
            "offset": offset,
            "limit": limit,
            "elapsed_s": round(elapsed, 1),
            "s_per_task": round(elapsed / max(len(tasks), 1), 1),
        }
    )
    print(f"summary: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--per-task-budget", type=float, default=150.0)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--no-ttt", action="store_true")
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        per_task_budget_s=args.per_task_budget,
        limit=args.limit,
        offset=args.offset,
        split=args.split,
        use_ttt=not args.no_ttt,
    )
