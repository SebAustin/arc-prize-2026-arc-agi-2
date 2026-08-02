"""Generate a synthetic ARC corpus for base fine-tuning, and sanity-check it.

Writes a JSONL of (prompt, completion) examples (stage this as a Kaggle Dataset
for offline base fine-tuning) and optionally measures the DSL solve-rate over a
sample — a quick proof that the generated tasks are genuinely solvable.

Usage:
    python scripts/build_synthetic.py --n 2000 --out artifacts/synth.jsonl --sanity 200
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

from arc.eval.metrics import score_predictions
from arc.pipeline import default_solvers
from arc.pipeline import run as run_pipeline
from arc.synth.build_dataset import (
    build_synthetic_tasks,
    save_examples_jsonl,
    tasks_to_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/synth.jsonl")
    parser.add_argument("--sanity", type=int, default=200, help="DSL solve-rate sample size")
    parser.add_argument(
        "--hard-only",
        action="store_true",
        help="restrict to DSL-provably-unsolvable generators and drop any easy re-rolls",
    )
    args = parser.parse_args()

    t0 = time.monotonic()
    tasks = build_synthetic_tasks(args.n, seed=args.seed, hard_only=args.hard_only)
    examples = tasks_to_examples(tasks)
    path = save_examples_jsonl(examples, args.out)
    gen_s = time.monotonic() - t0

    dist = Counter(t.task_id.split("-")[1] for t in tasks)
    print(f"generated {len(tasks)} tasks -> {len(examples)} examples in {gen_s:.1f}s")
    print(f"written: {path}")
    print("rule distribution:", dict(sorted(dist.items())))

    if args.sanity > 0:
        sample = {t.task_id: t for t in tasks[: args.sanity]}
        solutions = {tid: [t.test[0].output] for tid, t in sample.items()}
        preds = run_pipeline(sample, solvers=default_solvers(), per_task_budget_s=5.0)
        summary = score_predictions(preds, solutions)
        print(
            f"\nDSL+heuristic solve-rate on {summary['total']} synthetic tasks: "
            f"{summary['score']:.3f}  ({summary['correct']}/{summary['total']})"
        )
        print("(in-vocabulary concepts should be solved; gravity/border/mirror are not)")


if __name__ == "__main__":
    main()
