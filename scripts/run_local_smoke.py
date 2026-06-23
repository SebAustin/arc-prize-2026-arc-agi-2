"""End-to-end smoke run (CPU, no GPU) — the M0 acceptance check.

Runs the full pipeline over a slice of the public evaluation split, writes a
schema-valid submission.json, validates it against the challenges, and reports
the exact-match top-2 score of the CPU ensemble. Proves the entire plumbing —
load -> solve -> vote -> submit -> score — works without a GPU.

Usage:
    python scripts/run_local_smoke.py [--split evaluation] [--limit 40]
"""

from __future__ import annotations

import argparse
import time

from arc.config import get_config
from arc.eval.harness import load_split
from arc.io.submission import build_submission, validate_submission
from arc.eval.metrics import score_predictions
from arc.pipeline import default_solvers, run as run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--per-task-budget", type=float, default=10.0)
    parser.add_argument(
        "--use-mock-llm",
        action="store_true",
        help="Include the LLM solver backed by the CPU MockModel (plumbing check).",
    )
    parser.add_argument(
        "--use-mock-ttt",
        action="store_true",
        help="Include the TTT solver with a no-op MockTTTRunner (flow check).",
    )
    args = parser.parse_args()

    cfg = get_config()
    print(f"mode={cfg.mode}  data_dir={cfg.data_dir}")

    solvers = None
    if args.use_mock_ttt:
        from arc.solvers.dsl.solver import DSLSolver
        from arc.solvers.identity import CHEAP_SOLVERS
        from arc.solvers.llm import MockModel, MockTTTRunner, TTTSolver

        ttt = TTTSolver(
            MockTTTRunner(MockModel()),
            llm_kwargs={"num_augs": 3},
            ttt_data_kwargs={"num_augs": 8, "max_examples": 100},
        )
        solvers = [DSLSolver(), *CHEAP_SOLVERS, ttt]
        print("ensemble: DSL + heuristics + TTT(MockTTTRunner)")
    elif args.use_mock_llm:
        from arc.solvers.llm import MockModel

        solvers = default_solvers(llm_model=MockModel(), llm_kwargs={"num_augs": 3})
        print("ensemble: DSL + heuristics + LLM(MockModel)")

    tasks, solutions = load_split(args.split, cfg=cfg, limit=args.limit)
    print(f"loaded {len(tasks)} tasks from split '{args.split}'")

    t0 = time.monotonic()
    predictions = run_pipeline(
        tasks,
        solvers=solvers,
        output_path=cfg.submission_path,
        per_task_budget_s=args.per_task_budget,
        verbose=True,
    )
    elapsed = time.monotonic() - t0

    submission = build_submission(predictions)
    problems = validate_submission(submission, tasks)
    print(f"\nsubmission written to {cfg.submission_path}")
    print(f"schema problems: {len(problems)}")
    for p in problems[:5]:
        print("  -", p)

    if solutions is not None:
        summary = score_predictions(predictions, solutions)
        print(
            f"\nscore (top-2 exact match): {summary['score']:.3f}  "
            f"({summary['correct']}/{summary['total']} outputs over "
            f"{summary['num_tasks']} tasks)"
        )
    print(f"elapsed: {elapsed:.1f}s  ({elapsed / max(len(tasks), 1):.2f}s/task)")


if __name__ == "__main__":
    main()
