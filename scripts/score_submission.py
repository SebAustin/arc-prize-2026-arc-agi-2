"""Score a competition `submission.json` against a split's known solutions.

The go/no-go ANCHOR scorer. NVARC's released notebook, run on L4x4, emits a
`submission.json` in the competition schema; this turns that file into OUR canary
numbers — top-2 exact-match PLUS the partial-credit signals (cell accuracy,
shape-correct rate) — so an untouched third-party notebook's output is measured on
exactly the same yardstick as our own stack. That shared yardstick is what makes
the reproduction gate meaningful (see docs/runbooks/nvarc-anchor-l4x4.md and the
2026-10-02 research note).

CPU-only, no GPU, spends nothing. Run it on a submission downloaded from Kaggle:

    python scripts/score_submission.py path/to/submission.json
    python scripts/score_submission.py submission.json --split evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc.config import get_config
from arc.eval.harness import load_split
from arc.eval.metrics import score_predictions
from arc.io.grid import from_lists
from arc.io.submission import Attempt, Predictions


def parse_submission(raw: dict) -> Predictions:
    """Competition JSON ({task_id: [{attempt_1, attempt_2}, ...]}) -> Predictions.

    Malformed entries are skipped (never raise): a third-party submission is
    untrusted input, and a scorer that crashes on one bad grid tells us nothing.
    """
    preds: Predictions = {}
    if not isinstance(raw, dict):
        return preds
    for task_id, entries in raw.items():
        if not isinstance(entries, list):
            continue
        attempts: list[Attempt] = []
        for out in entries:
            if not isinstance(out, dict) or "attempt_1" not in out or "attempt_2" not in out:
                continue
            try:
                attempts.append(Attempt(from_lists(out["attempt_1"]), from_lists(out["attempt_2"])))
            except (TypeError, ValueError):
                continue  # a single unparseable grid shouldn't void the whole task
        if attempts:
            preds[str(task_id)] = attempts
    return preds


def main(submission_path: str, split: str = "evaluation") -> dict:
    if not Path(submission_path).is_file():
        raise SystemExit(
            f"no submission file at {submission_path!r}.\n"
            "This scorer runs on a submission.json produced by an actual eval/anchor run "
            "(see docs/runbooks/nvarc-anchor-l4x4.md) — there isn't one yet until that run "
            "completes. Point it at a real downloaded submission.json."
        )
    with open(submission_path, encoding="utf-8") as f:
        raw = json.load(f)
    preds = parse_submission(raw)

    _, solutions = load_split(split, cfg=get_config())
    if solutions is None:
        raise ValueError(f"split {split!r} has no solutions; cannot score")

    scored = {k: v for k, v in solutions.items() if k in preds}
    summary = score_predictions(preds, scored)
    summary["scored_tasks"] = len(scored)
    summary["missing_from_submission"] = len(set(solutions) - set(preds))
    summary["split"] = split
    print(f"summary: {summary}")
    if not scored:
        print(
            "WARNING: 0 tasks scored — the submission's task_ids don't match the "
            f"{split!r} split. Did the notebook run on the eval split, not the hidden test?"
        )
    elif summary["missing_from_submission"]:
        print(
            f"NOTE: {summary['missing_from_submission']} {split} tasks are absent from the "
            "submission and were not scored (expected if the run was limited/sliced)."
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="path to submission.json")
    parser.add_argument("--split", default="evaluation", help="split to score against")
    args = parser.parse_args()
    main(args.submission, split=args.split)
