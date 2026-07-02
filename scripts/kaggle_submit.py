"""Kaggle offline inference entrypoint (M1+).

Runs on the L4x4 GPU with NO internet: loads a base model from a pre-staged
Kaggle Dataset, runs the DSL + LLM ensemble over the test challenges under the
global time watchdog, and writes /kaggle/working/submission.json.

The final competition notebook is a thin wrapper that calls `main()`. Kept as a
plain script so it is importable and lint-clean off-GPU; torch only loads when an
HFModel is actually constructed.

Usage (inside the Kaggle notebook):
    import sys; sys.path.append('/kaggle/input/<code-dataset>/src')
    from kaggle_submit import main
    main(model_path='/kaggle/input/<model-dataset>')
"""

from __future__ import annotations

import argparse
import json
import os
import time

from arc.config import get_config
from arc.io.loader import MalformedTaskError, load_challenges
from arc.io.submission import (
    build_submission,
    fallback_from_raw,
    validate_submission,
    write_submission,
)
from arc.pipeline import run as run_pipeline


def _require_data(challenges_path) -> None:
    """Fail fast with an actionable message if the competition data is missing —
    the common cause is simply that the competition dataset was never attached to
    the notebook, which otherwise surfaces as a cryptic FileNotFoundError."""
    if os.path.exists(challenges_path):
        return
    inp = "/kaggle/input"
    mounted = sorted(os.listdir(inp)) if os.path.isdir(inp) else []
    listing = "\n".join(f"    - {inp}/{m}" for m in mounted) or (
        "    (nothing mounted under /kaggle/input)"
    )
    raise FileNotFoundError(
        f"Competition data not found: {challenges_path}\n"
        f"On Kaggle: use Add Input -> Competitions -> 'ARC Prize 2026 - ARC-AGI-2' "
        f"so the data mounts under /kaggle/input/. Or set ARC_DATA_DIR to the folder "
        f"that contains {os.path.basename(str(challenges_path))}.\n"
        f"Currently mounted under /kaggle/input:\n{listing}"
    )


def _pre_write_fallback(challenges_path, submission_path) -> int:
    """Write a complete, schema-valid fallback submission BEFORE any parsing or
    model work, so a crash/OOM anywhere downstream still leaves a scoreable file
    on disk. Returns the number of tasks covered (0 if the file is unreadable)."""
    try:
        with open(challenges_path, encoding="utf-8") as f:
            raw = json.load(f)
        preds = fallback_from_raw(raw)
        write_submission(build_submission(preds), submission_path)
        return len(preds)
    except Exception as exc:  # noqa: BLE001 — last-resort guard, never fatal
        print(f"WARNING: could not pre-write fallback submission: {exc}")
        return 0


# Per-test-input augmentations / samples — tuned per model on Kaggle.
DEFAULT_LLM_KWARGS = {
    "num_augs": 8,
    "num_samples": 1,
    "max_new_tokens": 1024,
    "temperature": 0.0,
}
# Corpus size for per-task test-time training (leave-one-out x augmentation).
DEFAULT_TTT_DATA_KWARGS = {"num_augs": 16, "max_examples": 250}


def _build_solvers(model, use_ttt: bool, llm_kwargs: dict):
    """Assemble the GPU ensemble: DSL + heuristics + (TTT or plain LLM)."""
    from arc.solvers.dsl.solver import DSLSolver
    from arc.solvers.identity import CHEAP_SOLVERS

    if use_ttt:
        from arc.solvers.llm import LoraTTTRunner, TTTConfig, TTTSolver

        ttt = TTTSolver(
            LoraTTTRunner(model, TTTConfig()),
            llm_kwargs=llm_kwargs,
            ttt_data_kwargs=DEFAULT_TTT_DATA_KWARGS,
        )
        print("ensemble: DSL + heuristics + TTT(LoRA)")
        return [DSLSolver(), *CHEAP_SOLVERS, ttt]

    from arc.solvers.llm import LLMSolver

    print("ensemble: DSL + heuristics + LLM(HFModel)")
    return [DSLSolver(), *CHEAP_SOLVERS, LLMSolver(model, **llm_kwargs)]


def main(
    model_path: str | None = None,
    adapter_path: str | None = None,
    per_task_budget_s: float = 150.0,
    llm_kwargs: dict | None = None,
    use_ttt: bool = True,
) -> dict:
    cfg = get_config()
    model_path = model_path or os.environ.get("ARC_MODEL_PATH")
    adapter_path = adapter_path or os.environ.get("ARC_ADAPTER_PATH")
    print(f"mode={cfg.mode}  data_dir={cfg.data_dir}  model_path={model_path}")
    print(f"adapter_path={adapter_path}  use_ttt={use_ttt}")

    challenges_path = cfg.challenges_path("test")
    _require_data(challenges_path)  # clear error if the data isn't attached
    covered = _pre_write_fallback(challenges_path, cfg.submission_path)
    print(f"pre-wrote fallback submission for {covered} tasks -> {cfg.submission_path}")

    try:
        tasks = load_challenges(challenges_path)
    except MalformedTaskError as exc:
        # The pre-written fallback is already a complete, scoreable submission;
        # keep it rather than crashing the kernel with an empty output.
        print(f"ERROR: could not parse challenges ({exc}); keeping fallback submission")
        return {"problems": [str(exc)], "elapsed_s": 0.0, "num_tasks": 0}
    print(f"loaded {len(tasks)} test tasks")

    solvers = None
    if model_path:
        from arc.solvers.llm import HFModel  # lazy: imports torch

        # adapter_path = the base-fine-tuned (synthetic-corpus) adapter; TTT
        # adapts further on top of it per task.
        model = HFModel(model_path, adapter_path=adapter_path)
        solvers = _build_solvers(model, use_ttt, llm_kwargs or DEFAULT_LLM_KWARGS)
    else:
        print("WARNING: no model_path — running DSL/heuristic ensemble only")

    t0 = time.monotonic()
    predictions = run_pipeline(
        tasks,
        solvers=solvers,
        output_path=cfg.submission_path,
        per_task_budget_s=per_task_budget_s,
        verbose=True,
    )
    elapsed = time.monotonic() - t0

    submission = build_submission(predictions)
    problems = validate_submission(submission, tasks)
    print(f"submission: {cfg.submission_path}  schema_problems={len(problems)}")
    for p in problems[:5]:
        print("  -", p)
    print(f"elapsed: {elapsed / 60:.1f} min  ({elapsed / max(len(tasks),1):.1f}s/task)")
    return {"problems": problems, "elapsed_s": elapsed, "num_tasks": len(tasks)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--per-task-budget", type=float, default=150.0)
    parser.add_argument(
        "--no-ttt",
        action="store_true",
        help="Disable test-time training (plain LLM transduction).",
    )
    args = parser.parse_args()
    main(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        per_task_budget_s=args.per_task_budget,
        use_ttt=not args.no_ttt,
    )
