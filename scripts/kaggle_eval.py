"""Public-eval canary: score the ensemble on the PUBLIC evaluation split.

The measurement instrument for the score-climb plan: runs on Kaggle GPU against
`arc-agi_evaluation_challenges.json` (+solutions), printing a per-task PASS/FAIL
line (for paired comparisons between rungs) and the exact-match summary. Spends
GPU hours but ZERO competition submissions.

Usage (Kaggle notebook, after the bootstrap cell):
    from kaggle_eval import main
    main(model_path=MODEL_DS, limit=None)          # FULL public eval split (all tasks)
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
from arc.solvers.factory import build_solvers
from kaggle_submit import DEFAULT_LLM_KWARGS


def _gpu_mem_stats() -> dict:
    """Best-effort peak-VRAM stats (GiB) for this run; {} on CPU / no torch.

    Part of the "no information-free run" instrumentation: pairing peak usage with
    the card's total tells us the HEADROOM — the difference between "fit with room
    to grow the recipe" and "OOM'd by 200 MiB" — which is the exact axis the NVARC
    4B+TTT fit question turns on.
    """
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return {}
        gib = 1024**3
        free, total = torch.cuda.mem_get_info()
        return {
            "vram_peak_alloc_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
            "vram_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 2),
            "vram_total_gib": round(total / gib, 2),
        }
    except Exception:  # pragma: no cover — CPU environments / torch quirks
        return {}


def _reset_gpu_peak_stats() -> None:
    """Zero CUDA peak counters so `_gpu_mem_stats` measures THIS run, not a prior arm."""
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # pragma: no cover — CPU environments
        pass


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
    limit: int | None = 40,
    offset: int = 0,
    split: str = "evaluation",
    dfs_selftest: bool = False,
) -> dict:
    cfg = get_config()
    model_path = model_path or os.environ.get("ARC_MODEL_PATH")
    adapter_path = adapter_path or os.environ.get("ARC_ADAPTER_PATH")
    print(f"eval: split={split} offset={offset} limit={limit} model={model_path}")

    tasks, solutions = load_split(split, cfg=cfg)
    if solutions is None:
        raise ValueError(f"split {split!r} has no solutions; cannot score")
    # limit=None -> full split (islice treats a None stop as "to the end"); an
    # int caps the slice at offset+limit. The autopilot passes None to score the
    # whole 120-task public eval — sub-1% gains are invisible on a 40-task slice.
    stop = None if limit is None else offset + limit
    items = itertools.islice(tasks.items(), offset, stop)
    tasks = dict(items)
    solutions = {k: v for k, v in solutions.items() if k in tasks}
    print(f"scoring {len(tasks)} tasks")

    solvers = None
    if model_path:
        from arc.solvers.llm import HFModel  # noqa: PLC0415 — lazy: imports torch

        model = HFModel(model_path, adapter_path=adapter_path)
        if dfs_selftest and (llm_kwargs or {}).get("decode") == "dfs" and tasks:
            # Advisory KV-cache-integrity check: DFS's first leaf must equal
            # generate() greedy. A mismatch means cache/position corruption —
            # warn loudly (results below would be suspect) but never crash.
            from arc.serialize.prompt import build_prompt  # noqa: PLC0415
            from arc.solvers.llm.dfs_decode import greedy_selftest  # noqa: PLC0415

            first = next(iter(tasks.values()))
            prompt = build_prompt(first.train, first.test[0].input)
            ok, detail = greedy_selftest(model, prompt)
            print(f"dfs_selftest: {'OK' if ok else 'WARNING — MISMATCH'} ({detail})")
        solvers = build_solvers(model, use_ttt, llm_kwargs or DEFAULT_LLM_KWARGS, ttt_config)
    else:
        from arc.pipeline import default_solvers  # noqa: PLC0415

        solvers = default_solvers()
        print("WARNING: no model_path — DSL/heuristic ensemble only")

    predictions = {}
    solved = 0
    _reset_gpu_peak_stats()  # measure peak VRAM of THIS eval, not model-load/prior-arm
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

    # Capture peak VRAM BEFORE freeing the model (max_memory_allocated resets with
    # the cache flush below) — this is the headroom reading the fit question needs.
    gpu_stats = _gpu_mem_stats()

    # Release this run's model before returning: paired A/B evals call main()
    # twice in one kernel, and a resident 15GB model from arm 1 starves arm 2's
    # GPU headroom (observed: arm 2 OOM'd at TTT step 1 on every large task,
    # invalidating the comparison). Explicit free + cache flush prevents it.
    if model_path:
        del solvers, model
        import gc  # noqa: PLC0415

        gc.collect()
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover — CPU environments
            pass

    elapsed = time.monotonic() - t0
    summary = score_predictions(predictions, solutions)
    summary.update(
        {
            "split": split,
            "offset": offset,
            "limit": limit,
            "elapsed_s": round(elapsed, 1),
            "s_per_task": round(elapsed / max(len(tasks), 1), 1),
            **gpu_stats,
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
