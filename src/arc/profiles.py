"""Run-recipe profiles: the single, runtime-chosen T4-safe vs L4-full decision.

One deep module behind a tiny interface. Given the GPU count (and an optional
explicit override) it returns the WHOLE recipe — worker fan-out, per-task budget,
TTT knobs, and LLM kwargs — so "are we on the cheap T4 gate or the competitive
L4x4 box?" is decided once, in one place, instead of being split across
`T4_SAFE_TTT` (API path), the notebook's `device_count()` branch (UI path), and
`TTTConfig` defaults.

Pure and torch-free at import: `select_profile` can run inside a pytest, on a CPU
dev box, and inside a generated Kaggle source string. The only torch touch is
lazily inside `detect_device_count`.

Safety invariants (encoded + tested in tests/test_profiles.py):
  * the autopilot/T4 can never be handed the 4-GPU fan-out — in auto mode
    num_workers > 1 implies device_count >= 4;
  * num_workers never exceeds the physical GPU count (run_parallel pins one GPU
    per worker, so over-provisioning would pin to a nonexistent card);
  * an unknown override raises rather than silently picking a wrong-cost recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

# Auto-mode threshold: at least this many GPUs selects the L4-full recipe.
_L4_MIN_GPUS = 4
# L4-full fans out one model replica per GPU, capped here (NVARC's nprocs=4 pattern).
_L4_BASE_WORKERS = 4

_PROFILE_NAMES = ("t4_safe", "l4_full")


@dataclass(frozen=True)
class RunRecipe:
    """The full recipe for one run. `ttt_config`/`llm_kwargs` are the BASE a
    per-run config layers on top of (per-run keys win)."""

    name: str
    num_workers: int  # 0/1 => sequential; >1 => arc.parallel.run_parallel fan-out
    per_task_budget_s: float
    ttt_config: dict
    llm_kwargs: dict | None  # None => the entrypoint's DEFAULT_LLM_KWARGS


def _t4_safe_recipe() -> tuple[int, float, dict, dict | None]:
    # Byte-identical to the historical autopilot_config.T4_SAFE_TTT, sequential.
    return 0, 150.0, {"batch_size": 1, "max_seq_len": 1536}, None


def _l4_full_recipe(device_count: int) -> tuple[int, float, dict, dict | None]:
    # Deeper-but-safe knobs the current (non-Unsloth) stack can run; the exact
    # NVARC structural recipe (r=256, seq 8192, Unsloth+FA2) is a separate lever.
    workers = min(_L4_BASE_WORKERS, device_count)
    ttt = {"max_steps": 96, "batch_size": 2, "max_seq_len": 4096}
    # The no-new-deps NVARC method levers (complete LLMSolver kwargs, not a merge):
    #   * scoreagg selection over 8 shared augs (occurrence count + geomean logprob),
    #   * keep_zero=True so colour 0/black is fixed in augmentations (NVARC parity),
    #   * DFS-as-default decode (num_samples=1/temp=0 as the dfs invariant requires).
    llm = {
        "num_augs": 8,
        "num_samples": 1,
        "max_new_tokens": 1024,
        "temperature": 0.0,
        "selection": "scoreagg",
        "poe_augs": 8,
        "keep_zero": True,
        "decode": "dfs",
        "dfs_eps": 0.12,
    }
    return workers, 550.0, ttt, llm


def select_profile(device_count: int, override: str | None = None) -> RunRecipe:
    """Return the run recipe for `device_count` GPUs.

    `override` forces a named profile (still clamped to the physical GPU count);
    `None` auto-selects l4_full at >= 4 GPUs, else t4_safe. An unknown override
    name raises ValueError.
    """
    if override is not None and override not in _PROFILE_NAMES:
        raise ValueError(
            f"unknown profile override {override!r}; expected one of {_PROFILE_NAMES} or None"
        )
    name = override or ("l4_full" if device_count >= _L4_MIN_GPUS else "t4_safe")
    if name == "l4_full":
        workers, budget, ttt, llm = _l4_full_recipe(device_count)
    else:
        workers, budget, ttt, llm = _t4_safe_recipe()
    # Return fresh dict copies so a caller's merge can never mutate the shared base.
    return RunRecipe(
        name=name,
        num_workers=workers,
        per_task_budget_s=budget,
        ttt_config=dict(ttt),
        llm_kwargs=None if llm is None else dict(llm),
    )


def detect_device_count() -> int:
    """Visible CUDA device count; 0 on CPU / if torch is unavailable. The ONE
    lazy torch touch — kept out of module import so `select_profile` stays pure."""
    try:
        import torch  # noqa: PLC0415 — lazy so importing this module needs no torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:  # pragma: no cover — CPU / no-torch environments
        pass
    return 0


def runtime_recipe(override: str | None = None) -> RunRecipe:
    """`select_profile` keyed off the real runtime GPU count (the notebook/entrypoint
    convenience; the autopilot pins `select_profile(2, "t4_safe")` directly)."""
    return select_profile(detect_device_count(), override)
