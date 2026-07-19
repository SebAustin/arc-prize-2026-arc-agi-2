"""Kernel-folder builders and the experiment backlog for the daily autopilot.

Two things live here:

1. `write_submission_kernel` / `write_train_kernel` — regenerate the relevant
   notebook (so the embedded `arc` package is current, per
   `scripts/build_kaggle_notebook.py`), patch in a caller-supplied run cell,
   and write a clean `kernel-metadata.json` (never `machine_shape`/
   `docker_image` — see DEPLOYMENT.md Sec 3) into a push-ready folder.
2. `BACKLOG` — the ordered list of cheap-gate-first experiments
   `daily_autopilot.py` works through when idle. Each item is
   self-contained (`build(state) -> Path`, `config(state) -> dict`); append
   to the list to add more.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from autopilot_config import (
    ADAPTER_DATASET_SLUG,
    ADAPTER_MOUNT,
    BASE_MODEL_MOUNT,
    BASE_MODEL_SOURCE,
    COMPETITION_SLUG,
    FULL_EVAL_LIMIT,
    HARD_CORPUS_DATASET,
    HARD_CORPUS_FILE,
    KERNEL_RUNS_DIR,
    SUBMISSION_KERNEL,
    T4_SAFE_TTT,
    TRAIN_KERNEL,
)
from build_kaggle_notebook import build_submission, build_train_adapter

# ---------------------------------------------------------------------------
# Notebook patching + kernel-metadata
# ---------------------------------------------------------------------------


def submission_kernel_for(state: dict) -> str:
    """The CURRENT submission-kernel id — state-owned, not fixed config.

    Kaggle's 2026-07-14 API has a busy-create bug: a push that creates a NEW
    kernel id but gets rejected on the GPU-session cap half-creates a broken
    server record, and every later push to that id fails 'Notebook not found'
    (verified empirically). The id is therefore a rotating sequence: base id,
    then `-1`, `-2`, ... — `daily_autopilot` bumps `submission_kernel_seq`
    whenever a create attempt burns an id, and pins the id once a push succeeds
    (`submission_kernel_created`; successfully-created kernels update fine)."""
    seq = int(state.get("submission_kernel_seq", 0))
    return SUBMISSION_KERNEL if seq == 0 else f"{SUBMISSION_KERNEL}-{seq}"


def _slug_title(kernel_id: str) -> str:
    """Kaggle hard-rejects (409 'kernel title does not resolve to the specified
    id', enforced ~2026-07-14; previously a warning) any push whose title does
    not slugify to the kernel id's slug. The one title guaranteed to survive
    any slugification rule is the slug itself — lowercase alphanumerics and
    hyphens are a fixed point — so default titles are derived from the id."""
    return kernel_id.split("/")[-1]


def _notebook_code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _patch_notebook(nb_path: Path, run_cell_src: str) -> dict:
    """Load a freshly-built notebook and replace its config+run cells (the
    last two cells, per `build_kaggle_notebook.py`'s 5-cell layout) with a
    single self-contained run cell carrying the desired call+config."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    cells[-2] = _notebook_code_cell("# autopilot: config folded into the run cell below\n")
    cells[-1] = _notebook_code_cell(run_cell_src)
    return nb


def _write_kernel_metadata(
    folder: Path,
    *,
    kernel_id: str,
    title: str,
    code_file: str,
    dataset_sources: list[str],
    model_sources: list[str],
    competition_sources: list[str],
) -> None:
    # NEVER include machine_shape/docker_image here: a pushed explicit value
    # (even one pulled straight from the kernel's own metadata) can silently
    # reset the kernel to an unusable default P100 — see DEPLOYMENT.md Sec 3.
    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": dataset_sources,
        "competition_sources": competition_sources,
        "kernel_sources": [],
        "model_sources": model_sources,
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_submission_kernel(
    folder,
    run_cell_src: str,
    dataset_sources: list[str] | None = None,
    model_sources: tuple[str, ...] = (BASE_MODEL_SOURCE,),
    competition_sources: tuple[str, ...] = (COMPETITION_SLUG,),
    kernel_id: str = SUBMISSION_KERNEL,
    title: str | None = None,  # None -> derived from kernel_id (must slug-resolve to it)
) -> Path:
    """Build a push-ready submission-kernel folder with `run_cell_src` as its
    run cell. Used for both the real commit-run (submit_commit) and
    eval-only pushes (same notebook shape, different run cell)."""
    nb_path = build_submission()
    nb = _patch_notebook(nb_path, run_cell_src)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    notebook_name = "submission_selfcontained.ipynb"
    (folder / notebook_name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    _write_kernel_metadata(
        folder,
        kernel_id=kernel_id,
        title=title or _slug_title(kernel_id),
        code_file=notebook_name,
        dataset_sources=list(dataset_sources or []),
        model_sources=list(model_sources),
        competition_sources=list(competition_sources),
    )
    return folder


def write_train_kernel(
    folder,
    run_cell_src: str,
    dataset_sources: list[str] | None = None,
    model_sources: tuple[str, ...] = (BASE_MODEL_SOURCE,),
    kernel_id: str = TRAIN_KERNEL,
    title: str | None = None,  # None -> derived from kernel_id (must slug-resolve to it)
) -> Path:
    """Same as `write_submission_kernel` but for the Rung-4 training notebook
    (no competition data needed — training only touches the corpus dataset)."""
    nb_path = build_train_adapter()
    nb = _patch_notebook(nb_path, run_cell_src)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    notebook_name = "train_adapter.ipynb"
    (folder / notebook_name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    _write_kernel_metadata(
        folder,
        kernel_id=kernel_id,
        title=title or _slug_title(kernel_id),
        code_file=notebook_name,
        dataset_sources=list(dataset_sources or []),
        model_sources=list(model_sources),
        competition_sources=[],
    )
    return folder


def write_dataset_metadata(folder: Path, slug: str, title: str) -> None:
    meta = {"title": title, "id": slug, "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def stage_dir(tag: str) -> Path:
    """A fresh, empty push-staging folder for `tag` (cleared each call so a
    stale file from a previous tick never leaks into a new kernel push)."""
    d = KERNEL_RUNS_DIR / "push_staging" / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def dataset_sources_for(config: dict) -> list[str]:
    if config.get("adapter_path") == ADAPTER_MOUNT:
        return [ADAPTER_DATASET_SLUG]
    return []


# ---------------------------------------------------------------------------
# Run-cell source generators — the actual `kaggle_*.main(...)` calls
# ---------------------------------------------------------------------------


def eval_run_src(config: dict) -> str:
    kwargs = {
        "model_path": BASE_MODEL_MOUNT,
        "adapter_path": config.get("adapter_path"),
        "per_task_budget_s": config.get("per_task_budget_s", 150.0),
        "llm_kwargs": config.get("llm_kwargs"),
        # T4-safe TTT memory settings as the base; the run's own ttt_config wins.
        "ttt_config": {**T4_SAFE_TTT, **(config.get("ttt_config") or {})},
        # Full 120-task public eval by default (None = all) so the promotion gate
        # can actually detect a sub-1% gain; a config may still override eval_limit.
        "limit": config.get("eval_limit", FULL_EVAL_LIMIT),
        # Advisory KV-cache-integrity check for DFS-decode runs (no-op otherwise).
        "dfs_selftest": config.get("dfs_selftest", False),
    }
    return (
        "import logging\n"
        "logging.basicConfig(level=logging.INFO, force=True)\n"
        "from kaggle_eval import main\n"
        f"CONFIG = {kwargs!r}\n"
        "result = main(**CONFIG)\n"
        "print('summary:', result)\n"
        "with open('/kaggle/working/eval_summary.txt', 'w') as f:\n"
        "    f.write('summary: ' + repr(result) + chr(10))\n"
    )


def submission_run_src(config: dict) -> str:
    kwargs = {
        "model_path": BASE_MODEL_MOUNT,
        "adapter_path": config.get("adapter_path"),
        "llm_kwargs": config.get("llm_kwargs"),
        "use_ttt": True,
        # T4-safe TTT memory settings as the base; the run's own ttt_config wins.
        "ttt_config": {**T4_SAFE_TTT, **(config.get("ttt_config") or {})},
        "num_workers": config.get("num_workers", 0),
    }
    return (
        "import logging\n"
        "logging.basicConfig(level=logging.INFO, force=True)\n"
        "from kaggle_submit import main\n"
        f"CONFIG = {kwargs!r}\n"
        "result = main(**CONFIG)\n"
        "print(result)\n"
        "assert result['problems'] == [], result['problems']\n"
    )


def train_run_src(max_examples: int, config_overrides: dict) -> str:
    kwargs = {
        "base_model_path": BASE_MODEL_MOUNT,
        "corpus_path": HARD_CORPUS_FILE,
        "max_examples": max_examples,
        "epochs": 1,
        "config_overrides": config_overrides,
    }
    return (
        "import logging\n"
        "logging.basicConfig(level=logging.INFO, force=True)\n"
        "from kaggle_train import main\n"
        f"CONFIG = {kwargs!r}\n"
        "adapter_dir = main(**CONFIG)\n"
        "print('adapter_dir:', adapter_dir)\n"
    )


# ---------------------------------------------------------------------------
# Backlog — cheapest-gate-first experiments the autopilot works through when
# idle. Append to this list to add more; each item is self-contained.
# ---------------------------------------------------------------------------

_HARD_TRAIN_OVERRIDES = {"batch_size": 1, "grad_accum": 16, "max_seq_len": 1024}


def _build_adapter_hard_2500(state: dict) -> Path:
    folder = stage_dir("adapter_hard_2500")
    run_src = train_run_src(max_examples=2500, config_overrides=_HARD_TRAIN_OVERRIDES)
    return write_train_kernel(folder, run_src, dataset_sources=[HARD_CORPUS_DATASET])


def _config_adapter_hard_2500(state: dict) -> dict:
    return {**state["live_config"], "adapter_path": None}


def _build_poe_regate(state: dict) -> Path:
    folder = stage_dir("poe_regate")
    cfg = _config_poe_regate(state)
    sources = dataset_sources_for(cfg)
    return write_submission_kernel(
        folder, eval_run_src(cfg), dataset_sources=sources,
        kernel_id=submission_kernel_for(state),
    )


def _config_poe_regate(state: dict) -> dict:
    live = state.get("live_config", {})
    llm_kwargs = {**(live.get("llm_kwargs") or {}), "selection": "poe"}
    # eval_limit: PoE's extra score_sum passes overran Kaggle's 12 h cap on the
    # full 120 (v2 CANCELed 2026-07-17 with zero results). 80 tasks ≈ 8 h.
    return {**live, "llm_kwargs": llm_kwargs, "eval_limit": 80}


def _build_ttt_steps_sweep(state: dict) -> Path:
    folder = stage_dir("ttt_steps_sweep")
    cfg = _config_ttt_steps_sweep(state)
    sources = dataset_sources_for(cfg)
    return write_submission_kernel(
        folder, eval_run_src(cfg), dataset_sources=sources,
        kernel_id=submission_kernel_for(state),
    )


def _config_ttt_steps_sweep(state: dict) -> dict:
    live = state.get("live_config", {})
    ttt_config = {**(live.get("ttt_config") or {}), "max_steps": 96}
    # eval_limit: 96 TTT adapt steps add minutes/task; the full 120 overran the
    # 12 h cap (v4 CANCELed 2026-07-18 with zero results). 60 tasks ≈ 7-8 h.
    return {**live, "ttt_config": ttt_config, "eval_limit": 60}


def _build_dfs_regate(state: dict) -> Path:
    folder = stage_dir("dfs_regate")
    cfg = _config_dfs_regate(state)
    sources = dataset_sources_for(cfg)
    return write_submission_kernel(
        folder, eval_run_src(cfg), dataset_sources=sources,
        kernel_id=submission_kernel_for(state),
    )


def _config_dfs_regate(state: dict) -> dict:
    """Rung 6: DFS/threshold decoding vs the live greedy config, A/B'd for free
    by the standard promotion gate."""
    live = state.get("live_config", {})
    llm_kwargs = {**(live.get("llm_kwargs") or {}), "decode": "dfs", "dfs_eps": 0.12}
    # eval_limit: DFS ~2x decode cost; the full 120 would flirt with the 12 h
    # cap that CANCELed the poe/ttt evals. 80 tasks ≈ 7 h with headroom.
    return {**live, "llm_kwargs": llm_kwargs, "dfs_selftest": True, "eval_limit": 80}


def _build_adapter_hard_6000(state: dict) -> Path:
    folder = stage_dir("adapter_hard_6000")
    run_src = train_run_src(max_examples=6000, config_overrides=_HARD_TRAIN_OVERRIDES)
    return write_train_kernel(folder, run_src, dataset_sources=[HARD_CORPUS_DATASET])


def _config_adapter_hard_6000(state: dict) -> dict:
    return {**state["live_config"], "adapter_path": None}


# ---------------------------------------------------------------------------
# Exploration variants — untried configs worth an otherwise-idle day's
# submission slot. Kaggle ranks the account's BEST submission, so an ungated
# submission can never lower the final standing; at <1% ability the eval gate
# almost never fires, which left the free daily slot unused for 6 straight
# days (2026-07-08 -> 07-14). Each variant is submitted at most once
# (state["explored"] bookkeeping in daily_autopilot).
# ---------------------------------------------------------------------------


def exploration_variants(state: dict) -> list[dict]:
    """Ordered [{name, config}] rotation: static config tweaks first, then any
    gate-failed candidates queued by `daily_autopilot` (a trained adapter that
    can't prove itself on the canary still gets its lottery ticket)."""
    live = state.get("live_config", {})
    variants = [
        {
            "name": "poe",
            "config": {
                **live,
                "llm_kwargs": {**(live.get("llm_kwargs") or {}), "selection": "poe"},
            },
        },
        {
            "name": "ttt96",
            "config": {**live, "ttt_config": {**(live.get("ttt_config") or {}), "max_steps": 96}},
        },
        {
            "name": "dfs",
            "config": {
                **live,
                "llm_kwargs": {
                    **(live.get("llm_kwargs") or {}), "decode": "dfs", "dfs_eps": 0.12,
                },
            },
        },
    ]
    variants.extend(state.get("extra_exploration_variants", []))
    return variants


BACKLOG: list[dict] = [
    {
        "name": "adapter_hard_2500",
        "kind": "train",
        "build": _build_adapter_hard_2500,
        "config": _config_adapter_hard_2500,
        "estimated_hours": 4.0,
    },
    {
        "name": "poe_regate",
        "kind": "eval",
        "build": _build_poe_regate,
        "config": _config_poe_regate,
        # Full 120-task public eval on T4 ~5 h (was 1 h for the old 40-task slice).
        "estimated_hours": 5.0,
    },
    {
        "name": "ttt_steps_sweep",
        "kind": "eval",
        "build": _build_ttt_steps_sweep,
        "config": _config_ttt_steps_sweep,
        # Full 120-task eval with 96 TTT adapt steps/task ~6 h.
        "estimated_hours": 6.0,
    },
    {
        "name": "dfs_regate",
        "kind": "eval",
        "build": _build_dfs_regate,
        "config": _config_dfs_regate,
        # Full 120-task eval; DFS decode ~2x greedy decode cost (not TTT/scoring).
        "estimated_hours": 7.0,
    },
    {
        "name": "adapter_hard_6000",
        "kind": "train",
        "build": _build_adapter_hard_6000,
        "config": _config_adapter_hard_6000,
        "estimated_hours": 8.0,
    },
]
