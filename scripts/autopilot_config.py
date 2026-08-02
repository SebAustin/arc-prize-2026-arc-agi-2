"""Shared constants for the daily autopilot (`daily_autopilot.py` +
`autopilot_kernels.py`) — kernel/dataset slugs, mount paths, and quotas. Split
out from `daily_autopilot.py` purely to keep individual files small; nothing
here is environment-aware or has side effects at import time.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "artifacts" / "autopilot_state.json"
SUBMISSION_LOG_PATH = REPO_ROOT / "SUBMISSION_LOG.md"
KERNEL_RUNS_DIR = REPO_ROOT / "artifacts" / "kernel_runs"

COMPETITION_SLUG = "arc-prize-2026-arc-agi-2"
# FRESH kernel ids (2026-07-14): Kaggle's api.kaggle.com migration left kernels
# created before it broken server-side — GetKernel 500s and every push to them
# fails "Kernel push error: Notebook not found" — while newly-created kernels
# push fine (verified: create + v2 update both work). The pre-migration kernels
# (arc-agi-2-self-contained-submission-notebook, arc-agi-2-train-adapter-rung-4)
# keep their history (incl. the 0.83 submission) but can no longer be pushed to.
SUBMISSION_KERNEL = "sebmontreal/arc-agi-2-autopilot-submit"
TRAIN_KERNEL = "sebmontreal/arc-agi-2-autopilot-train"

# The only push-time accelerator value that reliably works via the API — see
# the module docstring / DEPLOYMENT.md Sec 3 for the L4x4-resets-to-P100 landmine.
GPU_ACCELERATOR = "NvidiaTeslaT4"

MODEL_MOUNT_ROOT = "/kaggle/input/models/"

BASE_MODEL_MOUNT = "/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/7b/1"
# model_sources entries are the mount path minus the "/kaggle/input/models/" root.
BASE_MODEL_SOURCE = "qwen-lm/qwen2.5-coder/transformers/7b/1"

# NVARC (ARC Prize 2025 1st place, 24.03 private with THIS model + their
# harness): Qwen3-4B after their 139k-example grid SFT, published as a public
# Kaggle model by the winners (mounted straight from their winning notebook's
# metadata). 4B fp16 ~8GB also leaves real TTT headroom on a 14.5GB T4, unlike
# our 7B. License: prize rules required winners to open-source permissively —
# recorded as an assumption to verify in ASSUMPTIONS.md.
# Framework segment MUST be lowercase "transformers": Kaggle's API resolves model
# refs case-insensitively, but the kernel MOUNT path is case-sensitive and uses the
# canonical lowercase framework (the working Qwen base above proves this). A capital
# "Transformers" here yields /kaggle/input/models/.../Transformers/... — a path that
# doesn't exist at mount time, so the model fails to load (same failure class as the
# week-long adapter-path bug). Verified against the live instance via the Kaggle CLI.
NVARC_SFT_SOURCE = "sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
NVARC_SFT_MOUNT = MODEL_MOUNT_ROOT + NVARC_SFT_SOURCE

# Kaggle mounts an attached DATASET at /kaggle/input/<slug> — the owner is
# STRIPPED (only MODELS nest under /kaggle/input/models/<owner>/...). Building
# the mount as /kaggle/input/datasets/<owner>/<slug> yields a path that does not
# exist, so PEFT raised "Can't find 'adapter_config.json'" and every eval kernel
# ERROR'd for a week. Derive the mount from the slug's basename only.
HARD_CORPUS_DATASET = "sebmontreal/arc-synth-hard"
HARD_CORPUS_FILE = f"/kaggle/input/{HARD_CORPUS_DATASET.split('/')[-1]}/synth_hard_50k.jsonl"

# The adapter dataset this autopilot stages+versions itself, each time a
# training backlog item completes. One slug, versioned in place.
ADAPTER_DATASET_SLUG = "sebmontreal/arc-agi-2-adapter-autopilot"
ADAPTER_MOUNT = f"/kaggle/input/{ADAPTER_DATASET_SLUG.split('/')[-1]}"

WEEKLY_GPU_HOUR_QUOTA = 25.0

# An exploration submission's commit-run cost: the placeholder fast-path canary
# (~12 tasks) on T4. The hidden-set scoring rerun after `competitions submit`
# runs on Kaggle's competition compute, not our quota.
EXPLORE_COMMIT_HOURS = 1.5

# Public-eval canary size for every autopilot eval (candidate gate + backlog evals).
# The model solves <1% of tasks, so a 40-task slice reads 0/40 even for a working
# pipeline and the promotion gate (correct >= best_eval_correct + 1) can almost never
# fire. Scoring the FULL public evaluation split (120 tasks) ~triples the chance of
# registering a solved task — the sensitivity the gate needs to ever auto-promote —
# at ~3x the GPU cost (~5 h on T4). None = all tasks in the split (kaggle_eval treats
# a None `limit` as "to the end", so this stays correct if the split ever grows).
FULL_EVAL_LIMIT: int | None = None

# The autopilot can only get T4x2 via the API (L4x4 is UI-only — landmine). A 7B
# in fp16 fills ~13 of a T4's 14.5 GB, leaving almost nothing for per-task TTT's
# LoRA activations — the default TTT config OOMs every task and the whole eval
# degrades to DSL-only (and can flip the kernel to ERROR). These memory-safe TTT
# settings (proven on the v7 canary) are merged UNDER any run's own ttt_config so
# batch/seq stay T4-survivable while callers can still tune max_steps etc.
T4_SAFE_TTT = {"batch_size": 1, "max_seq_len": 1536}


# ---- Kaggle mount-path convention guard ------------------------------------
# Two DISTINCT mount bugs each caused a week-long silent failure (adapter dataset
# nested under an owner; model framework segment capitalized). Both are the same
# root cause: a `/kaggle/input/...` constant that doesn't match how Kaggle
# actually mounts the asset, discovered only after days of ERROR kernels. These
# structural checks encode the conventions so the mistake fails a test (and the
# autopilot's startup guard) instead of a week of runs. `scripts/verify_kaggle_
# mounts.py` confirms the same constants resolve against the LIVE Kaggle API.

_INPUT_PREFIX = "/kaggle/input/"


def _mount_specs() -> list[tuple[str, str, str, str]]:
    """(constant_name, mount_path, slug_or_source, kind) for every mount constant."""
    return [
        ("BASE_MODEL_MOUNT", BASE_MODEL_MOUNT, BASE_MODEL_SOURCE, "model"),
        ("NVARC_SFT_MOUNT", NVARC_SFT_MOUNT, NVARC_SFT_SOURCE, "model"),
        ("ADAPTER_MOUNT", ADAPTER_MOUNT, ADAPTER_DATASET_SLUG, "dataset"),
        ("HARD_CORPUS_FILE", HARD_CORPUS_FILE, HARD_CORPUS_DATASET, "dataset"),
    ]


def check_one_mount(name: str, mount: str, ref: str, kind: str) -> list[str]:
    """Structural violations for a single mount constant (empty == well-formed)."""
    if not mount.startswith(_INPUT_PREFIX):
        return [f"{name}: {mount!r} must start with {_INPUT_PREFIX!r}"]
    segs = mount[len(_INPUT_PREFIX) :].split("/")
    problems: list[str] = []
    if kind == "model":
        # Canonical: models/<owner>/<model>/<framework>/<variation>/<version>.
        expected = MODEL_MOUNT_ROOT + ref
        if mount != expected:
            problems.append(f"{name}: {mount!r} != MODEL_MOUNT_ROOT + source ({expected!r})")
        src = ref.split("/")
        if len(src) != 5:
            problems.append(
                f"{name}: model source {ref!r} must be owner/model/framework/variation/version"
            )
        elif src[2] != src[2].lower():
            # Kaggle mounts are case-sensitive and the framework slug is lowercase;
            # a capitalized framework points at a path that never exists.
            problems.append(
                f"{name}: framework {src[2]!r} must be lowercase — use {src[2].lower()!r}"
            )
    else:  # dataset — mounts FLAT at /kaggle/input/<slug-basename>, owner stripped.
        if segs[0] in ("models", "datasets"):
            problems.append(f"{name}: dataset must not nest under /{segs[0]}/ (owner is stripped)")
        basename = ref.split("/")[-1]
        if segs[0] != basename:
            problems.append(
                f"{name}: first segment {segs[0]!r} must be the slug basename {basename!r}"
                f" (owner stripped) — got {mount!r}"
            )
    return problems


def check_mount_conventions() -> list[str]:
    """Return all Kaggle mount-path convention violations across the autopilot's
    path constants; empty means every constant is well-formed. See the block
    comment above for the two conventions this enforces."""
    problems: list[str] = []
    for spec in _mount_specs():
        problems.extend(check_one_mount(*spec))
    return problems
