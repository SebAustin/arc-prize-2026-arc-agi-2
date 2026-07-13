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
SUBMISSION_KERNEL = "sebmontreal/arc-agi-2-self-contained-submission-notebook"
TRAIN_KERNEL = "sebmontreal/arc-agi-2-train-adapter-rung-4"

# The only push-time accelerator value that reliably works via the API — see
# the module docstring / DEPLOYMENT.md Sec 3 for the L4x4-resets-to-P100 landmine.
GPU_ACCELERATOR = "NvidiaTeslaT4"

BASE_MODEL_MOUNT = "/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/7b/1"
# model_sources entries are the mount path minus the "/kaggle/input/models/" root.
BASE_MODEL_SOURCE = "qwen-lm/qwen2.5-coder/transformers/7b/1"

HARD_CORPUS_DATASET = "sebmontreal/arc-synth-hard"
HARD_CORPUS_FILE = "/kaggle/input/datasets/sebmontreal/arc-synth-hard/synth_hard_50k.jsonl"

# The adapter dataset this autopilot stages+versions itself, each time a
# training backlog item completes. One slug, versioned in place.
ADAPTER_DATASET_SLUG = "sebmontreal/arc-agi-2-adapter-autopilot"
ADAPTER_MOUNT = f"/kaggle/input/datasets/{ADAPTER_DATASET_SLUG}"

WEEKLY_GPU_HOUR_QUOTA = 25.0
