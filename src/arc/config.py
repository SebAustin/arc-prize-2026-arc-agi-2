"""Environment configuration — the ONLY environment-aware module in the codebase.

Resolves data/output paths and the runtime mode (SMOKE on a local CPU box vs
KAGGLE on the L4x4 GPU). Everything else in `arc` receives paths/objects and is
environment-agnostic, so the identical code path runs in both places.

Detection order (first match wins):
  1. Explicit env vars (`ARC_DATA_DIR`, `ARC_OUTPUT_DIR`, `ARC_MODE`).
  2. Kaggle, if `/kaggle/input` exists.
  3. Local fallback (the user's Downloads copy of the competition data).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---- Runtime modes ---------------------------------------------------------
MODE_SMOKE = "SMOKE"   # local CPU, tiny/mock model, fast plumbing checks
MODE_KAGGLE = "KAGGLE"  # offline L4x4 GPU, real model + TTT

# Kaggle mounts competition data read-only under this directory.
_KAGGLE_INPUT = Path("/kaggle/input")
_KAGGLE_DATA = _KAGGLE_INPUT / "arc-prize-2026-arc-agi-2"
_KAGGLE_WORKING = Path("/kaggle/working")

# Local fallback: the user's downloaded competition bundle.
_LOCAL_DATA = Path(
    "/Users/sebastienhenry/Downloads/arc-prize-2026-arc-agi-2"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_OUTPUT = _REPO_ROOT / "artifacts"

# Canonical file names within a data directory.
FILE_NAMES = {
    "training_challenges": "arc-agi_training_challenges.json",
    "training_solutions": "arc-agi_training_solutions.json",
    "evaluation_challenges": "arc-agi_evaluation_challenges.json",
    "evaluation_solutions": "arc-agi_evaluation_solutions.json",
    "test_challenges": "arc-agi_test_challenges.json",
    "sample_submission": "sample_submission.json",
}


def _detect_mode() -> str:
    forced = os.environ.get("ARC_MODE")
    if forced:
        return forced.upper()
    return MODE_KAGGLE if _KAGGLE_INPUT.exists() else MODE_SMOKE


def _find_kaggle_data_dir() -> Path | None:
    """Locate the competition data under /kaggle/input.

    Kaggle mounts the competition under a folder named after its slug, but rather
    than trust a single hardcoded name we prefer the documented slug and
    otherwise scan for whichever mounted folder actually contains the
    test-challenges file (a couple of nesting levels deep). Returns None if
    nothing matching is mounted.
    """
    if not _KAGGLE_INPUT.exists():
        return None
    if _KAGGLE_DATA.exists():
        return _KAGGLE_DATA
    fname = FILE_NAMES["test_challenges"]
    for pattern in (f"*/{fname}", f"*/*/{fname}"):
        for match in sorted(_KAGGLE_INPUT.glob(pattern)):
            return match.parent
    return None


def _detect_data_dir() -> Path:
    override = os.environ.get("ARC_DATA_DIR")
    if override:
        return Path(override)
    if _KAGGLE_INPUT.exists():
        # On Kaggle: use the mounted competition data wherever it is; never fall
        # back to a developer-machine path (that produces a baffling
        # FileNotFoundError pointing at a box that isn't even running). If the
        # data isn't attached yet, return the documented slug so the eventual
        # error references a real /kaggle path — the entrypoint turns that into
        # an actionable "attach the competition data" message.
        found = _find_kaggle_data_dir()
        return found if found is not None else _KAGGLE_DATA
    return _LOCAL_DATA


def _detect_output_dir() -> Path:
    override = os.environ.get("ARC_OUTPUT_DIR")
    if override:
        return Path(override)
    if _KAGGLE_WORKING.exists():
        return _KAGGLE_WORKING
    return _LOCAL_OUTPUT


@dataclass(frozen=True)
class Config:
    """Resolved, immutable view of the runtime environment."""

    mode: str
    data_dir: Path
    output_dir: Path

    @property
    def is_kaggle(self) -> bool:
        return self.mode == MODE_KAGGLE

    @property
    def is_smoke(self) -> bool:
        return self.mode == MODE_SMOKE

    def challenges_path(self, split: str) -> Path:
        """split in {'training', 'evaluation', 'test'}."""
        return self.data_dir / FILE_NAMES[f"{split}_challenges"]

    def solutions_path(self, split: str) -> Path:
        """split in {'training', 'evaluation'} (test solutions are hidden)."""
        return self.data_dir / FILE_NAMES[f"{split}_solutions"]

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / FILE_NAMES["sample_submission"]

    @property
    def submission_path(self) -> Path:
        return self.output_dir / "submission.json"


def get_config() -> Config:
    """Resolve the active configuration from the environment."""
    cfg = Config(
        mode=_detect_mode(),
        data_dir=_detect_data_dir(),
        output_dir=_detect_output_dir(),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# ---- Global inference budget (used by the time watchdog in pipeline.py) ----
# 240 test tasks / 12 h ≈ 3 min/task. Keep margin for model load + I/O.
TOTAL_RUNTIME_BUDGET_S = 11.0 * 3600  # leave ~1h headroom under the 12h cap
DEFAULT_PER_TASK_BUDGET_S = 150.0     # 2.5 min target per task
