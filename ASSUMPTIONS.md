# Assumptions

Working assumptions made during the build, to revisit as milestones land.

- **Local data location.** Competition JSON stays in
  `/Users/sebastienhenry/Downloads/arc-prize-2026-arc-agi-2/`; the repo reads it
  via `arc.config` (not copied or committed). On Kaggle it auto-switches to
  `/kaggle/input/arc-prize-2026-arc-agi-2/`. Override with `ARC_DATA_DIR`.
- **Python.** Dev venv is Python 3.13 (Anaconda) with numpy + pytest only. Torch
  and friends are an optional `gpu` extra installed on Kaggle, never locally
  (the Mac has no CUDA).
- **Grid representation.** `tuple[tuple[int,...],...]` — immutable + hashable, so
  grids can be vote keys. Solvers may compute in numpy and convert back.
- **`test_challenges.json` is a placeholder.** Per competition docs it currently
  mirrors evaluation tasks; the real 240 hidden tasks are swapped in at rerun.
  We never rely on its contents, only its schema.
- **Base model (M1).** Default Qwen2.5 (3B → 7B) unless an open ARC-finetuned
  checkpoint clearly beats it; finalised after the M1 baseline.
- **Submission discipline.** 1 submission/day is reserved for notebooks that
  passed a local no-internet dry run; no speculative submits.
- **Score target.** ~15–25% (the proven Kaggle-offline band). The 85% bonus is
  out of scope.
