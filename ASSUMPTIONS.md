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
- **Base model.** Currently running **Qwen2.5-Coder-3B-Instruct** on Kaggle
  (first real GPU run, 2026-07-01). ⚠️ **Licensing:** the Qwen2.5 **3B** sizes
  (both general and Coder) are under the **Qwen Research License, NOT Apache-2.0**
  — fine for scoring/leaderboard, but **NOT prize-eligible**. For a prize-eligible
  final submission use an **Apache-2.0** size: Qwen2.5-Coder-**7B**-Instruct
  (recommended — fits one L4, keeps the code-model edge) or **1.5B** if 7B is too
  slow within the 12h budget. The solver is model-size-agnostic (generic
  `AutoModel` + raw-text prompts), so switching sizes needs no code change — just
  re-attach the model and update `MODEL_DS`.
- **Submission discipline.** 1 submission/day is reserved for notebooks that
  passed a local no-internet dry run; no speculative submits.
- **Score target.** ~15–25% (the proven Kaggle-offline band). The 85% bonus is
  out of scope.
