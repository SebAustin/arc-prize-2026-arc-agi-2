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

## 2026-07-19 — NVARC (ARC Prize 2025 winner) asset adoption

- **Assumption:** NVARC's released assets (github.com/1ytic/NVARC; Kaggle model
  `sorokin/qwen3_4b_grids15_sft139`; datasets `sorokin/nvarc-synthetic-puzzles`,
  `nvarc-augmented-puzzles`) are permissively licensed. Basis: ARC Prize 2025
  rules required winners to open-source solutions under a permissive license to
  receive the prize, and NVARC was paid 1st place (arcprize.org 2025 results).
  The Kaggle dataset license TAGS read "unknown" and the repo page shows no
  LICENSE file, so this is inferred, not confirmed.
- **Action before relying on it for OUR final prize claim:** confirm the exact
  license in the NVARC paper (nvarc_2025.pdf in their repo) or Kaggle writeup;
  if unclear, ask in the competition forum. Using the public Kaggle model for
  leaderboard experiments meanwhile is standard Kaggle practice (attach-only).
- Verdict recorded here per the Rung-5 license-gating protocol.
