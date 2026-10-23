# Runbook: NVARC reproduction anchor on L4×4

**Goal.** Get the project's first *trustworthy* number: run the ARC Prize 2025
winner's own notebook, untouched except for its input path, on the hardware it was
built for (4× L4), and score its output on our 120-task canary with our metrics.

**Why this run.** In 3+ months our pipeline has never produced a non-zero canary
signal. This run is the **reproduction gate** (ADR context + the 2026-10-02 research
note): NVARC's released notebook scored 24% on the 2025 private set, so if *their*
code reproduces ≈15–20%+ on our eval canary, the **method and our canary harness are
proven sound**, and any gap our own stack shows later is localised to *our* code.
This is the anchor three months of unanchored iteration never had.

**Who runs it.** You, in the Kaggle UI — L4×4 is UI-only; the autopilot (T4×2 via
API) cannot drive it. It needs GPU quota, so it's a post-reset action.

---

## Preconditions

1. **Weekly GPU quota available.** Kaggle's quota resets on a fixed weekly cycle,
   **Saturday ~00:00 UTC** (the *amount* floats with demand — this account has seen
   45 h). Confirm remaining hours first:
   ```bash
   .venv/bin/kaggle quota
   ```
   (If that errors — the wrapper is buggy in some CLI versions — check the accelerator
   panel in any Kaggle notebook editor, or your account usage page.)
2. This repo checked out locally (for scoring afterward). No GPU needed locally.

## Before the quota window (no compute — read-only recon)

Open NVARC's notebook in the browser and find the **single line that reads the
test-challenges file**. Record its exact **variable name, cell, and current path**
(it points at the 2025 *hidden* test, e.g. `.../arc-agi_test_challenges.json`). Also
confirm the notebook builds its submission dict keys from the **challenges it reads**,
not from a 240-key `sample_submission.json`. The repo holds no copy of this notebook,
so this recon is the only thing that makes the one input edit (step 5) reliable.

## Steps (Kaggle web UI)

1. Open NVARC's released submission notebook and **Copy & Edit**:
   `https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue`
2. **Accelerator → GPU L4 ×4**, and confirm the Settings → Accelerator panel reads
   **exactly "GPU L4 ×4"** — this is the *only* authoritative source (the API/CLI
   cannot set or report L4×4). The notebook is hard-wired to 4 GPUs
   (`mp.spawn(..., nprocs=4)`) and, being untouched, has **no GPU guard**; a Copy &
   Edit fork does not reliably inherit L4×4, and T4/P100/CPU all break it (P100 can't
   even run Kaggle's torch). Set **Internet = Off**.
3. Confirm the notebook's shipped inputs are still attached (model
   `sorokin/qwen3_4b_grids15_sft139` + the NVARC code/datasets it imports); re-add any
   that dropped during the fork.
4. **Add Input → Competitions → ARC Prize 2026** (`arc-prize-2026-arc-agi-2`). Do
   **not** attach a separate third-party "eval" dataset — only the competition mount
   guarantees the task-ids **and per-test ordering** match the local solutions the
   scorer reads (scoring is positional, and 52 of 172 eval outputs sit at a non-zero
   test index, so order must match).
5. **Make the one edit from your recon:** repoint that test-challenges variable to
   `/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json`. You are
   changing **both** the competition slug (2025 → `arc-prize-2026-arc-agi-2`) **and**
   the filename (`test` → `evaluation`). Copy the folder name from the attached-input
   sidebar rather than typing it. Change nothing else — recipe, LoRA, decoding,
   selection stay as released.
6. **Save Version → Save & Run All (Commit).** Do **not** rely on an interactive "Run
   All" left open: a 6–12 h session drops on an idle timeout or closed browser and
   loses all output — only the committed batch run gets the full 12 h cap and persists
   a downloadable `submission.json`. In the first ~2 min of the committed run's log,
   confirm **4 workers start (ranks 0–3)** with no CUDA/device-count or OOM error; if
   it is not 4× L4, **stop and fix the accelerator** before the window drains.
7. When the committed version finishes, **download `submission.json` from that
   version's output** (not from an interactive session). Note the wall-clock.

## Score it (local, no GPU)

```bash
cd "<repo root>"
PYTHONPATH=src:scripts .venv/bin/python scripts/score_submission.py path/to/submission.json
```

This prints our canary numbers on exactly the yardstick we use for our own stack:
- `score` / `correct` / `total` — **top-2 exact-match** (the competition metric),
- `cell_accuracy`, `shape_correct_rate` — partial-credit signals for the sub-24% regime,
- `scored_tasks` / `missing_from_submission` — coverage. **`scored_tasks` MUST be 120.**
  If it is 0 the scorer **exits non-zero with a "FAIL — wrong split" banner**: that is
  *not* a 0% score, it means the notebook ran on the wrong split/keys — re-check steps
  4–5 and do **not** record it as a result.

## The gate

- **Anchor PASS:** exact-match ≳ **15–20%**. The method + our harness are sound.
  Record the number; proceed to reproduce through *our* stack and localise the gap.
- **Anchor FAIL (≈0):** something in the harness/input is wrong (their code is known
  to score 24%) — debug the run (split, paths, accelerator) before trusting any of
  our own numbers.

Record the result (number, wall-clock, date) in `.remember/` and update the plan.

## Notes

- "Untouched except the input path" is deliberate: pointing at the eval split is the
  minimum change needed to get a *scorable* number; the recipe itself is unmodified,
  so the result is a faithful reproduction read.
- Our own-stack reproduction (next step) runs at best-effort fidelity and is **not**
  held to this exact gate — the gate lives on NVARC's untouched notebook. See
  [ADR-0003](../adr/0003-optimize-for-the-private-rerun-not-the-public-leaderboard.md)
  for why the public leaderboard is not the target.
