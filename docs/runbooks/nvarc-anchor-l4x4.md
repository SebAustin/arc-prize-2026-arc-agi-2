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

## Steps (Kaggle web UI)

1. Open NVARC's released submission notebook and **Copy & Edit**:
   `https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue`
2. **Accelerator → GPU L4 ×4.** The notebook is hard-wired to 4 GPUs
   (`mp.spawn(..., nprocs=4)`); anything else will misbehave. Internet can stay off.
3. Confirm these inputs are attached (they ship with the notebook; re-add if missing):
   - model `sorokin/qwen3_4b_grids15_sft139` (the SFT'd Qwen3-4B base),
   - the NVARC code/datasets the notebook imports.
4. **Point the input at our scorable canary** (the one unavoidable edit — the stock
   notebook reads the competition's *hidden* test, which has no public solutions, so
   it can't be scored). Attach the **ARC-AGI-2 evaluation split**
   (`arc-agi_evaluation_challenges.json` + `..._solutions.json`) and change the
   notebook's test-challenges input path to the evaluation challenges file.
   Change nothing else — recipe, LoRA, decoding, selection all stay as released.
5. **Run All.** Budget is Kaggle's 12 h code-competition cap; NVARC used the full 12 h
   on 4× L4 for the 4B config, so expect a long run. Note the wall-clock it reports.
6. When it finishes, **download `submission.json`** from the notebook output.

## Score it (local, no GPU)

```bash
cd "<repo root>"
PYTHONPATH=src:scripts .venv/bin/python scripts/score_submission.py path/to/submission.json
```

This prints our canary numbers on exactly the yardstick we use for our own stack:
- `score` / `correct` / `total` — **top-2 exact-match** (the competition metric),
- `cell_accuracy`, `shape_correct_rate` — partial-credit signals for the sub-24% regime,
- `scored_tasks` / `missing_from_submission` — sanity on coverage (if `scored_tasks`
  is 0, the notebook ran on the wrong split — re-check step 4).

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
  held to this exact gate — the gate lives on NVARC's untouched notebook. See the
  grill decisions recorded in `.remember/now.md`.
