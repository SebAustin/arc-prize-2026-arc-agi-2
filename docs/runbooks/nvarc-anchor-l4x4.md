# Runbook: NVARC reproduction anchor on L4×4

**Goal.** Get the project's first *trustworthy* number: run the ARC Prize 2025
winner's own notebook, untouched except for removing its 4-task dev filter, on the
hardware it was built for (4× L4), and score its output on our 120-task canary with
our metrics.

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

## Recon (DONE 2026-10-26 — pulled the notebook source; findings below)

Recon was run by pulling the source (`kaggle kernels pull sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue`,
8 code cells). The notebook selects its input split by an **env var, not a hardcoded path**:

```python
rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")   # cells 4, 5, 7
if rerun_mode: test_path = ".../arc-agi_test_challenges.json"        # hidden test (private rerun only)
else:          test_path = ".../arc-agi_evaluation_challenges.json"  # eval split
```

`KAGGLE_IS_COMPETITION_RERUN` is set **only** during Kaggle's private rerun. A normal
**Save & Run All (Commit)** is NOT a rerun, so the notebook **auto-selects the eval split**
and cell 7 self-scores it against `arc-agi_evaluation_solutions.json` (prints
`*** Reload score: ...`). **No path edit is needed to reach the eval split** — the earlier
"repoint the path" instruction was wrong and has been removed.

**THE ONE REQUIRED EDIT — delete the 4-task dev filter.** In the `starter.py` cell (the one
ending `mp.spawn(local_worker, ..., nprocs=4)`), the non-rerun path queues only 4 task ids:

```python
for key in sorted(data.keys()):
    if not rerun_mode:                                   # <-- DEV SUBSET: delete this block
        if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
            continue
    queue.put(key)
```

Left in place, a commit infers only **4 of ~120** eval tasks; `get_submission` still emits all
~120 and defaults the other ~116 to `[[0]]` → a submission that scores ~0 and *looks like a
failed run* (this is the likely source of the Oct-2 0/0.12/0.25 baseline). Delete the
`if not rerun_mode: ... continue` block so every eval task is queued. This single edit is the
difference between a real anchor and a wasted window.

Other facts (no edit needed): model mounts at
`/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1`; input is the **arc-prize-2025**
competition (its eval ids overlap our 2026 canary — `0934a4d8` etc.); `global_end_time =
now + 12h − 10min` (fits the commit cap); `nprocs=4` (needs L4×4); `n_guesses=2` (real pass@2,
so `attempt_1 ≠ attempt_2` on a genuine run).

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
4. **Confirm the `arc-prize-2025` competition input is still attached** (it ships the
   eval challenges + solutions the notebook reads and self-scores against). Do **not**
   swap it for 2026 or a third-party "eval" dataset — the notebook is wired to the 2025
   mount and its eval ids overlap our canary. (Only if the later cross-check with our
   scorer shows partial coverage do you also need the 2026 mount — see "Score it".)
5. **Make the one required edit from Recon: delete the 4-task dev filter** in the
   `starter.py` / `mp.spawn` cell — the `if not rerun_mode: if key not in [...]: continue`
   block — so all ~120 eval tasks are queued, not 4. Leave everything else exactly as
   released: recipe, LoRA, decoding, selection, and the `rerun_mode` path logic are all
   already correct (a Commit auto-takes the eval branch).
6. **Save Version → Save & Run All (Commit).** Do **not** rely on an interactive "Run
   All" left open: a 6–12 h session drops on an idle timeout or closed browser and
   loses all output — only the committed batch run gets the full 12 h cap and persists
   a downloadable `submission.json`. In the first ~2 min of the committed run's log,
   confirm **4 workers start (ranks 0–3)** with no CUDA/device-count or OOM error; if
   it is not 4× L4, **stop and fix the accelerator** before the window drains. As it
   runs, confirm the `[Rank N] finished <key>` lines cover **far more than 4 distinct
   keys** (proof the dev filter is gone); at the end, cell 7 prints `*** Reload score:`
   — that printed number is the anchor read, independent of our scorer.
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

Two cross-checks against the notebook's own `*** Reload score` (the primary anchor read):
- **Coverage:** if our scorer reports `scored_tasks` between 1 and 119 (partial, not 0),
  the 2025 and 2026 eval sets differ on a few ids — trust the notebook's self-score and,
  if an exactly-comparable canary number is needed, re-run with the 2026 competition mounted
  and the `else`-branch slug changed to `arc-prize-2026-arc-agi-2`.
- **Not degenerate:** a genuine pass@2 run has `attempt_1 ≠ attempt_2` on most tasks. If a
  downloaded `submission.json` has them identical everywhere (as the Oct-2 baseline did),
  the decode/selection path didn't run — the dev filter is still in, or the run timed out.

## The gate

- **Anchor PASS:** exact-match ≳ **15–20%**. The method + our harness are sound.
  Record the number; proceed to reproduce through *our* stack and localise the gap.
- **Anchor FAIL (≈0):** something in the harness/input is wrong (their code is known
  to score 24%) — debug the run (split, paths, accelerator) before trusting any of
  our own numbers.

Record the result (number, wall-clock, date) in `.remember/` and update the plan.

## Notes

- "Untouched except the dev filter" is deliberate: removing the 4-task subset is the
  minimum change needed to get a *full-eval scorable* number (the eval split is already
  auto-selected on a non-rerun commit); the recipe itself is unmodified, so the result
  is a faithful reproduction read.
- Our own-stack reproduction (next step) runs at best-effort fidelity and is **not**
  held to this exact gate — the gate lives on NVARC's untouched notebook. See
  [ADR-0003](../adr/0003-optimize-for-the-private-rerun-not-the-public-leaderboard.md)
  for why the public leaderboard is not the target.
