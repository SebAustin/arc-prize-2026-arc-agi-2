# DEPLOYMENT.md — Kaggle Deployment Runbook

Target: **ARC Prize 2026 / ARC-AGI-2** Kaggle competition kernel — Internet **Off**,
GPU **L4×4** (4×24 GB), **12 h** hard wall-clock cap, ~240 hidden test tasks,
exact-match scoring, 2 attempts per test output, **1 submission/day**.

This is an operator runbook. It does not replace `README.md` or
`scripts/stage_kaggle_assets.md` — it sequences them into a go/no-go pipeline per the
`deployment-readiness` skill. Read-only repo; nothing here edits source or triggers a
production submission — the final "Submit to Competition" click is yours.

---

## 1. TL;DR — fastest path (5 minutes, zero setup)

Validates plumbing end-to-end with **no model, no dataset, no API token** — pure DSL
ensemble.

1. Kaggle → competition page → **Code** → **New Notebook**.
2. **File → Upload Notebook** → `notebooks/submission_selfcontained.ipynb`.
   (If `src/` changed since this file was generated, regenerate it first —
   see §3.0.)
3. Notebook Settings (right panel):
   - **Accelerator**: GPU T4×2 or L4×4 (either works for Phase A; DSL-only is CPU-cheap).
   - **Internet**: **Off**.
   - Competition data is auto-attached on a competition notebook (confirm a
     `arc-prize-2026-arc-agi-2` folder is listed under Input; Kaggle may mount it
     under `/kaggle/input/competitions/…` — the code auto-detects either).
4. Leave the **config cell**'s `MODEL_DS = None` (default — do not edit unless doing Phase B).
5. **Run All** (the first code cell auto-removes an incompatible `torchao` if present — see §7).
6. Confirm the **run cell** prints `problems: []` and `/kaggle/working/submission.json` exists.

That's a valid, scoreable (DSL-only) submission. Go to §6 for the gated submit step,
or continue to §3 Phase B for a real TTT score.

---

## 2. Prerequisites

- [ ] Kaggle account, **joined the competition** (Rules accepted — required before any
      submission counts).
- [ ] GPU quota available (Settings → Accelerator; Kaggle grants ~30 GPU-hrs/week by
      default — confirm you have headroom for a 12 h run before using it here).
- [ ] For Phase B / a real score: a model attached via **Add Input → Models** search
      `Qwen2.5-3B-Instruct` (community/Kaggle-hosted mirror — no upload, no
      internet needed at run time, since Kaggle stages Model assets locally). Note
      the mount path it reports, e.g. `/kaggle/input/qwen2.5-3b-instruct/transformers/...`.
- [ ] Licensing: everything staged must be open-source-compatible (Apache/MIT/CC-BY)
      for **prize eligibility**. ⚠️ Most Qwen2.5 sizes are Apache-2.0, but the **3B**
      variants (general *and* Coder) are under the **Qwen Research License — NOT
      prize-eligible**. For a prize submission use an Apache-2.0 size:
      Qwen2.5-Coder-**7B**-Instruct (fits one L4) or **1.5B**. The 3B is fine only for
      leaderboard/testing. Re-check any non-Qwen model before swapping it in.
- [ ] (Optional, Phase B extra) a base-fine-tune LoRA adapter trained on the synthetic
      corpus via `src/arc/train/finetune.py`, staged as its own Kaggle Dataset, if you
      want `ADAPTER_DS` populated.

---

## 3. Path A — self-contained notebook (recommended)

`notebooks/submission_selfcontained.ipynb` base64-embeds the entire `arc` package plus
`scripts/kaggle_submit.py` in the "bootstrap" cell, so the kernel needs no code dataset
and no Kaggle API token. It has 5 cells: intro (markdown) → **env-prep** (removes an
incompatible `torchao` so PEFT/LoRA can run — see §7) → bootstrap (writes the embedded
package to `/kaggle/working/arc_code`, adds it to `sys.path`) → config (`MODEL_DS`,
`ADAPTER_DS`) → run (`kaggle_submit.main(...)`).

**Canary-first workflow (recommended before any full run):** set `MAX_TASKS = 8` in the
config cell (optionally `ttt_config={'max_steps': 16}` in the run cell) → a ~20-30 min
run validates GPU presence, model load, and TTT end-to-end. The canary's
`submission.json` is still schema-complete (unsolved tasks keep the fallback grid).
Check the log for `GPU N:` lines (nvidia-smi), `HFModel: loading ... dtype=...`, no
`solver llm_ttt failed` lines, and `schema_problems=0` — then flip `MAX_TASKS = None`
for the full scored run. Two earlier 10-hour-scale failures (accelerator off; torchao
conflict) would each have been caught by a 20-minute canary.

**Pushing via the API (`kaggle kernels push`) — two landmines (learned the hard way):**
- `--accelerator` **overrides** `enable_gpu` from the metadata, and the server
  **silently accepts invalid names** — a typo yields a CPU-only run with no error.
  Only `NvidiaTeslaT4` and `NvidiaTeslaP100` are documented/valid GPU names; the
  **L4×4 enum name is not exposed by the API** — L4×4 can only be selected in the
  notebook UI (Settings → Accelerator), after which it persists for future versions.
- A pulled `kernel-metadata.json` may contain `"machine_shape": "None"` (a literal
  string) — **delete that key before pushing** or it can override `enable_gpu: true`.
- Worse: the API **reports** values it does not **accept** — a kernel with L4×4 set in
  the UI pulls as `"machine_shape": "NvidiaL4"`, but pushing that value back is treated
  as invalid and **resets the session to the default P100** (whose sm_60 Kaggle's torch
  cannot run: "no kernel image is available"). Rule: **always strip `machine_shape`
  (and `docker_image`) from pulled metadata before every push** — an omitted value
  preserves the kernel's stored UI accelerator; any explicit value risks a silent reset.
  The only API-pushable GPU value that works reliably is `--accelerator NvidiaTeslaT4`.
- The model is hardware-adaptive either way: bf16 on L4/A100, fp16 on T4/P100, and
  `device_map="auto"` shards a 7B across multiple cards (2×T4 works).

### 3.0 Regenerate before every deploy if `src/` changed (REQUIRED)

The notebook is a build artifact, not hand-maintained source. **Never hand-edit
`notebooks/submission_selfcontained.ipynb`** — any fix belongs in `src/` or
`scripts/kaggle_submit.py`, then rebuild:

```bash
python scripts/build_kaggle_notebook.py
# -> wrote notebooks/submission_selfcontained.ipynb  (NN KB)
```

Run this any time `src/arc/**` or `scripts/kaggle_submit.py` changes and before every
upload to Kaggle. Treat a notebook that predates the latest `src/` commit as stale —
diff-worthy the same way a stale build artifact would be in any other pipeline.

### 3.1 Phase A — zero setup (validate plumbing, DSL-only)

1. Upload the notebook (§1 steps 1–3).
2. Cell 3 (config): leave as generated —
   ```python
   MODEL_DS = None      # e.g. '/kaggle/input/qwen2.5-3b-instruct'  (None = DSL-only Phase A)
   ADAPTER_DS = None    # e.g. '/kaggle/input/arc-base-ft-adapter'  (optional)
   ```
3. **Internet = Off**. **Run All**.
4. Expect in the output log: `WARNING: no model_path — running DSL/heuristic ensemble
   only`, then `submission: /kaggle/working/submission.json  schema_problems=0`.
5. This is the safety-net path — DSL + cheap heuristics only, no LLM, validated
   end-to-end locally over all 240 public/eval tasks. Low score, but a guaranteed
   clean submission.

### 3.2 Phase B — attach the model for a real TTT score

1. Notebook editor → **Add Input** → **Models** tab → search for an **Apache-2.0**
   Qwen2.5 size — **Qwen2.5-Coder-7B-Instruct** (recommended) or **1.5B**; avoid the
   3B (Qwen Research License, not prize-eligible) → **Add**. Kaggle mounts it read-only;
   copy the exact path shown (varies by variant/framework, typically
   `/kaggle/input/<owner>/<model>/transformers/<variant>/<version>`).
2. Edit the **config cell** only:
   ```python
   MODEL_DS = '/kaggle/input/qwen2.5/transformers/3b-instruct/1'   # <- your mount path
   ADAPTER_DS = None   # or '/kaggle/input/<your-adapter-dataset>' if you built one
   ```
3. **Internet = Off** (must stay off — the mounted Model mocks the download, and
   competition rules disallow live internet during scoring). **Run All**.
4. Expect: `mode=KAGGLE ... model_path=/kaggle/input/...`, `ensemble: DSL + heuristics
   + TTT(LoRA)`, then a real elapsed-time-per-task figure and `schema_problems=0`.
5. *(Optional)* If you trained a base adapter with `src/arc/train/finetune.py` on the
   synthetic corpus, stage it as its own Kaggle Dataset (same
   `kaggle datasets create -p .` flow as §4) and point `ADAPTER_DS` at its mount —
   TTT then adapts further on top of that base fine-tune per task.

Cell 4 always runs `assert result['problems'] == [], result['problems']` — a non-empty
`problems` list fails the cell loudly rather than silently shipping a bad file.

### 3.3 Multi-GPU full runs (Rung 3 — L4×4)

By default a single model replica is sharded across every visible GPU
(`HFModel(device_map="auto")`), which wastes 3 of 4 L4 cards on a 7B model that
fits comfortably on ONE. `num_workers=4` instead spawns 4 processes, each
pinned to its own GPU (`CUDA_VISIBLE_DEVICES`) with its own full model replica,
each solving its own shard of tasks concurrently — turning ~150s/task
(1 model, 4 idle-ish GPUs) into ~600s of effective parallel throughput.

Edit the **run cell** for a full-run push:

```python
from kaggle_submit import main
result = main(
    model_path=MODEL_DS, adapter_path=ADAPTER_DS,
    per_task_budget_s=575.0, use_ttt=True, max_tasks=MAX_TASKS,
    ttt_config={"max_steps": 112},   # retuned up from the single-replica default (64)
    num_workers=4,                    # L4x4: one full model replica per GPU
)
```

Retuned knobs for the 4-worker path (each worker now owns a whole GPU instead
of sharing one sharded model, so it can afford a heavier per-task budget):
- `per_task_budget_s`: **~550-600s** (up from the single-process 150s default)
  — each worker still solves its shard sequentially, so the parallel wall-clock
  win comes from 4x concurrency, not from a smaller per-task budget.
- `TTTConfig.max_steps`: **96-128** via `ttt_config={"max_steps": ...}` (up from
  the default 64) — more LoRA fitting steps per task now fit in the budget.
- `DEFAULT_LLM_KWARGS["num_augs"]`: **12-16** (up from 8) for a richer
  transduction/TTT corpus per task.

**Kill-switch:** `num_workers=0` (the default) always keeps the sequential
single-process path — if the parallel path misbehaves on Kaggle (a worker
hangs, GPU pinning doesn't take, etc.), drop `num_workers` from the run cell
(or set it to `0`/`1`) and rerun; nothing else about the call changes. The
parallel path shares every other invariant with the sequential one: a complete
fallback submission is written before any worker starts, results checkpoint to
disk every `checkpoint_every_s` (default 60s), and the global time watchdog
(`total_budget_s`, same `TOTAL_RUNTIME_BUDGET_S` as the sequential path)
terminates any still-running workers with a safety grace period before the
12h hard cap.

---

## 4. Path B — code-as-dataset (alternative)

Use this if you'd rather manage `src/`/`scripts/` as a versioned Kaggle Dataset instead
of a self-contained notebook (e.g. iterating on code without rebuilding the notebook
each time). Full detail in `scripts/stage_kaggle_assets.md`; summary:

1. **Model dataset** (on a machine WITH internet):
   ```bash
   pip install -U "huggingface_hub[cli]"
   hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./qwen2.5-3b-instruct
   pip install kaggle
   cd qwen2.5-3b-instruct
   kaggle datasets init -p .    # edit dataset-metadata.json: title/id, keep private
   kaggle datasets create -p . --dir-mode zip
   ```
2. **Code dataset** (from repo root): `kaggle datasets create -p .` to package `src/` +
   `scripts/`. Requires a Kaggle API token (`~/.kaggle/kaggle.json`) — do not commit it.
3. In the notebook (`notebooks/submission.ipynb`), edit the config cell:
   ```python
   CODE_DS = '/kaggle/input/<arc-code-dataset>'
   MODEL_DS = '/kaggle/input/<qwen25-3b-instruct>'
   ADAPTER_DS = None
   ```
   then `sys.path.append(f'{CODE_DS}/src')` / `.../scripts` and
   `from kaggle_submit import main`.
4. Extra wheels (only if a specific `peft`/`unsloth`/`vllm` version is required beyond
   what Kaggle images ship): stage as a dataset, `pip install --no-index
   --find-links /kaggle/input/<wheels>` offline.
5. Same Phase A/B split applies (`MODEL_DS`/`ADAPTER_DS` = None vs set) and the same
   pre-submit gate in §5.

Trade-off: Path A has zero external dependencies (no API token, single upload); Path B
lets you version code independently of the notebook but needs the Kaggle CLI/token and
a rebuild-the-dataset step on every code change.

---

## 5. Pre-submit gate (REQUIRED — go/no-go)

Per the `deployment-readiness` skill: treat every unchecked item below as a **blocker**.
Do not proceed to §6 until all are green, on a run with **Internet = Off** (the actual
scoring condition — never gate on an internet-enabled run).

| # | Check | How to verify | Pass condition |
|---|-------|----------------|-----------------|
| 1 | Model loads from mount (Phase B only) | Log line `mode=KAGGLE ... model_path=/kaggle/input/...` and `ensemble: DSL + heuristics + TTT(LoRA)` with no exception | No traceback; ensemble line printed |
| 2 | Submission written & schema-clean | Log line `submission: /kaggle/working/submission.json  schema_problems=0` | `schema_problems=0` (the run cell also hard-asserts this) |
| 2b | TTT actually ran (not silently degraded) | No repeated `solver llm_ttt failed …` lines in the log | zero TTT failures — if you see them, check §7 torchao |
| 3 | Wall-clock comfortably under 12 h | Log line `elapsed: NN.N min  (N.Ns/task)` | Total run (incl. model load) < ~10–10.5 h observed, matching the 11 h internal budget in `config.py` with margin left over |
| 4 | Accelerator = GPU | Notebook Settings panel | GPU L4×4 (or T4×2) selected, not CPU |
| 5 | Internet = Off | Notebook Settings panel | Toggled Off — this is the actual scoring condition; an internet-enabled dry run does not validate offline behavior |
| 6 | Output file present at the exact path | `/kaggle/working/submission.json` exists post-run | File present, non-empty, one entry per test `task_id` |

If any row fails: fix per §7 Troubleshooting, re-run, re-check all 6 rows — don't
selectively re-check only the row you patched (a timeout fix can regress the schema
row, etc.).

---

## 6. The gated final action — Submit to Competition

This is the **only** step this runbook prepares but never executes.

1. Confirm §5 is fully green on the notebook version you intend to submit.
2. Kaggle notebook page → **Save Version** → **Save & Run All (Commit)** → wait for
   the committed run to finish (this is the run that gets scored, not your interactive
   session — re-verify §5 against *this* run's logs/output).
3. Competition page → **Submit Predictions** → select the committed notebook version's
   `submission.json` output → **Submit to Competition**.

```
# This is the exact gated action — a manual UI click, not a CLI command, because
# Kaggle competition submission has no offline/dry-run CLI equivalent that spends
# the daily quota safely. Do this yourself when ready:
#
#   Kaggle UI -> Competition -> Submit Predictions -> pick the committed version -> Submit
```

**Do not** run this from an agent, script, or automation. It consumes the **1
submission/day** quota and is irreversible for that slot.

**Rollback note:** a bad submission is not destructive — Kaggle lets you pick which of
your submitted versions is your **final scored entry** at competition close. If a
submission scores poorly or errors, the fix is simply: don't select it: fix the
notebook, wait for tomorrow's quota (or use it if unused today), resubmit, and select
the best-scoring version before the deadline. There is no data or environment to roll
back — only a choice of which submission "counts."

---

## 7. Troubleshooting

**TTT silently disabled — repeated `solver llm_ttt failed …` + `ImportError: incompatible
version of torchao` (run finishes in seconds, DSL-only score)**
- Cause: the Kaggle image ships a `torchao` (e.g. 0.10.0) older than the installed
  `peft` requires (>0.16), so `get_peft_model()` raises on every task. The pipeline
  catches it and degrades to the DSL/heuristic ensemble — you get a valid but
  weak submission and the log fills with the failure (which is how you spot it).
- Fix: the notebook's **env-prep cell** now removes the incompatible `torchao`
  automatically (we don't use it). If you're on an older notebook, add a first cell:
  ```python
  import subprocess, sys
  subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
  ```
  then **Factory reset → Run All** (the reset clears the already-imported broken `peft`).
  `pip uninstall` needs no internet.

**`FileNotFoundError` on `load_challenges` (path points at a local/developer folder)**
- Cause: the **competition data isn't attached to the notebook**, so nothing mounts under
  `/kaggle/input/arc-prize-2026-arc-agi-2/`. The entrypoint now fails fast with an
  actionable "Competition data not found …" message that lists what *is* mounted.
- Fix: **Add Input → Competitions → "ARC Prize 2026 - ARC-AGI-2"** (a competition notebook
  created from the competition page attaches it automatically; an *imported* notebook does
  not). Then Run All again.
- `config.py` auto-detects the data folder wherever Kaggle mounts it (it scans
  `/kaggle/input/*` for the challenges file), and on Kaggle it never falls back to a
  developer-machine path. If your data lives somewhere non-standard, set `ARC_DATA_DIR`
  in the config cell to the folder containing `arc-agi_test_challenges.json`.

**OOM (CUDA out of memory)**
- Confirm accelerator is actually GPU (not CPU) and the right count (L4×4).
- Qwen2.5-3B in bf16 needs ~6 GB — with 4×24 GB there's large headroom, so OOM usually
  means TTT batch size or `max_seq_len` is too aggressive. Lower `TTTConfig.batch_size`
  (default 2) or `max_seq_len` (default 2048) in `src/arc/solvers/llm/ttt.py`, or drop
  `DEFAULT_LLM_KWARGS["max_new_tokens"]` (default 1024) in `scripts/kaggle_submit.py`.
- Falling back to a smaller/quantized model is the last resort — re-verify Apache/MIT/
  CC-BY licensing (§2) before swapping.

**Timeouts (run not finishing inside 12 h, or a handful of tasks eating the budget)**
- Lower `per_task_budget_s` passed to `main()` (default 150.0s / the run cell's
  `per_task_budget_s=150.0`) — this is a *per-task* cap enforced by the shared time
  watchdog in `arc/pipeline.py`, so lowering it trades score for safety margin.
- Lower `num_augs` in `DEFAULT_LLM_KWARGS` (default 8) or
  `DEFAULT_TTT_DATA_KWARGS["num_augs"]` (default 16) in `scripts/kaggle_submit.py` —
  fewer augmentations per task means less TTT corpus / fewer transduction samples but
  faster per-task wall-clock.
- Lower `TTTConfig.max_steps` (default 64) for a lighter per-task LoRA fit.
- The global budget (`TOTAL_RUNTIME_BUDGET_S = 11h` in `src/arc/config.py`) already
  reserves 1h under the 12h cap — if you're still breaching it, the per-task or
  per-run knobs above are the levers, not the global constant.

**Missing or empty `submission.json`**
- Should not happen: `kaggle_submit.main()` calls `_pre_write_fallback()` *before* any
  parsing or model work, writing a schema-valid all-1x1-zero-grid submission for every
  task it can find in the raw challenges JSON. Even a hard crash mid-run, an OOM, or a
  malformed challenges file leaves a scoreable file at `/kaggle/working/submission.json`.
- If the file is still missing: the challenges path itself is wrong (check
  `cfg.data_dir` in the printed `mode=... data_dir=...` log line) or the competition
  data Input wasn't attached — re-check Add Input on the notebook.
- If `schema_problems` is non-zero on the *final* (non-fallback) submission: read the
  first 5 problems printed (`kaggle_submit.py` logs up to 5) — they name the exact
  task_id/attempt at fault; cross-check against `validate_submission()` in
  `src/arc/io/submission.py` for the specific rule violated (missing task_id, wrong
  test-output count, missing attempt key, or invalid grid shape).

**License / eligibility concerns**
- Only ship models/adapters/data under Apache-2.0/MIT/CC-BY-style licenses to stay
  prize-eligible. Qwen2.5-3B-Instruct (Apache-2.0) is pre-cleared. Any adapter you
  train yourself (`src/arc/train/finetune.py`) is your own output and fine. If you
  swap the base model, re-verify its license before Phase B.

---

## 8. Budget math

- **240 hidden test tasks / 11 h internal budget** (`TOTAL_RUNTIME_BUDGET_S` in
  `src/arc/config.py`, deliberately 1h under Kaggle's 12h hard cap) **≈ 165 s/task**
  of headroom.
- **Default `per_task_budget_s=150.0`** (`DEFAULT_PER_TASK_BUDGET_S` in `config.py`,
  and the literal passed in both notebooks' run cell) leaves **~15s/task** of slack
  inside the 165s ceiling.
- The remaining ~1h margin (12h cap − 11h internal budget) absorbs model load time,
  Kaggle Dataset/Model mount I/O, tokenizer/weights initialization, and the final
  submission-file write/validation pass — none of which are charged against individual
  task budgets.
- If §5 row 3 shows you're tracking close to 11h+model-load, that's your earliest
  signal to trim `per_task_budget_s`, `num_augs`, or `TTTConfig.max_steps` (§7) rather
  than waiting for an actual 12h timeout.

---

## Summary of what this runbook governs

| Artifact | Role |
|---|---|
| `notebooks/submission_selfcontained.ipynb` | Path A entrypoint — generated, never hand-edited |
| `scripts/build_kaggle_notebook.py` | Regenerates the above from `src/` + `scripts/kaggle_submit.py` |
| `scripts/kaggle_submit.py` | Shared `main()` entrypoint both notebook paths call |
| `notebooks/submission.ipynb` | Path B entrypoint (code-as-dataset) |
| `scripts/stage_kaggle_assets.md` | Path B asset-staging detail |
| `src/arc/config.py` | Env-seam: KAGGLE mode auto-detect, global time budgets |
| `/kaggle/working/submission.json` | The single deploy artifact that gets scored |
