# Research: Kaggle GPU weekly-quota mechanics & NVARC method details

- **Date:** 2026-10-02
- **Author:** research task (for ARC-AGI-2 go/no-go on NVARC 4B + TTT)
- **Scope:** Two questions — (1) how Kaggle's weekly GPU quota resets and whether it is
  queryable via API/CLI; (2) the exact NVARC (ARC Prize 2025, 1st place) model + TTT recipe
  and whether a 4B model + per-task LoRA TTT fits a single 16 GB T4.
- **Source discipline:** Claims are tied to primary sources — the official `Kaggle/kaggle-api`
  source, Kaggle staff product announcements, the NVARC authors' own repo/paper/notebook, and
  the Kaggle model identifier. Where a Kaggle web page is JavaScript-gated and could not be
  fetched verbatim, that is flagged explicitly as an open item.
- **Note:** No compute-spending or kernel-push Kaggle commands were run. The only Kaggle calls
  were read-only (`kaggle kernels pull`, inspecting the locally installed `kaggle` package).

---

## Question 1 — Kaggle GPU weekly-quota reset mechanics

### Bottom line

- The weekly GPU quota is a **fixed weekly reset, Saturday ~00:00 UTC** — **not** a rolling
  7-day window. The word "floating" that Kaggle uses refers to the **amount** of hours (which
  varies week to week by demand — this is why this account sees **45.00 h** rather than the
  commonly-cited 30 h), **not** to the reset timing, which Kaggle says resets "weekly on
  Saturday morning (midnight UTC) the same way it always has."
- **There IS an API/CLI way to query remaining quota and the reset time** — contrary to the
  common "web-UI-only" belief. The official `kaggle` package ships a top-level **`kaggle quota`**
  command (and `KaggleApi.quota_view()` for the Python API) that returns GPU/TPU
  **used / remaining / total hours** plus a **`refreshAt` / `quota_refresh_time`** timestamp.
  **This command is already present in the version installed in this project's `.venv`
  (kaggle 2.2.2).**
- Practical upshot: the autopilot can stop guessing the reset time — it can read the exact
  per-account `refreshAt` and remaining hours directly from `kaggle quota` before deciding to
  push a GPU kernel.

### Details & evidence

**Quota amount is variable ("floating"); reset cadence is fixed (Saturday 00:00 UTC).**
Kaggle's staff announcement "[Notebooks update] More GPU hours — Introducing a 'floating' GPU
quota" states the quota amount floats with demand (it can exceed 30 h) while "the quota resets
weekly on Saturday morning (midnight UTC) the same way it always has," and that in the two
months after launch quotas varied between ~36 and ~43 h. This directly explains the
"Maximum weekly GPU quota of 45.00 hours reached" message in `SUBMISSION_LOG.md`.
- https://www.kaggle.com/product-feedback/173129
- Corroborating Q&A threads (fixed Saturday 00:00 UTC reset; variable amount):
  https://www.kaggle.com/general/135810 ,
  https://www.kaggle.com/general/208138 ,
  https://www.kaggle.com/docs/efficient-gpu-usage

> ⚠️ Verification caveat: Kaggle forum/announcement pages are JavaScript-rendered and could not
> be fetched verbatim via automated fetch; the "Saturday 00:00 UTC" wording above comes from
> search-index snippets of the staff post plus multiple corroborating forum answers, and from
> the 2020 announcement (old). Treat the **exact weekday/time as "very likely Saturday 00:00
> UTC" rather than freshly re-confirmed verbatim.** This uncertainty is moot in practice because
> the API returns the exact per-account refresh timestamp — see below.

**API/CLI quota query — authoritative and already installed.**
The official `Kaggle/kaggle-api` exposes a `quota` command:
- `cli.py` registers it: `parser_quota = subparsers.add_parser("quota", ...)` with help text
  **"Show the current user's weekly GPU and TPU accelerator quota"**, dispatching to
  `api.quota_view_cli`.
- `kaggle_api_extended.py`: `quota_view()` calls
  `kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())`
  and returns `ApiGetAcceleratorQuotaStatisticsResponse` with **`quota_refresh_time`,
  `gpu_quota`, `tpu_quota`** fields; each quota has **`time_used`** and
  **`total_time_allowed`** (timedeltas).
- `quota_view_cli()` prints a table with columns **`resource, used, remaining, total, refreshAt`**
  (remaining = `total_time_allowed − time_used`; `refreshAt` = `quota_refresh_time.isoformat()`).
- Source: https://github.com/Kaggle/kaggle-api (files `src/kaggle/cli.py`,
  `src/kaggle/api/kaggle_api_extended.py`)

**Confirmed present in this project's environment.** The locally installed `kaggle` package
(`.venv`, version **2.2.2**) contains `KaggleApi.quota_view` / `quota_view_cli` and the
`parse_quota` CLI registration with the same implementation (verified by reading
`.venv/lib/python3.13/site-packages/kaggle/cli.py` and `.../api/kaggle_api_extended.py`).
So `kaggle quota` (and `kaggle quota --csv` / `-v`) should work with this account's credentials
today. Latest PyPI release is 2.2.4; the command is in both.

**What `quota_refresh_time` means for the fixed-vs-rolling question.** The API returns a
**single** next-refresh timestamp per account. That single value is the operational source of
truth: if it reads as the upcoming Saturday 00:00 UTC it confirms the fixed weekly reset; if it
slid forward with usage it would indicate a rolling window. Recommend the autopilot read
`refreshAt` directly rather than hard-coding a weekday.

**Also noted (not the account quota):** `max_gpu_runtime_minutes` in the kaggle-api source is a
**per-competition code-competition rerun setting**, not the weekly account quota — do not
confuse the two. The widely-documented per-session cap is ~12 h (ARC Prize 2025 rerun limit is
12 h; see Q2).

### Open questions (Q1)

- Exact reset weekday/time not re-confirmed verbatim from a live, current Kaggle page (JS-gated).
  Mitigation: `kaggle quota` → `refreshAt` gives the authoritative per-account value — verify
  empirically from the account rather than trusting the 2020 "Saturday 00:00 UTC" statement.
- Whether `quota_refresh_time` is populated for every account tier (it may be `None`, in which
  case the CLI prints the row with an empty `refreshAt`). Worth a one-time read to confirm.

---

## Question 2 — NVARC (ARC Prize 2025, 1st place) method details

### Bottom line

- **A single 16 GB T4 (~14.56 GiB usable) cannot run NVARC's released 4B + per-task TTT recipe
  as-is.** NVARC's own paper states the solution had to run within **"Kaggle compute limit: 12
  hours with 4 L4 GPUs"**, and their released submission notebook is **hard-wired to 4 GPUs**
  (`mp.spawn(..., nprocs=4)`, one worker per GPU). Each worker runs the 4B model in **bfloat16
  with NO 4-bit quantization and NO gradient checkpointing**, LoRA **rank 256** on all
  projections **plus `embed_tokens` and `lm_head`**, seq length **8192**, `adamw_torch`
  optimizer — a per-worker footprint that overflows a single T4. This matches this project's
  repeated `torch.OutOfMemoryError` on a 14.56 GiB GPU in `SUBMISSION_LOG.md`.
- **Implication for the go/no-go eval:** running NVARC's recipe unchanged on a single T4 will
  OOM (as observed). To complete a go/no-go on T4 hardware you must either (a) use the 4×L4
  accelerator NVARC actually targeted, or (b) materially shrink the recipe (re-enable 4-bit
  quantization, drop `embed_tokens`/`lm_head` from LoRA targets, lower rank, shorten seq,
  enable gradient checkpointing) — which diverges from the winning configuration and changes
  what the eval is actually testing.
- **Model identity confirmed:** `sorokin/qwen3_4b_grids15_sft139` is the **base SFT checkpoint**
  — a **full fine-tune of Qwen3-4B-Thinking-2507** (extended with 16 grid tokens) on the
  **~3.2 M-sample** augmented ARC corpus. It is **not** the TTT adapter; TTT (LoRA) happens per
  task at inference time inside the submission notebook.

### 2.1 Model identity, base, size, SFT corpus

- **Base model:** **Qwen3-4B-Thinking-2507** (4B params), with a tiny custom tokenizer of **16
  tokens** (10 digits, newline, `user`/`assistant` grid markers, 2 special tokens, 1 padding) —
  hence the `-16t` variant name. Confirmed in the SFT config `policy.model_name:
  "/models/Qwen3-4B-Thinking-2507"` / run script `"/models/Qwen3-4B-Thinking-2507-16t"`, in the
  paper (Table 2 lists "Qwen3-4B-Thinking-2507", public LB 29.72), and in the notebook which
  loads `/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1`.
  - https://github.com/1ytic/NVARC (files `ARChitects/sft_mg.yaml`, `ARChitects/run_sft_4b.sh`,
    `README.md`), paper `nvarc_2025.pdf` §3.1 & Table 2.
- **What `qwen3_4b_grids15_sft139` is:** the experiment name `EXP_NAME="qwen3_4b_grids15_sft139"`
  is defined in `ARChitects/run_sft_4b.sh` for the **supervised fine-tuning** run. `grids15` =
  the `data/grids_v15/*` dataset; `sft139` is an experiment tag. So the Kaggle model is the
  **post-SFT base weights**, published as a `transformers/bfloat16` variant.
  - Kaggle model: https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139
- **SFT corpus size:** final training set = **3,255,481 augmented samples** from **104,539
  unique puzzles** (paper Table 1), each sample up to 7 input/output pairs. Built from NVARC
  synthetic puzzles (**103,253** synthetic; "103k") + augmented sets (**3.2 M**) + real sets
  (ARC-AGI-2 train/eval, RE-ARC, MINI-ARC, ConceptARC). Datasets:
  `sorokin/nvarc-synthetic-puzzles` (103k), `sorokin/nvarc-augmented-puzzles` (3.2 M),
  `sorokin/nvarc-artifacts-puzzles`.
  - Paper `nvarc_2025.pdf` §2–§3.1, Table 1; repo `README.md`.
- **SFT compute (base checkpoint, done OFFLINE on a cluster — not on Kaggle):** full fine-tune
  of the 4B on **4 nodes × 8× H100 for 27 h**, Megatron backend, tensor-parallel + context-
  parallel, sequence packing to **256,000 tokens** from 256 examples; lr 1e-4, 1 epoch,
  ~12,716 steps, global batch 256, bfloat16.
  - Paper §3.1; `ARChitects/sft_mg.yaml` + `ARChitects/run_sft_4b.sh`.

### 2.2 Test-time-training (TTT) recipe — per-task LoRA

From the paper (§3.2) and, authoritatively, the released submission notebook
`sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue` (pulled read-only). TTT is applied **per
puzzle independently**:

**LoRA (`FastLanguageModel.get_peft_model`):**
- `r = 256`, `lora_alpha = 32`, `lora_dropout = 0.0`, `use_rslora = True`
- `target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj,
  embed_tokens, lm_head]` — **note it adapts embeddings + LM head**, which is memory-heavy
- `use_gradient_checkpointing = False`, `load_in_4bit = False` (paper: "removed gradient
  checkpointing … removed 4-bit quantization too")

**Training args (`UnslothTrainingArguments`):**
- `learning_rate = 5e-5`, `lr_scheduler_type = "cosine"`, `warmup_ratio = 0.1`,
  `warmup_steps = 0`
- `num_train_epochs = 1`, `per_device_train_batch_size = 1`, `gradient_accumulation_steps = 1`
  (so #steps ≈ number of augmented training rows for that task; "batch4" in the notebook name
  refers to the DFS/decoding batching, not the TTT micro-batch)
- `optim = "adamw_torch"`, `weight_decay = 0.0`, `bf16 = True`, `fp16 = False`,
  `gradient_checkpointing = False`
- `max_seq_length = 8192`
- Framework: **Unsloth + Flash Attention 2**, base loaded from the `qwen3_4b_grids15_sft139`
  bf16 checkpoint.
- Source: notebook cells (verified locally); paper §3.2 "LoRA test-time fine-tuning for each
  puzzle independently with r=256 and alpha=32 … bfloat16 … Flash Attention 2 with Unsloth."
  - https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue
  - Paper `nvarc_2025.pdf` §3.2; repo `README.md`.

**Augmentation strategy:**
- Candidate **re-scoring uses 8 augmentations** (`m = 8`), with the *same* 8 augmentations
  applied to every candidate so scores are comparable (paper §3.4; notebook splits the 8 into
  two batches of 4 in `calc_scores`). The augmentation family is the ARChitects-2024 scheme
  (D4 geometric symmetries + color permutations) referenced via [Franzen et al. 2025].
  For the **TRM** component the paper spells it out explicitly ("geometric rotation and
  transpose, as well as color permutations with 0/black fixed").
  - Paper §3.4 & §4 (`nvarc_2025.pdf`).

**Candidate generation + selection/voting:**
- Decoding: **batched Depth-First Search (DFS)** over token sequences (`turbo_dfs` /
  `inference_turbo_dfs`), scaling ~linearly with batch size.
- Selection (`run_selection_algo`, `n_guesses = 2` → pass@2): score each candidate by
  **(count of DFS occurrences) + geometric-mean of log-probs across the 8 augmentations**
  (`scoreagg` in paper §3.4); pick the top-scored candidate per test input.
- Source: notebook (`arc_decoder.ArcDecoder(..., n_guesses=2)`, `run_selection_algo`);
  paper §3.3–§3.4.

### 2.3 Hardware fit — does 4B + TTT fit a single 16 GB T4?

**No — not as released.** Evidence:
- **Paper (authors' own):** "the solution we built had to run within Kaggle compute limit:
  **12 hours with 4 L4 GPUs** for solving 240 tasks. As a result we had to use small models
  (4B at most)…" (`nvarc_2025.pdf` §1, p.2). 4× L4 = 4 × 24 GB = **96 GB** aggregate.
- **Released notebook is hard-wired to 4 GPUs:** `mp.spawn(local_worker, args=(...), nprocs=4)`
  with each worker pinning `os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)` (ranks 0–3) and a
  file-based barrier (`/kaggle/worker{rank}`). Each of the 4 workers independently loads the
  bf16 4B model and runs TTT + DFS + rescoring on **its own** L4. There is no model sharding —
  it's data-parallel over tasks, one full 4B model per GPU.
- **Per-worker memory is sized for a 24 GB L4, not a 14.56 GiB T4.** bf16 4B weights alone are
  ~8 GB; add LoRA r=256 **including `embed_tokens`+`lm_head`** (large, because vocab×hidden),
  `adamw_torch` fp32 optimizer moments over all trainable params, activations at seq 8192, and a
  KV cache for DFS decoding — with **no 4-bit quantization and no gradient checkpointing** this
  comfortably exceeds ~14.5 GB. This is exactly the failure signature in `SUBMISSION_LOG.md`
  ("GPU 0 has a total capacity of 14.56 GiB … torch.OutOfMemoryError").
- ARC Prize 2025 is a code competition with a **12 h rerun limit**; the winners used the full
  12 h across 4 L4 for the 4B config (paper Table 2: Qwen3-4B-Thinking-2507 → 12 h → 29.72 LB).

**Consequences for the go/no-go eval on a single T4:**
1. As-is, the eval will OOM on one T4 (confirmed empirically in the log). A clean go/no-go of
   the *unmodified* NVARC recipe requires the **4×L4** accelerator Kaggle provides for this
   competition.
2. If the eval must run on a single T4, the recipe has to be shrunk, which deviates from the
   winning setup: candidate levers — re-enable `load_in_4bit=True` (QLoRA), remove
   `embed_tokens`/`lm_head` from `target_modules`, reduce `r` (256 → 32–64), cut
   `max_seq_length` (8192 → e.g. 2048–4096), enable gradient checkpointing, run `nprocs=1`.
   Each lever trades memory for fidelity/score, so a "fits-on-T4" result would not be directly
   comparable to NVARC's 4×L4 numbers.

### Open questions (Q2)

- The paper's released artifacts are described as open (ARC Prize requires winners to open-
  source to claim the prize), but an explicit LICENSE file for the Kaggle model/datasets was not
  confirmed here — consistent with the existing note in `ASSUMPTIONS.md` (2026-07-19). Verify
  the license text on the model/dataset pages before redistribution.
- A 2B variant (`Qwen3-VL-2B-Instruct` LLM part) scored 22.22 LB in ~6 h (paper Table 2) and is
  the natural smaller-footprint fallback if single-GPU memory is the hard constraint — still a
  4×L4 run in the paper, but far lighter per task; worth checking whether a 2B + reduced-TTT
  config fits a single T4 for a cheaper go/no-go.

---

## Primary sources

- Kaggle API source (quota command): https://github.com/Kaggle/kaggle-api
  (`src/kaggle/cli.py`, `src/kaggle/api/kaggle_api_extended.py`)
- Kaggle "floating" GPU quota staff announcement: https://www.kaggle.com/product-feedback/173129
- Kaggle GPU reset Q&A (corroborating): https://www.kaggle.com/general/135810 ,
  https://www.kaggle.com/general/208138
- Kaggle efficient GPU usage docs: https://www.kaggle.com/docs/efficient-gpu-usage
- NVARC repo: https://github.com/1ytic/NVARC
- NVARC paper: `nvarc_2025.pdf` in the repo above (Sorokin & Puget, Nov 2025)
- NVARC submission notebook: https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue
- NVARC base model: https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139
- NVARC datasets: https://www.kaggle.com/datasets/sorokin/nvarc-synthetic-puzzles ,
  https://www.kaggle.com/datasets/sorokin/nvarc-augmented-puzzles ,
  https://www.kaggle.com/datasets/sorokin/nvarc-artifacts-puzzles
- ARC Prize 2025 results/analysis: https://arcprize.org/blog/arc-prize-2025-results-analysis
- ARC Prize 2025 competition: https://www.kaggle.com/competitions/arc-prize-2025
