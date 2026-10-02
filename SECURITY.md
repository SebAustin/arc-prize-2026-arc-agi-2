# SECURITY.md — ARC-AGI-2 Solver

Security / supply-chain / robustness review of an **offline ML batch pipeline** that runs
inside the Kaggle scoring kernel (Internet = Off, no user auth, no network-facing surface).
The threat model is deliberately calibrated for that environment: the classic web/OWASP
attack surface is largely **not applicable**. The real risks here are (a) untrusted
deserialization when loading model/adapter weights, (b) supply-chain reproducibility of
unpinned GPU dependencies, and (c) robustness — a malformed input crashing the run and
forfeiting the submission.

Review scope: `src/arc/**`, `scripts/**`, `notebooks/**`, `pyproject.toml`. READ-ONLY audit;
no code was modified.

---

## System decomposition

**Entry points**
- `scripts/kaggle_submit.py :: main()` — the offline inference entrypoint (called by the notebook).
- `scripts/run_local_smoke.py`, `scripts/build_synthetic.py` — local dev CLIs (not in the scored kernel).
- `scripts/build_kaggle_notebook.py` — build-time generator for the self-contained notebook.
- `notebooks/submission_selfcontained.ipynb` — the Run-All artifact executed on Kaggle.

**Trust boundaries**
1. **Model / LoRA-adapter files** (Kaggle Dataset mount) → `HFModel` / `PeftModel` /
   `finetune()` via `transformers`/`peft` `from_pretrained`. This is the one genuine
   code-execution trust boundary: loading a pickled checkpoint can execute arbitrary code.
2. **Competition task JSON** (`/kaggle/input/...`) → `io/loader.py`. Data boundary; not a
   code boundary, but a robustness boundary.
3. **Model-generated text** → `serialize/prompt.parse_completion` → `tokenizer.str_to_grid`.
   Parsed as digits-only grid text; never `eval`'d. Not a code boundary.

**Data stores**: read-only competition JSON (in); `/kaggle/working/submission.json` (out).
No DB, no cache, no network egress.

**Sensitive data**: none. No PII, no credentials, no secrets. Submission JSON is grids of 0–9.

---

## STRIDE-lite (honest N/A calls for an offline batch job)

| Category | Applies? | Assessment |
|---|---|---|
| **S**poofing | N/A | No auth, no identities, no sessions. Kaggle owns identity outside the kernel. |
| **T**ampering | **Partial** | No network in-transit tampering. Real vector = a **tampered/poisoned model or adapter file** deserialized via `from_pretrained` (F1). Input-JSON tampering is a robustness issue (F4), not a privilege one. |
| **R**epudiation | N/A | Single-run batch job, no multi-user actions to audit. |
| **I**nformation disclosure | N/A | No secrets or PII in scope; nothing to leak. Confirmed no hardcoded credentials. |
| **D**enial of service | **Partial (self-inflicted)** | No external attacker. But a malformed input file (F4) or an unbounded loop can crash/hang the kernel and forfeit the 12h run. Pipeline already has a time watchdog + up-front fallback submission — good. |
| **E**levation of privilege | **Partial** | Only meaningful as arbitrary-code-execution via untrusted checkpoint deserialization (F1). No OS/user privilege model in the kernel. |
| Prompt injection / tool scope (LLM) | **Low** | The LLM is a pure transducer: it emits grid text that is parsed digits-only and never executed. No tools, no function-calling, no shell. Task content can only influence *which grid* is predicted — it cannot escape into code. |

**Scans run (all clean):** no `eval`/`exec`/`compile`/`__import__`; no `pickle`/`marshal`/`dill`;
no `subprocess`/`os.system`/`shell=True`; no `requests`/`urllib`/`socket` (offline confirmed);
no `trust_remote_code`; no hardcoded Kaggle/HF/API tokens or key material in any tracked file;
`.venv`, `kaggle.json`, `.env`, `artifacts/` all untracked / gitignored.

---

## Findings

| # | Severity | File:line | Issue | Remediation |
|---|---|---|---|---|
| F1 | Medium | `src/arc/solvers/llm/model.py:94-105`; `train/finetune.py:85-88` | Model/adapter loaded via `from_pretrained` with no `safetensors`-only enforcement. A `.bin`/`.pt` checkpoint triggers `torch.load` (pickle) → arbitrary code execution if the staged Dataset is tampered with. | Pass `use_safetensors=True` and prefer safetensors-format model + adapter datasets; the ecosystem defaults to `weights_only=True` but do not rely on that — pin to safetensors artifacts you staged yourself. |
| F2 | Medium | `pyproject.toml:15-20` (`[gpu]`) | `torch`, `transformers`, `peft`, `accelerate` are fully **unpinned**. Non-reproducible reruns and supply-chain exposure to a newly-published malicious/broken version at stage time. | Pin exact versions (`torch==2.x.y`, etc.) and record hashes; a competition that must rerun deterministically should freeze the GPU stack. |
| F3 | Medium→**Fix now** | `scripts/kaggle_submit.py:74` (loader path) + `src/arc/io/loader.py:41-59` | `load_challenges` uses hard `body["train"]`/`body["test"]`/`raw["input"]` indexing with **no schema validation**. A malformed/renamed key in `test_challenges.json` raises `KeyError`/`TypeError` **before** the up-front fallback submission is written (that happens later inside `run_pipeline`), so a single bad task forfeits the entire run with an empty `/kaggle/working`. | Validate each task's shape and skip/repair bad entries; and/or write the empty fallback submission *before* `load_challenges` so a valid file always exists. See "Recommended safe fix" below. |
| F4 | Low | `src/arc/solvers/dsl/primitives.py:66-82` | `scale_program`/`tile_program` build outputs via `np.repeat`/`np.tile` with factors inferred from train pairs; a pathological ratio (e.g. huge output) could allocate a large array. Bounded in practice by 30×30 competition grids and per-task time budget. | Cap output dimensions at `MAX_DIM` before allocating; reject programs whose predicted shape exceeds the grid limit. |
| F5 | Info | `scripts/build_kaggle_notebook.py:22-52`; `notebooks/submission_selfcontained.ipynb` | The self-contained notebook base64-embeds the whole `arc` package + entrypoint and writes it to `/kaggle/working` at runtime. **Verified safe**: 37 embedded files, **zero drift** vs source, no paths outside the `src/` + `scripts/kaggle_submit.py` allowlist, byte-reproducible from the generator. The embedded code is first-party only; nothing untrusted is executed. | No fix required. Keep regenerating from source (`build_kaggle_notebook.py`) rather than hand-editing the notebook, so the embed stays inspectable and reproducible. |

**Not findings (explicitly cleared):** no secrets committed; no prompt-injection→code path
(model output is digits-only, never executed); no `eval`/`exec`/`subprocess`; no network I/O;
`trust_remote_code` unused (safe default); submission layer already fails safe with a complete
1×1 fallback and validates its own schema (`io/submission.py`).

---

## Recommended safe fix (F3) — highest value for an offline batch job

Two independent, low-risk mitigations:
1. In `scripts/kaggle_submit.py :: main()`, write the empty fallback submission from
   `empty_predictions(tasks)` immediately, and guard `load_challenges` so a per-task parse
   failure is skipped (logged) rather than aborting the whole run.
2. In `io/loader.py`, validate `body`/`raw` keys and grid shapes before constructing `Task`,
   dropping or repairing malformed tasks instead of raising.

Result: a corrupt input file degrades to a lower score, never to an empty submission.

---

## Severity counts

| Severity | Count |
|---|---|
| Critical | 0 |
| High | 0 |
| Medium | 3 (F1, F2, F3) |
| Low | 1 (F4) |
| Info | 1 (F5) |

Overall posture: **good for an offline competition pipeline.** No web attack surface, no
secrets, no code-injection, no network egress. The residual risk is concentrated in
untrusted-checkpoint deserialization (F1), dependency pinning (F2), and input-robustness
(F3) — all standard ML-supply-chain hygiene rather than application vulnerabilities.
