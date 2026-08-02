# ARC Prize 2026 — ARC-AGI-2 Solver

A hybrid solver for the [ARC Prize 2026 / ARC-AGI-2](https://kaggle.com/competitions/arc-prize-2026-arc-agi-2)
Kaggle competition: solve *novel* abstract-reasoning grid tasks, 2 attempts per
test output, exact-match scoring.

## Approach

ARC-AGI-2 is memorization-proof, so the system **adapts at inference time**. We
combine two complementary solvers under a voting ensemble:

1. **DSL micro-solver** — searches compositions of grid primitives that reproduce
   every demonstration pair, then applies the simplest survivors. Millisecond
   cost, high precision, always-on safety net.
2. **Test-time-training LLM** *(M2+)* — per-task LoRA fine-tuning of a small
   (~3–7B) model on augmented demonstrations, then transduction with
   augmentation-based candidate selection. This is the main score lever, mirroring
   the open-source 2025 winners (NVARC, ARChitects).

A time watchdog guarantees a complete, valid `submission.json` within the 12-hour
Kaggle budget even on timeout or OOM.

## Layout

```
src/arc/
  io/         data loading, grid type, submission build/validate
  serialize/  grid <-> text, prompt assembly
  augment/    D4 symmetry, colour permutation, task augmentation (all invertible)
  solvers/    Solver interface; cheap heuristics; DSL search; (LLM in M2+)
  eval/       metric + evaluation harness
  pipeline.py ensemble voting + time-budgeted run
  config.py   the only environment-aware module (SMOKE vs KAGGLE)
```

## Two environments, one code path

- **Smoke mode (local CPU):** full pipeline over public eval with the DSL +
  heuristic ensemble. No GPU required.
- **Kaggle mode (L4x4 GPU):** identical code with the real model + TTT; weights
  and wheels pre-staged as offline Kaggle Datasets.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check src/ scripts/ tests/    # lint gate (also enforced in CI)
pytest                              # unit + property tests (149)
python scripts/run_local_smoke.py --limit 40   # end-to-end smoke + score
```

Lint (`ruff`) and tests run automatically on push/PR via
[GitHub Actions](.github/workflows/ci.yml) across Python 3.10 and 3.13.

## Status

- [x] **M0** — foundation, harness, DSL + heuristic ensemble, smoke test
- [x] **M1 (code)** — LLM transduction solver + augmentation selection + CPU mock,
      all unit-tested; Kaggle entrypoint (`scripts/kaggle_submit.py`) + asset
      staging guide ready.
- [x] **M2 (code)** — test-time training: per-task corpus builder (leave-one-out ×
      augmentation), `TTTRunner` protocol with CPU mock + GPU `LoraTTTRunner`, and
      `TTTSolver`, all unit-tested. Entrypoint defaults to TTT.
- [x] **M3 (code)** — 9 procedural task generators (`synth/`) + corpus builder/persistence;
      GPU base-fine-tune script (`train/finetune.py`). Generators verified: balanced
      concepts, ~3k tasks/s, 67% DSL solve-rate = exactly the in-vocabulary fraction.
- [x] **M4 (code)** — likelihood-weighted selection (`select.score_candidates`), shared
      per-task time budgeting across solvers, base-FT adapter loading, final
      `notebooks/submission.ipynb`.

- [x] **Review & hardening (agency pass)** — multi-perspective review (`CODEBASE.md`,
      `SECURITY.md`), then robustness fixes on the scoring path: TTT now charges adaptation
      time to its own budget and always restores the shared model on failure; a schema-valid
      fallback submission is written *before* parsing so a malformed rerun file can't forfeit
      the run; validated challenge loading, time-based checkpointing, structured
      solver-failure logging, a DSL output-size cap, safetensors-only model loading, and a
      `ruff` + CI gate. See `ACCEPTANCE.md`.

**The full pipeline (M0–M4) is built and CPU-verified — 149 passing tests.** What remains is
GPU execution on Kaggle (not buildable locally — the Mac has no CUDA).

### Easiest path: the self-contained notebook (no code dataset, no API token)

`notebooks/submission_selfcontained.ipynb` embeds the entire `arc` package (base64), so the
kernel needs nothing but itself. Regenerate with `python scripts/build_kaggle_notebook.py`.

- **Phase A (zero setup):** new competition notebook → upload the self-contained notebook →
  leave `MODEL_DS = None` → Internet Off → Run All. Produces a valid submission from the DSL
  ensemble (validated end-to-end locally over all 240 tasks).
- **Phase B (real score):** Add Input → **Models** → attach a Kaggle-hosted `Qwen2.5-3B-Instruct`
  (no upload needed) → set `MODEL_DS` to its mount path → Run All → first real TTT score.
- *(optional)* base fine-tune on the synthetic corpus → adapter (`train/finetune.py`), attach
  it, set `ADAPTER_DS`.

(The dataset-based `notebooks/submission.ipynb` + `scripts/stage_kaggle_assets.md` remain as an
alternative if you prefer uploading code as a dataset.)

Local CPU checks (no GPU):
`python scripts/run_local_smoke.py --use-mock-ttt` ·
`python scripts/build_synthetic.py --n 2000 --sanity 200`

See the build plan for milestone detail. Licensed Apache-2.0 (open-source per
competition rules).
