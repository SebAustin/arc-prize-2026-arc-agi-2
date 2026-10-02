# Acceptance — Agency Review & Hardening Pass

**Input:** "review the projects and find improvements if needed."
**Mode:** existing-project enhancement, in place, on branch `enhance/agency-review` (off `main` @ `ad50032`).
**Constraint honored:** no regressions — the pre-existing suite stayed green throughout; every
change is additive or behavior-preserving. GPU code paths (`torch`/`transformers`/`peft`) are
not executable locally, so those edits are additive and marked unverifiable-locally below.

## Baseline vs. final

| | Baseline | Final |
|---|---|---|
| `pytest` | 130 passed | **149 passed** |
| `ruff check src/ scripts/ tests/` | no config; 5 F401 | **All checks passed!** |
| CI | none | GitHub Actions (ruff + pytest, Py 3.10 & 3.13) |

Verbatim final gate:

```
$ ruff check src/ scripts/ tests/
All checks passed!

$ pytest
149 passed in 0.95s
```

## Review deliverables

- `CODEBASE.md` — architecture map, data flow, two-environment mechanism, ranked extension/risk points (`codebase-analyst`).
- `SECURITY.md` — STRIDE-lite for an offline ML pipeline + findings table (`security-auditor`).
- Independent correctness review (`python-reviewer`) — **verified sound**: voting/top-2,
  D4 + colour augmentation invertibility, submission schema, log-prob alignment. No correctness
  bugs in the scoring math. Two HIGH robustness findings in the TTT path (both fixed below).

## Improvements delivered — pass/fail per success criterion

| # | Improvement | Impact | Status | Evidence |
|---|-------------|--------|--------|----------|
| 1 | TTT charges corpus-build + `adapt()` time to the inner solver's budget (no more per-task overrun) | ROBUSTNESS/SCORE | ✅ PASS | `test_ttt_budget.py::test_adaptation_time_is_charged_to_inner_budget`, `::test_remaining_budget_is_clamped_at_zero` |
| 2 | `adapt()` moved inside `try/finally` → `reset()` always runs; `LoraTTTRunner` unloads a partial adapter on failure | ROBUSTNESS/SCORE | ✅ PASS | `test_ttt_budget.py::test_reset_runs_when_adaptation_raises` (runner-level); adapter-unload guard is GPU-only ⚠️ |
| 3 | Entrypoint writes a schema-valid fallback submission **before** parsing/model work; graceful degrade on a malformed file | ROBUSTNESS (zero-score guard) | ✅ PASS | `test_robustness.py::test_entrypoint_writes_valid_submission`, `::test_entrypoint_keeps_fallback_on_malformed_file` |
| 4 | Challenge loader validates schema and raises a clear, task-named `MalformedTaskError` (was a bare `KeyError`); `skip_invalid` option | ROBUSTNESS/CORRECTNESS | ✅ PASS | `test_loader_validation.py` (6 cases) |
| 5 | Structured logging on solver failure (silent ensemble degradation is now visible) | ROBUSTNESS | ✅ PASS | `test_robustness.py::test_failing_solver_is_logged_and_skipped` |
| 6 | Warn once when a solver has no registered vote weight | CORRECTNESS/MAINT | ✅ PASS | `test_robustness.py::test_unregistered_solver_name_warns` |
| 7 | Time-based checkpointing (every N tasks **or** N seconds) — a slow run no longer risks losing ~1h of answers on a crash | ROBUSTNESS | ✅ PASS | `test_robustness.py::test_time_based_checkpoint_flushes` |
| 8 | DSL `scale`/`tile` refuse to allocate beyond the 30×30 ARC cap | ROBUSTNESS | ✅ PASS | `test_robustness.py::test_{scale,tile}_program_*` (3 cases) |
| 9 | GPU generation accepts a wall-clock `max_time_s`, wired from the per-task deadline | ROBUSTNESS | ⚠️ ADDITIVE (GPU-only) | plumbed through mock protocol; effect only on `HFModel` |
| 10 | `use_safetensors=True` on model/adapter load + safe adapter serialization (refuse pickle `.bin` RCE surface) | SECURITY | ⚠️ ADDITIVE (GPU-only) | `SECURITY.md` F1 |
| 11 | `gpu` extras given a compatibility floor; freeze to exact `==` from the staged wheel set before submit | SECURITY/REPRO | ✅ PARTIAL | `pyproject.toml`; exact hash-pin deferred to stage time (needs Kaggle wheel set) |
| 12 | `ruff` config + fixed 5 F401 and all lint findings; CI runs lint + tests on push/PR | TOOLING | ✅ PASS | gate output above; `.github/workflows/ci.yml` |

## Verification method

- Local venv: Python 3.13.9, numpy + pytest + ruff (no GPU stack — matches project design).
- Gate run as the last step after the final edit, on every created/modified file.
- Auto-fixed lint changes (import order, `zip(strict=)`, redundant modes, unused imports,
  `typing`→`collections.abc`) individually reviewed as behavior-preserving; confirmed by the
  suite staying green.

## Deferred / next (documented, not done)

- **Exact dependency pinning with hashes** (F2) — requires the real Kaggle GPU wheel set;
  inventing exact versions blind risks breaking the staged environment. Floor bounds added now.
- **GPU-path execution coverage** — `HFModel.generate`, `LoraTTTRunner.adapt`,
  `train/finetune.py`, and the `max_time_s`/safetensors/adapter-unload edits are only reachable
  with real torch/peft. Fixes #2 (adapter unload), #9, #10 are unverified until a real Kaggle
  run; they are additive and low-risk by construction.
- **TTT ↔ base-finetune dedup** — `LoraTTTRunner` and `train/finetune.py` share tokenize/collate/
  loop code; a shared helper was left out of this pass to avoid refactoring untested GPU code.
- **Salvage-and-merge on partial malformation** — currently a malformed file keeps the complete
  pre-written fallback and degrades gracefully; running the well-formed subset while preserving
  the malformed tasks' fallback is a possible future refinement.
