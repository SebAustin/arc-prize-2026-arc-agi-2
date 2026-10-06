# ARC-AGI-2 Solver

The project's shared vocabulary: an entry for ARC Prize 2026 (ARC-AGI-2) that
solves abstract-reasoning grid tasks with an LLM, test-time training, and a
self-driving Kaggle loop. This file is a glossary only — no implementation
details. Mechanics live in code comments; decisions live in `docs/adr/`.

## Solving method

**Test-Time Training (TTT)**:
Per-task fine-tuning of the base model, on that one task's demonstration pairs
(augmented), performed at inference time rather than ahead of it.
_Avoid_: online training, fine-tune-at-test.

**Base model**:
The frozen LLM the solver loads beneath any adapter and TTT (default: the
NVARC-SFT'd Qwen3-4B). The thing TTT adapts; not itself task-specific.
_Avoid_: foundation model, LLM.

**Adapter**:
The LoRA weight-delta trained on top of a frozen base model, staged as a Kaggle
dataset and mounted at eval. An adapter is base-specific — one trained on a 7B
base cannot load on a 4B base.
_Avoid_: LoRA (reserve for the technique), weights, checkpoint.

**Augmentation voting**:
Re-expressing a task under invertible transforms (D4 symmetry + colour
permutation), predicting in each frame, inverting predictions back to the
canonical frame, and voting for the consensus answer.
_Avoid_: ensembling, test-time augmentation.

**DSL micro-solver**:
The rule-based, non-LLM solver path; the fallback that runs when the model
cannot load and part of the candidate mix otherwise.
_Avoid_: heuristic solver, symbolic solver.

**Rung**:
A numbered capability tier of the solver stack (Rung 0 = DSL … Rung 6 = DFS
decoding), each a self-contained increment over the one below.
_Avoid_: stage, level, phase, milestone.

## Autopilot

**Autopilot**:
The self-driving daily loop that advances the project one step at a time with no
human in the loop: reconcile results, poll in-flight work, gate, submit, launch
the next experiment.
_Avoid_: cron, scheduler, bot, daemon.

**Tick**:
One step of the autopilot for a given day — deterministic given the prior state
and an injected date.
_Avoid_: run, cycle, iteration, pass.

**Candidate**:
A trained-adapter configuration awaiting evaluation on the canary. Its eval
either promotes it (gated) or discards it.
_Avoid_: model, entry, submission.

**Gate** / **gated candidate**:
The promotion test (an eval must beat the best-so-far by at least one solved
task) and a candidate that has passed it and is queued for competition submission.
_Avoid_: threshold, filter, cutoff.

**Backlog**:
The ordered, cheap-gate-first list of experiments the autopilot works through
when otherwise idle; a cursor marks the next one.
_Avoid_: queue, todo, pipeline.

**Canary**:
The held-out public-evaluation slice (120 tasks) that every eval scores, used as
the generalization signal — distinct from the competition's hidden test set.
_Avoid_: validation set, test set, holdout, dev set.

**Inflight**:
The single Kaggle kernel the autopilot is currently waiting on (a train, eval, or
submit run); at most one per lane.
_Avoid_: running job, active task, current run.

**Exploration lane**:
The second, narrow lane that spends an otherwise-idle day's submission slot on an
untried variant — free upside, since the competition ranks an entrant's best
submission.
_Avoid_: side channel, secondary slot.

**Partial credit**:
Sub-exact-match signals (per-cell accuracy, shape-correct rate) reported
alongside the competition's top-2 exact-match, to give a gradient below the
all-zero exact-match floor.
_Avoid_: soft score, fuzzy match, similarity.

## Environments

**SMOKE**:
The local environment (Mac, no CUDA) used for fast, tiny-slice correctness checks.
_Avoid_: dev, local, test-env.

**KAGGLE**:
The Kaggle-kernel runtime (GPU, competition data mounted) where real evals and
submissions execute.
_Avoid_: prod, remote, cloud.

## Kaggle environment (landmine vocabulary)

**Kernel**:
A Kaggle notebook run — the unit the autopilot pushes, polls, and submits.
_Avoid_: notebook (reserve for the file), script, job.

**Mount convention**:
The fixed rule for where Kaggle exposes an attached asset: datasets mount flat at
`/kaggle/input/<slug-basename>` (owner stripped); models nest under
`/kaggle/input/models/<owner>/<model>/<framework-lowercase>/<variation>/<version>`.
_Avoid_: input path, attach path.

**Kernel-id burn**:
The state of a kernel id that a create-push left half-created (rejected on the
session cap), after which every later push to that id fails; recovered by rotating
to the next sequential id.
_Avoid_: dead kernel, broken kernel.

**Session cap**:
Kaggle's account-wide limit of two concurrent batch-GPU sessions. A push beyond it
is a capacity rejection (retry), never a failure of the work.
_Avoid_: slot limit, concurrency limit.

**Weekly quota**:
Kaggle's account-wide weekly GPU-hour allowance. When spent, every GPU push is
rejected until it resets — a capacity condition (retry), never an item failure.
_Avoid_: budget (reserve for the autopilot's own self-imposed weekly GPU-hour cap).

**12h cap**:
Kaggle's hard per-kernel wall-clock limit; a run that exceeds it is cancelled with
zero output, so heavy evals must bound their work to fit.
_Avoid_: timeout, deadline.
