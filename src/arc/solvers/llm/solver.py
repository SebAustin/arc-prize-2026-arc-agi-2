"""LLMSolver — transduction with augmentation-based candidate selection.

For each test input, the task is re-expressed under several invertible
augmentations (D4 symmetry + colour permutation). The model predicts in each
augmented frame; predictions are inverted back to the canonical frame and voted.
Augmenting both diversifies the model's inputs and provides a consensus signal
that is far more reliable than a single greedy decode.

This is the M1 baseline (no test-time training); M2 plugs a per-task fine-tuned
model into the same solver via the `model` argument.
"""

from __future__ import annotations

import time

from ...augment.task_aug import distinct_augs
from ...io.loader import Task
from ..base import Candidates, Solver
from .infer import generate_candidates
from .model import LanguageModel
from .select import rank_by_votes, score_candidates


class LLMSolver(Solver):
    name = "llm"

    def __init__(
        self,
        model: LanguageModel,
        num_augs: int = 4,
        num_samples: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        keep_zero: bool = False,
        max_candidates: int = 4,
        aug_seed: int = 0,
        use_likelihood: bool = False,
    ):
        self.model = model
        self.num_augs = num_augs
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.keep_zero = keep_zero
        self.max_candidates = max_candidates
        self.aug_seed = aug_seed
        self.use_likelihood = use_likelihood
        self.last_telemetry: dict = {}

    def solve(self, task: Task, budget_s: float) -> Candidates:
        deadline = time.monotonic() + budget_s
        augs = distinct_augs(self.num_augs, seed=self.aug_seed, keep_zero=self.keep_zero)
        per_test: Candidates = []
        augs_completed = 0
        decode_s = 0.0
        score_s = 0.0

        for i in range(len(task.test)):
            weighted: list[tuple] = []
            for j, aug in enumerate(augs):
                # Always run the first (identity) augmentation; stop adding more
                # once the per-task budget is spent.
                if j > 0 and time.monotonic() > deadline:
                    break
                atask = aug.apply_task(task)
                t_dec = time.monotonic()
                grids = generate_candidates(
                    self.model,
                    atask.train,
                    atask.test[i].input,
                    num_samples=self.num_samples,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    # Cap this decode at the time still left in the per-task
                    # budget so one slow generation can't overrun it.
                    max_time_s=max(0.0, deadline - time.monotonic()),
                )
                decode_s += time.monotonic() - t_dec
                augs_completed += 1
                for grid in grids:
                    weighted.append((aug.invert_grid(grid), 1.0))

            ranked = rank_by_votes(weighted)
            voted = [g for g, _ in ranked]
            if self.use_likelihood and voted:
                # Re-rank the voted candidates by the model's own confidence
                # under the canonical (un-augmented) prompt.
                t_sc = time.monotonic()
                scored = score_candidates(
                    self.model, task.train, task.test[i].input, voted
                )
                score_s += time.monotonic() - t_sc
                voted = [g for g, _ in scored]
            per_test.append(voted[: self.max_candidates])

        # Budget-split instrument: read this to see whether decode is being
        # starved (augs_completed << num_tests * num_augs) before tuning knobs.
        self.last_telemetry = {
            "augs_completed": augs_completed,
            "augs_planned": len(augs) * len(task.test),
            "decode_s": round(decode_s, 2),
            "score_s": round(score_s, 2),
        }
        return per_test
