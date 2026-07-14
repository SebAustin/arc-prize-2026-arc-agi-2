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

import logging
import time

from ...augment.task_aug import distinct_augs
from ...io.loader import Task
from ..base import Candidates, Solver
from .infer import generate_candidates, generate_candidates_batch, generate_candidates_dfs
from .model import LanguageModel
from .select import rank_by_votes, score_candidates, score_candidates_poe

_log = logging.getLogger(__name__)

# Candidate selection modes, in increasing strength (see select.py).
SELECTION_MODES = ("votes", "likelihood", "poe")
# Decode modes: one greedy completion per prompt, or a DFS token-tree search
# returning every completion above a cumulative-probability threshold
# (see dfs_decode.py). Greedy is the default — DFS is opt-in per config.
DECODE_MODES = ("greedy", "dfs")


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
        selection: str = "votes",
        poe_augs: int = 4,
        decode: str = "greedy",
        dfs_eps: float = 0.12,
        dfs_max_expansions: int = 3072,
    ):
        self.model = model
        self.num_augs = num_augs
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.keep_zero = keep_zero
        self.max_candidates = max_candidates
        self.aug_seed = aug_seed
        if selection not in SELECTION_MODES:
            raise ValueError(f"selection must be one of {SELECTION_MODES}")
        # `use_likelihood` predates `selection`; keep accepting it as an alias.
        if use_likelihood and selection == "votes":
            selection = "likelihood"
        self.selection = selection
        self.poe_augs = poe_augs
        if decode not in DECODE_MODES:
            raise ValueError(f"decode must be one of {DECODE_MODES}")
        if decode == "dfs" and (num_samples != 1 or temperature > 0.0):
            raise ValueError("decode='dfs' requires num_samples=1, temperature=0.0")
        self.decode = decode
        self.dfs_eps = dfs_eps
        self.dfs_max_expansions = dfs_max_expansions
        self.last_telemetry: dict = {}

    def solve(self, task: Task, budget_s: float) -> Candidates:
        deadline = time.monotonic() + budget_s
        augs = distinct_augs(self.num_augs, seed=self.aug_seed, keep_zero=self.keep_zero)
        per_test: Candidates = []
        augs_completed = 0
        decode_s = 0.0
        score_s = 0.0
        dfs_nodes = 0
        dfs_leaves = 0
        # DFS needs a step-capable model; fall back to greedy otherwise so a
        # config typo can never zero a Kaggle run (kill-switch direction).
        use_dfs = self.decode == "dfs" and hasattr(self.model, "as_step_model")
        if self.decode == "dfs" and not use_dfs:
            _log.warning(
                "decode='dfs' but %s has no as_step_model(); using greedy",
                type(self.model).__name__,
            )
        # One batched generate covers all augs when the model supports it and
        # the config is greedy single-sample (the production default). Checked
        # AFTER dfs: a DFS config is greedy-shaped and would match this gate.
        use_batch = (
            not use_dfs
            and self.num_samples == 1
            and self.temperature == 0.0
            and hasattr(self.model, "generate_batch")
        )

        for i in range(len(task.test)):
            weighted: list[tuple] = []
            if use_dfs:
                for j, aug in enumerate(augs):
                    # Same discipline as the sequential loop below: the first
                    # (identity) aug always runs; later augs stop past deadline.
                    if j > 0 and time.monotonic() > deadline:
                        break
                    atask = aug.apply_task(task)
                    t_dec = time.monotonic()
                    pairs, stats = generate_candidates_dfs(
                        self.model,
                        atask.train,
                        atask.test[i].input,
                        eps=self.dfs_eps,
                        max_new_tokens=self.max_new_tokens,
                        max_expansions=self.dfs_max_expansions,
                        # Enough leaves that voting sees real alternatives, but
                        # bounded so one aug can't flood the tree budget.
                        max_candidates=4 * self.max_candidates,
                        deadline_s=deadline,
                    )
                    decode_s += time.monotonic() - t_dec
                    augs_completed += 1
                    dfs_nodes += stats.expansions
                    dfs_leaves += stats.leaves
                    for grid, weight in pairs:
                        weighted.append((aug.invert_grid(grid), weight))
            elif use_batch:
                atasks = [aug.apply_task(task) for aug in augs]
                t_dec = time.monotonic()
                grid_lists = generate_candidates_batch(
                    self.model,
                    [(at.train, at.test[i].input) for at in atasks],
                    max_new_tokens=self.max_new_tokens,
                    max_time_s=max(0.0, deadline - time.monotonic()),
                )
                decode_s += time.monotonic() - t_dec
                augs_completed += len(augs)
                for aug, grids in zip(augs, grid_lists, strict=True):
                    for grid in grids:
                        weighted.append((aug.invert_grid(grid), 1.0))
            else:
                for j, aug in enumerate(augs):
                    # Always run the first (identity) augmentation; stop adding
                    # more once the per-task budget is spent.
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
            if voted and self.selection == "likelihood":
                # Re-rank the voted candidates by the model's own confidence
                # under the canonical (un-augmented) prompt.
                t_sc = time.monotonic()
                scored = score_candidates(
                    self.model, task.train, task.test[i].input, voted
                )
                score_s += time.monotonic() - t_sc
                voted = [g for g, _ in scored]
            elif voted and self.selection == "poe":
                # Product-of-experts: judge every candidate under several
                # augmented framings; wrong-but-plausible candidates rarely
                # survive all of them.
                t_sc = time.monotonic()
                scored = score_candidates_poe(
                    self.model,
                    task.train,
                    task.test[i].input,
                    voted,
                    augs[: self.poe_augs],
                    deadline_s=deadline,
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
            # DFS work accounting (0 under greedy) — the instrument behind the
            # "~2x greedy cost" claim on the Kaggle canary.
            "dfs_nodes": dfs_nodes,
            "dfs_leaves": dfs_leaves,
        }
        return per_test
