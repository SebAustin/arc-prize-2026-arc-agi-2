"""Candidate selection: aggregate weighted grid votes into a ranked list.

Three selectors, in increasing strength:
  * `rank_by_votes` — augmentation-consensus voting (a grid that survives many
    independent augmentations is more trustworthy).
  * `score_candidates` — re-rank by model log-likelihood under the single
    canonical prompt.
  * `score_candidates_poe` — product-of-experts: sum log-probs across several
    augmented prompts. A wrong candidate that happens to look plausible under
    one framing rarely stays plausible under all of them; this was one of the
    top selection levers in the 2025 winners' pipelines.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable, Sequence

from ...augment.task_aug import TaskAug
from ...io.grid import Grid
from ...io.loader import Pair
from ...serialize.prompt import build_prompt
from ...serialize.tokenizer import grid_to_str
from .model import LanguageModel
from .ttt_data import COMPLETION_PREFIX


def score_candidates(
    model: LanguageModel,
    train: Sequence[Pair],
    test_input: Grid,
    candidates: Sequence[Grid],
) -> list[tuple[Grid, float]]:
    """Rank candidates by the model's log-likelihood under the canonical prompt.

    Each candidate grid is scored as the completion the model would assign after
    the clean (un-augmented) few-shot prompt — a confidence signal independent of
    how the candidate was generated. Best (highest log-prob) first.
    """
    prompt = build_prompt(train, test_input)
    scored = [
        (grid, model.score(prompt, COMPLETION_PREFIX + grid_to_str(grid)))
        for grid in candidates
    ]
    scored.sort(key=lambda kv: (-kv[1], _size(kv[0]), kv[0]))
    return scored


def score_candidates_poe(
    model: LanguageModel,
    train: Sequence[Pair],
    test_input: Grid,
    candidates: Sequence[Grid],
    augs: Sequence[TaskAug],
    deadline_s: float | None = None,
) -> list[tuple[Grid, float]]:
    """Rank candidates by summed log-prob across augmented prompts (PoE).

    For each augmentation the train pairs, test input, AND candidate are mapped
    into that frame, so the model judges a self-consistent view. Augmentations
    are processed whole (every candidate scored under an aug before the deadline
    is checked) so partial-time results stay comparable. Best first.
    """
    totals = [0.0] * len(candidates)
    for k, aug in enumerate(augs):
        # Always score under the first frame; stop adding frames past deadline.
        if k > 0 and deadline_s is not None and time.monotonic() >= deadline_s:
            break
        atrain = [aug.apply_pair(p) for p in train]
        prompt = build_prompt(atrain, aug.apply_grid(test_input))
        for i, grid in enumerate(candidates):
            completion = COMPLETION_PREFIX + grid_to_str(aug.apply_grid(grid))
            totals[i] += model.score_sum(prompt, completion)
    scored = list(zip(candidates, totals, strict=True))
    scored.sort(key=lambda kv: (-kv[1], _size(kv[0]), kv[0]))
    return scored


def rank_by_votes(weighted: Iterable[tuple[Grid, float]]) -> list[tuple[Grid, float]]:
    """Sum weights per distinct grid and return them best-first.

    Ties break toward the smaller grid (a mild Occam prior), then lexicographically
    for determinism.
    """
    agg: dict[Grid, float] = defaultdict(float)
    for grid, weight in weighted:
        agg[grid] += weight
    return sorted(agg.items(), key=lambda kv: (-kv[1], _size(kv[0]), kv[0]))


def _size(grid: Grid) -> int:
    return len(grid) * (len(grid[0]) if grid else 0)
