"""Candidate selection: aggregate weighted grid votes into a ranked list.

In M1 the weight is a simple count across augmentations (a grid that survives
many independent augmentations is more trustworthy). M2 will add model
log-likelihood as an additional weight signal — same aggregation, richer weights.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

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
