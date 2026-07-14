"""Run a language model on one (train, test_input) and parse grid candidates."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ...io.grid import Grid
from ...io.loader import Pair
from ...serialize.prompt import build_prompt, parse_completion
from .dfs_decode import DfsStats, dfs_decode, top_k_for_eps
from .model import LanguageModel


def generate_candidates(
    model: LanguageModel,
    train: Sequence[Pair],
    test_input: Grid,
    *,
    num_samples: int = 1,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    max_time_s: float | None = None,
) -> list[Grid]:
    """Prompt the model and return the valid grids parsed from its completions."""
    prompt = build_prompt(train, test_input)
    completions = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        temperature=temperature,
        max_time_s=max_time_s,
    )
    grids: list[Grid] = []
    for completion in completions:
        grid = parse_completion(completion)
        if grid is not None:
            grids.append(grid)
    return grids


def generate_candidates_batch(
    model: LanguageModel,
    items: Sequence[tuple[Sequence[Pair], Grid]],
    *,
    max_new_tokens: int = 1024,
    max_time_s: float | None = None,
) -> list[list[Grid]]:
    """Greedy candidates for many (train, test_input) items in ONE batched call.

    Batching the per-augmentation decodes recovers a 3-5x throughput factor over
    the sequential loop; result list is index-aligned with `items` (an item whose
    completion doesn't parse yields an empty list).
    """
    prompts = [build_prompt(train, test_input) for train, test_input in items]
    completions = model.generate_batch(
        prompts, max_new_tokens=max_new_tokens, max_time_s=max_time_s
    )
    out: list[list[Grid]] = []
    for completion in completions:
        grid = parse_completion(completion)
        out.append([grid] if grid is not None else [])
    return out


def generate_candidates_dfs(
    model,
    train: Sequence[Pair],
    test_input: Grid,
    *,
    eps: float,
    max_new_tokens: int = 1024,
    max_expansions: int = 3072,
    max_candidates: int = 16,
    deadline_s: float | None = None,
) -> tuple[list[tuple[Grid, float]], DfsStats]:
    """DFS-decode one (train, test_input): many grids, each with its
    probability mass as the vote weight.

    Leaves that parse to the SAME grid pool their mass (`exp(cum_logprob)` —
    positive, so `rank_by_votes`' descending sort stays correct; raw negative
    log-probs would invert it). `model` must expose `as_step_model()`
    (HFModel does); callers gate on that.
    """
    prompt = build_prompt(train, test_input)
    step_model = model.as_step_model(top_k=top_k_for_eps(eps))
    cands, stats = dfs_decode(
        step_model,
        prompt,
        eps=eps,
        max_new_tokens=max_new_tokens,
        max_expansions=max_expansions,
        max_candidates=max_candidates,
        deadline_s=deadline_s,
    )
    weights: dict[Grid, float] = {}
    for cand in cands:
        grid = parse_completion(cand.text)
        if grid is None:
            continue
        weights[grid] = weights.get(grid, 0.0) + math.exp(cand.cum_logprob)
    pairs = sorted(weights.items(), key=lambda kv: -kv[1])
    return pairs, stats
