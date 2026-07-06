"""Run a language model on one (train, test_input) and parse grid candidates."""

from __future__ import annotations

from collections.abc import Sequence

from ...io.grid import Grid
from ...io.loader import Pair
from ...serialize.prompt import build_prompt, parse_completion
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
