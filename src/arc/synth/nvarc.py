"""Convert NVARC synthetic puzzles into our fine-tuning corpus format.

The ARC Prize 2025 winners published curated synthetic puzzles as the public
dataset `sorokin/nvarc-synthetic-puzzles`: per-task JSON files, each a flat list
of `{"input", "output"}` demonstration pairs that all share one transformation
rule. This module turns each file into leave-one-out `(prompt, completion)`
`TrainExample`s — the exact shape our base-fine-tune / TTT pipeline already
consumes — so we can retrain the adapter on the winners' higher-quality corpus
instead of (or alongside) our procedurally-generated 50k.

Each demonstration pair becomes a query once, with up to `max_support` of the
other pairs as few-shot context (capped so prompts stay near the inference-time
budget rather than ballooning to all ~13 pairs).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..io.grid import from_lists, is_valid_grid
from ..io.loader import Pair
from ..serialize.prompt import build_prompt
from ..serialize.tokenizer import grid_to_str
from ..solvers.llm.ttt_data import COMPLETION_PREFIX, TrainExample


def load_nvarc_pairs(path: str | Path) -> list[Pair]:
    """Load one NVARC puzzle file into validated demonstration pairs.

    Silently drops any malformed pair (non-rectangular / out-of-range) rather
    than failing the whole file — a single bad grid shouldn't lose a task."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs: list[Pair] = []
    for item in raw:
        try:
            grid_in = from_lists(item["input"])
            grid_out = from_lists(item["output"])
        except (KeyError, TypeError, ValueError):
            continue
        if is_valid_grid(grid_in) and is_valid_grid(grid_out):
            pairs.append(Pair(input=grid_in, output=grid_out))
    return pairs


def nvarc_examples(
    pairs: list[Pair],
    *,
    max_support: int = 4,
    max_queries: int | None = 6,
    seed: int = 0,
) -> list[TrainExample]:
    """Leave-one-out `(prompt, completion)` examples from one puzzle's pairs.

    Each of up to `max_queries` pairs is predicted from up to `max_support` of the
    remaining pairs. Deterministic in `seed`. Returns [] if there are <2 pairs
    (no support possible)."""
    if len(pairs) < 2:
        return []
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    if max_queries is not None and max_queries < len(indices):
        indices = sorted(rng.sample(indices, max_queries))

    examples: list[TrainExample] = []
    for qi in indices:
        others = [p for j, p in enumerate(pairs) if j != qi]
        if len(others) > max_support:
            others = rng.sample(others, max_support)
        query = pairs[qi]
        prompt = build_prompt(others, query.input)
        completion = COMPLETION_PREFIX + grid_to_str(query.output)
        examples.append(TrainExample(prompt=prompt, completion=completion))
    return examples


def nvarc_file_to_examples(path: str | Path, **kwargs) -> list[TrainExample]:
    """Convenience: load one file and convert it in a single call."""
    return nvarc_examples(load_nvarc_pairs(path), **kwargs)
