"""Build a per-task fine-tuning corpus for test-time training.

A task carries only a handful of demonstration pairs — too few to fine-tune on
directly. We expand them two ways:

  * leave-one-out: each demo pair becomes a (support -> query) prediction
    problem, so N pairs yield N self-supervised examples;
  * augmentation: every view is replicated under invertible D4 x colour
    augmentations, turning N pairs into hundreds of training examples.

Each example is a (prompt, completion) text pair where the prompt ends with the
open `Output:` tag and the completion is the target grid. Pure string assembly —
no model required — so the whole builder is unit-testable on CPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ...augment.task_aug import distinct_augs, leave_one_out
from ...io.loader import Pair, Task
from ...serialize.prompt import build_prompt
from ...serialize.tokenizer import grid_to_str

# The completion starts on the line after "Output:" (mirrors the demo format).
COMPLETION_PREFIX = "\n"


@dataclass(frozen=True)
class TrainExample:
    """One supervised fine-tuning example for TTT."""

    prompt: str
    completion: str


def _example(support: tuple[Pair, ...], query: Pair) -> TrainExample:
    prompt = build_prompt(support, query.input)
    completion = COMPLETION_PREFIX + grid_to_str(query.output)
    return TrainExample(prompt=prompt, completion=completion)


def _views(task: Task) -> list[tuple[tuple[Pair, ...], Pair]]:
    """Leave-one-out views; fall back to self-views if <2 demo pairs."""
    views = leave_one_out(task)
    if views:
        return views
    return [(task.train, p) for p in task.train]


def build_ttt_examples(
    task: Task,
    num_augs: int = 16,
    *,
    seed: int = 0,
    keep_zero: bool = False,
    max_examples: int | None = 500,
) -> list[TrainExample]:
    """Assemble the TTT corpus for a task: leave-one-out views over many
    augmentations, deduped and capped."""
    augs = distinct_augs(num_augs, seed=seed, keep_zero=keep_zero)
    examples: list[TrainExample] = []
    seen: set[tuple[str, str]] = set()
    for aug in augs:
        atask = aug.apply_task(task)
        for support, query in _views(atask):
            if query.output is None:
                continue
            ex = _example(support, query)
            key = (ex.prompt, ex.completion)
            if key in seen:
                continue
            seen.add(key)
            examples.append(ex)

    if max_examples is not None and len(examples) > max_examples:
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_examples]
    return examples
