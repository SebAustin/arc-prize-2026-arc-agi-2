"""Test-time-training corpus builder tests (pure CPU)."""

from __future__ import annotations

from arc.serialize.prompt import parse_completion
from arc.solvers.llm.ttt_data import build_ttt_examples


def _task3(grid_factory, task_factory):
    # 3 demonstration pairs (output = input recoloured trivially) + 1 test input.
    pairs = []
    for s in range(3):
        g = grid_factory(3, 3, ncolors=5, seed=s)
        out = tuple(tuple((c + 1) % 5 for c in row) for row in g)
        pairs.append((g, out))
    return task_factory(pairs, [grid_factory(3, 3, ncolors=5, seed=9)])


def test_examples_prompt_and_completion_format(grid_factory, task_factory):
    task = _task3(grid_factory, task_factory)
    examples = build_ttt_examples(task, num_augs=1)
    assert examples
    for ex in examples:
        assert ex.prompt.rstrip().endswith("Output:")  # open for the target
        # completion must parse back to a valid grid (the held-out query output)
        assert parse_completion(ex.completion) is not None


def test_leave_one_out_count_with_single_aug(grid_factory, task_factory):
    task = _task3(grid_factory, task_factory)
    examples = build_ttt_examples(task, num_augs=1)
    # 3 demo pairs -> 3 leave-one-out views under the identity augmentation.
    assert len(examples) == 3


def test_augmentation_increases_corpus(grid_factory, task_factory):
    task = _task3(grid_factory, task_factory)
    few = build_ttt_examples(task, num_augs=2)
    many = build_ttt_examples(task, num_augs=8)
    assert len(many) > len(few)


def test_max_examples_cap(grid_factory, task_factory):
    task = _task3(grid_factory, task_factory)
    capped = build_ttt_examples(task, num_augs=16, max_examples=5)
    assert len(capped) == 5


def test_examples_are_deduplicated(grid_factory, task_factory):
    task = _task3(grid_factory, task_factory)
    examples = build_ttt_examples(task, num_augs=8, max_examples=None)
    keys = {(e.prompt, e.completion) for e in examples}
    assert len(keys) == len(examples)
