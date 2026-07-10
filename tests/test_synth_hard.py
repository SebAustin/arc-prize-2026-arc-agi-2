"""Rung-5 "hard" generator tests: DSL-unsolvability proof and the corpus filter.

Complements `tests/test_synth.py` (which already exercises well-formedness for
every generator in `GENERATORS`, including these) with the extra guarantee
this milestone is about: each new generator's concept is provably outside the
DSL micro-solver's vocabulary, and `exclude_dsl_solvable=True` actually removes
whatever slips through.
"""

from __future__ import annotations

import pytest

from arc.serialize.prompt import parse_completion
from arc.synth.build_dataset import (
    build_synthetic_tasks,
    is_dsl_solvable,
    tasks_to_examples,
)
from arc.synth.generators import GENERATORS, HARD_GENERATORS, build_task

_SEEDS = (0, 1, 2)


@pytest.mark.parametrize("gen", HARD_GENERATORS, ids=lambda g: g.name)
@pytest.mark.parametrize("seed", _SEEDS)
def test_hard_generator_is_deterministic(gen, seed):
    a = build_task(gen, seed=seed)
    b = build_task(gen, seed=seed)
    assert a.task == b.task


@pytest.mark.parametrize("gen", HARD_GENERATORS, ids=lambda g: g.name)
@pytest.mark.parametrize("seed", _SEEDS)
def test_hard_generator_is_not_dsl_solvable(gen, seed):
    gt = build_task(gen, seed=seed)
    assert not is_dsl_solvable(gt.task, budget_s=2.0)


def test_is_dsl_solvable_true_for_recolor():
    gen = next(g for g in GENERATORS if g.name == "recolor")
    gt = build_task(gen, seed=7)
    assert is_dsl_solvable(gt.task)


def test_is_dsl_solvable_false_for_symmetry_repair():
    gen = next(g for g in HARD_GENERATORS if g.name == "symmetry_repair")
    gt = build_task(gen, seed=7)
    assert not is_dsl_solvable(gt.task)


def test_exclude_dsl_solvable_filters_mixed_corpus():
    tasks = build_synthetic_tasks(10, seed=9, generators=GENERATORS, exclude_dsl_solvable=True)
    assert len(tasks) == 10
    for t in tasks:
        assert not is_dsl_solvable(t)


def test_hard_only_round_trip_examples():
    tasks = build_synthetic_tasks(6, seed=4, hard_only=True)
    assert len(tasks) == 6
    for t in tasks:
        assert not is_dsl_solvable(t)
    examples = tasks_to_examples(tasks)
    assert len(examples) == 6
    for ex, task in zip(examples, tasks, strict=False):
        assert ex.prompt.rstrip().endswith("Output:")
        assert parse_completion(ex.completion) == task.test[0].output
