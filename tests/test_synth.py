"""Synthetic generator tests: well-formedness, solvability, determinism, corpus."""

from __future__ import annotations

import pytest

from arc.serialize.prompt import parse_completion
from arc.solvers.dsl.solver import DSLSolver
from arc.synth.build_dataset import (
    build_synthetic_tasks,
    load_examples_jsonl,
    save_examples_jsonl,
    tasks_to_examples,
)
from arc.synth.generators import GENERATORS, build_task, is_well_formed

# Generators whose concept lies within the DSL micro-solver's vocabulary.
DSL_SOLVABLE = {"recolor", "color_swap", "symmetry", "scale", "tile", "crop_to_content"}


@pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: g.name)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generator_produces_well_formed_task(gen, seed):
    gt = build_task(gen, seed=seed)
    # >=2 demo pairs, one labelled test pair, all grids valid, rule reproduces outputs.
    assert len(gt.task.train) >= 2
    assert gt.task.num_test == 1 and gt.task.test[0].output is not None
    assert is_well_formed(gt)


@pytest.mark.parametrize("gen", [g for g in GENERATORS if g.name in DSL_SOLVABLE], ids=lambda g: g.name)
def test_dsl_solvable_generators_are_solved(gen):
    gt = build_task(gen, seed=7)
    candidates = DSLSolver().solve(gt.task, budget_s=5.0)
    expected = gt.transform(gt.task.test[0].input)
    assert expected in candidates[0]


def test_build_synthetic_tasks_count_and_validity():
    tasks = build_synthetic_tasks(20, seed=3)
    assert len(tasks) == 20
    for t in tasks:
        assert len(t.train) >= 2 and t.num_test == 1


def test_build_synthetic_tasks_is_deterministic():
    a = build_synthetic_tasks(10, seed=5)
    b = build_synthetic_tasks(10, seed=5)
    assert [t.task_id for t in a] == [t.task_id for t in b]
    assert [t.train for t in a] == [t.train for t in b]


def test_tasks_to_examples_format():
    tasks = build_synthetic_tasks(5, seed=2)
    examples = tasks_to_examples(tasks)
    assert len(examples) == 5
    for ex, task in zip(examples, tasks):
        assert ex.prompt.rstrip().endswith("Output:")
        assert parse_completion(ex.completion) == task.test[0].output


def test_jsonl_roundtrip(tmp_path):
    examples = tasks_to_examples(build_synthetic_tasks(4, seed=1))
    path = save_examples_jsonl(examples, tmp_path / "corpus.jsonl")
    loaded = load_examples_jsonl(path)
    assert loaded == examples
