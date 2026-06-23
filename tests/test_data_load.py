"""Integration check against the real competition files (skips if absent)."""

from __future__ import annotations

import pytest

from arc.config import get_config
from arc.io.loader import load_challenges, load_solutions

CFG = get_config()


def _require(path):
    if not path.exists():
        pytest.skip(f"competition data not found at {path}")


def test_training_challenges_load_and_shape():
    path = CFG.challenges_path("training")
    _require(path)
    tasks = load_challenges(path)
    assert len(tasks) == 1000
    sample = next(iter(tasks.values()))
    assert len(sample.train) >= 2
    assert sample.num_test >= 1
    # demonstration pairs carry outputs; test inputs do not.
    assert all(p.output is not None for p in sample.train)


def test_training_solutions_align_with_tests():
    cpath = CFG.challenges_path("training")
    spath = CFG.solutions_path("training")
    _require(cpath)
    _require(spath)
    tasks = load_challenges(cpath)
    sols = load_solutions(spath)
    # every task has a solution list whose length matches its test count.
    for task_id, task in list(tasks.items())[:50]:
        assert task_id in sols
        assert len(sols[task_id]) == task.num_test


def test_test_challenges_have_no_outputs():
    path = CFG.challenges_path("test")
    _require(path)
    tasks = load_challenges(path)
    sample = next(iter(tasks.values()))
    assert all(p.output is None for p in sample.test)
