"""Grid type, conversions, and submission schema tests."""

from __future__ import annotations

from arc.io.grid import (
    from_lists,
    from_numpy,
    is_valid_grid,
    to_lists,
    to_numpy,
)
from arc.io.submission import (
    Attempt,
    build_submission,
    empty_predictions,
    validate_submission,
)


def test_from_to_lists_roundtrip():
    rows = [[0, 1, 2], [3, 4, 5]]
    grid = from_lists(rows)
    assert grid == ((0, 1, 2), (3, 4, 5))
    assert to_lists(grid) == rows


def test_numpy_roundtrip(grid_factory):
    grid = grid_factory(4, 7, seed=3)
    assert from_numpy(to_numpy(grid)) == grid


def test_is_valid_grid_accepts_well_formed(grid_factory):
    assert is_valid_grid(grid_factory(1, 1))
    assert is_valid_grid(grid_factory(30, 30))


def test_is_valid_grid_rejects_malformed():
    assert not is_valid_grid(())                       # empty
    assert not is_valid_grid(((0, 1), (2,)))           # ragged
    assert not is_valid_grid(((0, 10),))               # symbol out of range
    assert not is_valid_grid(tuple((0,) * 5 for _ in range(31)))  # too tall


def test_build_and_validate_submission(task_factory):
    g = ((1, 2), (3, 4))
    task = task_factory(train_pairs=[(g, g)], test_inputs=[g])
    tasks = {task.task_id: task}
    preds = {task.task_id: [Attempt(g, g)]}
    submission = build_submission(preds)
    assert validate_submission(submission, tasks) == []


def test_validate_catches_missing_task(task_factory):
    g = ((1,),)
    task = task_factory(train_pairs=[(g, g)], test_inputs=[g])
    tasks = {task.task_id: task}
    problems = validate_submission({}, tasks)
    assert any("missing" in p for p in problems)


def test_empty_predictions_is_schema_valid(task_factory):
    g = ((1, 1), (1, 1))
    t1 = task_factory([(g, g)], [g, g], task_id="a")  # 2 test inputs
    t2 = task_factory([(g, g)], [g], task_id="b")
    tasks = {"a": t1, "b": t2}
    preds = empty_predictions(tasks)
    assert len(preds["a"]) == 2 and len(preds["b"]) == 1
    assert validate_submission(build_submission(preds), tasks) == []
