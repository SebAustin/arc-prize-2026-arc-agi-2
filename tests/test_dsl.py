"""DSL micro-solver tests: synthesised tasks with known programs."""

from __future__ import annotations

from arc.augment import symmetry
from arc.solvers.base import verify_program
from arc.solvers.dsl.primitives import colormap_program, scale_program
from arc.solvers.dsl.solver import DSLSolver

SOLVER = DSLSolver()
BUDGET = 5.0


def _rot90(g):
    return symmetry.apply("rot90", g)


def test_verify_program_accepts_correct(grid_factory, task_factory):
    inputs = [grid_factory(3, 3, seed=s) for s in range(3)]
    task = task_factory([(g, _rot90(g)) for g in inputs], [grid_factory(3, 3, seed=99)])
    assert verify_program(_rot90, task)
    assert not verify_program(lambda g: g, task)


def test_solves_rotation_task(grid_factory, task_factory):
    inputs = [grid_factory(3, 3, ncolors=6, seed=s) for s in range(3)]
    test_in = grid_factory(3, 3, ncolors=6, seed=50)
    task = task_factory([(g, _rot90(g)) for g in inputs], [test_in])
    candidates = SOLVER.solve(task, BUDGET)
    assert _rot90(test_in) in candidates[0]


def test_solves_colormap_task(grid_factory, task_factory):
    mapping = {0: 3, 1: 4, 2: 5}
    recolor = colormap_program(mapping)
    inputs = [grid_factory(4, 4, ncolors=3, seed=s) for s in range(3)]
    test_in = grid_factory(4, 4, ncolors=3, seed=70)
    task = task_factory([(g, recolor(g)) for g in inputs], [test_in])
    candidates = SOLVER.solve(task, BUDGET)
    assert recolor(test_in) in candidates[0]


def test_solves_scale_task(grid_factory, task_factory):
    scale = scale_program(2, 2)
    inputs = [grid_factory(2, 3, ncolors=5, seed=s) for s in range(3)]
    test_in = grid_factory(2, 3, ncolors=5, seed=80)
    task = task_factory([(g, scale(g)) for g in inputs], [test_in])
    candidates = SOLVER.solve(task, BUDGET)
    assert scale(test_in) in candidates[0]


def test_unsolvable_task_returns_empty(grid_factory, task_factory):
    # Random unrelated outputs: no simple program should verify.
    inputs = [grid_factory(3, 3, seed=s) for s in range(3)]
    outputs = [grid_factory(3, 3, seed=100 + s) for s in range(3)]
    task = task_factory(list(zip(inputs, outputs, strict=False)), [grid_factory(3, 3, seed=200)])
    candidates = SOLVER.solve(task, BUDGET)
    assert candidates[0] == []
