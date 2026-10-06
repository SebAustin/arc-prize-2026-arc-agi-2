"""TTTSolver orchestration tests via the no-op MockTTTRunner (CPU)."""

from __future__ import annotations

from arc.augment import symmetry
from arc.io.submission import build_submission, validate_submission
from arc.pipeline import run
from arc.solvers.dsl.solver import DSLSolver
from arc.solvers.llm import MockModel, MockTTTRunner, TTTSolver


def _rot90(g):
    return symmetry.apply("rot90", g)


def test_ttt_solver_builds_corpus_and_adapts(grid_factory, task_factory):
    g = grid_factory(3, 3, seed=1)
    task = task_factory([(g, g), (g, g)], [g])
    runner = MockTTTRunner(MockModel())
    solver = TTTSolver(runner, llm_kwargs={"num_augs": 2})
    cands = solver.solve(task, budget_s=5.0)
    assert runner.adapt_calls == 1
    assert runner.reset_calls == 1          # reset always runs (finally)
    assert runner.last_examples             # corpus was built
    assert cands[0][0] == g                 # identity model -> echoes input


def test_ttt_solver_recovers_transform(grid_factory, task_factory):
    g = grid_factory(3, 3, seed=2)
    task = task_factory([(g, _rot90(g)), (g, _rot90(g))], [g])
    runner = MockTTTRunner(MockModel(transform=_rot90))
    solver = TTTSolver(runner, llm_kwargs={"num_augs": 1})
    cands = solver.solve(task, budget_s=5.0)
    assert cands[0][0] == _rot90(g)


def test_ttt_solver_resets_even_on_inference(grid_factory, task_factory):
    g = grid_factory(2, 2, seed=3)
    task = task_factory([(g, g), (g, g)], [g])
    runner = MockTTTRunner(MockModel())
    TTTSolver(runner, llm_kwargs={"num_augs": 1}).solve(task, budget_s=5.0)
    assert runner.reset_calls == 1


def test_ttt_in_ensemble_produces_valid_submission(grid_factory, task_factory):
    g = grid_factory(3, 3, seed=4)
    tasks = {"t": task_factory([(g, g), (g, g)], [g], task_id="t")}
    runner = MockTTTRunner(MockModel())
    solvers = [DSLSolver(), TTTSolver(runner, llm_kwargs={"num_augs": 2})]
    preds = run(tasks, solvers=solvers, per_task_budget_s=5.0)
    assert validate_submission(build_submission(preds), tasks) == []
    assert preds["t"][0].attempt_1 == g
