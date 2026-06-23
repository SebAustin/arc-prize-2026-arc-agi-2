"""LLM transduction path tests, driven by the deterministic CPU MockModel."""

from __future__ import annotations

from arc.augment import symmetry
from arc.io.submission import build_submission, validate_submission
from arc.pipeline import default_solvers, run
from arc.serialize.prompt import build_prompt
from arc.solvers.llm import LLMSolver, MockModel
from arc.solvers.llm.infer import generate_candidates
from arc.solvers.llm.select import score_candidates
from arc.io.loader import Pair


def _rot90(g):
    return symmetry.apply("rot90", g)


def test_mock_model_echoes_test_input(grid_factory):
    g = grid_factory(3, 3, seed=1)
    model = MockModel()  # identity transform
    cands = generate_candidates(model, train=[Pair(g, g)], test_input=g)
    assert cands == [g]


def test_mock_model_applies_transform(grid_factory):
    g = grid_factory(3, 4, seed=2)
    model = MockModel(transform=_rot90)
    cands = generate_candidates(model, train=[Pair(g, _rot90(g))], test_input=g)
    assert cands == [_rot90(g)]


def test_llm_solver_recovers_transform_through_identity_aug(grid_factory, task_factory):
    g = grid_factory(3, 3, seed=3)
    task = task_factory([(g, _rot90(g))], [g])
    solver = LLMSolver(MockModel(transform=_rot90), num_augs=1)
    cands = solver.solve(task, budget_s=5.0)
    assert cands[0][0] == _rot90(g)


def test_llm_solver_invert_consistent_across_augmentations(grid_factory, task_factory):
    # An identity-predicting model must yield the test input back under EVERY
    # augmentation once inverted — proves the augment/invert round-trip in-solver.
    g = grid_factory(4, 5, ncolors=7, seed=4)
    task = task_factory([(g, g)], [g])
    solver = LLMSolver(MockModel(), num_augs=6)
    cands = solver.solve(task, budget_s=5.0)
    assert cands[0][0] == g
    # all augmentations agreed on a single candidate
    assert len(cands[0]) == 1


def test_llm_solver_handles_multi_test_inputs(grid_factory, task_factory):
    g1 = grid_factory(2, 2, seed=5)
    g2 = grid_factory(3, 3, seed=6)
    task = task_factory([(g1, g1)], [g1, g2])
    solver = LLMSolver(MockModel(), num_augs=2)
    cands = solver.solve(task, budget_s=5.0)
    assert len(cands) == 2
    assert cands[0][0] == g1 and cands[1][0] == g2


def test_pipeline_with_mock_llm_produces_valid_submission(grid_factory, task_factory):
    g = grid_factory(3, 3, seed=7)
    tasks = {"t": task_factory([(g, g)], [g], task_id="t")}
    solvers = default_solvers(llm_model=MockModel(), llm_kwargs={"num_augs": 3})
    preds = run(tasks, solvers=solvers, per_task_budget_s=5.0)
    assert validate_submission(build_submission(preds), tasks) == []
    assert preds["t"][0].attempt_1 == g


def test_score_candidates_prefers_model_target(grid_factory):
    g = grid_factory(3, 3, seed=10)
    other = grid_factory(3, 3, seed=11)
    model = MockModel()  # identity -> the model's target for input g is g
    scored = score_candidates(model, [Pair(g, g)], g, [other, g])
    assert scored[0][0] == g  # exact-match scores 0.0, beats other's -1.0


def test_llm_solver_with_likelihood_selection(grid_factory, task_factory):
    g = grid_factory(3, 3, seed=12)
    task = task_factory([(g, g)], [g])
    cands = LLMSolver(MockModel(), num_augs=2, use_likelihood=True).solve(task, 5.0)
    assert cands[0][0] == g
