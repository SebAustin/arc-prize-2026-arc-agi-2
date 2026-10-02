"""Product-of-experts selection + batched decoding (Rung 2), CPU via MockModel."""

from __future__ import annotations

import time

from arc.augment import symmetry
from arc.augment.task_aug import distinct_augs
from arc.io.loader import Pair
from arc.solvers.llm import LLMSolver, MockModel
from arc.solvers.llm.select import score_candidates_poe


def _rot90(g):
    return symmetry.apply("rot90", g)


class _CountingModel(MockModel):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.score_sum_calls = 0

    def score_sum(self, prompt: str, completion: str) -> float:
        self.score_sum_calls += 1
        return super().score_sum(prompt, completion)


def test_mock_score_sum_semantics(grid_factory):
    from arc.serialize.prompt import build_prompt
    from arc.serialize.tokenizer import grid_to_str
    from arc.solvers.llm.ttt_data import COMPLETION_PREFIX

    g = grid_factory(2, 2, seed=1)
    model = MockModel()  # identity transform
    prompt = build_prompt([Pair(g, g)], g)
    right = COMPLETION_PREFIX + grid_to_str(g)
    wrong = COMPLETION_PREFIX + grid_to_str(grid_factory(2, 2, seed=2))
    assert model.score_sum(prompt, right) == 0.0
    assert model.score_sum(prompt, wrong) == -10.0


def test_poe_ranks_consistent_candidate_first(grid_factory):
    # Model predicts rot90; the "right" candidate is consistent under EVERY
    # augmented framing, the "wrong" one under none.
    g = grid_factory(3, 3, seed=3)
    train = [Pair(g, _rot90(g))]
    test_input = grid_factory(3, 3, seed=4)
    right = _rot90(test_input)
    wrong = grid_factory(3, 3, seed=5)
    augs = distinct_augs(3, seed=0)
    model = _CountingModel(transform=_rot90)

    scored = score_candidates_poe(model, train, test_input, [wrong, right], augs)
    grids = [g_ for g_, _ in scored]
    assert grids[0] == right
    assert scored[0][1] > scored[1][1]  # strictly higher summed log-prob
    assert model.score_sum_calls == len(augs) * 2  # every candidate x every aug


def test_poe_deadline_stops_after_first_aug(grid_factory):
    g = grid_factory(2, 2, seed=6)
    train = [Pair(g, g)]
    candidates = [g, grid_factory(2, 2, seed=7)]
    augs = distinct_augs(4, seed=0)
    model = _CountingModel()
    past = time.monotonic() - 1.0
    scored = score_candidates_poe(model, train, g, candidates, augs, deadline_s=past)
    # First aug always runs (comparable partial results); later augs skipped.
    assert model.score_sum_calls == len(candidates)
    assert scored[0][0] == g  # identity candidate still wins on the one frame


def test_solver_poe_selection_end_to_end(grid_factory, task_factory):
    g = grid_factory(3, 3, seed=8)
    task = task_factory([(g, _rot90(g)), (g, _rot90(g))], [g])
    solver = LLMSolver(MockModel(transform=_rot90), num_augs=3, selection="poe")
    cands = solver.solve(task, budget_s=5.0)
    assert cands[0][0] == _rot90(g)


def test_batched_equals_looped(grid_factory, task_factory):
    g = grid_factory(3, 3, seed=9)
    task = task_factory([(g, _rot90(g)), (g, _rot90(g))], [g])

    class _NoBatchModel(MockModel):
        generate_batch = property()  # hasattr() -> AttributeError -> False

    batched = LLMSolver(MockModel(transform=_rot90), num_augs=4)
    looped = LLMSolver(_NoBatchModel(transform=_rot90), num_augs=4)
    assert batched.solve(task, 5.0) == looped.solve(task, 5.0)
    assert batched.last_telemetry["augs_completed"] == 4


def test_use_likelihood_maps_to_selection_alias():
    m = MockModel()
    assert LLMSolver(m, use_likelihood=True).selection == "likelihood"
    assert LLMSolver(m).selection == "votes"
    assert LLMSolver(m, selection="poe", use_likelihood=False).selection == "poe"


def test_invalid_selection_rejected():
    import pytest

    with pytest.raises(ValueError):
        LLMSolver(MockModel(), selection="bogus")
