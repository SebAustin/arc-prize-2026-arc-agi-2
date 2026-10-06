"""NVARC-style 'scoreagg' selection (occurrence count + geomean-logprob over augs).

CPU via MockModel, whose `score` is 0.0 for an aug-consistent completion and -1.0
otherwise — so the geomean term is an independent oracle, not the code recomputed.
"""

from __future__ import annotations

from arc.augment import symmetry
from arc.augment.task_aug import distinct_augs
from arc.io.loader import Pair
from arc.solvers.llm import LLMSolver, MockModel
from arc.solvers.llm.select import score_candidates_scoreagg


def _rot90(g):
    return symmetry.apply("rot90", g)


def test_equal_counts_broken_by_aug_consistency(grid_factory):
    # Same occurrence count -> the geomean term decides -> the aug-consistent
    # candidate (geomean 0.0) beats the inconsistent one (geomean -1.0).
    g = grid_factory(3, 3, seed=3)
    train = [Pair(g, _rot90(g))]
    test_input = grid_factory(3, 3, seed=4)
    right = _rot90(test_input)
    wrong = grid_factory(3, 3, seed=5)
    augs = distinct_augs(3, seed=0)
    model = MockModel(transform=_rot90)

    scored = score_candidates_scoreagg(
        model, train, test_input, [(wrong, 2.0), (right, 2.0)], augs
    )
    assert scored[0][0] == right  # equal counts -> higher geomean (more consistent) wins
    assert scored[0][1] > scored[1][1]  # strictly, via the geomean tiebreak


def test_higher_count_wins_when_geomean_tied(grid_factory):
    # Two equally-(in)consistent candidates -> the occurrence count orders them.
    g = grid_factory(2, 2, seed=6)
    train = [Pair(g, g)]
    test_input = grid_factory(2, 2, seed=7)
    a = grid_factory(2, 2, seed=8)  # both wrong under identity transform -> geomean -1 each
    b = grid_factory(2, 2, seed=9)
    augs = distinct_augs(2, seed=0)
    model = MockModel()  # identity; neither a nor b equals test_input

    scored = score_candidates_scoreagg(
        model, train, test_input, [(a, 1.0), (b, 3.0)], augs
    )
    assert scored[0][0] == b  # 3.0 - 1.0 = 2.0 beats 1.0 - 1.0 = 0.0


def test_solver_scoreagg_selection_end_to_end(grid_factory, task_factory):
    g = grid_factory(3, 3, seed=8)
    task = task_factory([(g, _rot90(g)), (g, _rot90(g))], [g])
    solver = LLMSolver(MockModel(transform=_rot90), num_augs=3, selection="scoreagg", poe_augs=3)
    cands = solver.solve(task, budget_s=5.0)
    assert cands[0][0] == _rot90(g)


def test_scoreagg_is_a_valid_selection_mode():
    from arc.solvers.llm.solver import SELECTION_MODES

    assert "scoreagg" in SELECTION_MODES
    LLMSolver(MockModel(), selection="scoreagg")  # constructor must accept it


def test_l4_full_profile_kwargs_build_a_valid_solver():
    """Guard against drift: every key the l4_full profile sets must be a valid
    LLMSolver kwarg AND satisfy the decode='dfs' invariant — else the competitive
    L4x4 run would crash at solver construction."""
    from arc.profiles import select_profile

    LLMSolver(MockModel(), **select_profile(4).llm_kwargs)  # must not raise
