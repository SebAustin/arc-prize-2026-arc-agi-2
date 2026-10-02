"""LLMSolver DFS wiring (decode="dfs"), pure CPU.

`_TrieLM` composes the repo's MockModel with a canned `TrieStepModel` (pattern
of test_select_poe.py's _NoBatchModel/_CountingModel) so the full
solver -> generate_candidates_dfs -> dfs_decode -> vote chain runs without
torch. Grid texts use the canonical digit-per-cell rows ("11\n11").
"""

from __future__ import annotations

import logging
import math

import pytest

from arc.solvers.llm.dfs_decode import TrieStepModel
from arc.solvers.llm.infer import generate_candidates_dfs
from arc.solvers.llm.model import MockModel
from arc.solvers.llm.solver import LLMSolver

G1 = ((1, 1), (1, 1))  # text "11\n11"
G2 = ((2,),)  # text "2"


def _lp(p: float) -> float:
    return math.log(p)


class _TrieLM(MockModel):
    """MockModel that also decodes via a canned token trie."""

    def __init__(self, tree, detok, transform=None):
        super().__init__(transform)
        self._tree = tree
        self._detok = detok
        self.step_models: list[TrieStepModel] = []

    def as_step_model(self, top_k: int = 16) -> TrieStepModel:
        sm = TrieStepModel(self._tree, detok=self._detok, top_k=top_k)
        self.step_models.append(sm)
        return sm


def _two_grid_lm() -> _TrieLM:
    """G1 with mass 0.6, G2 with 0.3, and a below-threshold third branch."""
    tree = {
        (): {1: _lp(0.6), 2: _lp(0.3), 3: _lp(0.09)},
        (1,): {0: _lp(1.0)},
        (2,): {0: _lp(1.0)},
        (3,): {0: _lp(1.0)},
    }
    return _TrieLM(tree, detok={1: "11\n11", 2: "2", 3: "2"})


# ---- generate_candidates_dfs --------------------------------------------------


def test_dfs_weights_are_probability_mass(task_factory):
    task = task_factory([(G1, G1)], [G2])
    pairs, stats = generate_candidates_dfs(
        _two_grid_lm(), task.train, task.test[0].input, eps=0.1
    )

    assert [g for g, _ in pairs] == [G1, G2]
    assert pairs[0][1] == pytest.approx(0.6)
    assert pairs[1][1] == pytest.approx(0.3)
    assert stats.leaves == 2  # the 0.09 branch was pruned


def test_same_grid_leaves_pool_mass(task_factory):
    # Two textually-different leaves parse to the SAME grid -> one candidate
    # holding their summed probability mass.
    tree = {
        (): {1: _lp(0.4), 2: _lp(0.35), 3: _lp(0.2)},
        (1,): {0: _lp(1.0)},
        (2,): {0: _lp(1.0)},
        (3,): {0: _lp(1.0)},
    }
    lm = _TrieLM(tree, detok={1: "11\n11", 2: "1 1\n1 1", 3: "2"})
    task = task_factory([(G1, G1)], [G2])
    pairs, _ = generate_candidates_dfs(lm, task.train, task.test[0].input, eps=0.05)

    assert pairs[0][0] == G1
    assert pairs[0][1] == pytest.approx(0.75)  # 0.4 + 0.35 pooled
    assert pairs[1] == (G2, pytest.approx(0.2))


def test_unparseable_leaves_dropped(task_factory):
    tree = {
        (): {1: _lp(0.5), 2: _lp(0.3)},
        (1,): {0: _lp(1.0)},
        (2,): {0: _lp(1.0)},
    }
    lm = _TrieLM(tree, detok={1: "no digits at all!", 2: "2"})
    task = task_factory([(G1, G1)], [G2])
    pairs, stats = generate_candidates_dfs(lm, task.train, task.test[0].input, eps=0.1)

    assert pairs == [(G2, pytest.approx(0.3))]
    assert stats.leaves == 2  # DFS found both; the parse layer dropped one


# ---- LLMSolver wiring ----------------------------------------------------------


def test_solver_dfs_ranks_by_mass(task_factory):
    solver = LLMSolver(_two_grid_lm(), num_augs=1, decode="dfs", dfs_eps=0.1)
    task = task_factory([(G1, G1)], [G2])
    per_test = solver.solve(task, budget_s=5.0)

    assert per_test[0][:2] == [G1, G2]


def test_decode_validation():
    with pytest.raises(ValueError, match="decode"):
        LLMSolver(MockModel(), decode="beam")
    with pytest.raises(ValueError, match="dfs"):
        LLMSolver(MockModel(), decode="dfs", temperature=0.7)
    with pytest.raises(ValueError, match="dfs"):
        LLMSolver(MockModel(), decode="dfs", num_samples=2)


def test_dfs_fallback_to_greedy_without_step_model(task_factory, caplog):
    # MockModel has no as_step_model: decode="dfs" must degrade to the exact
    # greedy behavior (kill-switch direction — a typo can't zero a Kaggle run).
    task = task_factory([(G1, G1)], [G1])
    with caplog.at_level(logging.WARNING, logger="arc.solvers.llm.solver"):
        dfs_result = LLMSolver(MockModel(), num_augs=1, decode="dfs").solve(task, 5.0)
    greedy_result = LLMSolver(MockModel(), num_augs=1).solve(task, 5.0)

    assert dfs_result == greedy_result
    assert any("as_step_model" in r.message for r in caplog.records)


def test_use_batch_gate_off_for_dfs(task_factory):
    class _CountingTrieLM(_TrieLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batch_calls = 0

        def generate_batch(self, *args, **kwargs):
            self.batch_calls += 1
            return super().generate_batch(*args, **kwargs)

    lm = _CountingTrieLM(
        {(): {1: _lp(0.9)}, (1,): {0: _lp(1.0)}}, detok={1: "2"}
    )
    task = task_factory([(G1, G1)], [G2])
    LLMSolver(lm, num_augs=1, decode="dfs").solve(task, 5.0)

    assert lm.batch_calls == 0  # DFS configs must not fall into the batch path
    assert lm.step_models  # ... and actually decoded via the step model


def test_greedy_default_unchanged(task_factory):
    # Identity MockModel predicts the test input back; the default (greedy)
    # path is untouched by the DFS addition.
    task = task_factory([(G1, G1)], [G2])
    per_test = LLMSolver(MockModel(), num_augs=1).solve(task, 5.0)

    assert per_test[0][0] == G2


def test_dfs_telemetry_keys(task_factory):
    task = task_factory([(G1, G1)], [G2])
    solver = LLMSolver(_two_grid_lm(), num_augs=1, decode="dfs", dfs_eps=0.1)
    solver.solve(task, 5.0)
    assert solver.last_telemetry["dfs_nodes"] > 0
    assert solver.last_telemetry["dfs_leaves"] > 0

    greedy = LLMSolver(MockModel(), num_augs=1)
    greedy.solve(task, 5.0)
    assert greedy.last_telemetry["dfs_nodes"] == 0
    assert greedy.last_telemetry["dfs_leaves"] == 0


def test_dfs_with_poe_selection(task_factory):
    # End-to-end weighted-votes -> PoE chain: the test input IS G1, so
    # MockModel's pseudo-score prefers G1 and PoE keeps it on top.
    solver = LLMSolver(
        _two_grid_lm(), num_augs=1, decode="dfs", dfs_eps=0.1,
        selection="poe", poe_augs=1,
    )
    task = task_factory([(G1, G1)], [G1])
    per_test = solver.solve(task, budget_s=5.0)

    assert per_test[0][0] == G1
    assert solver.last_telemetry["score_s"] >= 0.0
