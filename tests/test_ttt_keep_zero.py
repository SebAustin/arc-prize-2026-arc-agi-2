"""TTTSolver threads the keep_zero (fix-black) flag from llm_kwargs into the TTT
data augmentation, so one profile flag governs BOTH inference and TTT corpus augs
(NVARC parity). An explicit ttt_data_kwargs value still wins.
"""

from __future__ import annotations

import arc.solvers.llm.ttt as ttt_mod
from arc.solvers.llm import MockModel


class _StubRunner:
    def adapt(self, examples, deadline_s=None):
        return MockModel()

    def reset(self):  # called in TTTSolver.solve's finally
        pass


def _capture_build(monkeypatch):
    captured: dict = {}

    def fake_build(task, **kw):
        captured.update(kw)
        return []

    monkeypatch.setattr(ttt_mod, "build_ttt_examples", fake_build)
    return captured


def test_keep_zero_flows_from_llm_kwargs_to_ttt_data(monkeypatch, grid_factory, task_factory):
    captured = _capture_build(monkeypatch)
    g = grid_factory(2, 2, seed=1)
    task = task_factory([(g, g)], [g])
    solver = ttt_mod.TTTSolver(
        _StubRunner(), llm_kwargs={"keep_zero": True}, ttt_data_kwargs={"num_augs": 2}
    )
    solver.solve(task, 5.0)
    assert captured.get("keep_zero") is True


def test_explicit_ttt_data_keep_zero_wins(monkeypatch, grid_factory, task_factory):
    captured = _capture_build(monkeypatch)
    g = grid_factory(2, 2, seed=2)
    task = task_factory([(g, g)], [g])
    solver = ttt_mod.TTTSolver(
        _StubRunner(),
        llm_kwargs={"keep_zero": True},
        ttt_data_kwargs={"num_augs": 2, "keep_zero": False},
    )
    solver.solve(task, 5.0)
    assert captured.get("keep_zero") is False


def test_default_keep_zero_is_false(monkeypatch, grid_factory, task_factory):
    captured = _capture_build(monkeypatch)
    g = grid_factory(2, 2, seed=3)
    task = task_factory([(g, g)], [g])
    solver = ttt_mod.TTTSolver(_StubRunner(), llm_kwargs={}, ttt_data_kwargs={"num_augs": 2})
    solver.solve(task, 5.0)
    assert captured.get("keep_zero") is False  # behavior-preserving for existing callers
