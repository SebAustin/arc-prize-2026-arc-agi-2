"""Regression tests for the two TTT robustness fixes:

  * the inner transduction solver is charged for adaptation wall-clock, so a slow
    `adapt()` no longer hands a fresh full budget to `LLMSolver` (budget overrun);
  * `reset()` runs even when `adapt()` raises, so a mid-adaptation failure (e.g.
    CUDA OOM) cannot leave the shared model corrupted for later tasks.

Both paths were previously exercised only by the no-op `MockTTTRunner`, which
never consumes time nor raises.
"""

from __future__ import annotations

import time

import pytest

import arc.solvers.llm.ttt as ttt_mod
from arc.solvers.llm import MockModel, MockTTTRunner, TTTSolver


class _SlowRunner(MockTTTRunner):
    """A runner whose adapt() burns a known amount of wall-clock before returning."""

    def __init__(self, model, delay_s: float):
        super().__init__(model)
        self.delay_s = delay_s

    def adapt(self, examples, deadline_s=None):
        time.sleep(self.delay_s)
        return super().adapt(examples, deadline_s)


class _RaisingRunner:
    """adapt() raises; reset() is counted so we can assert it still fires."""

    def __init__(self):
        self.reset_calls = 0

    def adapt(self, examples, deadline_s=None):
        raise RuntimeError("simulated CUDA OOM during adaptation")

    def reset(self):
        self.reset_calls += 1


def _record_inner_budget(monkeypatch, captured):
    class _RecordingLLMSolver:
        def __init__(self, model, **kwargs):
            self.model = model

        def solve(self, task, budget_s):
            captured["budget"] = budget_s
            return [[grid] for grid in [captured["echo"]]]

    monkeypatch.setattr(ttt_mod, "LLMSolver", _RecordingLLMSolver)


def test_adaptation_time_is_charged_to_inner_budget(
    grid_factory, task_factory, monkeypatch
):
    g = grid_factory(3, 3, seed=1)
    task = task_factory([(g, g), (g, g)], [g])
    captured = {"echo": g}
    _record_inner_budget(monkeypatch, captured)

    delay = 0.05
    TTTSolver(_SlowRunner(MockModel(), delay_s=delay)).solve(task, budget_s=10.0)

    # The inner solver sees the budget minus (at least) the adaptation delay.
    assert captured["budget"] < 10.0
    assert captured["budget"] <= 10.0 - delay + 0.02


def test_remaining_budget_is_clamped_at_zero(grid_factory, task_factory, monkeypatch):
    g = grid_factory(2, 2, seed=2)
    task = task_factory([(g, g)], [g])
    captured = {"echo": g}
    _record_inner_budget(monkeypatch, captured)

    TTTSolver(_SlowRunner(MockModel(), delay_s=0.02)).solve(task, budget_s=0.0)
    assert captured["budget"] == 0.0  # never negative


def test_reset_runs_when_adaptation_raises(grid_factory, task_factory):
    g = grid_factory(2, 2, seed=3)
    task = task_factory([(g, g)], [g])
    runner = _RaisingRunner()
    with pytest.raises(RuntimeError):
        TTTSolver(runner).solve(task, budget_s=5.0)
    assert runner.reset_calls == 1  # finally-block reset fired despite the raise


def test_adapt_deadline_is_ttt_fraction_of_budget(grid_factory, task_factory):
    g = grid_factory(2, 2, seed=4)
    task = task_factory([(g, g), (g, g)], [g])
    runner = MockTTTRunner(MockModel())
    before = time.monotonic()
    TTTSolver(runner, ttt_fraction=0.4).solve(task, budget_s=10.0)
    # adapt received an absolute deadline ~= t0 + 0.4 * 10s
    assert runner.last_deadline_s is not None
    offset = runner.last_deadline_s - before
    assert 3.5 <= offset <= 4.5


def test_telemetry_populated(grid_factory, task_factory):
    g = grid_factory(2, 2, seed=5)
    task = task_factory([(g, g), (g, g)], [g])
    solver = TTTSolver(MockTTTRunner(MockModel()), llm_kwargs={"num_augs": 2})
    solver.solve(task, budget_s=5.0)
    t = solver.last_telemetry
    assert t["task_id"] == task.task_id
    assert t["corpus_n"] > 0
    assert t["augs_completed"] >= 1  # identity aug always runs
    assert t["augs_planned"] == 2  # 2 augs x 1 test input
    assert "adapt_s" in t and "decode_s" in t
