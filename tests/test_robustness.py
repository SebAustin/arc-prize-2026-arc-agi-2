"""Robustness tests for the submission-guarantee path: solver-failure logging,
unregistered-weight warnings, time-based checkpointing, the DSL output-size cap,
the raw fallback builder, and the Kaggle entrypoint's pre-write safety net.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import pytest

import arc.pipeline as pipeline_mod
from arc.io.submission import FALLBACK_GRID, fallback_from_raw
from arc.pipeline import run, solve_task
from arc.solvers.base import Candidates, Solver
from arc.solvers.dsl.primitives import scale_program, tile_program


class _FailingSolver(Solver):
    name = "boom"

    def solve(self, task, budget_s: float) -> Candidates:
        raise RuntimeError("kaboom")


class _StubSolver(Solver):
    def __init__(self, out, name: str = "dsl"):
        self.out = out
        self.name = name

    def solve(self, task, budget_s: float) -> Candidates:
        return [[self.out] for _ in task.test]


class _RankedStubSolver(Solver):
    def __init__(self, outs, name: str):
        self.outs = outs
        self.name = name

    def solve(self, task, budget_s: float) -> Candidates:
        return [list(self.outs) for _ in task.test]


def test_llm_ttt_second_choice_beats_identity_echo(grid_factory, task_factory):
    # Regression: identity's rank-0 vote (was 8.0) used to outbid llm_ttt's
    # rank-1 vote (9/2=4.5), wasting attempt_2 on "output = input".
    g = grid_factory(2, 2, seed=1)
    a, b = grid_factory(2, 2, seed=2), grid_factory(2, 2, seed=3)
    task = task_factory([(g, g)], [g])
    llm = _RankedStubSolver([a, b], name="llm_ttt")
    identity_echo = _StubSolver(g, name="identity")
    attempts = solve_task(task, [llm, identity_echo], budget_s=5.0)
    assert attempts[0].attempt_1 == a
    assert attempts[0].attempt_2 == b  # llm_ttt's 2nd candidate, not the echo


# ---- pipeline: failure logging + weight warning ---------------------------
def test_failing_solver_is_logged_and_skipped(grid_factory, task_factory, caplog):
    g = grid_factory(2, 2, seed=1)
    task = task_factory([(g, g)], [g])
    with caplog.at_level(logging.ERROR, logger="arc.pipeline"):
        attempts = solve_task(task, [_FailingSolver(), _StubSolver(g)], budget_s=5.0)
    assert attempts[0].attempt_1 == g  # the surviving solver still won
    assert "failed on task" in caplog.text


def test_unregistered_solver_name_warns(grid_factory, task_factory, caplog):
    g = grid_factory(2, 2, seed=1)
    tasks = {"t": task_factory([(g, g)], [g], task_id="t")}
    with caplog.at_level(logging.WARNING, logger="arc.pipeline"):
        run(tasks, solvers=[_StubSolver(g, name="mystery")], per_task_budget_s=5.0)
    assert "mystery" in caplog.text
    assert "no registered vote weight" in caplog.text


# ---- pipeline: time-based checkpointing -----------------------------------
def test_time_based_checkpoint_flushes(grid_factory, task_factory, tmp_path, monkeypatch):
    g = grid_factory(2, 2, seed=1)
    tasks = {f"t{i}": task_factory([(g, g)], [g], task_id=f"t{i}") for i in range(3)}

    writes: list[int] = []
    real_write = pipeline_mod.write_submission
    monkeypatch.setattr(
        pipeline_mod,
        "write_submission",
        lambda sub, path: (writes.append(1), real_write(sub, path))[1],
    )
    monkeypatch.setattr(pipeline_mod, "_CHECKPOINT_SECONDS", 0.0)  # every task is due

    out = tmp_path / "submission.json"
    run(tasks, solvers=[_StubSolver(g)], output_path=str(out), per_task_budget_s=5.0)
    # 1 up-front + 3 per-task (time-due) + 1 final
    assert len(writes) == 5


# ---- DSL output-size cap ---------------------------------------------------
def test_scale_program_passes_through_when_oversized(grid_factory):
    g = grid_factory(20, 20, seed=1)  # 20*2 = 40 > 30
    assert scale_program(2, 2)(g) == g  # oversized -> passthrough (fails to verify)


def test_scale_program_allocates_within_cap(grid_factory):
    g = grid_factory(10, 10, seed=1)  # 10*2 = 20 <= 30
    out = scale_program(2, 2)(g)
    assert len(out) == 20 and len(out[0]) == 20


def test_tile_program_passes_through_when_oversized(grid_factory):
    g = grid_factory(16, 16, seed=1)  # 16*2 = 32 > 30
    assert tile_program(2, 2)(g) == g


# ---- raw fallback builder --------------------------------------------------
def test_fallback_from_raw_counts_test_inputs():
    raw = {
        "a": {"test": [{"input": [[1]]}, {"input": [[2]]}]},
        "b": {"test": [{"input": [[3]]}]},
        "c": {"garbage": True},  # undeterminable count -> default 1
        "d": "not a dict",  # -> default 1
    }
    preds = fallback_from_raw(raw)
    assert [len(preds[k]) for k in ("a", "b", "c", "d")] == [2, 1, 1, 1]
    assert preds["a"][0].attempt_1 == FALLBACK_GRID


def test_fallback_from_raw_non_dict_is_empty():
    assert fallback_from_raw([1, 2, 3]) == {}


# ---- Kaggle entrypoint pre-write safety net --------------------------------
def _load_entrypoint():
    path = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_submit.py"
    spec = importlib.util.spec_from_file_location("kaggle_submit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env(monkeypatch, tmp_path, challenges):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    (data / "arc-agi_test_challenges.json").write_text(json.dumps(challenges))
    monkeypatch.setenv("ARC_MODE", "SMOKE")
    monkeypatch.setenv("ARC_DATA_DIR", str(data))
    monkeypatch.setenv("ARC_OUTPUT_DIR", str(out))
    return out / "submission.json"


def test_entrypoint_writes_valid_submission(tmp_path, monkeypatch):
    challenges = {
        "t1": {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}],
            "test": [{"input": [[5, 6], [7, 8]]}],
        }
    }
    sub = _env(monkeypatch, tmp_path, challenges)
    result = _load_entrypoint().main()  # no model -> DSL/heuristic ensemble only
    assert sub.exists()
    loaded = json.loads(sub.read_text())
    assert loaded["t1"][0]["attempt_1"] and loaded["t1"][0]["attempt_2"]
    assert result["num_tasks"] == 1


def test_entrypoint_keeps_fallback_on_malformed_file(tmp_path, monkeypatch):
    # 'train' missing -> parse error, but 'test' present so the fallback covers it.
    sub = _env(monkeypatch, tmp_path, {"t1": {"test": [{"input": [[1]]}]}})
    result = _load_entrypoint().main()
    assert sub.exists()  # pre-written fallback survives the parse failure
    loaded = json.loads(sub.read_text())
    assert "t1" in loaded  # task still present -> submission is scoreable
    assert result["num_tasks"] == 0  # graceful degrade, no crash


def test_entrypoint_falls_back_to_dsl_when_model_load_fails(tmp_path, monkeypatch):
    # model_path is set but the model can't load (locally: torch absent; on Kaggle
    # with no GPU: "Torch not compiled with CUDA enabled"). Must degrade to the
    # DSL ensemble and still write a real submission — never crash into all-zeros.
    challenges = {
        "t1": {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[3, 4], [1, 2]]}],
            "test": [{"input": [[5, 6], [7, 8]]}],
        }
    }
    sub = _env(monkeypatch, tmp_path, challenges)
    result = _load_entrypoint().main(model_path="/nonexistent/model")
    assert sub.exists()
    loaded = json.loads(sub.read_text())
    assert loaded["t1"][0]["attempt_1"]  # a real submission was produced
    assert result["problems"] == []  # DSL fallback -> schema-clean, no crash
    assert result["num_tasks"] == 1


def test_entrypoint_canary_max_tasks_keeps_submission_complete(tmp_path, monkeypatch):
    # max_tasks=1 solves only the first task, but the written submission must
    # still cover EVERY task_id (unsolved keep the 1x1 fallback grid).
    challenges = {
        "solved": {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}],
            "test": [{"input": [[5, 6], [7, 8]]}],
        },
        "untouched": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        },
    }
    sub = _env(monkeypatch, tmp_path, challenges)
    result = _load_entrypoint().main(max_tasks=1)
    loaded = json.loads(sub.read_text())
    assert set(loaded) == {"solved", "untouched"}  # schema-complete
    assert loaded["solved"][0]["attempt_1"] == [[5, 6], [7, 8]]  # identity solved
    assert loaded["untouched"][0]["attempt_1"] == [[0]]  # fallback retained
    assert result["problems"] == []  # validates against the FULL task set
    assert result["num_tasks"] == 1  # only the canary subset was solved


def _three_task_challenges():
    t = {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]]}],
    }
    return {"a": t, "b": t, "c": t}


def test_commit_fast_path_canaries_on_placeholder_hash(tmp_path, monkeypatch):
    sub = _env(monkeypatch, tmp_path, _three_task_challenges())
    mod = _load_entrypoint()
    # Make the written file's hash "the placeholder" and shrink the canary size.
    data_file = tmp_path / "data" / "arc-agi_test_challenges.json"
    monkeypatch.setattr(mod, "PLACEHOLDER_SHA256", mod._sha256_file(data_file))
    monkeypatch.setattr(mod, "_COMMIT_CANARY_TASKS", 1)
    result = mod.main()
    assert result["num_tasks"] == 1  # canary subset
    assert json.loads(sub.read_text()).keys() == {"a", "b", "c"}  # still complete


def test_commit_fast_path_full_run_on_hash_mismatch(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path, _three_task_challenges())
    mod = _load_entrypoint()  # real PLACEHOLDER_SHA256 won't match the temp file
    assert mod.main()["num_tasks"] == 3  # full run (fail-safe direction)


def test_commit_fast_path_force_full_override(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path, _three_task_challenges())
    mod = _load_entrypoint()
    data_file = tmp_path / "data" / "arc-agi_test_challenges.json"
    monkeypatch.setattr(mod, "PLACEHOLDER_SHA256", mod._sha256_file(data_file))
    monkeypatch.setenv("ARC_FORCE_FULL", "1")
    assert mod.main()["num_tasks"] == 3  # override wins over the hash match


def test_entrypoint_errors_clearly_when_data_missing(tmp_path, monkeypatch):
    # data dir exists but has NO challenges file (e.g. competition not attached)
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    monkeypatch.setenv("ARC_MODE", "SMOKE")
    monkeypatch.setenv("ARC_DATA_DIR", str(data))
    monkeypatch.setenv("ARC_OUTPUT_DIR", str(out))
    with pytest.raises(FileNotFoundError) as ei:
        _load_entrypoint().main()
    assert "Competition data not found" in str(ei.value)  # actionable, not cryptic
