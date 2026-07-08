"""Tests for the Rung 3 multi-process fan-out (`arc.parallel.run_parallel`):
merge-equivalence with the sequential path, fallback-first, crash tolerance,
watchdog termination, the num_workers<=1 kill-switch, jsonl robustness, and
the torch-stays-lazy invariant.

Model factories used as the `model_factory` test seam must be module-level
(not closures) so they survive pickling across the `spawn` start method.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

from arc.augment import symmetry
from arc.io.submission import build_submission, validate_submission
from arc.parallel import run_parallel
from arc.pipeline import default_solvers
from arc.pipeline import run as run_sequential
from arc.solvers.base import Candidates, Solver

# ---- module-level fixtures/helpers (spawn-picklable) -----------------------


def _rot90(g):
    return symmetry.apply("rot90", g)


def _make_tasks(grid_factory, task_factory, n: int = 6):
    """n deterministic rot90 tasks the DSL solver reliably finds."""
    tasks = {}
    for i in range(n):
        inputs = [grid_factory(3, 3, seed=100 * i + s) for s in range(3)]
        test_in = grid_factory(3, 3, seed=100 * i + 50)
        tasks[f"t{i}"] = task_factory(
            [(g, _rot90(g)) for g in inputs], [test_in], task_id=f"t{i}"
        )
    return tasks


def default_solvers_factory(worker_index: int):
    """model_factory test seam: plain DSL/heuristic ensemble, no model at all."""
    return default_solvers()


class _CrashingSolver(Solver):
    """Raises for every task — simulates a worker whose model/solver
    construction is broken. Used via `_crash_on_worker_1_factory` below."""

    name = "crash_solver"

    def solve(self, task, budget_s: float) -> Candidates:
        raise RuntimeError("simulated worker-1 solver failure")


def _crash_on_worker_1_factory(worker_index: int):
    """Worker 1 gets a solver that raises on every task (still handled inside
    `solve_task`'s per-solver try/except, so the WORKER process itself stays
    alive and simply writes empty/fallback-losing candidates for its shard —
    exercising the "one worker's tasks never solve" degradation path without
    an actual process crash)."""
    if worker_index == 1:
        return [_CrashingSolver()]
    return default_solvers()


class _SlowSolver(Solver):
    """Sleeps well past a tiny total_budget_s so the watchdog must fire."""

    name = "slow_solver"

    def solve(self, task, budget_s: float) -> Candidates:
        time.sleep(2.0)
        return self._empty(task)


def _slow_factory(worker_index: int):
    return [_SlowSolver()]


# ---- (a) merged output == sequential output --------------------------------


def test_run_parallel_matches_sequential(grid_factory, task_factory, tmp_path):
    tasks = _make_tasks(grid_factory, task_factory, n=6)

    seq = run_sequential(dict(tasks), per_task_budget_s=5.0)

    par = run_parallel(
        dict(tasks),
        worker_config={"model_path": None},
        num_workers=2,
        per_task_budget_s=5.0,
        work_dir=tmp_path / "work",
        model_factory=default_solvers_factory,
    )

    assert set(par) == set(seq)
    for task_id in tasks:
        assert par[task_id] == seq[task_id]


# ---- (b) fallback-first / schema-complete afterwards -----------------------


def test_run_parallel_writes_schema_complete_submission(grid_factory, task_factory, tmp_path):
    tasks = _make_tasks(grid_factory, task_factory, n=4)
    out = tmp_path / "submission.json"

    preds = run_parallel(
        dict(tasks),
        worker_config={"model_path": None},
        output_path=out,
        num_workers=2,
        per_task_budget_s=5.0,
        work_dir=tmp_path / "work",
        model_factory=default_solvers_factory,
    )

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert set(loaded) == set(tasks)
    assert validate_submission(build_submission(preds), tasks) == []


# ---- (c) one worker's solver breaks -> submission still schema-complete ----


def test_run_parallel_survives_one_broken_worker(grid_factory, task_factory, tmp_path):
    tasks = _make_tasks(grid_factory, task_factory, n=6)
    out = tmp_path / "submission.json"

    preds = run_parallel(
        dict(tasks),
        worker_config={"model_path": None},
        output_path=out,
        num_workers=2,
        per_task_budget_s=5.0,
        work_dir=tmp_path / "work",
        model_factory=_crash_on_worker_1_factory,
    )

    # Every task_id present and schema-valid, even the ones sharded to the
    # broken worker (they keep the fallback grid because their solver failed
    # on every call, never because the process died).
    assert set(preds) == set(tasks)
    assert validate_submission(build_submission(preds), tasks) == []
    loaded = json.loads(out.read_text())
    assert set(loaded) == set(tasks)


# ---- (d) watchdog terminates promptly --------------------------------------


def test_run_parallel_watchdog_terminates_promptly(grid_factory, task_factory, tmp_path):
    tasks = _make_tasks(grid_factory, task_factory, n=4)
    out = tmp_path / "submission.json"

    t0 = time.monotonic()
    preds = run_parallel(
        dict(tasks),
        worker_config={"model_path": None},
        output_path=out,
        num_workers=2,
        total_budget_s=0.1,
        watchdog_grace_s=0.0,
        checkpoint_every_s=0.05,
        per_task_budget_s=30.0,
        work_dir=tmp_path / "work",
        model_factory=_slow_factory,
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 8.0  # watchdog fired well before the 2s-per-task sleep loop finished
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert set(loaded) == set(tasks)  # schema-complete (fallback grids for un-solved tasks)
    assert validate_submission(build_submission(preds), tasks) == []


# ---- (e) num_workers<=1 delegates to the sequential path -------------------


def test_num_workers_le_1_delegates_to_sequential(grid_factory, task_factory, monkeypatch):
    tasks = _make_tasks(grid_factory, task_factory, n=2)

    calls = []

    def _fake_run_sequential(tasks_arg, **kwargs):
        calls.append((tasks_arg, kwargs))
        return {}

    # run_parallel's num_workers<=1 branch does `from .pipeline import run as
    # run_sequential` locally, so patching the function on arc.pipeline (the
    # module it resolves the name from) is what actually takes effect.
    import arc.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "run", _fake_run_sequential)

    run_parallel(dict(tasks), worker_config={"model_path": None}, num_workers=1)
    assert len(calls) == 1
    assert calls[0][0] == tasks


# ---- (f) torch is never imported at module scope ---------------------------


def test_importing_parallel_does_not_import_torch():
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            del sys.modules[mod]
    sys.modules.pop("arc.parallel", None)

    import arc.parallel  # noqa: F401

    assert "torch" not in sys.modules


# ---- (g) jsonl robustness: truncated last line -----------------------------


def test_merge_tolerates_truncated_jsonl_line(tmp_path, grid_factory, task_factory):
    from arc.parallel import _merge_from_workers

    tasks = _make_tasks(grid_factory, task_factory, n=2)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    good_line = json.dumps(
        {"task_id": "t0", "attempts": [[[[1]], [[1]]]]}
    )
    (work_dir / "worker_0.jsonl").write_text(good_line + "\n" + '{"task_id": "t1", "att')

    merged = _merge_from_workers(dict(tasks), work_dir, num_workers=1)
    assert set(merged) == set(tasks)  # both task_ids present
    assert merged["t0"][0].attempt_1 == ((1,),)  # parsed from the good line
    # t1's truncated line was skipped -> keeps the fallback grid
    from arc.io.submission import FALLBACK_GRID

    assert merged["t1"][0].attempt_1 == FALLBACK_GRID


# ---- (h) kaggle_submit.main(num_workers=...) plumb-through -----------------


def _load_entrypoint():
    path = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_submit.py"
    spec = importlib.util.spec_from_file_location("kaggle_submit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entrypoint_env(monkeypatch, tmp_path, challenges):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    (data / "arc-agi_test_challenges.json").write_text(json.dumps(challenges))
    monkeypatch.setenv("ARC_MODE", "SMOKE")
    monkeypatch.setenv("ARC_DATA_DIR", str(data))
    monkeypatch.setenv("ARC_OUTPUT_DIR", str(out))
    return out / "submission.json"


def test_entrypoint_num_workers_without_model_stays_sequential(tmp_path, monkeypatch):
    # No model_path -> num_workers is accepted (signature + CLI flag exist)
    # but there's no model to parallelize, so the sequential DSL path runs.
    challenges = {
        "t1": {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}],
            "test": [{"input": [[5, 6], [7, 8]]}],
        }
    }
    sub = _entrypoint_env(monkeypatch, tmp_path, challenges)
    mod = _load_entrypoint()
    result = mod.main(num_workers=2)
    assert sub.exists()
    loaded = json.loads(sub.read_text())
    assert loaded["t1"][0]["attempt_1"] and loaded["t1"][0]["attempt_2"]
    assert result["num_tasks"] == 1


def test_entrypoint_has_num_workers_cli_flag():
    mod = _load_entrypoint()
    import inspect

    sig = inspect.signature(mod.main)
    assert "num_workers" in sig.parameters
    assert sig.parameters["num_workers"].default == 0


# ---- model localization (network mount -> local disk, once, parent-side) ----
def test_localize_model_copies_once(tmp_path):
    from arc.parallel import _localize_model

    src = tmp_path / "model"
    src.mkdir()
    (src / "weights.safetensors").write_bytes(b"w" * 64)
    (src / "config.json").write_text("{}")
    cache = tmp_path / "cache"

    local1 = _localize_model(str(src), cache_root=cache)
    assert local1 != str(src)
    assert (Path(local1) / "weights.safetensors").read_bytes() == b"w" * 64
    assert (Path(local1) / ".arc_copy_complete").exists()

    # Second call reuses the completed copy (marker present -> no re-copy).
    before = (Path(local1) / "weights.safetensors").stat().st_mtime_ns
    local2 = _localize_model(str(src), cache_root=cache)
    assert local2 == local1
    assert (Path(local2) / "weights.safetensors").stat().st_mtime_ns == before


def test_localize_model_recovers_from_half_copy(tmp_path):
    from arc.parallel import _localize_model

    src = tmp_path / "model"
    src.mkdir()
    (src / "weights.safetensors").write_bytes(b"good")
    cache = tmp_path / "cache"
    # Simulate a crashed prior copy: dst exists but no completion marker.
    stale = cache / "arc_model_local" / "model"
    stale.mkdir(parents=True)
    (stale / "weights.safetensors").write_bytes(b"trunc")

    local = _localize_model(str(src), cache_root=cache)
    assert (Path(local) / "weights.safetensors").read_bytes() == b"good"


def test_localize_model_falls_back_on_failure(tmp_path):
    from arc.parallel import _localize_model

    src = tmp_path / "model"
    src.mkdir()
    (src / "w").write_bytes(b"x")
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory must go")
    # cache_root is a FILE -> mkdir/copytree fails -> original path returned.
    assert _localize_model(str(src), cache_root=blocker) == str(src)


def test_localize_model_passthrough_for_non_dir(tmp_path):
    from arc.parallel import _localize_model

    assert _localize_model("Qwen/some-hub-id", cache_root=tmp_path) == "Qwen/some-hub-id"
