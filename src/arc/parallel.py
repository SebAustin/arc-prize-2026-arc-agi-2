"""Multi-process fan-out: N worker processes, each holding one FULL model
replica on its own GPU, instead of one model sharded across every GPU.

On Kaggle's L4x4 (4x24GB) a 7B model fits comfortably on a single L4, so
sharding one replica across all four cards (`HFModel(device_map="auto")`)
wastes 3 of them on communication overhead for a model that didn't need
splitting. Rung 3 instead spawns `num_workers` processes (default 4), each
pinned to one GPU via `CUDA_VISIBLE_DEVICES`, each running its own complete
ensemble over a round-robin shard of the task IDs — turning ~150s/task
(1 model, 4 GPUs busy-waiting on shard boundaries) into ~600s of *effective*
parallel throughput (4 models, each independently solving its own shard).

Design invariants (each is covered by a test in tests/test_parallel.py):
  * The parent writes a complete FALLBACK submission up front — the same
    invariant `arc.pipeline.run` upholds — before any worker is spawned.
  * Each worker appends one JSON line per solved task to its own
    `worker_{i}.jsonl` immediately after solving it (crash-safe incremental
    results: a killed/OOM'd worker never loses previously solved tasks).
  * The parent periodically re-reads every worker's jsonl, merges over the
    fallback, and re-checkpoints the submission file — mirroring
    `arc.pipeline.run`'s time/count-based checkpointing.
  * A global watchdog terminates any still-running workers once the total
    budget (minus a safety grace period) elapses, then does one final merge.
  * `num_workers <= 1` is a kill-switch: delegate straight to the sequential
    `arc.pipeline.run` (no multiprocessing at all).

No torch import at module scope (nor in anything imported here at module
scope): this module must be importable on a CPU dev box with no GPU libs
installed, and worker bodies only import torch (via HFModel) inside the
spawned child process.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DEFAULT_PER_TASK_BUDGET_S, TOTAL_RUNTIME_BUDGET_S
from .io.grid import from_lists, to_lists
from .io.submission import (
    Attempt,
    Predictions,
    build_submission,
    empty_predictions,
    write_submission,
)

if TYPE_CHECKING:
    from .io.loader import Task

_log = logging.getLogger(__name__)

# How often the parent re-reads worker jsonl files and re-checkpoints the
# submission while workers are still running.
DEFAULT_CHECKPOINT_EVERY_S = 60.0
# Safety margin subtracted from total_budget_s before the watchdog fires, so
# terminate() + final merge + write always completes before the hard cap.
DEFAULT_WATCHDOG_GRACE_S = 600.0
# How long the parent blocks per poll iteration waiting on each worker — keeps
# the checkpoint loop responsive instead of blocking indefinitely on join().
_POLL_TIMEOUT_S = 0.5


def _shard(task_ids: list[str], num_workers: int) -> list[list[str]]:
    """Round-robin static sharding of task IDs across `num_workers` shards."""
    shards: list[list[str]] = [[] for _ in range(num_workers)]
    for i, task_id in enumerate(task_ids):
        shards[i % num_workers].append(task_id)
    return shards


def _worker_jsonl_path(work_dir: Path, worker_index: int) -> Path:
    return work_dir / f"worker_{worker_index}.jsonl"


def _localize_model(model_path: str, cache_root: str | Path | None = None) -> str:
    """Copy the model directory to fast local disk ONCE, for all workers.

    Kaggle mounts /kaggle/input over network storage (GCS FUSE): N workers
    concurrently streaming the same ~15GB checkpoint share one pipe and can
    spend 30-60+ min just loading (observed on the first L4x4 run). One
    sequential copy to local scratch, then N local reads, removes the
    contention. Any failure (disk quota, exotic layout) falls back to the
    original path — slower but always correct.
    """
    src = Path(model_path)
    if not src.is_dir():
        return model_path
    root = Path(cache_root) if cache_root is not None else Path(tempfile.gettempdir())
    dst = root / "arc_model_local" / src.name
    marker = dst / ".arc_copy_complete"
    try:
        if not marker.exists():
            if dst.exists():
                shutil.rmtree(dst)  # half-finished copy from a crashed attempt
            _log.info("localizing model %s -> %s", src, dst)
            t0 = time.monotonic()
            shutil.copytree(src, dst)
            marker.touch()
            _log.info("model localized in %.1fs", time.monotonic() - t0)
        return str(dst)
    except Exception as exc:  # noqa: BLE001 — never let the cache sink the run
        _log.warning("model localization failed (%s); using original path", exc)
        shutil.rmtree(dst, ignore_errors=True)
        return model_path


def _default_model_factory(worker_index: int, worker_config: dict):
    """Build this worker's solver list: pin to one GPU, load one full model
    replica, then assemble the ensemble via `arc.solvers.factory.build_solvers`.

    Runs INSIDE the spawned child process, so the torch/transformers imports
    inside `HFModel` never touch the parent or a torch-less dev machine.
    """
    from .pipeline import default_solvers  # noqa: PLC0415
    from .solvers.factory import build_solvers  # noqa: PLC0415

    model_path = worker_config.get("model_path")
    if not model_path:
        return default_solvers()

    # Pin this worker to exactly one physical GPU BEFORE constructing the
    # model, so device_map={"": 0} (this process's only visible device) lands
    # the whole replica on worker_index's card instead of sharding again.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index)
    from .solvers.llm import HFModel  # noqa: PLC0415 — lazy: imports torch

    model = HFModel(
        model_path,
        adapter_path=worker_config.get("adapter_path"),
        device_map={"": 0},
    )
    return build_solvers(
        model,
        use_ttt=worker_config.get("use_ttt", True),
        llm_kwargs=worker_config.get("llm_kwargs") or {},
        ttt_config=worker_config.get("ttt_config"),
    )


def _worker_main(
    worker_index: int,
    tasks: dict[str, Task],
    task_ids: list[str],
    worker_config: dict,
    work_dir: str,
    per_task_budget_s: float,
    model_factory,
) -> None:
    """Spawn-picklable worker entrypoint: solve this shard, appending one JSON
    line per completed task to `work_dir/worker_{worker_index}.jsonl`.

    Module-level (not a closure/lambda) so it survives `multiprocessing`'s
    spawn-context pickling. Any exception solving an individual task is caught
    by `solve_task` itself (never sinks the whole worker); an exception
    building the solvers/model here IS allowed to propagate and end the
    worker process early — the parent's merge-over-fallback covers whatever
    that worker didn't get to.
    """
    from .pipeline import solve_task  # noqa: PLC0415

    solvers = (
        model_factory(worker_index) if model_factory is not None
        else _default_model_factory(worker_index, worker_config)
    )

    out_path = _worker_jsonl_path(Path(work_dir), worker_index)
    with open(out_path, "a", encoding="utf-8") as f:
        for task_id in task_ids:
            task = tasks[task_id]
            attempts = solve_task(task, solvers, per_task_budget_s)
            line = {
                "task_id": task_id,
                "attempts": [
                    [to_lists(a.attempt_1), to_lists(a.attempt_2)] for a in attempts
                ],
            }
            f.write(json.dumps(line) + "\n")
            f.flush()


def _read_worker_jsonl(path: Path) -> dict[str, list[Attempt]]:
    """Parse one worker's incremental results file, tolerating a truncated or
    corrupt final line (the worker may be mid-write when the parent polls)."""
    results: dict[str, list[Attempt]] = {}
    if not path.exists():
        return results
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            attempts = [
                Attempt(from_lists(a1), from_lists(a2)) for a1, a2 in record["attempts"]
            ]
            results[record["task_id"]] = attempts
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Partial write (crash mid-flush) or malformed line: skip it. The
            # task will simply keep its fallback grid until a complete line
            # for it appears on a later poll (or forever, if the worker died).
            continue
    return results


def _merge_from_workers(
    tasks: dict[str, Task], work_dir: Path, num_workers: int
) -> Predictions:
    """Merge every worker's jsonl over a complete fallback set."""
    merged = empty_predictions(tasks)
    for i in range(num_workers):
        merged.update(_read_worker_jsonl(_worker_jsonl_path(work_dir, i)))
    return merged


def run_parallel(
    tasks: dict[str, Task],
    worker_config: dict,
    output_path=None,
    num_workers: int = 4,
    total_budget_s: float = TOTAL_RUNTIME_BUDGET_S,
    per_task_budget_s: float = DEFAULT_PER_TASK_BUDGET_S,
    work_dir: str | Path | None = None,
    model_factory=None,
    checkpoint_every_s: float = DEFAULT_CHECKPOINT_EVERY_S,
    watchdog_grace_s: float = DEFAULT_WATCHDOG_GRACE_S,
) -> Predictions:
    """Solve `tasks` across `num_workers` processes, each a full model replica
    pinned to one GPU, and merge their incremental results into `Predictions`.

    `worker_config` is forwarded to the default per-worker model/solver
    builder: `{"model_path", "adapter_path", "use_ttt", "llm_kwargs",
    "ttt_config"}`. `model_factory`, if given, is a spawn-picklable
    module-level callable `(worker_index) -> list[Solver]` that overrides the
    default HFModel construction entirely — the test seam.

    `num_workers <= 1` delegates straight to `arc.pipeline.run` (sequential
    kill-switch: no multiprocessing spun up at all).
    """
    if num_workers <= 1:
        from .pipeline import run as run_sequential  # noqa: PLC0415

        return run_sequential(
            tasks,
            output_path=output_path,
            total_budget_s=total_budget_s,
            per_task_budget_s=per_task_budget_s,
        )

    # Fallback-first: a complete, schema-valid submission exists before any
    # worker is spawned — identical invariant to arc.pipeline.run.
    predictions = empty_predictions(tasks)
    if output_path is not None:
        write_submission(build_submission(predictions), output_path)

    if not tasks:
        return predictions

    work_root = Path(work_dir) if work_dir is not None else (
        Path(output_path).parent if output_path is not None else Path.cwd()
    )
    work_root.mkdir(parents=True, exist_ok=True)
    # Start from a clean slate: stale jsonl from a previous run in the same
    # work_dir must not leak into this run's merge.
    for i in range(num_workers):
        _worker_jsonl_path(work_root, i).unlink(missing_ok=True)

    # One sequential model copy to local disk beats N workers contending for
    # the network mount (see _localize_model). Parent-side so it happens once.
    if model_factory is None and worker_config.get("model_path"):
        worker_config = {
            **worker_config,
            "model_path": _localize_model(worker_config["model_path"]),
        }

    shards = _shard(list(tasks.keys()), num_workers)
    ctx = get_context("spawn")
    procs = []
    for i, shard_ids in enumerate(shards):
        if not shard_ids:
            continue
        p = ctx.Process(
            target=_worker_main,
            args=(
                i,
                tasks,
                shard_ids,
                worker_config,
                str(work_root),
                per_task_budget_s,
                model_factory,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    start = time.monotonic()
    deadline = start + max(0.0, total_budget_s - watchdog_grace_s)
    last_checkpoint = start
    try:
        while any(p.is_alive() for p in procs):
            if time.monotonic() > deadline:
                _log.warning(
                    "watchdog: total budget exhausted, terminating %d live worker(s)",
                    sum(p.is_alive() for p in procs),
                )
                for p in procs:
                    if p.is_alive():
                        p.terminate()
                break
            for p in procs:
                p.join(_POLL_TIMEOUT_S)
            now = time.monotonic()
            if now - last_checkpoint >= checkpoint_every_s:
                predictions = _merge_from_workers(tasks, work_root, num_workers)
                if output_path is not None:
                    write_submission(build_submission(predictions), output_path)
                last_checkpoint = now
    finally:
        # Always reap: terminate() only requests death, join() confirms it, so
        # a terminated-but-not-yet-dead worker cannot leave a zombie process.
        for p in procs:
            p.join(timeout=5.0)

    predictions = _merge_from_workers(tasks, work_root, num_workers)
    if output_path is not None:
        write_submission(build_submission(predictions), output_path)
    return predictions
