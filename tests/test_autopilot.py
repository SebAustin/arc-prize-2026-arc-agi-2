"""scripts/daily_autopilot.py: the tick() state machine, fully offline.

Every Kaggle-touching call goes through `FakeKaggleClient` (records calls,
scriptable status/push/output/submit/submissions) or, for the real
`KaggleClient`'s CLI-text parsing, a monkeypatched `_run` that returns a
canned `subprocess.CompletedProcess` — neither path ever shells out to the
real `kaggle` binary or touches the network. Filesystem touch points (state
file, SUBMISSION_LOG.md, kernel-run staging, notebook build) are redirected
under `tmp_path` via the `autopilot` fixture, so nothing here mutates the
real repo tree.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_autopilot():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))  # daily_autopilot imports sibling scripts/modules
    autopilot_path = _SCRIPTS / "daily_autopilot.py"
    spec = importlib.util.spec_from_file_location("daily_autopilot", autopilot_path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: `PushResult`/`Submission` are dataclasses under
    # `from __future__ import annotations`, and dataclass's postponed-annotation
    # resolution looks the module up via `sys.modules[cls.__module__]` — it
    # must already be registered by the time the class body executes.
    sys.modules["daily_autopilot"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_notebook_builder(tmp_path: Path, filename: str):
    """A fast stand-in for `build_kaggle_notebook.build_submission`/
    `build_train_adapter` — same 5-cell shape, none of the real base64
    embedding, so kernel-folder tests never touch the real notebooks/ dir."""

    def _build() -> Path:
        nb = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": ["intro\n"]},
                {
                    "cell_type": "code", "metadata": {}, "outputs": [],
                    "execution_count": None, "source": ["env prep\n"],
                },
                {
                    "cell_type": "code", "metadata": {}, "outputs": [],
                    "execution_count": None, "source": ["bootstrap\n"],
                },
                {
                    "cell_type": "code", "metadata": {}, "outputs": [],
                    "execution_count": None, "source": ["config\n"],
                },
                {
                    "cell_type": "code", "metadata": {}, "outputs": [],
                    "execution_count": None, "source": ["run\n"],
                },
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path = tmp_path / filename
        path.write_text(json.dumps(nb))
        return path

    return _build


@pytest.fixture
def autopilot(tmp_path, monkeypatch):
    """A freshly loaded `daily_autopilot` module with every filesystem touch
    point redirected under `tmp_path`."""
    mod = _load_autopilot()
    ak = sys.modules["autopilot_kernels"]
    monkeypatch.setattr(mod, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(mod, "SUBMISSION_LOG_PATH", tmp_path / "SUBMISSION_LOG.md")
    monkeypatch.setattr(mod, "KERNEL_RUNS_DIR", tmp_path / "kernel_runs")
    monkeypatch.setattr(ak, "KERNEL_RUNS_DIR", tmp_path / "kernel_runs")
    monkeypatch.setattr(
        ak, "build_submission", _stub_notebook_builder(tmp_path, "submission.ipynb")
    )
    monkeypatch.setattr(
        ak, "build_train_adapter", _stub_notebook_builder(tmp_path, "train_adapter.ipynb")
    )
    return mod


@dataclass
class FakeKaggleClient:
    """Records every call; each behavior is scripted per test via the fields
    below (never shells out, never touches the network)."""

    status_queue: dict = field(default_factory=dict)  # kernel -> [status, ...] (last repeats)
    push_queue: list = field(default_factory=list)  # [PushResult, ...] (last repeats)
    output_files: dict = field(default_factory=dict)  # kernel -> {relative_path: text}
    submissions_result: list = field(default_factory=list)
    submit_ok: bool = True
    dataset_ok: bool = True
    calls: list = field(default_factory=list)

    def kernel_status(self, kernel):
        self.calls.append(("kernel_status", kernel))
        queue = self.status_queue.get(kernel) or ["COMPLETE"]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def kernel_push(self, folder, accelerator="NvidiaTeslaT4"):
        self.calls.append(("kernel_push", str(folder), accelerator))
        assert self.push_queue, "test must seed push_queue before a push can happen"
        return self.push_queue.pop(0) if len(self.push_queue) > 1 else self.push_queue[0]

    def kernel_output(self, kernel, dest_dir):
        self.calls.append(("kernel_output", kernel, str(dest_dir)))
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for rel, text in self.output_files.get(kernel, {}).items():
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return dest

    def dataset_create_or_version(self, folder):
        self.calls.append(("dataset_create_or_version", str(folder)))
        return self.dataset_ok

    def competition_submit(self, kernel, version, message):
        self.calls.append(("competition_submit", kernel, version, message))
        return self.submit_ok

    def competition_submissions(self):
        self.calls.append(("competition_submissions",))
        return self.submissions_result


# ---------------------------------------------------------------------------
# (a) train COMPLETE -> adapter staged -> candidate set -> next tick pushes eval
# ---------------------------------------------------------------------------


def test_train_complete_stages_adapter_and_sets_candidate(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["inflight"] = {
        "kernel": mod.TRAIN_KERNEL,
        "version": 3,
        "kind": "train",
        "purpose": "backlog:adapter_hard_2500",
        "config": dict(state["live_config"]),
    }
    client = FakeKaggleClient(
        status_queue={mod.TRAIN_KERNEL: ["COMPLETE"]},
        output_files={mod.TRAIN_KERNEL: {"adapter/adapter_model.safetensors": "fake-weights"}},
    )
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["inflight"] is None
    assert new_state["candidate"]["adapter_dataset"] == mod.ADAPTER_DATASET_SLUG
    assert new_state["candidate"]["config"]["adapter_path"] == mod.ADAPTER_MOUNT
    assert any(c[0] == "dataset_create_or_version" for c in client.calls)


def test_candidate_set_pushes_eval_next_tick(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["candidate"] = {
        "config": {**state["live_config"], "adapter_path": mod.ADAPTER_MOUNT},
        "adapter_dataset": mod.ADAPTER_DATASET_SLUG,
    }
    push = mod.PushResult(ok=True, busy=False, version=11, error=None)
    client = FakeKaggleClient(push_queue=[push])
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["inflight"]["kind"] == "eval"
    assert new_state["inflight"]["version"] == 11
    assert new_state["inflight"]["kernel"] == mod.SUBMISSION_KERNEL
    assert any(c[0] == "kernel_push" for c in client.calls)


def test_train_complete_with_no_adapter_file_discards(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["inflight"] = {
        "kernel": mod.TRAIN_KERNEL, "version": 3, "kind": "train",
        "purpose": "backlog:adapter_hard_2500", "config": dict(state["live_config"]),
    }
    busy = mod.PushResult(ok=False, busy=True, version=None, error="busy")
    client = FakeKaggleClient(
        status_queue={mod.TRAIN_KERNEL: ["COMPLETE"]},
        output_files={},
        push_queue=[busy],  # the now-idle tick may try an explore launch; stand down
    )
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["inflight"] is None
    assert new_state["candidate"] is None
    assert not any(c[0] == "dataset_create_or_version" for c in client.calls)


# ---------------------------------------------------------------------------
# (b) eval COMPLETE: correct > best -> gated_candidate; correct <= best -> discard
# ---------------------------------------------------------------------------


def test_eval_complete_with_improvement_sets_gated_candidate(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["best_eval_correct"] = 10
    cfg = {**state["live_config"], "adapter_path": mod.ADAPTER_MOUNT}
    state["candidate"] = {"config": cfg, "adapter_dataset": mod.ADAPTER_DATASET_SLUG}
    state["inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL, "version": 7, "kind": "eval",
        "purpose": "measure candidate", "config": cfg,
    }
    summary_text = "summary: {'score': 0.3, 'correct': 12, 'total': 40, 'num_tasks': 40}\n"
    client = FakeKaggleClient(
        status_queue={mod.SUBMISSION_KERNEL: ["COMPLETE"]},
        output_files={mod.SUBMISSION_KERNEL: {"eval_summary.txt": summary_text}},
    )
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["candidate"] is None
    assert new_state["inflight"] is None
    assert new_state["gated_candidate"]["eval_correct"] == 12
    assert new_state["gated_candidate"]["version"] == 7


def test_eval_complete_without_improvement_discards_candidate(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["best_eval_correct"] = 10
    cfg = {**state["live_config"], "adapter_path": mod.ADAPTER_MOUNT}
    state["candidate"] = {"config": cfg, "adapter_dataset": mod.ADAPTER_DATASET_SLUG}
    state["inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL, "version": 7, "kind": "eval",
        "purpose": "measure candidate", "config": cfg,
    }
    summary_text = "summary: {'score': 0.2, 'correct': 9, 'total': 40, 'num_tasks': 40}\n"
    busy = mod.PushResult(ok=False, busy=True, version=None, error="busy")
    client = FakeKaggleClient(
        status_queue={mod.SUBMISSION_KERNEL: ["COMPLETE"]},
        output_files={mod.SUBMISSION_KERNEL: {"eval_summary.txt": summary_text}},
        push_queue=[busy],  # the now-idle tick may try an explore launch; stand down
    )
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["candidate"] is None
    assert new_state["inflight"] is None
    assert new_state["gated_candidate"] is None
    # Discarded from the GATED path — but the adapter candidate keeps its
    # exploration lottery ticket (Kaggle ranks the best submission).
    assert [v["name"] for v in new_state["extra_exploration_variants"]] == ["adapter-eval-v7"]


def test_eval_complete_with_no_summary_discards_candidate(autopilot):
    mod = autopilot
    state = mod.default_state()
    cfg = {**state["live_config"], "adapter_path": mod.ADAPTER_MOUNT}
    state["candidate"] = {"config": cfg, "adapter_dataset": mod.ADAPTER_DATASET_SLUG}
    state["inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL, "version": 7, "kind": "eval",
        "purpose": "measure candidate", "config": cfg,
    }
    busy = mod.PushResult(ok=False, busy=True, version=None, error="busy")
    client = FakeKaggleClient(
        status_queue={mod.SUBMISSION_KERNEL: ["COMPLETE"]},
        output_files={},
        push_queue=[busy],  # the now-idle tick may try an explore launch; stand down
    )
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["candidate"] is None
    assert new_state["gated_candidate"] is None


# ---------------------------------------------------------------------------
# (c) gated + not-submitted-today -> submit_commit pushed; COMPLETE -> submit
# ---------------------------------------------------------------------------


def test_gated_candidate_pushes_submit_commit(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["gated_candidate"] = {"config": state["live_config"], "version": 7, "eval_correct": 15}
    state["last_submit_date"] = "2026-07-11"
    push = mod.PushResult(ok=True, busy=False, version=21, error=None)
    client = FakeKaggleClient(push_queue=[push])
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["inflight"]["kind"] == "submit_commit"
    assert new_state["inflight"]["version"] == 21
    assert new_state["gated_candidate"] is not None  # not yet cleared — clears on COMPLETE


def test_submit_commit_complete_calls_competition_submit(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["gated_candidate"] = {"config": state["live_config"], "version": 7, "eval_correct": 15}
    state["inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL, "version": 21, "kind": "submit_commit",
        "purpose": "commit-run the gated candidate", "config": state["gated_candidate"]["config"],
    }
    client = FakeKaggleClient(status_queue={mod.SUBMISSION_KERNEL: ["COMPLETE"]})
    new_state = mod.tick(client, state, "2026-07-12")

    submit_calls = [c for c in client.calls if c[0] == "competition_submit"]
    assert len(submit_calls) == 1
    assert submit_calls[0][1] == mod.SUBMISSION_KERNEL
    assert submit_calls[0][2] == 21
    assert new_state["pending_lb"] is not None
    assert new_state["pending_lb"]["eval_correct"] == 15
    assert new_state["last_submit_date"] == "2026-07-12"
    assert new_state["best_eval_correct"] == 15
    assert new_state["gated_candidate"] is None
    assert new_state["inflight"] is None


def test_submit_commit_failure_keeps_gated_candidate_for_retry(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["gated_candidate"] = {"config": state["live_config"], "version": 7, "eval_correct": 15}
    state["inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL, "version": 21, "kind": "submit_commit",
        "purpose": "commit-run the gated candidate", "config": state["gated_candidate"]["config"],
    }
    client = FakeKaggleClient(status_queue={mod.SUBMISSION_KERNEL: ["COMPLETE"]}, submit_ok=False)
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["inflight"] is None
    assert new_state["pending_lb"] is None
    assert new_state["gated_candidate"] is not None  # retry candidate next tick
    assert new_state["last_submit_date"] != "2026-07-12"


# ---------------------------------------------------------------------------
# (d) already submitted today -> no second submit
# ---------------------------------------------------------------------------


def test_no_second_submit_same_day(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["gated_candidate"] = {"config": state["live_config"], "version": 7, "eval_correct": 15}
    state["last_submit_date"] = "2026-07-12"
    # Falls through to the backlog branch instead — seed a push result so that
    # branch can complete without error; the point under test is that neither
    # a submit_commit push nor a competition_submit call happens.
    push = mod.PushResult(ok=True, busy=False, version=99, error=None)
    client = FakeKaggleClient(push_queue=[push])
    new_state = mod.tick(client, state, "2026-07-12")

    assert not any(c[0] == "competition_submit" for c in client.calls)
    assert new_state["gated_candidate"] == state["gated_candidate"]
    if new_state["inflight"] is not None:
        assert new_state["inflight"]["kind"] != "submit_commit"


# ---------------------------------------------------------------------------
# (e) push returns busy -> inflight unchanged, no crash
# ---------------------------------------------------------------------------


def test_backlog_push_busy_is_not_a_crash_and_state_is_unchanged(autopilot):
    mod = autopilot
    state = mod.default_state()
    busy = mod.PushResult(
        ok=False, busy=True, version=None,
        error="Maximum batch GPU session count of 2 reached.",
    )
    client = FakeKaggleClient(push_queue=[busy])
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["inflight"] is None
    assert new_state["backlog_index"] == state["backlog_index"]
    assert new_state["gpu_hours_week"] == state["gpu_hours_week"]


def test_eval_push_busy_leaves_candidate_and_inflight_alone(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["candidate"] = {
        "config": {**state["live_config"], "adapter_path": mod.ADAPTER_MOUNT},
        "adapter_dataset": mod.ADAPTER_DATASET_SLUG,
    }
    busy = mod.PushResult(
        ok=False, busy=True, version=None,
        error="Maximum batch GPU session count of 2 reached.",
    )
    client = FakeKaggleClient(push_queue=[busy])
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["inflight"] is None
    assert new_state["candidate"] == state["candidate"]


# ---------------------------------------------------------------------------
# (f) pending_lb reconciliation: better LB -> promote, worse LB -> reject
# ---------------------------------------------------------------------------


def test_pending_lb_promotes_on_better_score(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["best_lb"] = 0.83
    new_cfg = {**state["live_config"], "llm_kwargs": {"selection": "poe"}}
    ref = "owner/kernel:v5:2026-07-10"
    state["pending_lb"] = {"ref": ref, "config": new_cfg, "eval_correct": 20}
    # Reconciliation falls through to a backlog launch afterwards (no
    # inflight/candidate/gated_candidate pending) — seed a push result so
    # that branch doesn't crash; it's not what this test is about.
    client = FakeKaggleClient(
        submissions_result=[
            mod.Submission(
                ref=ref, date="2026-07-11", description=f"autopilot {ref} correct=20",
                public_score="0.90", status="complete",
            )
        ],
        push_queue=[mod.PushResult(ok=True, busy=False, version=1, error=None)],
    )
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["pending_lb"] is None
    assert new_state["best_lb"] == 0.90
    assert new_state["live_config"] == new_cfg


def test_pending_lb_rejects_on_worse_score(autopilot):
    mod = autopilot
    state = mod.default_state()
    original_cfg = state["live_config"]
    state["best_lb"] = 0.83
    new_cfg = {**state["live_config"], "llm_kwargs": {"selection": "poe"}}
    ref = "owner/kernel:v6:2026-07-10"
    state["pending_lb"] = {"ref": ref, "config": new_cfg, "eval_correct": 5}
    client = FakeKaggleClient(
        submissions_result=[
            mod.Submission(
                ref=ref, date="2026-07-11", description=f"autopilot {ref} correct=5",
                public_score="0.50", status="complete",
            )
        ],
        push_queue=[mod.PushResult(ok=True, busy=False, version=1, error=None)],
    )
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["pending_lb"] is None
    assert new_state["best_lb"] == 0.83
    assert new_state["live_config"] == original_cfg


def test_pending_lb_waits_when_not_yet_scored(autopilot):
    mod = autopilot
    state = mod.default_state()
    ref = "owner/kernel:v7:2026-07-12"
    state["pending_lb"] = {"ref": ref, "config": state["live_config"], "eval_correct": 20}
    client = FakeKaggleClient(
        submissions_result=[],
        push_queue=[mod.PushResult(ok=True, busy=False, version=1, error=None)],
    )
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["pending_lb"] == state["pending_lb"]


# ---------------------------------------------------------------------------
# (g) quota exhausted -> no new backlog launch
# ---------------------------------------------------------------------------


def test_quota_exhausted_blocks_backlog_launch(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["gpu_hours_week"] = mod.WEEKLY_GPU_HOUR_QUOTA
    state["week_start"] = "2026-07-12"  # same day as `today` -> no weekly reset
    client = FakeKaggleClient()
    new_state = mod.tick(client, state, "2026-07-12")

    assert new_state["inflight"] is None
    assert not any(c[0] == "kernel_push" for c in client.calls)


def test_week_rollover_resets_quota_after_seven_days(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["week_start"] = "2026-07-01"
    state["gpu_hours_week"] = 20.0
    busy = mod.PushResult(ok=False, busy=True, version=None, error="busy")
    client = FakeKaggleClient(push_queue=[busy])
    new_state = mod.tick(client, state, "2026-07-12")  # 11 days later

    assert new_state["week_start"] == "2026-07-12"
    assert new_state["gpu_hours_week"] == 0.0


# ---------------------------------------------------------------------------
# (h) state round-trips through save/load
# ---------------------------------------------------------------------------


def test_state_round_trips_through_save_and_load(autopilot, tmp_path):
    mod = autopilot
    state = mod.default_state()
    state["gated_candidate"] = {"config": {"a": 1}, "version": 3, "eval_correct": 9}
    state["inflight"] = {
        "kernel": "k", "version": 1, "kind": "eval", "purpose": "p", "config": {"x": 1},
    }
    path = tmp_path / "state_roundtrip.json"

    mod.save_state(state, path)
    loaded = mod.load_state(path)

    assert loaded == state


def test_load_state_defaults_when_missing(autopilot, tmp_path):
    mod = autopilot
    loaded = mod.load_state(tmp_path / "does_not_exist.json")
    assert loaded == mod.default_state()


def test_load_state_defaults_when_corrupt(autopilot, tmp_path):
    mod = autopilot
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")
    loaded = mod.load_state(path)
    assert loaded == mod.default_state()


def test_load_state_backfills_new_keys(autopilot, tmp_path):
    """An older, partial state file must not lose any fields on load."""
    mod = autopilot
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"best_eval_correct": 4}), encoding="utf-8")
    loaded = mod.load_state(path)
    assert loaded["best_eval_correct"] == 4
    assert loaded["best_lb"] == mod.default_state()["best_lb"]


# ---------------------------------------------------------------------------
# tick() bookkeeping — SUBMISSION_LOG.md status line
# ---------------------------------------------------------------------------


def test_tick_appends_one_line_status_to_submission_log(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["gpu_hours_week"] = mod.WEEKLY_GPU_HOUR_QUOTA
    client = FakeKaggleClient()
    mod.tick(client, state, "2026-07-12")

    text = mod.SUBMISSION_LOG_PATH.read_text(encoding="utf-8")
    assert "# Submission Log" in text
    assert "2026-07-12" in text
    assert "quota exhausted" in text


# ---------------------------------------------------------------------------
# KaggleClient — defensive CLI-text parsing, monkeypatching `_run` so no
# subprocess is ever spawned and the real `kaggle` binary is never invoked.
# ---------------------------------------------------------------------------


def _fake_run(stdout: str = "", stderr: str = "", returncode: int = 0):
    def _run(self, args):
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)

    return _run


def test_kernel_push_parses_version_on_success(autopilot, monkeypatch):
    mod = autopilot
    stdout = "Kernel version 42 successfully pushed.  Please check progress at https://x\n"
    monkeypatch.setattr(mod.KaggleClient, "_run", _fake_run(stdout=stdout))
    result = mod.KaggleClient().kernel_push("some/folder")

    assert result.ok
    assert result.version == 42
    assert not result.busy


def test_kernel_push_detects_busy_from_gpu_session_error(autopilot, monkeypatch):
    mod = autopilot
    stdout = "Kernel push error: Maximum batch GPU session count of 2 reached.\n"
    monkeypatch.setattr(mod.KaggleClient, "_run", _fake_run(stdout=stdout))
    result = mod.KaggleClient().kernel_push("some/folder")

    assert result.busy
    assert not result.ok
    assert result.version is None


def test_kernel_push_reports_other_errors(autopilot, monkeypatch):
    mod = autopilot
    stdout = "Kernel push error: some other kernel-metadata problem\n"
    monkeypatch.setattr(mod.KaggleClient, "_run", _fake_run(stdout=stdout))
    result = mod.KaggleClient().kernel_push("some/folder")

    assert not result.ok
    assert not result.busy
    assert "kernel-metadata" in result.error


def test_kernel_status_parses_complete(autopilot, monkeypatch):
    mod = autopilot
    monkeypatch.setattr(
        mod.KaggleClient, "_run", _fake_run(stdout='owner/kernel has status "complete"\n')
    )
    assert mod.KaggleClient().kernel_status("owner/kernel") == "COMPLETE"


def test_kernel_status_unrecognized_output_is_unknown(autopilot, monkeypatch):
    mod = autopilot
    monkeypatch.setattr(mod.KaggleClient, "_run", _fake_run(stdout="garbled nonsense\n"))
    assert mod.KaggleClient().kernel_status("owner/kernel") == "UNKNOWN"


def test_kernel_status_404_is_unknown_not_error(autopilot, monkeypatch):
    # "404 Client Error" contains the substring "ERROR" — the phantom-404 case
    # (transient after a push; any no-active-session kernel since Kaggle's
    # ~2026-07-14 API change) must NOT be misread as a run failure, or healthy
    # runs get abandoned on the spot. UNKNOWN flows into the stale-guard.
    mod = autopilot
    stderr = (
        "404 Client Error: Not Found for url: "
        "https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus\n"
    )
    monkeypatch.setattr(mod.KaggleClient, "_run", _fake_run(stderr=stderr, returncode=1))
    assert mod.KaggleClient().kernel_status("owner/kernel") == "UNKNOWN"


def test_competition_submissions_parses_json(autopilot, monkeypatch):
    mod = autopilot
    row = {
        "ref": "123", "date": "2026-07-01", "description": "d",
        "publicScore": "0.5", "status": "complete",
    }
    payload = json.dumps([row])
    monkeypatch.setattr(mod.KaggleClient, "_run", _fake_run(stdout=payload))
    subs = mod.KaggleClient().competition_submissions()

    assert len(subs) == 1
    assert subs[0].ref == "123"
    assert subs[0].public_score == "0.5"


def test_competition_submissions_returns_empty_on_garbage(autopilot, monkeypatch):
    mod = autopilot
    monkeypatch.setattr(mod.KaggleClient, "_run", _fake_run(stdout="not json at all"))
    assert mod.KaggleClient().competition_submissions() == []


def test_dry_run_client_never_sets_inflight(autopilot):
    mod = autopilot
    state = mod.default_state()
    client = mod.DryRunKaggleClient()
    new_state = mod.tick(client, state, "2026-07-12")
    assert new_state["inflight"] is None


# ---------------------------------------------------------------------------
# Kernel-folder builders + backlog shape
# ---------------------------------------------------------------------------


def test_write_submission_kernel_metadata_has_no_machine_shape(autopilot, tmp_path):
    ak = sys.modules["autopilot_kernels"]
    folder = tmp_path / "kernel_folder"
    ak.write_submission_kernel(folder, "print('hi')\n", dataset_sources=["owner/ds"])
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert "machine_shape" not in meta
    assert "docker_image" not in meta
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is False
    assert meta["dataset_sources"] == ["owner/ds"]


def test_write_train_kernel_metadata_has_no_competition_sources(autopilot, tmp_path):
    ak = sys.modules["autopilot_kernels"]
    folder = tmp_path / "train_kernel_folder"
    ak.write_train_kernel(folder, "print('train')\n", dataset_sources=["owner/corpus"])
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert "machine_shape" not in meta
    assert "docker_image" not in meta
    assert meta["competition_sources"] == []
    assert meta["dataset_sources"] == ["owner/corpus"]


def _kaggle_slugify(title: str) -> str:
    """Kaggle's title->slug rule (lowercase; non-alphanumeric runs -> hyphen)."""
    import re

    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def test_kernel_titles_slugify_to_their_kernel_ids(autopilot, tmp_path):
    # Kaggle hard-rejects (409, enforced ~2026-07-14) a push whose title does
    # not slug-resolve to the kernel id — this killed the 07-14 08:00 tick.
    ak = sys.modules["autopilot_kernels"]
    sub = tmp_path / "sub_folder"
    train = tmp_path / "train_folder"
    ak.write_submission_kernel(sub, "print('run')\n")
    ak.write_train_kernel(train, "print('train')\n")

    for folder in (sub, train):
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        id_slug = meta["id"].split("/")[-1]
        assert _kaggle_slugify(meta["title"]) == id_slug, meta


def test_explicit_kernel_title_still_wins(autopilot, tmp_path):
    ak = sys.modules["autopilot_kernels"]
    folder = tmp_path / "custom_title_folder"
    ak.write_submission_kernel(folder, "print('run')\n", title="my custom title")
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["title"] == "my custom title"


def test_backlog_items_have_expected_shape(autopilot):
    ak = sys.modules["autopilot_kernels"]
    names = [item["name"] for item in ak.BACKLOG]

    assert names == ["adapter_hard_2500", "poe_regate", "ttt_steps_sweep", "adapter_hard_6000"]
    for item in ak.BACKLOG:
        assert item["kind"] in ("train", "eval")
        assert callable(item["build"])
        assert callable(item["config"])
        assert item["estimated_hours"] > 0


def test_backlog_round_robins_and_wraps(autopilot):
    mod = autopilot
    ak = sys.modules["autopilot_kernels"]
    state = mod.default_state()
    kernels_seen = []
    for i, _ in enumerate(ak.BACKLOG * 2):
        push = mod.PushResult(ok=True, busy=False, version=i, error=None)
        client = FakeKaggleClient(push_queue=[push])
        state = mod.tick(client, state, "2026-07-12")
        kernels_seen.append(state["inflight"]["purpose"])
        state["inflight"] = None  # simulate the run finishing/clearing without full dispatch
        state["gpu_hours_week"] = 0.0  # isolate cursor-wrapping from quota (tested separately)

    expected_names = [f"backlog:{item['name']}" for item in ak.BACKLOG]
    assert kernels_seen == expected_names * 2


def test_adapter_train_is_first_backlog_item(autopilot):
    # The "adapter training runs next" invariant: a fresh cursor (backlog_index=0)
    # must launch the hard-corpus adapter TRAIN, not a cheap eval.
    ak = sys.modules["autopilot_kernels"]
    first = ak.BACKLOG[0]
    assert first["name"] == "adapter_hard_2500"
    assert first["kind"] == "train"


def test_eval_run_src_defaults_to_full_public_split(autopilot):
    # Widened canary: with no eval_limit in the config, the generated run cell
    # must ask kaggle_eval for the FULL split (limit=None), not the old 40-slice.
    ak = sys.modules["autopilot_kernels"]
    src = ak.eval_run_src({})
    assert "'limit': None" in src
    assert "'limit': 40" not in src


def test_eval_run_src_respects_explicit_eval_limit(autopilot):
    # A config may still pin a cheaper slice (e.g. a smoke run) — the override wins.
    ak = sys.modules["autopilot_kernels"]
    src = ak.eval_run_src({"eval_limit": 40})
    assert "'limit': 40" in src


# ---------------------------------------------------------------------------
# push-failure retry semantics: a transient failure (e.g. the 2026-07-14 409)
# must NOT knock a priority item out of line; 3 strikes still bounds it.
# ---------------------------------------------------------------------------


def test_push_failure_retries_same_backlog_item(autopilot):
    mod = autopilot
    state = mod.default_state()
    push = mod.PushResult(ok=False, busy=False, version=None, error="409 Conflict")
    client = FakeKaggleClient(push_queue=[push])
    state = mod.tick(client, state, "2026-07-14")

    assert state["backlog_index"] == 0  # NOT advanced — same item retries next tick
    assert state["push_failures"] == 1
    assert state["inflight"] is None


def test_push_failure_skips_item_after_three_strikes(autopilot):
    mod = autopilot
    state = mod.default_state()
    push = mod.PushResult(ok=False, busy=False, version=None, error="409 Conflict")
    for day in ("2026-07-14", "2026-07-15", "2026-07-16"):
        client = FakeKaggleClient(push_queue=[push])
        state = mod.tick(client, state, day)

    assert state["backlog_index"] == 1  # skipped only after 3 consecutive failures
    assert state["push_failures"] == 0  # counter reset for the next item


def test_push_success_resets_failure_counter(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["push_failures"] = 2
    push = mod.PushResult(ok=True, busy=False, version=9, error=None)
    client = FakeKaggleClient(push_queue=[push])
    state = mod.tick(client, state, "2026-07-14")

    assert state["inflight"]["purpose"] == "backlog:adapter_hard_2500"
    assert state["push_failures"] == 0


def test_busy_push_does_not_count_as_failure(autopilot):
    mod = autopilot
    state = mod.default_state()
    push = mod.PushResult(ok=False, busy=True, version=None, error="2 GPU sessions")
    client = FakeKaggleClient(push_queue=[push])
    state = mod.tick(client, state, "2026-07-14")

    assert state["push_failures"] == 0
    assert state["backlog_index"] == 0


# ---------------------------------------------------------------------------
# exploration submissions — the second (submit-commit-only) lane that stops
# the daily submission slot going unused while the eval gate can't fire
# ---------------------------------------------------------------------------


def test_exploration_variants_order_and_extras(autopilot):
    ak = sys.modules["autopilot_kernels"]
    state = {
        "live_config": {"llm_kwargs": {"selection": "votes"}, "ttt_config": None},
        "extra_exploration_variants": [{"name": "adapter-eval-v9", "config": {"a": 1}}],
    }
    variants = ak.exploration_variants(state)
    assert [v["name"] for v in variants] == ["poe", "ttt96", "adapter-eval-v9"]
    assert variants[0]["config"]["llm_kwargs"]["selection"] == "poe"
    assert variants[1]["config"]["ttt_config"]["max_steps"] == 96


def test_explore_commit_launches_alongside_running_train(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["inflight"] = {
        "kernel": mod.TRAIN_KERNEL,
        "version": 8,
        "kind": "train",
        "purpose": "backlog:adapter_hard_2500",
        "config": {},
        "since": "2026-07-14",
    }
    push = mod.PushResult(ok=True, busy=False, version=21, error=None)
    client = FakeKaggleClient(status_queue={mod.TRAIN_KERNEL: ["RUNNING"]}, push_queue=[push])
    state = mod.tick(client, state, "2026-07-14")

    assert state["inflight"]["kind"] == "train"  # main slot untouched
    assert state["explore_inflight"]["variant"] == "poe"
    assert state["explore_inflight"]["kernel"] == mod.SUBMISSION_KERNEL
    assert "poe" in state["explored"]  # tried is marked at launch


def test_explore_skipped_when_main_slot_uses_submission_kernel(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL,
        "version": 14,
        "kind": "eval",
        "purpose": "backlog:poe_regate",
        "config": {},
        "since": "2026-07-14",
    }
    client = FakeKaggleClient(status_queue={mod.SUBMISSION_KERNEL: ["RUNNING"]})
    state = mod.tick(client, state, "2026-07-14")

    assert state["explore_inflight"] is None
    assert all(call[0] != "kernel_push" for call in client.calls)


def test_explore_complete_submits_and_sets_pending_lb(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["explore_inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL,
        "version": 21,
        "variant": "poe",
        "config": {"llm_kwargs": {"selection": "poe"}},
    }
    state["explored"] = ["poe"]
    busy = mod.PushResult(ok=False, busy=True, version=None, error="busy")
    client = FakeKaggleClient(
        status_queue={mod.SUBMISSION_KERNEL: ["COMPLETE"]}, push_queue=[busy]
    )
    state = mod.tick(client, state, "2026-07-15")

    assert state["explore_inflight"] is None
    assert state["last_submit_date"] == "2026-07-15"
    assert state["pending_lb"]["ref"].endswith(":2026-07-15")
    submits = [c for c in client.calls if c[0] == "competition_submit"]
    assert submits and submits[0][1] == mod.SUBMISSION_KERNEL and submits[0][2] == 21


def test_explore_respects_one_submission_per_day(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["last_submit_date"] = "2026-07-15"
    push = mod.PushResult(ok=True, busy=False, version=9, error=None)
    client = FakeKaggleClient(push_queue=[push])
    state = mod.tick(client, state, "2026-07-15")

    assert state["inflight"]["kind"] == "train"  # backlog still launches
    assert state["explore_inflight"] is None  # but no second submission today
    assert state["explored"] == []


def test_explore_rotation_exhausted_no_launch(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["explored"] = ["poe", "ttt96"]
    push = mod.PushResult(ok=True, busy=False, version=9, error=None)
    client = FakeKaggleClient(push_queue=[push])
    state = mod.tick(client, state, "2026-07-16")

    assert state["explore_inflight"] is None


def test_explore_error_clears_slot_and_variant_stays_tried(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["explore_inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL,
        "version": 21,
        "variant": "poe",
        "config": {},
    }
    state["explored"] = ["poe"]
    busy = mod.PushResult(ok=False, busy=True, version=None, error="busy")
    client = FakeKaggleClient(
        status_queue={mod.SUBMISSION_KERNEL: ["ERROR"]}, push_queue=[busy]
    )
    state = mod.tick(client, state, "2026-07-15")

    assert state["explore_inflight"] is None
    assert "poe" in state["explored"]  # no unattended retry loop
    assert all(c[0] != "competition_submit" for c in client.calls)


def test_gate_failed_adapter_candidate_queued_for_exploration(autopilot):
    mod = autopilot
    state = mod.default_state()
    adapter_config = {
        "model_path": "m",
        "adapter_path": "/kaggle/input/datasets/x/adapter",
        "adapter_dataset": "x/adapter",
    }
    state["candidate"] = {"config": adapter_config, "adapter_dataset": "x/adapter"}
    state["inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL,
        "version": 13,
        "kind": "eval",
        "purpose": "measure trained candidate",
        "config": adapter_config,
        "since": "2026-07-15",
    }
    busy = mod.PushResult(ok=False, busy=True, version=None, error="busy")
    client = FakeKaggleClient(
        status_queue={mod.SUBMISSION_KERNEL: ["COMPLETE"]},
        output_files={
            mod.SUBMISSION_KERNEL: {
                "eval_summary.txt": "summary: {'correct': 0, 'num_tasks': 120}"
            }
        },
        push_queue=[busy],
    )
    state = mod.tick(client, state, "2026-07-15")

    extras = state["extra_exploration_variants"]
    assert [v["name"] for v in extras] == ["adapter-eval-v13"]
    assert extras[0]["config"]["adapter_dataset"] == "x/adapter"
    ak = sys.modules["autopilot_kernels"]
    assert "adapter-eval-v13" in [v["name"] for v in ak.exploration_variants(state)]


# ---------------------------------------------------------------------------
# rotating submission-kernel id — Kaggle busy-create burn bug (2026-07-14):
# a create push rejected on the session cap half-creates a write-dead kernel;
# the id must rotate, never strike the backlog item, never consume a variant.
# ---------------------------------------------------------------------------


def test_submission_kernel_for_rotates_by_seq(autopilot):
    ak = sys.modules["autopilot_kernels"]
    assert ak.submission_kernel_for({}) == ak.SUBMISSION_KERNEL
    assert ak.submission_kernel_for({"submission_kernel_seq": 0}) == ak.SUBMISSION_KERNEL
    assert ak.submission_kernel_for({"submission_kernel_seq": 3}) == f"{ak.SUBMISSION_KERNEL}-3"


def test_busy_create_burns_and_rotates_then_next_tick_uses_new_id(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["inflight"] = {
        "kernel": mod.TRAIN_KERNEL, "version": 1, "kind": "train",
        "purpose": "backlog:adapter_hard_2500", "config": {}, "since": "2026-07-14",
    }
    busy = mod.PushResult(ok=False, busy=True, version=None, error="2 GPU sessions")
    client = FakeKaggleClient(status_queue={mod.TRAIN_KERNEL: ["RUNNING"]}, push_queue=[busy])
    state = mod.tick(client, state, "2026-07-14")

    assert state["submission_kernel_seq"] == 1  # busy-on-create burned the id
    assert state["explore_inflight"] is None
    assert state["explored"] == []  # the variant's ticket was NOT consumed

    ok = mod.PushResult(ok=True, busy=False, version=1, error=None)
    client2 = FakeKaggleClient(status_queue={mod.TRAIN_KERNEL: ["RUNNING"]}, push_queue=[ok])
    state = mod.tick(client2, state, "2026-07-14")

    assert state["explore_inflight"]["kernel"] == f"{mod.SUBMISSION_KERNEL}-1"
    assert state["submission_kernel_created"] is True


def test_created_submission_kernel_busy_does_not_rotate(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["submission_kernel_created"] = True
    state["inflight"] = {
        "kernel": mod.TRAIN_KERNEL, "version": 1, "kind": "train",
        "purpose": "backlog:adapter_hard_2500", "config": {}, "since": "2026-07-14",
    }
    busy = mod.PushResult(ok=False, busy=True, version=None, error="2 GPU sessions")
    client = FakeKaggleClient(status_queue={mod.TRAIN_KERNEL: ["RUNNING"]}, push_queue=[busy])
    state = mod.tick(client, state, "2026-07-14")

    assert state["submission_kernel_seq"] == 0  # ordinary busy retry, no burn


def test_notebook_not_found_on_eval_backlog_rotates_without_strike(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["backlog_index"] = 1  # poe_regate (kind=eval, pushes the submission kernel)
    fail = mod.PushResult(ok=False, busy=False, version=None,
                          error="Kernel push error: Notebook not found")
    client = FakeKaggleClient(push_queue=[fail])
    new_state, msg = mod._maybe_launch_backlog(client, state)

    assert new_state["submission_kernel_seq"] == 1
    assert new_state["backlog_index"] == 1  # item keeps its place in line
    assert new_state["push_failures"] == 0  # a burned id is not the item's fault
    assert "rotated" in msg


def test_rotation_helper_ignores_ordinary_failures(autopilot):
    mod = autopilot
    state = mod.default_state()
    fail = mod.PushResult(ok=False, busy=False, version=None, error="some metadata problem")
    assert mod._rotate_burned_submission_kernel(state, fail) is None


def test_eval_launch_defers_while_explore_holds_submission_kernel(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["candidate"] = {"config": {"model_path": "m"}, "adapter_dataset": None}
    state["explore_inflight"] = {
        "kernel": mod.SUBMISSION_KERNEL,
        "version": 21,
        "variant": "poe",
        "config": {},
    }
    client = FakeKaggleClient(status_queue={mod.SUBMISSION_KERNEL: ["RUNNING"]})
    state = mod.tick(client, state, "2026-07-15")

    assert state["inflight"] is None  # eval deferred, not pushed
    assert state["candidate"] is not None  # candidate preserved for next tick
    assert all(call[0] != "kernel_push" for call in client.calls)


# ---------------------------------------------------------------------------
# staleness guard: an unresolved inflight is abandoned after STALE_INFLIGHT_DAYS
# (protects the UNATTENDED loop from a phantom kernel — e.g. a 404 session)
# ---------------------------------------------------------------------------


def test_waiting_inflight_gets_since_stamp(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["inflight"] = {
        "kernel": mod.TRAIN_KERNEL, "version": 8, "kind": "train",
        "purpose": "x", "config": dict(state["live_config"]),
    }
    busy = mod.PushResult(ok=False, busy=True, version=None, error="busy")
    client = FakeKaggleClient(status_queue={mod.TRAIN_KERNEL: ["UNKNOWN"]}, push_queue=[busy])
    new_state = mod.tick(client, state, "2026-07-12")
    assert new_state["inflight"] is not None          # still waiting
    assert new_state["inflight"]["since"] == "2026-07-12"  # first-observed stamped


def test_stale_inflight_is_abandoned(autopilot):
    mod = autopilot
    state = mod.default_state()
    state["inflight"] = {
        "kernel": mod.TRAIN_KERNEL, "version": 8, "kind": "train",
        "purpose": "x", "config": dict(state["live_config"]),
        "since": "2026-07-10",  # 2 days before `today` -> stale
    }
    busy = mod.PushResult(ok=False, busy=True, version=None, error="busy")
    client = FakeKaggleClient(status_queue={mod.TRAIN_KERNEL: ["UNKNOWN"]}, push_queue=[busy])
    new_state = mod.tick(client, state, "2026-07-12")
    assert new_state["inflight"] is None  # abandoned, loop unwedged
