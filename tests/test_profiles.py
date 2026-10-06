"""Run-recipe profiles — the single runtime-chosen T4-safe vs L4-full decision.

Spec for `arc.profiles`: a pure, torch-free module that maps a GPU count (and an
optional explicit override) to the full run recipe (num_workers, budget, ttt_config,
llm_kwargs). The safety invariants (the autopilot/T4 can never be handed the 4-GPU
fan-out; workers never exceed physical GPUs) are the whole point, so they are tested.
"""

from __future__ import annotations

import dataclasses
import sys

import pytest

from arc.profiles import runtime_recipe, select_profile

# The live T4-safe TTT knobs this profile must stay byte-identical to (autopilot path).
_T4_SAFE_TTT = {"batch_size": 1, "max_seq_len": 1536}


def test_two_gpus_is_t4_safe_and_identical_to_today():
    r = select_profile(2)
    assert r.name == "t4_safe"
    assert r.num_workers == 0  # sequential — the autopilot kill-switch
    assert r.per_task_budget_s == 150.0
    assert r.ttt_config == _T4_SAFE_TTT  # byte-identical to the current T4_SAFE_TTT
    assert r.llm_kwargs is None  # -> entrypoint DEFAULT_LLM_KWARGS (current behavior)


def test_four_gpus_auto_selects_l4_full():
    r = select_profile(4)
    assert r.name == "l4_full"
    assert r.num_workers == 4  # one replica per L4 (run_parallel fan-out)
    assert r.per_task_budget_s == 550.0
    assert r.ttt_config.get("max_steps") == 96  # deeper-but-safe knobs
    assert r.ttt_config.get("max_seq_len") == 4096


def test_l4_full_turns_on_nvarc_no_dep_levers():
    """l4_full carries the no-new-deps NVARC method levers: scoreagg selection over
    8 shared augs, fix-black augmentation, and DFS-as-default (with the num_samples/
    temperature the solver's decode='dfs' invariant requires)."""
    llm = select_profile(4).llm_kwargs
    assert llm is not None
    assert llm["selection"] == "scoreagg"
    assert llm["poe_augs"] == 8
    assert llm["keep_zero"] is True
    assert llm["decode"] == "dfs"
    assert llm["num_samples"] == 1 and llm["temperature"] == 0.0


def test_cpu_and_single_gpu_fall_back_to_sequential_t4_safe():
    for dc in (0, 1):
        r = select_profile(dc)
        assert r.name == "t4_safe"
        assert r.num_workers == 0


def test_auto_never_fans_out_below_four_gpus():
    """The core safety invariant: auto mode only picks >1 worker on >=4 GPUs, and
    never asks for more workers than physical cards (run_parallel pins one GPU each)."""
    for dc in range(0, 9):
        r = select_profile(dc)
        if r.num_workers > 1:
            assert dc >= 4, f"dc={dc} fanned out in auto mode"
        assert r.num_workers <= dc, f"dc={dc} asked for {r.num_workers} workers"


def test_override_forces_profile_and_clamps_workers_to_gpus():
    # explicit l4_full on only 2 GPUs is allowed, but clamped to the physical count
    assert select_profile(2, "l4_full").num_workers == 2
    assert select_profile(8, "l4_full").num_workers == 4  # base cap, not 8
    # explicit t4_safe on a big box stays sequential
    assert select_profile(8, "t4_safe").name == "t4_safe"
    assert select_profile(8, "t4_safe").num_workers == 0


def test_unknown_override_raises():
    with pytest.raises(ValueError):
        select_profile(4, "l4_turbo")  # a typo must never silently pick a wrong-cost recipe


def test_recipe_is_immutable():
    r = select_profile(4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.num_workers = 1  # type: ignore[misc]


def test_runtime_recipe_uses_detected_device_count(monkeypatch):
    monkeypatch.setattr("arc.profiles.detect_device_count", lambda: 4)
    assert runtime_recipe().name == "l4_full"
    monkeypatch.setattr("arc.profiles.detect_device_count", lambda: 1)
    assert runtime_recipe().name == "t4_safe"
    # override still wins over detection
    monkeypatch.setattr("arc.profiles.detect_device_count", lambda: 4)
    assert runtime_recipe("t4_safe").name == "t4_safe"


def test_importing_profiles_does_not_import_torch():
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            del sys.modules[mod]
    sys.modules.pop("arc.profiles", None)

    import arc.profiles  # noqa: F401

    assert "torch" not in sys.modules
