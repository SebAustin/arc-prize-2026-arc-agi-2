"""Rung 4 base-fine-tune hardening: dtype plumbing, subset selection, resume
bookkeeping — all pure-Python helpers extracted from `arc.train.finetune` so
they're testable without torch (mirrors `tests/test_model_config.py`)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from arc.solvers.llm.ttt_data import TrainExample
from arc.train.finetune import (
    TrainConfig,
    _batches_to_skip,
    _load_state,
    _pick_dtype,
    _save_state,
    select_examples,
)


def _fake_torch(available: bool, bf16: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            is_bf16_supported=lambda: bf16,
        )
    )


# ---- dtype plumbing ---------------------------------------------------------


def test_pick_dtype_config_override_wins(monkeypatch):
    monkeypatch.delenv("ARC_MODEL_DTYPE", raising=False)
    cfg = TrainConfig(dtype="float32")
    assert _pick_dtype(cfg, _fake_torch(True, True)) == "float32"


def test_pick_dtype_falls_back_to_hardware_adaptive(monkeypatch):
    monkeypatch.delenv("ARC_MODEL_DTYPE", raising=False)
    cfg = TrainConfig(dtype=None)
    assert _pick_dtype(cfg, _fake_torch(True, True)) == "bfloat16"  # L4/A100
    assert _pick_dtype(cfg, _fake_torch(True, False)) == "float16"  # T4/P100


def test_pick_dtype_env_override(monkeypatch):
    monkeypatch.setenv("ARC_MODEL_DTYPE", "float16")
    cfg = TrainConfig(dtype=None)
    assert _pick_dtype(cfg, _fake_torch(True, True)) == "float16"


# ---- subset selection --------------------------------------------------------


def _examples(n: int) -> list[TrainExample]:
    return [TrainExample(prompt=f"p{i}", completion=f"c{i}") for i in range(n)]


def test_select_examples_returns_all_when_no_cap():
    exs = _examples(10)
    assert select_examples(exs, None, seed=0) == exs


def test_select_examples_returns_all_when_cap_exceeds_len():
    exs = _examples(10)
    assert select_examples(exs, 50, seed=0) == exs


def test_select_examples_deterministic_given_seed():
    exs = _examples(200)
    a = select_examples(exs, 20, seed=7)
    b = select_examples(exs, 20, seed=7)
    assert a == b
    assert len(a) == 20


def test_select_examples_different_seeds_differ():
    exs = _examples(200)
    a = select_examples(exs, 20, seed=1)
    b = select_examples(exs, 20, seed=2)
    assert a != b


def test_select_examples_preserves_relative_order():
    exs = _examples(50)
    sub = select_examples(exs, 10, seed=3)
    prompts = [ex.prompt for ex in sub]
    # Relative order of surviving examples must match the original list —
    # a representative-but-shuffled *subset*, not a reshuffled sequence.
    indices = [int(p[1:]) for p in prompts]
    assert indices == sorted(indices)


# ---- resume bookkeeping ------------------------------------------------------


def test_state_round_trip(tmp_path):
    _save_state(tmp_path, epoch=1, step=42, examples_seen=999)
    state = _load_state(tmp_path)
    assert state == {"epoch": 1, "step": 42, "examples_seen": 999}


def test_state_missing_returns_none(tmp_path):
    assert _load_state(tmp_path / "does-not-exist") is None


@pytest.mark.parametrize(
    ("order_len", "batch_size", "resume_step", "expected"),
    [
        (1000, 8, 10, 10),  # ordinary fast-forward
        (80, 8, 999, 10),  # clamp: only 10 batches exist
        (100, 8, 0, 0),  # fresh epoch, nothing to skip
        (17, 4, 3, 3),  # uneven final batch still counts as a whole batch
    ],
)
def test_batches_to_skip(order_len, batch_size, resume_step, expected):
    assert _batches_to_skip(order_len, batch_size, resume_step) == expected


def test_batches_to_skip_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        _batches_to_skip(100, 0, 5)


def test_grad_clip_default_present():
    # fp16 NaN guard: grad clipping must be on by default (T4 has no bf16).
    from arc.train.finetune import TrainConfig

    assert TrainConfig().grad_clip == 1.0


def test_finetune_accepts_max_train_seconds():
    import inspect

    from arc.train.finetune import finetune
    sig = inspect.signature(finetune)
    assert "max_train_seconds" in sig.parameters


def test_kaggle_train_defaults_max_train_seconds_under_12h():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_train.py"
    spec = importlib.util.spec_from_file_location("kaggle_train_mt", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import inspect
    default = inspect.signature(mod.main).parameters["max_train_seconds"].default
    assert 0 < default < 12 * 3600  # stops+saves before Kaggle's 12h kill
