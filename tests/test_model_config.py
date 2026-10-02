"""Hardware-adaptive model config: dtype resolution and TTT-config passthrough.

The GPU itself is unreachable in CI, but the *decisions* (which dtype for which
hardware, which TTTConfig the ensemble is built with) are pure logic — locked in
here so a Kaggle accelerator swap (L4 -> T4) can't silently reintroduce the
bf16-on-Turing failure mode.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from arc.solvers.llm.model import _resolve_dtype


def _fake_torch(available: bool, bf16: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            is_bf16_supported=lambda: bf16,
        )
    )


def test_explicit_override_wins(monkeypatch):
    monkeypatch.delenv("ARC_MODEL_DTYPE", raising=False)
    assert _resolve_dtype(_fake_torch(True, True), "float32") == "float32"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("ARC_MODEL_DTYPE", "float16")
    assert _resolve_dtype(_fake_torch(True, True), None) == "float16"


def test_bf16_on_supported_gpu(monkeypatch):
    monkeypatch.delenv("ARC_MODEL_DTYPE", raising=False)
    assert _resolve_dtype(_fake_torch(True, True), None) == "bfloat16"  # L4/A100


def test_fp16_on_turing(monkeypatch):
    monkeypatch.delenv("ARC_MODEL_DTYPE", raising=False)
    assert _resolve_dtype(_fake_torch(True, False), None) == "float16"  # T4/P100


def test_fp16_when_no_cuda(monkeypatch):
    monkeypatch.delenv("ARC_MODEL_DTYPE", raising=False)
    assert _resolve_dtype(_fake_torch(False, False), None) == "float16"


def test_build_solvers_applies_ttt_config():
    path = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_submit.py"
    spec = importlib.util.spec_from_file_location("kaggle_submit_mc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from arc.solvers.llm import TTTSolver

    dummy_model = SimpleNamespace(model=object())  # LoraTTTRunner stores .model
    solvers = mod._build_solvers(dummy_model, True, {}, {"max_steps": 7})
    ttt = next(s for s in solvers if isinstance(s, TTTSolver))
    assert ttt.runner.config.max_steps == 7  # override reached the runner
