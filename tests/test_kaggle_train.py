"""Rung 4 training entrypoint (scripts/kaggle_train.py): import safety off-GPU,
env-var fallback resolution, and corpus stats. Mirrors `tests/test_kaggle_eval.py`
for loading a standalone script via importlib."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from arc.solvers.llm.ttt_data import TrainExample

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_kaggle_train():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location("kaggle_train", _SCRIPTS / "kaggle_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_import_is_torch_free():
    """Importing the module must not pull in torch — it's only used inside
    finetune(), which this test never calls."""
    _load_kaggle_train()
    assert "torch" not in sys.modules


def test_default_output_dir_local_when_no_kaggle_working(monkeypatch):
    mod = _load_kaggle_train()
    monkeypatch.setattr("os.path.isdir", lambda p: False)
    assert mod._default_output_dir() == "artifacts/adapter"


def test_default_output_dir_kaggle_when_working_dir_present(monkeypatch):
    mod = _load_kaggle_train()
    monkeypatch.setattr("os.path.isdir", lambda p: p == "/kaggle/working")
    assert mod._default_output_dir() == "/kaggle/working/adapter"


def test_corpus_stats_empty():
    mod = _load_kaggle_train()
    assert mod._corpus_stats([]) == {
        "count": 0,
        "mean_prompt_chars": 0.0,
        "mean_completion_chars": 0.0,
    }


def test_corpus_stats_computes_means():
    mod = _load_kaggle_train()
    examples = [
        TrainExample(prompt="ab", completion="xyz"),
        TrainExample(prompt="abcd", completion="x"),
    ]
    stats = mod._corpus_stats(examples)
    assert stats["count"] == 2
    assert stats["mean_prompt_chars"] == 3.0  # (2 + 4) / 2
    assert stats["mean_completion_chars"] == 2.0  # (3 + 1) / 2


def test_main_requires_base_model_path(monkeypatch):
    mod = _load_kaggle_train()
    monkeypatch.delenv("ARC_MODEL_PATH", raising=False)
    monkeypatch.setenv("ARC_CORPUS_PATH", "/tmp/corpus.jsonl")
    try:
        mod.main()
    except ValueError as exc:
        assert "base_model_path" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing base_model_path")


def test_main_requires_corpus_path(monkeypatch):
    mod = _load_kaggle_train()
    monkeypatch.setenv("ARC_MODEL_PATH", "/models/qwen")
    monkeypatch.delenv("ARC_CORPUS_PATH", raising=False)
    try:
        mod.main()
    except ValueError as exc:
        assert "corpus_path" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing corpus_path")


def test_main_resolves_paths_from_env(monkeypatch, tmp_path):
    """env fallback resolution reaches load_examples_jsonl (fails past that
    boundary only because finetune() needs torch — proves the plumbing works
    without needing a GPU)."""
    mod = _load_kaggle_train()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"prompt": "p", "completion": "c"}\n')
    monkeypatch.setenv("ARC_MODEL_PATH", "/models/qwen")
    monkeypatch.setenv("ARC_CORPUS_PATH", str(corpus))
    try:
        mod.main(output_dir=str(tmp_path / "out"))
    except ModuleNotFoundError as exc:
        # Reached finetune()'s lazy `import torch` — env fallback + corpus
        # loading all worked; torch just isn't installed in this venv.
        assert "torch" in str(exc)
    else:
        raise AssertionError("expected ModuleNotFoundError from finetune()'s torch import")


def test_locate_corpus_passthrough_when_exists(tmp_path):
    mod = _load_kaggle_train()
    p = tmp_path / "synth.jsonl"
    p.write_text("{}")
    assert mod._locate_corpus(str(p)) == str(p)


def test_locate_corpus_raises_with_listing_when_missing(tmp_path):
    mod = _load_kaggle_train()
    import pytest

    with pytest.raises(FileNotFoundError) as ei:
        mod._locate_corpus(str(tmp_path / "nope" / "synth_50k.jsonl"))
    assert "Attach the corpus Dataset" in str(ei.value)
