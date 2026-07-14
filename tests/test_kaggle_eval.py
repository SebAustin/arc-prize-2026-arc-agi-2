"""The public-eval canary instrument (scripts/kaggle_eval.py), CPU/DSL-only.

Locks the measurement loop every rung promotion depends on: correct per-task
PASS/FAIL, offset/limit slicing, and a scored summary.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_eval():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))  # kaggle_eval imports kaggle_submit
    spec = importlib.util.spec_from_file_location("kaggle_eval", _SCRIPTS / "kaggle_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_env(monkeypatch, tmp_path):
    """Tiny evaluation split: one DSL-solvable (rot180) task, one unsolvable."""
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    challenges = {
        "solvable": {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
                {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
            ],
            "test": [{"input": [[1, 0], [0, 2]]}],
        },
        "unsolvable": {
            "train": [{"input": [[1]], "output": [[7, 7], [7, 7]]}],
            "test": [{"input": [[2]]}],
        },
    }
    solutions = {
        "solvable": [[[2, 0], [0, 1]]],  # rot180 of the test input
        "unsolvable": [[[9, 9], [9, 9]]],  # nothing in the ensemble finds this
    }
    (data / "arc-agi_evaluation_challenges.json").write_text(json.dumps(challenges))
    (data / "arc-agi_evaluation_solutions.json").write_text(json.dumps(solutions))
    monkeypatch.setenv("ARC_MODE", "SMOKE")
    monkeypatch.setenv("ARC_DATA_DIR", str(data))
    monkeypatch.setenv("ARC_OUTPUT_DIR", str(out))


def test_eval_scores_dsl_ensemble(tmp_path, monkeypatch, capsys):
    _eval_env(monkeypatch, tmp_path)
    summary = _load_eval().main(per_task_budget_s=5.0, limit=40)
    assert summary["split"] == "evaluation"
    assert summary["num_tasks"] == 2
    assert summary["correct"] == 1  # rot180 solved, decoy not
    out = capsys.readouterr().out
    assert "solvable  PASS" in out
    assert "unsolvable  fail" in out


def test_eval_offset_and_limit_slice(tmp_path, monkeypatch):
    _eval_env(monkeypatch, tmp_path)
    summary = _load_eval().main(per_task_budget_s=5.0, limit=1, offset=1)
    assert summary["num_tasks"] == 1  # second task only


def test_eval_limit_none_scores_full_split(tmp_path, monkeypatch):
    # The widened-canary contract: limit=None means "score every task in the
    # split" (islice treats a None stop as to-the-end), not a capped slice.
    _eval_env(monkeypatch, tmp_path)
    summary = _load_eval().main(per_task_budget_s=5.0, limit=None)
    assert summary["num_tasks"] == 2  # both tasks, no cap
    assert summary["limit"] is None
