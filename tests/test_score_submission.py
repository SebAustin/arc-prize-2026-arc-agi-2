"""scripts/score_submission.py — score a competition submission.json on a split.

Locks the ANCHOR scorer: NVARC's notebook output must score on exactly our canary
yardstick (exact-match + partial credit), with malformed third-party entries
tolerated and an id-mismatch surfaced loudly rather than silently reading 0.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "score_submission", _SCRIPTS / "score_submission.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_split(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    challenges = {
        "a": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[1, 2], [3, 4]]}]},
        "b": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[0]]}]},
    }
    solutions = {"a": [[[1, 2], [3, 4]]], "b": [[[9]]]}
    (data / "arc-agi_evaluation_challenges.json").write_text(json.dumps(challenges))
    (data / "arc-agi_evaluation_solutions.json").write_text(json.dumps(solutions))
    monkeypatch.setenv("ARC_MODE", "SMOKE")
    monkeypatch.setenv("ARC_DATA_DIR", str(data))
    monkeypatch.setenv("ARC_OUTPUT_DIR", str(tmp_path / "out"))


def test_scores_exact_match_against_eval_solutions(tmp_path, monkeypatch):
    _eval_split(tmp_path, monkeypatch)
    mod = _load()
    submission = {
        "a": [{"attempt_1": [[1, 2], [3, 4]], "attempt_2": [[0, 0], [0, 0]]}],  # exact in attempt_1
        "b": [{"attempt_1": [[1]], "attempt_2": [[2]]}],                        # right shape, wrong
    }
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(submission))

    summary = mod.main(str(path))
    assert summary["correct"] == 1 and summary["total"] == 2
    assert summary["score"] == 0.5
    assert summary["scored_tasks"] == 2
    assert summary["missing_from_submission"] == 0


def test_parse_skips_malformed_entries():
    mod = _load()
    raw = {
        "ok": [{"attempt_1": [[1]], "attempt_2": [[2]]}],
        "missing_attempt_2": [{"attempt_1": [[1]]}],  # dropped
        "not_a_list": {"x": 1},                        # dropped
    }
    preds = mod.parse_submission(raw)
    assert set(preds) == {"ok"}


def test_missing_file_exits_cleanly(tmp_path):
    """A nonexistent path (e.g. the runbook's placeholder before any run exists)
    must give a clear message, not a raw FileNotFoundError traceback."""
    import pytest

    mod = _load()
    with pytest.raises(SystemExit) as exc:
        mod.main(str(tmp_path / "does_not_exist.json"))
    assert "no submission file" in str(exc.value)


def test_id_mismatch_scores_zero_and_warns(tmp_path, monkeypatch, capsys):
    _eval_split(tmp_path, monkeypatch)
    mod = _load()
    path = tmp_path / "submission.json"
    path.write_text(json.dumps({"zzz": [{"attempt_1": [[1]], "attempt_2": [[2]]}]}))

    summary = mod.main(str(path))
    assert summary["scored_tasks"] == 0
    assert summary["missing_from_submission"] == 2
    assert "0 tasks scored" in capsys.readouterr().out
