"""Schema-validation tests for the challenge loader.

A malformed rerun `test_challenges.json` must surface as a clear, task-named
`MalformedTaskError` instead of a bare `KeyError` deep in parsing — and, with
`skip_invalid`, well-formed tasks must still load.
"""

from __future__ import annotations

import json

import pytest

from arc.io.loader import MalformedTaskError, load_challenges


def _write(tmp_path, obj) -> str:
    p = tmp_path / "challenges.json"
    p.write_text(json.dumps(obj))
    return str(p)


def _ok_task():
    return {
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3, 4]]}],
    }


def test_valid_file_loads(tmp_path):
    tasks = load_challenges(_write(tmp_path, {"t": _ok_task()}))
    assert set(tasks) == {"t"}
    assert tasks["t"].num_test == 1
    assert tasks["t"].train[0].output == ((2, 1),)


def test_missing_train_raises_clear_error(tmp_path):
    p = _write(tmp_path, {"t": {"test": [{"input": [[1]]}]}})
    with pytest.raises(MalformedTaskError) as ei:
        load_challenges(p)
    assert "t" in str(ei.value) and "train" in str(ei.value)


def test_pair_missing_input_raises(tmp_path):
    p = _write(tmp_path, {"t": {"train": [{"output": [[1]]}], "test": [{"input": [[1]]}]}})
    with pytest.raises(MalformedTaskError) as ei:
        load_challenges(p)
    assert "input" in str(ei.value)


def test_empty_test_raises(tmp_path):
    p = _write(tmp_path, {"t": {"train": [{"input": [[1]], "output": [[1]]}], "test": []}})
    with pytest.raises(MalformedTaskError):
        load_challenges(p)


def test_top_level_not_object_raises(tmp_path):
    with pytest.raises(MalformedTaskError):
        load_challenges(_write(tmp_path, [1, 2, 3]))


def test_skip_invalid_drops_bad_tasks_only(tmp_path):
    p = _write(tmp_path, {"good": _ok_task(), "bad": {"test": [{"input": [[1]]}]}})
    tasks = load_challenges(p, skip_invalid=True)
    assert set(tasks) == {"good"}  # malformed task dropped, good one kept
