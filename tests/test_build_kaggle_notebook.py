"""scripts/build_kaggle_notebook.py: both notebooks build, share the bootstrap
machinery correctly, and the pre-existing submission notebook's shape
(cell count/types) is unchanged by the Rung 4 training-notebook addition."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_builder():
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "build_kaggle_notebook", _SCRIPTS / "build_kaggle_notebook.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _embedded_files(cell) -> dict[str, str]:
    src = "".join(cell["source"])
    match = re.search(r"FILES = json\.loads\(r'''(.*)'''\)", src, re.S)
    assert match is not None, "bootstrap cell must embed a FILES payload"
    return json.loads(match.group(1))


def test_build_submission_has_five_cells(tmp_path, monkeypatch):
    mod = _load_builder()
    out = tmp_path / "submission.ipynb"
    monkeypatch.setattr(mod, "SUBMISSION_OUT", out)
    path = mod.build_submission()
    nb = json.loads(path.read_text())
    assert path == out
    assert len(nb["cells"]) == 5
    assert [c["cell_type"] for c in nb["cells"]] == [
        "markdown",
        "code",
        "code",
        "code",
        "code",
    ]


def test_build_submission_embeds_submission_entrypoints(tmp_path, monkeypatch):
    mod = _load_builder()
    out = tmp_path / "submission.ipynb"
    monkeypatch.setattr(mod, "SUBMISSION_OUT", out)
    nb = json.loads(mod.build_submission().read_text())
    files = _embedded_files(nb["cells"][2])
    assert "scripts/kaggle_submit.py" in files
    assert "scripts/kaggle_eval.py" in files
    assert "scripts/kaggle_train.py" not in files  # training code stays out


def test_build_train_adapter_has_five_cells(tmp_path, monkeypatch):
    mod = _load_builder()
    out = tmp_path / "train_adapter.ipynb"
    monkeypatch.setattr(mod, "TRAIN_OUT", out)
    path = mod.build_train_adapter()
    nb = json.loads(path.read_text())
    assert path == out
    assert len(nb["cells"]) == 5
    assert [c["cell_type"] for c in nb["cells"]] == [
        "markdown",
        "code",
        "code",
        "code",
        "code",
    ]


def test_build_train_adapter_embeds_kaggle_train(tmp_path, monkeypatch):
    mod = _load_builder()
    out = tmp_path / "train_adapter.ipynb"
    monkeypatch.setattr(mod, "TRAIN_OUT", out)
    nb = json.loads(mod.build_train_adapter().read_text())
    files = _embedded_files(nb["cells"][2])
    assert "scripts/kaggle_train.py" in files
    assert "scripts/kaggle_submit.py" not in files  # inference code stays out
    # The corpus itself must NOT be embedded — it's a staged Kaggle Dataset.
    assert not any("synth_50k" in k or "artifacts/" in k for k in files)


def test_train_adapter_mentions_resume_and_duration(tmp_path, monkeypatch):
    mod = _load_builder()
    out = tmp_path / "train_adapter.ipynb"
    monkeypatch.setattr(mod, "TRAIN_OUT", out)
    nb = json.loads(mod.build_train_adapter().read_text())
    intro = "".join(nb["cells"][0]["source"])
    assert "resume" in intro.lower()
    assert "duration" in intro.lower() or "h for the full" in intro.lower()


def test_build_all_writes_both_notebooks(tmp_path, monkeypatch):
    mod = _load_builder()
    sub_out = tmp_path / "submission.ipynb"
    train_out = tmp_path / "train_adapter.ipynb"
    monkeypatch.setattr(mod, "SUBMISSION_OUT", sub_out)
    monkeypatch.setattr(mod, "TRAIN_OUT", train_out)
    paths = mod.build_all()
    assert set(paths) == {sub_out, train_out}
    assert sub_out.exists()
    assert train_out.exists()


def test_collect_files_shared_by_both_builders_includes_src_package():
    mod = _load_builder()
    files = mod._collect_files(("kaggle_submit.py",))
    assert any(k.startswith("src/arc/") and k.endswith(".py") for k in files)
    assert "scripts/kaggle_submit.py" in files
