"""Data-dir resolution: robust on Kaggle, never falls back to a dev path there.

Regression guard for the deploy bug where a not-yet-attached competition dataset
made the Kaggle kernel resolve the data dir to a hardcoded developer-machine
path (a baffling FileNotFoundError pointing at a box that isn't running).
"""

from __future__ import annotations

import arc.config as config


def test_override_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_DATA_DIR", str(tmp_path))
    assert config._detect_data_dir() == tmp_path


def test_offline_uses_local_when_not_kaggle(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_DATA_DIR", raising=False)
    monkeypatch.setattr(config, "_KAGGLE_INPUT", tmp_path / "absent")  # not on Kaggle
    assert config._detect_data_dir() == config._LOCAL_DATA


def test_kaggle_prefers_documented_slug(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_DATA_DIR", raising=False)
    inp = tmp_path / "input"
    slug = inp / "arc-prize-2026-arc-agi-2"
    slug.mkdir(parents=True)
    monkeypatch.setattr(config, "_KAGGLE_INPUT", inp)
    monkeypatch.setattr(config, "_KAGGLE_DATA", slug)
    assert config._detect_data_dir() == slug


def test_kaggle_finds_data_under_nonstandard_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_DATA_DIR", raising=False)
    inp = tmp_path / "input"
    data = inp / "some-other-mount-name"
    data.mkdir(parents=True)
    (data / config.FILE_NAMES["test_challenges"]).write_text("{}")
    monkeypatch.setattr(config, "_KAGGLE_INPUT", inp)
    monkeypatch.setattr(config, "_KAGGLE_DATA", inp / "arc-prize-2026-arc-agi-2")  # absent
    assert config._detect_data_dir() == data


def test_kaggle_missing_data_never_uses_local(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_DATA_DIR", raising=False)
    inp = tmp_path / "input"
    inp.mkdir()  # on Kaggle, but competition data not attached
    documented = inp / "arc-prize-2026-arc-agi-2"
    monkeypatch.setattr(config, "_KAGGLE_INPUT", inp)
    monkeypatch.setattr(config, "_KAGGLE_DATA", documented)
    result = config._detect_data_dir()
    assert result == documented  # a /kaggle path...
    assert result != config._LOCAL_DATA  # ...never the developer machine path
