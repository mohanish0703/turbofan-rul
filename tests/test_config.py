from pathlib import Path

import pytest
from pydantic import ValidationError

from rul.config import config_for, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_and_dataset_override():
    assert config_for("FD001", config_dir=CONFIG_DIR).preprocess.n_conditions == 1
    cfg = config_for("FD002", config_dir=CONFIG_DIR)
    assert cfg.preprocess.n_conditions == 6
    assert cfg.preprocess.rul_cap == 125
    assert cfg.preprocess.window_length == 30


def test_smoke_points_at_synthetic_data():
    cfg = config_for("FD001", extra=CONFIG_DIR / "smoke.yaml", config_dir=CONFIG_DIR)
    assert "synthetic" in str(cfg.data.raw_dir)


def test_invalid_values_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("preprocess:\n  rul_cap: 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_typos_rejected(tmp_path):
    bad = tmp_path / "typo.yaml"
    bad.write_text("preprocess:\n  window_lenght: 50\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)
