from __future__ import annotations

from pathlib import Path

import pytest

from rul.config import config_for
from rul.synthetic import generate

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    """Small SYNTHETIC C-MAPSS-format dataset shared by all tests."""
    return generate(tmp_path_factory.mktemp("cmapss"), n_train_units=12, n_test_units=8, seed=123)


@pytest.fixture(scope="session")
def tiny_cfg(synthetic_dir):
    """Config factory for fast runs on the SYNTHETIC data: 3 folds, tiny tree ensembles."""

    def make(dataset: str = "FD001"):
        return config_for(
            dataset,
            config_dir=CONFIG_DIR,
            overrides={
                "data": {"raw_dir": str(synthetic_dir)},
                "cv": {"n_folds": 3},
                "models": {"random_forest": {"n_estimators": 10}, "xgboost": {"n_estimators": 20}},
            },
        )

    return make
