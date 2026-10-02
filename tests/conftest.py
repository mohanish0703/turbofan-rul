from __future__ import annotations

import pytest

from rul.synthetic import generate


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    """Small SYNTHETIC C-MAPSS-format dataset shared by all tests."""
    return generate(tmp_path_factory.mktemp("cmapss"), n_train_units=12, n_test_units=8, seed=123)
