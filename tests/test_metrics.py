import math

import numpy as np
import pytest

from rul.metrics import all_metrics, mae, nasa_score, rmse


def test_hand_computed_values():
    y_true = [100, 100, 50]
    y_pred = [87, 110, 50]  # d = -13 (early), +10 (late), 0 (exact)
    # NASA: (e^(13/13) - 1) + (e^(10/10) - 1) + 0 = 2(e - 1)
    assert nasa_score(y_true, y_pred) == pytest.approx(2 * (math.e - 1))
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt((169 + 100 + 0) / 3))
    assert mae(y_true, y_pred) == pytest.approx(23 / 3)


def test_nasa_single_terms():
    assert nasa_score([10], [10]) == 0.0
    assert nasa_score([30], [4]) == pytest.approx(math.exp(2) - 1)  # d = -26
    assert nasa_score([0], [20]) == pytest.approx(math.exp(2) - 1)  # d = +20


def test_nasa_penalises_late_more_than_early():
    late = nasa_score([50], [70])
    early = nasa_score([50], [30])
    assert late > early


def test_perfect_prediction_is_zero():
    y = np.array([1.0, 20.0, 125.0])
    assert all_metrics(y, y) == {"rmse": 0.0, "mae": 0.0, "nasa_score": 0.0}


def test_bad_inputs():
    with pytest.raises(ValueError, match="Shape mismatch"):
        rmse([1, 2], [1])
    with pytest.raises(ValueError, match="Empty"):
        mae([], [])
