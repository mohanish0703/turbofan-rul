import dataclasses

import numpy as np
import pytest

from rul.data import load_dataset
from rul.features import Preprocessor, cap_rul
from rul.metrics import rmse
from rul.models import MODEL_NAMES, build_model
from rul.models.tabular import summarise_windows
from rul.windows import Windows, last_windows, make_windows


def _one_window(rows, cycle=40, features=("a", "b")):
    x = np.asarray(rows, dtype=np.float32)[None]
    return Windows(X=x, y=None, units=np.array([1]), cycles=np.array([cycle]), features=features)


@pytest.fixture(scope="module")
def windows(synthetic_dir):
    """Training windows (capped target) and last test windows from SYNTHETIC FD001."""
    data = load_dataset(synthetic_dir, "FD001")
    pre = Preprocessor().fit(data.train)
    frame = pre.transform(data.train)
    frame["rul"] = cap_rul(frame["rul"], 125)
    train = make_windows(frame, pre.feature_cols_, 30, target_col="rul")
    test = last_windows(pre.transform(data.test), pre.feature_cols_, 30)
    truth = cap_rul(data.test_rul.loc[test.units].to_numpy(), 125)
    return train, test, truth


def test_summary_features_hand_computed():
    # a rises by 2 per cycle; b is flat, then jumps at the last cycle.
    table, names = summarise_windows(_one_window([[1, 0], [3, 0], [5, 0], [7, 4]]))
    assert names == [
        "a_last", "b_last", "a_mean", "b_mean", "a_std", "b_std", "a_slope", "b_slope", "cycle"
    ]  # fmt: skip
    # std: a deviates +-3, +-1 from its mean 4 -> variance 5; b deviates -1, -1, -1, +3 -> 3.
    # slope vs centred time (-1.5, -0.5, 0.5, 1.5): a -> 2; b -> (1.5 * 4) / 5 = 1.2.
    expected = [7, 4, 4, 1, np.sqrt(5), np.sqrt(3), 2, 1.2, 40]
    np.testing.assert_allclose(table[0], expected, rtol=1e-6)


def test_single_cycle_window_has_zero_slope():
    table, names = summarise_windows(_one_window([[5, 6]]))
    np.testing.assert_array_equal(table[0, [names.index("a_slope"), names.index("b_slope")]], 0)


def test_windows_must_name_their_features():
    with pytest.raises(ValueError, match="names 0 features"):
        summarise_windows(_one_window([[1, 2]], features=()))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_predict_one_value_per_window_in_target_range(name, windows, tiny_cfg):
    train, test, _ = windows
    pred = build_model(name, tiny_cfg()).fit(train).predict(test)
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()
    assert (pred <= 125).all()


def test_naive_predicts_the_training_mean(windows, tiny_cfg):
    train, test, _ = windows
    pred = build_model("naive", tiny_cfg()).fit(train).predict(test)
    np.testing.assert_allclose(pred, train.y.mean(), rtol=1e-5)


@pytest.mark.parametrize("name", ["linear", "random_forest", "xgboost"])
def test_learned_models_beat_naive_on_synthetic_test(name, windows, tiny_cfg):
    train, test, truth = windows
    naive = rmse(truth, build_model("naive", tiny_cfg()).fit(train).predict(test))
    assert rmse(truth, build_model(name, tiny_cfg()).fit(train).predict(test)) < naive


def test_predictions_are_clipped_to_zero_and_the_cap(tiny_cfg):
    # Linear model trained on RUL = cycle (0..99) extrapolates far outside [0, 125].
    cycles = np.arange(100)
    train = Windows(
        X=np.zeros((100, 3, 1), np.float32),
        y=cycles.astype(np.float32),
        units=np.ones(100, dtype=int),
        cycles=cycles,
        features=("a",),
    )
    model = build_model("linear", tiny_cfg()).fit(train)
    far = dataclasses.replace(
        train, X=train.X[:2], y=None, units=np.array([1, 2]), cycles=np.array([-50, 1000])
    )
    np.testing.assert_allclose(model.predict(far), [0, 125])


@pytest.mark.parametrize("name", ["random_forest", "xgboost"])
def test_tree_models_are_deterministic(name, windows, tiny_cfg):
    train, test, _ = windows
    first = build_model(name, tiny_cfg()).fit(train).predict(test)
    second = build_model(name, tiny_cfg()).fit(train).predict(test)
    # Same trees; the random forest sums them across threads, so only rounding (~1e-14) differs.
    np.testing.assert_allclose(first, second, rtol=1e-12)


def test_feature_importances(windows, tiny_cfg):
    train, _, _ = windows
    xgb = build_model("xgboost", tiny_cfg()).fit(train)
    importances = xgb.feature_importances()
    assert set(importances) == set(xgb.feature_names_)
    assert list(importances.values()) == sorted(importances.values(), reverse=True)
    assert build_model("linear", tiny_cfg()).fit(train).feature_importances() == {}


def test_fit_and_predict_guards(windows, tiny_cfg):
    train, test, _ = windows
    model = build_model("linear", tiny_cfg())
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(test)
    with pytest.raises(ValueError, match="need targets"):
        model.fit(test)
    model.fit(train)
    renamed = dataclasses.replace(test, features=tuple(f"x{i}" for i in range(len(test.features))))
    with pytest.raises(ValueError, match="different features"):
        model.predict(renamed)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("svm", tiny_cfg())
