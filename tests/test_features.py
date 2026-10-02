import numpy as np
import pandas as pd
import pytest

from rul.data import load_dataset
from rul.features import Preprocessor, cap_rul
from rul.synthetic import FLAT_SENSORS

EXPECTED_DROPPED = [f"s_{i}" for i in FLAT_SENSORS]


def test_cap_rul():
    rul = pd.Series([300, 126, 125, 124, 0])
    assert cap_rul(rul, 125).tolist() == [125, 125, 125, 124, 0]
    with pytest.raises(ValueError, match="positive"):
        cap_rul(rul, 0)


@pytest.mark.parametrize(("dataset", "n_conditions"), [("FD001", 1), ("FD002", 6)])
def test_drops_exactly_the_flat_sensors(synthetic_dir, dataset, n_conditions):
    train = load_dataset(synthetic_dir, dataset).train
    pre = Preprocessor(n_conditions=n_conditions).fit(train)
    assert pre.dropped_sensors_ == EXPECTED_DROPPED
    assert set(pre.feature_cols_).isdisjoint(EXPECTED_DROPPED)


def test_multi_condition_variance_needs_clusters(synthetic_dir):
    # Ignoring operating conditions on FD002, flat sensors look like they vary.
    train = load_dataset(synthetic_dir, "FD002").train
    pre = Preprocessor(n_conditions=1).fit(train)
    assert "s_1" not in pre.dropped_sensors_


def test_clusters_recover_operating_conditions(synthetic_dir):
    train = load_dataset(synthetic_dir, "FD002").train
    conditions = Preprocessor(n_conditions=6).fit(train).assign_conditions(train)
    per_cluster_spread = train.assign(c=conditions).groupby("c")["setting_1"].std()
    assert len(per_cluster_spread) == 6
    assert (per_cluster_spread < 0.01).all()


@pytest.mark.parametrize(("dataset", "n_conditions"), [("FD001", 1), ("FD004", 6)])
def test_train_is_standardised_per_condition(synthetic_dir, dataset, n_conditions):
    train = load_dataset(synthetic_dir, dataset).train
    pre = Preprocessor(n_conditions=n_conditions)
    out = pre.fit_transform(train)
    stats = out.groupby("condition")[pre.feature_cols_].agg(["mean", "std"])
    assert np.allclose(stats.xs("mean", axis=1, level=1), 0, atol=1e-4)
    assert np.allclose(stats.xs("std", axis=1, level=1), 1, atol=0.05)


def test_scaler_is_fit_on_train_only(synthetic_dir):
    data = load_dataset(synthetic_dir, "FD002")
    pre = Preprocessor(n_conditions=6).fit(data.train)
    means, stds = pre.means_.copy(), pre.stds_.copy()

    shifted_test = data.test.copy()
    shifted_test[pre.feature_cols_] = shifted_test[pre.feature_cols_] * 3 + 1000
    pre.transform(shifted_test)

    np.testing.assert_array_equal(pre.means_, means)
    np.testing.assert_array_equal(pre.stds_, stds)
    assert pre.n_train_rows_ == len(data.train)


def test_transform_keeps_ids_and_requires_fit(synthetic_dir):
    data = load_dataset(synthetic_dir, "FD001")
    with pytest.raises(RuntimeError, match="not fitted"):
        Preprocessor().transform(data.test)
    pre = Preprocessor().fit(data.train)
    out = pre.transform(data.test)
    assert list(out.columns[:4]) == ["unit", "cycle", "rul", "condition"]
    assert len(out) == len(data.test)
    with pytest.raises(ValueError, match="Missing columns"):
        pre.transform(data.test.drop(columns=["s_2"]))


def test_save_load_roundtrip(synthetic_dir, tmp_path):
    data = load_dataset(synthetic_dir, "FD002")
    pre = Preprocessor(n_conditions=6).fit(data.train)
    pre.save(tmp_path / "pre.joblib")
    loaded = Preprocessor.load(tmp_path / "pre.joblib")
    pd.testing.assert_frame_equal(loaded.transform(data.test), pre.transform(data.test))
